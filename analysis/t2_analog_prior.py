"""Task 2 onset: a STATE-CONDITIONED prior (analog / nearest-neighbour onset map) instead of only the static one.

Motivation (notes 17-21, 44, user's suggestion): the location base rate carries a lot of information (dropping
pos/priors hurts, 98-99% of onsets sit at train-frequent bottlenecks), but the static train frequency breaks under the
run shift; capacities are stable and sub-1% demand differences decide WHICH candidate breaks down. So condition the
prior on today's demand state: find train onset windows whose demand state resembles the current window's and take
their T+30 onset maps.

Demand state (history only, T-60..T-5): r = log(flow / train mean flow of the same weekday type, slot and link).
Two keys, compared:
    cand   per candidate bottleneck (train's 10 most frequent first-onset run ends), the mean r over the links within
           +-2 of it in the last 30 min, and its change (last 15 min - first 15 min); plus time of day
    pca    PCA (8 comps, fitted on the train library) of the last-30-min mean r over all links; plus time of day
Library: the train earliest-onset windows (the onset model's own training windows) of the same weekday type, with their
T+30 onset sets. Same-day library windows are excluded for train rows (as at scoring time).
Analog prior per link: p_a = (sum_j w_j y_ji + ALPHA p_g_i) / (sum_j w_j + ALPHA), K nearest, Gaussian weights; p_g is
the library frequency. Features: p_a, p_a smoothed over +-1 link, its rank in the window, p_a / (p_g + 0.01).
Zeroed for a random half of the training windows (like the static prior dropout, note 20).
Paired LightGBM in H's configuration (prior dropout, 5 bags): base / +cand / +pca, 3 seeds, scored on the blank-aware
official-like onset windows with masked histories and with T1-filled histories (the official histories are complete,
note 46).

    .venv/Scripts/python.exe analysis/t2_analog_prior.py
"""
from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "scripts"))
import t2_prior_robust  # noqa: E402
from t2_eda02 import T2_PANELS  # noqa: E402
from t2_eda03 import F, H  # noqa: E402
from t2_expected_iou import select  # noqa: E402
from t2_first_onset_eval import first_onsets, official_blank, smooth_links  # noqa: E402
from t2_m1_models import Ctx, hist_from_truth  # noqa: E402
from t2_m1_predict import feats_on, table_prior  # noqa: E402
from t2_onset_geometry import runs  # noqa: E402
from t2_pseudo_eval import iou  # noqa: E402
from t2_shift_remedies import PARAMS  # noqa: E402
from tfb.data import load  # noqa: E402
from tfb.t1eval import family_mean  # noqa: E402

SEEDS, N_BAGS, K, ALPHA, N_CAND, N_PC = (0, 1, 2), 5, 20, 2.0, 10, 8
A_NAMES = ["a_p", "a_p_s", "a_rank", "a_lift"]
VARIANTS = ("base", "cand", "pca")
DAY_SETS = {"d1-14": 14, "all": 31}
rng = np.random.default_rng(31)


def fmean(d):
    d = {k: v for k, v in d.items() if np.isfinite(v)}
    return family_mean(d) if d else float("nan")


# ------------------------------------------------------------------ demand state
def train_profile(ctx) -> np.ndarray:
    """(2, T, S) mean train flow per weekday type, slot and link (eligible cells)."""
    q = np.where(ctx.el, ctx.z["true_flow"], np.nan)
    with np.errstate(all="ignore"):
        prof = np.stack([np.nanmean(q[ctx.g == g], 0) for g in (0, 1)])
    return np.nan_to_num(prof, nan=np.nanmean(prof)).clip(1.0, None).astype(np.float32)


def log_ratio(prof, qf, ok, t0, grp) -> np.ndarray:
    """(N, H, S) log(flow / profile) on observed cells, NaN elsewhere."""
    hi = t0[:, None] + np.arange(-H, 0)
    ref = prof[grp[:, None], hi]                                               # (N, H, S)
    with np.errstate(all="ignore"):
        return np.where(ok & np.isfinite(qf), np.log(np.maximum(qf, 1.0) / ref), np.nan)


def cand_key(r, cands, t0, S) -> np.ndarray:
    with np.errstate(all="ignore"):
        last, a, b = np.nanmean(r[:, -6:], 1), np.nanmean(r[:, -3:], 1), np.nanmean(r[:, :3], 1)   # (N, S)
    cols = []
    for e in cands:
        blk = slice(max(0, e - 2), min(S, e + 3))
        cols.append(np.nanmean(last[:, blk], 1))
        cols.append(np.nanmean(a[:, blk], 1) - np.nanmean(b[:, blk], 1))
    cols.append(t0 / 12.0)
    return np.nan_to_num(np.stack(cols, 1), nan=0.0).astype(np.float32)


def pca_fit(r):
    with np.errstate(all="ignore"):
        X = np.nan_to_num(np.nanmean(r[:, -6:], 1), nan=0.0)
    mu = X.mean(0)
    _, _, vt = np.linalg.svd(X - mu, full_matrices=False)
    return mu, vt[:N_PC]


def pca_key(r, pca, t0) -> np.ndarray:
    with np.errstate(all="ignore"):
        X = np.nan_to_num(np.nanmean(r[:, -6:], 1), nan=0.0)
    return np.concatenate([(X - pca[0]) @ pca[1].T, (t0 / 12.0)[:, None]], 1).astype(np.float32)


def analog(lib, qkey, qgrp, qday=None, return_dist=False):
    """(N, S) analog onset probability. lib: dict(key, y, grp, day, pg, sd). With return_dist, also (N,) the mean
    standardised distance to the K neighbours (how far today's state is from the train library)."""
    out = np.empty((len(qkey), lib["y"].shape[1]), np.float32)
    dist = np.full(len(qkey), np.nan, np.float32)
    kz = lib["key"] / lib["sd"]
    for n in range(len(qkey)):
        ok = lib["grp"] == qgrp[n]
        if qday is not None:
            ok &= lib["day"] != qday[n]
        idx = np.flatnonzero(ok)
        if len(idx) == 0:
            out[n] = lib["pg"]
            continue
        d = np.sqrt(((kz[idx] - qkey[n] / lib["sd"]) ** 2).sum(1))
        nn = idx[np.argsort(d)[:K]]
        dn = np.sort(d)[:K]
        dist[n] = dn.mean()
        h = max(float(np.median(dn)), 1e-6)
        w = np.exp(-(dn / h) ** 2)
        out[n] = (w @ lib["y"][nn] + ALPHA * lib["pg"]) / (w.sum() + ALPHA)
    return (out, dist) if return_dist else out


def a_feats(pa, pg) -> np.ndarray:
    S = pa.shape[1]
    ps = smooth_links(pa) / 2.0
    rank = np.argsort(np.argsort(-pa, 1), 1) / max(S - 1, 1)
    return np.stack([pa, ps, rank, pa / (pg[None] + 0.01)], -1).astype(np.float32)   # (N, S, 4)


def filled(z, panel, split):
    with np.load(ROOT / "work" / "sub_m3ttt" / f"pred_{split}_{panel}.npz") as f:
        pv, pq = f["pv"], f["pq"]
    hole = z["elig"] & ~(np.isfinite(z["obs_speed"]) & np.isfinite(z["obs_flow"]))
    zs = dict(z)
    zs["obs_speed"] = np.where(hole, pv, z["obs_speed"]).astype(np.float32)
    zs["obs_flow"] = np.where(hole, pq, z["obs_flow"]).astype(np.float32)
    return zs


def rehist(z, d, t0):
    hi = t0[:, None] + np.arange(-H, 0)
    v, qf = z["obs_speed"][d[:, None], hi], z["obs_flow"][d[:, None], hi]
    return v, qf, z["elig"][d[:, None], hi] & np.isfinite(v) & np.isfinite(qf)


# ------------------------------------------------------------------ main
def main() -> None:
    t_start = time.time()
    Xb = {(s, b): [] for s in SEEDS for b in range(N_BAGS)}
    Aq = {v: [] for v in VARIANTS if v != "base"}
    ys, ctxs, names_on = [], [], None
    for i, p in enumerate(T2_PANELS):
        ctx = Ctx(p, i)
        ctx.fitd = np.ones_like(ctx.fitd)
        z = ctx.z
        d, t0 = ctx.eo
        v, qf, ok = hist_from_truth(ctx, d, t0)
        y = ctx.q[d, t0 + F]
        grp = ctx.g[d].astype(int)
        ctx.prof = train_profile(ctx)
        r = log_ratio(ctx.prof, qf, ok, t0, grp)
        ends = Counter()
        for c in y:
            for s, e in runs(c):
                ends[s] += 1; ends[e] += 1
        ctx.cands = [e for e, _ in ends.most_common(N_CAND)]
        ctx.pca = pca_fit(r)
        pg = y.mean(0).astype(np.float32)
        ctx.lib, ctx.dist_train = {}, {}
        for var, key in (("cand", cand_key(r, ctx.cands, t0, ctx.S)), ("pca", pca_key(r, ctx.pca, t0))):
            ctx.lib[var] = dict(key=key, y=y.astype(np.float32), grp=grp, day=d, pg=pg, sd=key.std(0) + 1e-6)
            pa, dist = analog(ctx.lib[var], key, grp, qday=d, return_dist=True)  # leave-same-day-out for train rows
            ctx.dist_train[var] = float(np.nanmean(dist))
            A = a_feats(pa, pg)
            drop = rng.random(len(d)) < 0.5                                     # analog-prior dropout
            A[drop] = 0.0
            Aq[var].append(A.reshape(-1, len(A_NAMES)))
        ys.append(y.reshape(-1))
        for sd in SEEDS:
            t2_prior_robust.rng = np.random.default_rng(1000 * (sd + 1) + i)
            for b in range(N_BAGS):
                names_on, X = feats_on(ctx, False, v, qf, ok, t0, ctx.g[d], t2_prior_robust.dropout_prior(ctx), d, z["weekday"][d])
                Xb[(sd, b)].append(X.reshape(-1, X.shape[-1]))
        ctx.z = {k: z[k] for k in ("link_ids", "weekday", "dates")}
        ctx.q = ctx.el = None
        ctxs.append(ctx)
        print(f"[{p}] library {len(d)} windows, candidates {ctx.cands} ({time.time() - t_start:.0f}s)", flush=True)
    y_all = np.concatenate(ys)
    A_all = {k: np.concatenate(v) for k, v in Aq.items()}
    models = {}
    for sd in SEEDS:
        for var in VARIANTS:
            ms = []
            for b in range(N_BAGS):
                X = np.concatenate(Xb[(sd, b)])
                if var != "base":
                    X = np.concatenate([X, A_all[var]], 1)
                nm = names_on + ([f"{n}_{var}" for n in A_NAMES] if var != "base" else [])
                ms.append(lgb.train(dict(PARAMS, num_leaves=63, seed=b + 100 * sd), lgb.Dataset(X, y_all, feature_name=nm), 600))
            models[(sd, var)] = ms
            if var != "base":
                imp = ms[0].feature_importance("gain")
                print(f"trained seed {sd} {var} ({time.time() - t_start:.0f}s); analog share of gain {imp[-len(A_NAMES):].sum() / imp.sum():.3f}", flush=True)
    del Xb
    res: dict = {}
    for ctx in ctxs:
        for split in ("validation", "private"):
            z = load(ctx.panel, split)
            blank = official_blank(ctx.panel, split, z)
            on = first_onsets(ctx.vf, z, blank=blank)
            if not len(on["d"]):
                continue
            grp = (z["weekday"][on["d"]] >= 5).astype(int)
            key = np.stack([on["d"], grp], 1)
            for hname, zz in (("masked", z), ("filled", filled(z, ctx.panel, split))):
                v, qf, ok = rehist(zz, on["d"], on["t0"])
                _, X = feats_on(ctx, False, v, qf, ok, on["t0"], grp, table_prior(ctx), key, z["weekday"][on["d"]].astype(int))
                r = log_ratio(ctx.prof, qf, ok, on["t0"], grp)
                A, dq, probs = {}, {}, {}
                for var, qk in (("cand", cand_key(r, ctx.cands, on["t0"], ctx.S)), ("pca", pca_key(r, ctx.pca, on["t0"]))):
                    pa, dq[var] = analog(ctx.lib[var], qk, grp, return_dist=True)
                    A[var] = a_feats(pa, ctx.lib[var]["pg"])
                # the analog map on its own (no model), as a diagnostic
                for var in ("cand", "pca"):
                    pr = np.zeros(on["fut"].shape, bool)
                    pr[:, F - 1] = np.stack([select(q, "ratio", 1.0) for q in A[var][..., 0]])
                    s = iou(pr, on["fut"], on["efut"])
                    for ds, dmax in DAY_SETS.items():
                        m = on["d"] < dmax
                        res.setdefault((hname, "map only", var, split, ds, 0), {})[ctx.panel] = s[m].mean() if m.any() else np.nan
                for (sd, var), ms in models.items():
                    Xo = np.concatenate([X, A[var]], -1) if var != "base" else X
                    pp = np.mean([m.predict(Xo.reshape(-1, Xo.shape[-1])) for m in ms], 0).reshape(Xo.shape[:2])
                    probs.setdefault(var, []).append(pp)
                    if (sd, var) == list(models)[-1]:  # all seeds and variants done: cache for offline blends
                        np.savez_compressed(ROOT / "work" / "cache" / f"t2analog_{hname}_{split}_{ctx.panel}.npz",
                                            fut=on["fut"], efut=on["efut"], d=on["d"],
                                            p_base=np.mean(probs["base"], 0), p_cand=np.mean(probs["cand"], 0),
                                            p_pca=np.mean(probs["pca"], 0), map_cand=A["cand"][..., 0], map_pca=A["pca"][..., 0],
                                            dist_cand=dq["cand"], dist_pca=dq["pca"],
                                            dist_train_cand=ctx.dist_train["cand"], dist_train_pca=ctx.dist_train["pca"])
                    pr = np.zeros(on["fut"].shape, bool)
                    pr[:, F - 1] = np.stack([select(q, "ratio", 1.0) for q in pp])
                    s = iou(pr, on["fut"], on["efut"])
                    for ds, dmax in DAY_SETS.items():
                        m = on["d"] < dmax
                        res.setdefault((hname, "lgbm", var, split, ds, sd), {})[ctx.panel] = s[m].mean() if m.any() else np.nan
        print(f"  scored {ctx.panel} ({time.time() - t_start:.0f}s)", flush=True)
    for hname in ("masked", "filled"):
        print(f"\n== onset IoU, {hname} histories (family mean; LightGBM: mean over 3 seeds, delta vs paired base, seeds > 0) ==")
        for var in ("cand", "pca"):
            cells = [f"{sp[:4]} {ds} {fmean(res[(hname, 'map only', var, sp, ds, 0)]):.3f}" for sp in ("validation", "private") for ds in DAY_SETS]
            print(f"  analog map alone ({var}): " + " | ".join(cells))
        for var in VARIANTS:
            cells = []
            for sp in ("validation", "private"):
                for ds in DAY_SETS:
                    vals = [fmean(res[(hname, "lgbm", var, sp, ds, sd)]) for sd in SEEDS]
                    base = [fmean(res[(hname, "lgbm", "base", sp, ds, sd)]) for sd in SEEDS]
                    dl = [a - b for a, b in zip(vals, base)]
                    cells.append(f"{sp[:4]} {ds} {np.mean(vals):.3f}" + ("" if var == "base" else f" ({np.mean(dl):+.3f}, {sum(x > 0 for x in dl)}/3)"))
            print(f"  lgbm {var:<5}: " + " | ".join(cells))
    print(f"total {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
