"""Task 2: global network-state features (Traffic4cast 2022 2nd place, Bolt: "global features dominate", note 42).

Every current T2 feature is per link. This adds a summary of the WHOLE panel's current state, computed only from the
window's own history (T-60 .. T-5; no priors, no future data), broadcast to every link of the window:
    g_fq_last, g_fq_trend      share of observed links queued (v <= 0.6 vf) in the last 3 slots, and last3 - first3
    g_vr_last, g_vr_trend      mean v/vf over observed cells, last 3 slots, and last3 - first3
    g_qc_last, g_qc_trend      mean q/cap, likewise
    g_heads                    number of queue heads at the last slots (queued link whose downstream link is not)
    g_pc1 .. g_pc5             principal components of the panel's v/vf profile (last-3-slot mean per link, unseen = 1),
                               PCA fitted per panel on train truth only (random 04-21 h origins)
The comparison is paired, exactly as analysis/t2_ramp_hist.py (identical rows and prior-dropout draws):
    onset   : H's configuration (prior dropout, 5 bags; expected-IoU ratio decision, c 1.0), with / without
    ongoing : I's configuration (no prior, demand stress; tau 0.65), with / without
Blank-aware official-like windows, days 1-14 and all, validation and private, family mean.
Saves the models and the PCA to work/models_gstate/ and the per-window probabilities (g0 = without, g1 = with) to
work/cache/t2gs_<split>_<panel>.npz, on the same windows as work/cache/t2probs_* (for the blend with the CNN).

    .venv/Scripts/python.exe analysis/t2_gstate.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "scripts"))
from t2_eda02 import T2_PANELS, candidates  # noqa: E402
from t2_eda03 import F, H  # noqa: E402
from t2_expected_iou import select  # noqa: E402
from t2_first_onset_eval import first_onsets, official_blank  # noqa: E402
from t2_m1_models import Ctx, hist_from_truth  # noqa: E402
from t2_m1_predict import feats_og, feats_on, table_prior, zero_prior  # noqa: E402
from t2_official_like_eval import ongoing_official  # noqa: E402
from t2_prior_robust import dropout_prior  # noqa: E402
from t2_pseudo_eval import iou  # noqa: E402
from t2_shift_remedies import PARAMS  # noqa: E402
from tfb.data import CACHE, load  # noqa: E402
from tfb.features import shift  # noqa: E402
from tfb.t1eval import family_mean  # noqa: E402

N_BAGS, MAX_ROWS, N_PC, N_FIT = 5, 400_000, 5, 4000
DAY_SETS = {"d1-14": 14, "all": 31}
G_NAMES = ["g_fq_last", "g_fq_trend", "g_vr_last", "g_vr_trend", "g_qc_last", "g_qc_trend", "g_heads"] + \
          [f"g_pc{k + 1}" for k in range(N_PC)]
rng = np.random.default_rng(17)


def fmean(d):
    d = {k: v for k, v in d.items() if np.isfinite(v)}
    return family_mean(d) if d else float("nan")


def profile(ctx, v, ok) -> np.ndarray:
    """(N, H, S) history -> (N, S) last-3-slot mean v/vf, unseen = 1 (free flow)."""
    vr = np.where(ok, v / ctx.vf, np.nan)[:, -3:]
    with np.errstate(all="ignore"):
        p = np.nanmean(vr, 1)
    return np.where(np.isfinite(p), np.clip(p, 0, 1.5), 1.0).astype(np.float32)


def fit_pca(ctx) -> tuple[np.ndarray, np.ndarray]:
    """PCA of the v/vf profile on random train origins (04-21 h), train truth only."""
    z = ctx.z
    D = z["true_speed"].shape[0]
    d = rng.integers(0, D, N_FIT)
    t0 = rng.integers(48, 252, N_FIT)
    hi = t0[:, None] + np.arange(-H, 0)
    P = profile(ctx, z["true_speed"][d[:, None], hi], ctx.el[d[:, None], hi])
    mu = P.mean(0)
    _, _, vt = np.linalg.svd(P - mu, full_matrices=False)
    return mu, vt[:N_PC]


def gstate(ctx, pca, v, qf, ok) -> np.ndarray:
    """(N, H, S) history -> (N, len(G_NAMES)) panel-wide state of the window."""
    vr = np.where(ok, v / ctx.vf, np.nan)
    qc = np.where(ok, qf / ctx.cap, np.nan)
    qd = np.where(ok, (np.nan_to_num(v, nan=1e9) <= 0.6 * ctx.vf).astype(np.float32), np.nan)
    with np.errstate(all="ignore"):
        m = lambda a, sl: np.nan_to_num(np.nanmean(a[:, sl], (1, 2)), nan=0.0)  # noqa: E731
        fq3, fq0 = m(qd, slice(-3, None)), m(qd, slice(0, 3))
        vr3, vr0 = m(vr, slice(-3, None)), m(vr, slice(0, 3))
        qc3, qc0 = m(qc, slice(-3, None)), m(qc, slice(0, 3))
        ql = np.nan_to_num(np.nanmax(qd[:, -3:], 1), nan=0.0) > 0.5          # (N, S) queued in the last 3 slots
    heads = (ql & ~shift(ql.astype(np.float32), -1, 1).astype(bool)).sum(1)  # downstream neighbour not queued
    pcs = (profile(ctx, v, ok) - pca[0]) @ pca[1].T
    cols = [fq3, fq3 - fq0, vr3, vr3 - vr0, qc3, qc3 - qc0, heads.astype(np.float32)]
    return np.concatenate([np.stack(cols, 1), pcs], 1).astype(np.float32)


def main() -> None:
    t_start = time.time()
    Xon = {(o, b): [] for o in (False, True) for b in range(N_BAGS)}
    yon, Xog_a, Xog_b, yog, ctxs = [], [], [], [], []
    names_on = names_og = None
    for i, p in enumerate(T2_PANELS):
        ctx = Ctx(p, i)
        ctx.fitd = np.ones_like(ctx.fitd)
        z = ctx.z
        ctx.pca = fit_pca(ctx)
        # onset rows
        d, t0 = ctx.eo
        hist = hist_from_truth(ctx, d, t0)
        g = np.broadcast_to(gstate(ctx, ctx.pca, *hist)[:, None], (len(d), ctx.S, len(G_NAMES)))
        for b in range(N_BAGS):
            pri = dropout_prior(ctx)  # one draw per bag, shared by both variants
            names_on, X = feats_on(ctx, False, *hist, t0, ctx.g[d], pri, d, z["weekday"][d])
            Xon[(False, b)].append(X.reshape(-1, X.shape[-1]))
            Xon[(True, b)].append(np.concatenate([X, g], -1).reshape(-1, X.shape[-1] + len(G_NAMES)))
        yon.append(ctx.q[d, t0 + F].reshape(-1))
        # ongoing rows
        c = candidates(z)
        d, t0 = c["d"][c["ongoing"]], c["t0"][c["ongoing"]]
        pa, pb, yl = [], [], []
        for j in range(0, len(d), 400):
            dd, tt = d[j:j + 400], t0[j:j + 400]
            v, qf, ok = hist_from_truth(ctx, dd, tt)
            names_og, Xh, base = feats_og(ctx, True, v, qf, ok, tt, ctx.g[dd], zero_prior(ctx), dd, z["weekday"][dd])
            y = ctx.q[dd[:, None], tt[:, None] + np.arange(1, F + 1)]
            e = ctx.el[dd[:, None], tt[:, None] + np.arange(1, F + 1)]
            near = np.minimum(base["dist_slow_up"], base["dist_slow_dn"]) <= 10
            keep = np.flatnonzero(e & (np.broadcast_to(near[:, None, :], y.shape) | y | (rng.random(y.shape) < 0.02)))
            gf = np.broadcast_to(gstate(ctx, ctx.pca, v, qf, ok)[:, None, None], Xh.shape[:3] + (len(G_NAMES),))
            pa.append(Xh.reshape(-1, Xh.shape[-1])[keep]); pb.append(gf.reshape(-1, len(G_NAMES))[keep])
            yl.append(y.reshape(-1)[keep])
        yp = np.concatenate(yl)
        sub = rng.choice(len(yp), MAX_ROWS, replace=False) if len(yp) > MAX_ROWS else np.arange(len(yp))
        Xog_a.append(np.concatenate(pa)[sub]); Xog_b.append(np.concatenate(pb)[sub]); yog.append(yp[sub])
        ctx.z = {k: z[k] for k in ("link_ids", "weekday", "dates")}
        ctx.q = ctx.el = None
        ctxs.append(ctx)
        print(f"[{p}] rows ({time.time() - t_start:.0f}s)", flush=True)
    y_on = np.concatenate(yon)
    m_on = {}
    for o in (False, True):
        nm = names_on + (G_NAMES if o else [])
        m_on[o] = [lgb.train(dict(PARAMS, num_leaves=63, seed=b), lgb.Dataset(np.concatenate(Xon[(o, b)]), y_on, feature_name=nm), 600)
                   for b in range(N_BAGS)]
        imp = m_on[o][0].feature_importance("gain")
        extra = f"; global share of gain {imp[-len(G_NAMES):].sum() / imp.sum():.3f}" if o else ""
        print(f"trained onset global={o} ({time.time() - t_start:.0f}s){extra}", flush=True)
    del Xon
    y_og = np.concatenate(yog)
    A = np.concatenate(Xog_a)
    m_og = {False: lgb.train(dict(PARAMS, num_leaves=127), lgb.Dataset(A, y_og, feature_name=names_og), 3300)}
    print(f"trained ongoing base ({time.time() - t_start:.0f}s)", flush=True)
    B = np.concatenate([A, np.concatenate(Xog_b)], 1)
    del A, Xog_a, Xog_b
    m_og[True] = lgb.train(dict(PARAMS, num_leaves=127), lgb.Dataset(B, y_og, feature_name=names_og + G_NAMES), 3300)
    imp = m_og[True].feature_importance("gain")
    print(f"trained ongoing +global ({time.time() - t_start:.0f}s); global share of gain {imp[-len(G_NAMES):].sum() / imp.sum():.3f}", flush=True)
    del B
    out = ROOT / "work" / "models_gstate"
    out.mkdir(parents=True, exist_ok=True)
    for o in (False, True):
        for b, m in enumerate(m_on[o]):
            m.save_model(str(out / f"onset_g{int(o)}_b{b}.txt"))
        m_og[o].save_model(str(out / f"ongoing_g{int(o)}.txt"))
    for ctx in ctxs:  # the PCA is part of the model: the same components must be used at scoring time
        np.savez(out / f"pca_{ctx.panel}.npz", mu=ctx.pca[0], vt=ctx.pca[1])
    res: dict = {}
    for ctx in ctxs:
        for split in ("validation", "private"):
            z = load(ctx.panel, split)
            blank = official_blank(ctx.panel, split, z)
            on = first_onsets(ctx.vf, z, blank=blank)
            pon = {o: np.zeros((0, ctx.S), np.float32) for o in (False, True)}
            if len(on["d"]):
                grp = (z["weekday"][on["d"]] >= 5).astype(int)
                key = np.stack([on["d"], grp], 1)
                _, X = feats_on(ctx, False, on["v"], on["qf"], on["ok"], on["t0"], grp, table_prior(ctx), key,
                                z["weekday"][on["d"]].astype(int))
                g = np.broadcast_to(gstate(ctx, ctx.pca, on["v"], on["qf"], on["ok"])[:, None], X.shape[:2] + (len(G_NAMES),))
                for o in (False, True):
                    Xo = np.concatenate([X, g], -1) if o else X
                    p = np.mean([m.predict(Xo.reshape(-1, Xo.shape[-1])) for m in m_on[o]], 0).reshape(Xo.shape[:2])
                    pon[o] = p.astype(np.float32)
                    pr = np.zeros(on["fut"].shape, bool)
                    pr[:, F - 1] = np.stack([select(q, "ratio", 1.0) for q in p])
                    s = iou(pr, on["fut"], on["efut"])
                    for ds, dmax in DAY_SETS.items():
                        m = on["d"] < dmax
                        res.setdefault(("onset", split, ds), {}).setdefault(o, {})[ctx.panel] = s[m].mean() if m.any() else np.nan
            og, _ = ongoing_official(ctx, z, blank)
            pga, pgb = [np.zeros((0, F, ctx.S), np.float32)], [np.zeros((0, F, ctx.S), np.float32)]
            if len(og["d"]):
                grp = (z["weekday"][og["d"]] >= 5).astype(int)
                key = np.stack([og["d"], grp], 1)
                wdo = z["weekday"][og["d"]].astype(int)
                sa, sb = [], []
                for j in range(0, len(og["d"]), 200):
                    b = slice(j, j + 200)
                    _, Xh, _ = feats_og(ctx, True, og["v"][b], og["qf"][b], og["ok"][b], og["t0"][b], grp[b], zero_prior(ctx), key[b], wdo[b])
                    gf = np.broadcast_to(gstate(ctx, ctx.pca, og["v"][b], og["qf"][b], og["ok"][b])[:, None, None],
                                         Xh.shape[:3] + (len(G_NAMES),))
                    pa = m_og[False].predict(Xh.reshape(-1, Xh.shape[-1])).reshape(Xh.shape[:3])
                    pb = m_og[True].predict(np.concatenate([Xh, gf], -1).reshape(-1, Xh.shape[-1] + len(G_NAMES))).reshape(Xh.shape[:3])
                    pga.append(pa.astype(np.float32)); pgb.append(pb.astype(np.float32))
                    sa.append(iou(pa >= 0.65, og["fut"][b], og["efut"][b])); sb.append(iou(pb >= 0.65, og["fut"][b], og["efut"][b]))
                for o, s in ((False, np.concatenate(sa)), (True, np.concatenate(sb))):
                    for ds, dmax in DAY_SETS.items():
                        m = og["d"] < dmax
                        res.setdefault(("ongoing", split, ds), {}).setdefault(o, {})[ctx.panel] = s[m].mean() if m.any() else np.nan
            np.savez_compressed(CACHE / f"t2gs_{split}_{ctx.panel}.npz", on_d=on["d"], og_d=og["d"],
                                on_p0=pon[False], on_p1=pon[True], og_p0=np.concatenate(pga), og_p1=np.concatenate(pgb))
        print(f"  scored {ctx.panel} ({time.time() - t_start:.0f}s)", flush=True)
    for (kind, split, ds), r in res.items():
        a, b = fmean(r[False]), fmean(r[True])
        diffs = [r[True][p] - r[False][p] for p in r[False] if np.isfinite(r[False][p]) and np.isfinite(r[True][p])]
        print(f"== {kind} {split} {ds}: base {a:.3f} -> +global {b:.3f} ({b - a:+.3f}; better/worse "
              f"{sum(x > 1e-9 for x in diffs)}/{sum(x < -1e-9 for x in diffs)})")
    print(f"total {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
