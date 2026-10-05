"""Task 2: replace link identity with transferable bottleneck structure and capacity margins.

The prior features are not the only train-run memory: `panel` + `pos` identify every link, so the trees can
learn "this link queues at 17:00" directly from the train run. That is the same shift problem as the priors
(notes 17-20), and prior dropout does not touch it. A queue starts at an active bottleneck when the arriving flow
reaches its discharge capacity, and that capacity transfers across runs to within about 0.5% (note 21):
  static structure (st_*)
    st_lane_drop1/4, st_lane_gain_up1, st_cap_ratio1/4, st_on, st_off, st_on_dn1, st_merge_dn, st_dist_bneck
    st_ceff_ratio  effective capacity / released capacity
  effective capacity ceff[s] = train median discharge flow at queue heads (>= 20 events), else the released cap
  window margins (mg_*), flows from the last 3 history slots (interpolated along the corridor where unseen)
    mg_q_own      flow[s] / ceff[s]
    mg_q_dn1      flow[s+1] / ceff[s+1];    mg_dn_max4   max_k=1..4 flow[s+k] / ceff[s+k]
    mg_tight      flow[s] / min_k=0..4 ceff[s+k]   (arriving flow vs the tightest capacity just downstream)
    mg_tight_proj mg_tight x profile growth to T+30
    mg_trend      (last-3 mean - first-3 mean of the history) / ceff[s]
One full feature matrix is built (production features + structure + margins + demand stress), and the variants
are column subsets, so all share identical samples:
    base, noid (- pos, panel), noid+struct, noid+struct+mg, noid+all (+ demand stress, t2_demand_stress)
Scored on in-month pseudo windows with days < 14 (validation and private).

    py analysis/t2_struct_noid.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "scripts"))
from t2_demand_stress import EXTRA_OG, EXTRA_ON, load_tq, ongoing_Xs, onset_Xs  # noqa: E402
from t2_eda02 import T2_PANELS, candidates  # noqa: E402
from t2_eda03 import F  # noqa: E402
from t2_m1_models import Ctx, hist_from_truth, onset_decide  # noqa: E402
from t2_m1_predict import table_prior, zero_prior  # noqa: E402
from t2_prior_robust import dropout_prior  # noqa: E402
from t2_pseudo_eval import iou, pseudo_windows  # noqa: E402
from t2_shift_remedies import fit, sample_og  # noqa: E402
from tfb.data import load  # noqa: E402
from tfb.features import interp_along  # noqa: E402
from tfb.physics import load_ramps  # noqa: E402
from tfb.t1eval import family_mean  # noqa: E402

EVAL_DAYS, MAX_OG = 14, 350_000
rng = np.random.default_rng(3)
IDENT = {"pos", "panel"}


def at(x: np.ndarray, k: int, axis: int = -1) -> np.ndarray:
    """Value k links downstream along `axis` (edge-padded)."""
    S = x.shape[axis]
    return np.take(x, np.clip(np.arange(S) + k, 0, S - 1), axis=axis)


def effective_capacity(ctx: Ctx) -> np.ndarray:
    z = ctx.z
    v, q = z["true_speed"], z["true_flow"]
    ok = z["elig"] & np.isfinite(v) & np.isfinite(q)
    qd = ok & (np.nan_to_num(v, nan=1e9) <= 0.6 * ctx.vf)
    fr = ok & (np.nan_to_num(v, nan=0) > 0.85 * ctx.vf)
    head = np.zeros_like(ok)
    head[:, :, 1:] = fr[:, :, 1:] & qd[:, :, :-1]
    n = head.sum((0, 1))
    dis = np.array([np.median(q[:, :, s][head[:, :, s]]) if n[s] >= 20 else np.nan for s in range(ctx.S)])
    return np.where(np.isfinite(dis), dis, ctx.cap).astype(np.float32)


def struct_vecs(ctx: Ctx, ceff: np.ndarray) -> dict[str, np.ndarray]:
    r = load_ramps(ctx.panel, "train", ctx.z)
    S = ctx.S
    on, off = r["on_att"].astype(np.float32), r["off_att"].astype(np.float32)
    lanes, cap = np.asarray(ctx.lanes, np.float32), np.asarray(ctx.cap, np.float32)
    bneck = (at(ceff, 1) < 0.95 * ceff) | (at(on, 1) > 0)
    dist, nxt = np.full(S, 99, np.float32), None
    for s in range(S - 1, -1, -1):
        if bneck[s]:
            nxt = s
        dist[s] = 99 if nxt is None else nxt - s
    return {
        "st_lane_drop1": lanes - at(lanes, 1), "st_lane_drop4": lanes - np.min([at(lanes, k) for k in range(1, 5)], 0),
        "st_lane_gain_up1": lanes - at(lanes, -1),
        "st_cap_ratio1": at(ceff, 1) / ceff, "st_cap_ratio4": np.min([at(ceff, k) for k in range(1, 5)], 0) / ceff,
        "st_on": on, "st_off": off, "st_on_dn1": at(on, 1), "st_merge_dn": np.max([at(on, k) for k in range(3)], 0),
        "st_dist_bneck": dist, "st_ceff_ratio": ceff / cap,
    }


def margins(ceff, tq, qf, ok, t0, wd) -> dict[str, np.ndarray]:
    with np.errstate(all="ignore"):
        q3 = np.nanmean(np.where(ok[:, -3:], qf[:, -3:], np.nan), 1)
        qa = np.nanmean(np.where(ok[:, :3], qf[:, :3], np.nan), 1)
    have = np.isfinite(q3)
    qfill, _ = interp_along(np.where(have, q3, np.nan), have, axis=1)
    qfill = np.nan_to_num(qfill, nan=0.0)
    tight = np.min([at(ceff, k) for k in range(5)], 0)
    growth = tq[wd, np.clip(t0 + F, 0, tq.shape[1] - 1)] / np.maximum(tq[wd, t0 - 1], 1.0)
    r = qfill / ceff
    return {
        "mg_q_own": r, "mg_q_dn1": at(r, 1), "mg_dn_max4": np.max([at(r, k) for k in range(1, 5)], 0),
        "mg_tight": qfill / tight, "mg_tight_proj": qfill / tight * growth,
        "mg_trend": np.nan_to_num((q3 - qa) / ceff, nan=0.0),
    }


def add_cols(names, X, cols: dict):
    """Append (S,) or (N, S) columns to X of shape (N, S, K) or (N, F, S, K)."""
    lead = X.shape[:-1]

    def fit_shape(a):
        a = np.asarray(a, np.float32)
        if a.ndim == 2 and len(lead) == 3:  # per-window value, same for every horizon step
            a = a[:, None, :]
        return np.broadcast_to(a, lead)
    keys = sorted(cols)
    return names + keys, np.concatenate([X, np.stack([fit_shape(cols[k]) for k in keys], -1)], -1)


def full_onset(ctx, sv, ceff, tq, v, qf, ok, t0, grp, pri, key, wd):
    names, X = onset_Xs(ctx, v, qf, ok, t0, grp, pri, key, wd, tq, True)
    names, X = add_cols(names, X, sv)
    return add_cols(names, X, margins(ceff, tq, qf, ok, t0, wd))


def full_ongoing(ctx, sv, ceff, tq, v, qf, ok, t0, grp, pri, key, wd):
    names, Xh, base = ongoing_Xs(ctx, v, qf, ok, t0, grp, pri, key, wd, tq, True)
    names, Xh = add_cols(names, Xh, sv)
    names, Xh = add_cols(names, Xh, margins(ceff, tq, qf, ok, t0, wd))  # (N, S) -> broadcast over the horizon
    return names, Xh, base


def subsets(names: list[str], stress: list[str]) -> dict[str, np.ndarray]:
    st = [n for n in names if n.startswith("st_") and n not in stress]
    mg = [n for n in names if n.startswith("mg_")]
    base = [n for n in names if not n.startswith(("st_", "mg_"))]
    noid = [n for n in base if n not in IDENT]
    col = {n: i for i, n in enumerate(names)}
    sets = {"base": base, "noid": noid, "noid+struct": noid + st, "noid+struct+mg": noid + st + mg,
            "noid+all": noid + st + mg + stress}
    return {k: np.array([col[n] for n in v]) for k, v in sets.items()}


def train_rows(ctx, sv, ceff, tq):
    z = ctx.z
    d, t0 = ctx.eo
    n_on, X = full_onset(ctx, sv, ceff, tq, *hist_from_truth(ctx, d, t0), t0, ctx.g[d], dropout_prior(ctx), d, z["weekday"][d])
    on = (X.reshape(-1, X.shape[-1]), ctx.q[d, t0 + F].reshape(-1))
    c = candidates(z)
    d, t0 = c["d"][c["ongoing"]], c["t0"][c["ongoing"]]
    xs, ys = [], []
    for i in range(0, len(d), 400):
        dd, tt = d[i:i + 400], t0[i:i + 400]
        n_og, Xh, base = full_ongoing(ctx, sv, ceff, tq, *hist_from_truth(ctx, dd, tt), tt, ctx.g[dd], zero_prior(ctx), dd,
                                      z["weekday"][dd])
        y = ctx.q[dd[:, None], tt[:, None] + np.arange(1, F + 1)]
        e = ctx.el[dd[:, None], tt[:, None] + np.arange(1, F + 1)]
        a, b = sample_og(Xh, y, e, base)
        xs.append(a); ys.append(b)
    X, y = np.concatenate(xs), np.concatenate(ys)
    if len(y) > MAX_OG:
        idx = rng.choice(len(y), MAX_OG, replace=False)
        X, y = X[idx], y[idx]
    return n_on, on, n_og, (X, y)


def main() -> None:
    t_start = time.time()
    ctxs, tqs, svs, cef, rows = [], {}, {}, {}, []
    for i, p in enumerate(T2_PANELS):
        ctx = Ctx(p, i)
        ctx.fitd = np.ones_like(ctx.fitd)
        tqs[p], cef[p] = load_tq(p), effective_capacity(ctx)
        svs[p] = struct_vecs(ctx, cef[p])
        rows.append(train_rows(ctx, svs[p], cef[p], tqs[p]))
        ctx.z = {"link_ids": ctx.z["link_ids"]}
        ctx.q = ctx.el = None
        ctxs.append(ctx)
        print(f"[{p}] train rows; ceff differs from cap on {int((np.abs(cef[p] / ctx.cap - 1) > 0.03).sum())} links "
              f"({time.time() - t_start:.0f}s)", flush=True)
    n_on, n_og = rows[0][0], rows[0][2]
    cols_on, cols_og = subsets(n_on, EXTRA_ON), subsets(n_og, EXTRA_OG)
    models = {}
    for v in cols_on:
        co, cg = cols_on[v], cols_og[v]
        m_on = fit([n_on[j] for j in co], [(r[1][0][:, co], r[1][1], 1.0) for r in rows], 600, 63)
        m_og = fit([n_og[j] for j in cg], [(r[3][0][:, cg], r[3][1], 1.0) for r in rows], 2000, 127)
        models[v] = (m_on, m_og)
        imp = m_on.feature_importance("gain")
        top = ", ".join(f"{n_on[co[j]]} {imp[j] / imp.sum():.2f}" for j in np.argsort(-imp)[:6])
        print(f"trained {v} ({time.time() - t_start:.0f}s); onset top gain: {top}", flush=True)
    del rows
    for split in ("validation", "private"):
        res = {v: {"onset": {}, "ongoing": {}} for v in models}
        for ctx in ctxs:
            z = load(ctx.panel, split)
            w = pseudo_windows(ctx, z)
            grp = (z["weekday"][w["d"]] >= 5).astype(int)
            key = np.stack([w["d"], grp], 1)
            wd = z["weekday"][w["d"]].astype(int)
            ev = w["d"] < EVAL_DAYS
            args = (svs[ctx.panel], cef[ctx.panel], tqs[ctx.panel])
            on = ev & w["onset"]
            if on.any():
                _, X = full_onset(ctx, *args, w["v"][on], w["qf"][on], w["ok"][on], w["t0"][on], grp[on], table_prior(ctx), key[on], wd[on])
                for v, (mo, _) in models.items():
                    Xv = X[..., cols_on[v]]
                    p = mo.predict(Xv.reshape(-1, Xv.shape[-1])).reshape(Xv.shape[:2])
                    pr = np.zeros((on.sum(), F, ctx.S), bool)
                    pr[:, F - 1] = onset_decide(p, ("sum", 1.2))
                    res[v]["onset"][ctx.panel] = iou(pr, w["fut"][on], w["efut"][on]).mean()
            idx = np.flatnonzero(ev & w["ongoing"])
            ious = {v: [] for v in models}
            for j in range(0, len(idx), 200):
                b = idx[j:j + 200]
                _, Xh, _ = full_ongoing(ctx, *args, w["v"][b], w["qf"][b], w["ok"][b], w["t0"][b], grp[b], zero_prior(ctx), key[b], wd[b])
                for v, (_, mg) in models.items():
                    Xv = Xh[..., cols_og[v]]
                    p = mg.predict(Xv.reshape(-1, Xv.shape[-1])).reshape(Xv.shape[:3])
                    ious[v].append(iou(p >= 0.65, w["fut"][b], w["efut"][b]))
            for v in models:
                if ious[v]:
                    res[v]["ongoing"][ctx.panel] = np.concatenate(ious[v]).mean()
            print(f"  scored {split} {ctx.panel} ({time.time() - t_start:.0f}s)", flush=True)
        print(f"\n== {split}: pseudo windows on days < {EVAL_DAYS}, family mean ==")
        for v, r in res.items():
            a, b = family_mean(r["onset"]), family_mean(r["ongoing"])
            print(f"  {v:<15} onset {a:.3f}  ongoing {b:.3f}  S_queue {(a + b) / 2:.3f}", flush=True)
    print(f"total {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
