"""Task 2: demand-vs-capacity "stress" features (physics-inspired, meant to transfer across simulation runs).

A queue starts where the arriving demand exceeds a bottleneck's capacity (Newell's bottleneck theory;
stochastic capacity). The train queue priors are demand x capacity of the *train run*, and they fail in the
new run (notes 17-20). Demand's time-of-day shape and link capacities should transfer, so this module projects
each link's flow over the horizon as
    today's level (last 3 history flows / train weekday x tod flow profile, blended link / corridor median)
    x train profile at T+h,
and divides it by capacity:
    pqc[h, s]     projected flow / capacity at T+5..T+30
    pqc_dn[h, s]  max of pqc over the 4 downstream links (the bottleneck a queue would spill back from)
    growth[s]     profile(T+30) / profile(T-5): how steeply demand is still rising
    level         today's corridor demand level
Experiment: the robust models (onset prior-dropout, ongoing no-prior; note 20) with and without these
features, scored on in-month pseudo windows with days < 14 in validation and private.

    py analysis/t2_demand_stress.py
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
from t2_eda02 import T2_PANELS, candidates  # noqa: E402
from t2_eda03 import F  # noqa: E402
from t2_m1_models import Ctx, hist_from_truth, onset_decide  # noqa: E402
from t2_m1_predict import ongoing_X, onset_X, table_prior, zero_prior  # noqa: E402
from t2_prior_robust import dropout_prior  # noqa: E402
from t2_pseudo_eval import iou, pseudo_windows  # noqa: E402
from t2_shift_remedies import fit, sample_og  # noqa: E402
from tfb.data import CACHE, T, load  # noqa: E402
from tfb.features import shift  # noqa: E402
from tfb.t1eval import family_mean  # noqa: E402

EVAL_DAYS, MAX_OG = 14, 350_000
rng = np.random.default_rng(2)
EXTRA_ON = ["st_pqc30", "st_pqc30_dn", "st_pqc_max", "st_growth", "st_level"]
EXTRA_OG = ["st_pqc_h", "st_pqc_dn_h", "st_growth", "st_level"]


def load_tq(panel: str) -> np.ndarray:
    with np.load(CACHE / f"{panel}_t1tables.npz") as f:
        return f["tq"]  # (7, T, S) weekday x slot x station mean flow over every train day


def stress(ctx: Ctx, tq: np.ndarray, qf: np.ndarray, ok: np.ndarray, t0: np.ndarray, wd: np.ndarray) -> dict:
    hi = t0[:, None] + np.arange(-3, 0)
    prof_hist = tq[wd[:, None], hi]                                    # (N, 3, S)
    q3 = np.where(ok[:, -3:], qf[:, -3:], np.nan)
    with np.errstate(all="ignore"):
        r_link = np.nanmean(q3 / np.maximum(prof_hist, 1.0), 1)        # (N, S)
        r_corr = np.nanmedian(r_link, 1)                               # (N,)
    r_corr = np.nan_to_num(r_corr, nan=1.0)
    ratio = np.where(np.isfinite(r_link), 0.5 * r_link + 0.5 * r_corr[:, None], r_corr[:, None])
    fut = np.clip(t0[:, None] + np.arange(1, F + 1), 0, T - 1)
    prof_fut = tq[wd[:, None], fut]                                    # (N, F, S)
    pqc = (ratio[:, None, :] * prof_fut / ctx.cap).astype(np.float32)
    dn = np.zeros_like(pqc)
    for k in range(1, 5):
        dn = np.maximum(dn, np.nan_to_num(shift(pqc, -k, 2), nan=0.0))
    growth = (prof_fut[:, -1] / np.maximum(prof_hist[:, -1], 1.0)).astype(np.float32)
    return dict(pqc=pqc, pqc_dn=dn, growth=growth, level=np.broadcast_to(r_corr[:, None], growth.shape).astype(np.float32))


def onset_Xs(ctx, v, qf, ok, t0, grp, pri, key, wd, tq, extra: bool):
    names, X = onset_X(ctx, v, qf, ok, t0, grp, pri, key)
    if not extra:
        return names, X
    st = stress(ctx, tq, qf, ok, t0, wd)
    E = np.stack([st["pqc"][:, -1], st["pqc_dn"][:, -1], st["pqc"].max(1), st["growth"], st["level"]], -1)
    return names + EXTRA_ON, np.concatenate([X, E], -1)


def ongoing_Xs(ctx, v, qf, ok, t0, grp, pri, key, wd, tq, extra: bool):
    names, Xh, base = ongoing_X(ctx, v, qf, ok, t0, grp, pri, key)
    if not extra:
        return names, Xh, base
    st = stress(ctx, tq, qf, ok, t0, wd)
    E = np.stack([st["pqc"], st["pqc_dn"], np.broadcast_to(st["growth"][:, None], st["pqc"].shape),
                  np.broadcast_to(st["level"][:, None], st["pqc"].shape)], -1)
    return names + EXTRA_OG, np.concatenate([Xh, E], -1), base


def train_rows(ctx, tq, extra):
    z = ctx.z
    d, t0 = ctx.eo
    names_on, X = onset_Xs(ctx, *hist_from_truth(ctx, d, t0), t0, ctx.g[d], dropout_prior(ctx), d, z["weekday"][d], tq, extra)
    on = (X.reshape(-1, X.shape[-1]), ctx.q[d, t0 + F].reshape(-1))
    c = candidates(z)
    d, t0 = c["d"][c["ongoing"]], c["t0"][c["ongoing"]]
    xs, ys = [], []
    for i in range(0, len(d), 400):
        dd, tt = d[i:i + 400], t0[i:i + 400]
        names_og, Xh, base = ongoing_Xs(ctx, *hist_from_truth(ctx, dd, tt), tt, ctx.g[dd], zero_prior(ctx), dd,
                                        z["weekday"][dd], tq, extra)
        y = ctx.q[dd[:, None], tt[:, None] + np.arange(1, F + 1)]
        e = ctx.el[dd[:, None], tt[:, None] + np.arange(1, F + 1)]
        a, b = sample_og(Xh, y, e, base)
        xs.append(a); ys.append(b)
    X, y = np.concatenate(xs), np.concatenate(ys)
    if len(y) > MAX_OG:
        idx = rng.choice(len(y), MAX_OG, replace=False)
        X, y = X[idx], y[idx]
    return names_on, on, names_og, (X, y)


def score(ctx, w, sel, grp, key, wd, tq, extra, m_on, m_og, pri_on):
    res = {}
    on = sel & w["onset"]
    if on.any():
        _, X = onset_Xs(ctx, w["v"][on], w["qf"][on], w["ok"][on], w["t0"][on], grp[on], pri_on, key[on], wd[on], tq, extra)
        p = m_on.predict(X.reshape(-1, X.shape[-1])).reshape(X.shape[:2])
        pr = np.zeros((on.sum(), F, ctx.S), bool)
        pr[:, F - 1] = onset_decide(p, ("sum", 1.2))
        res["onset"] = iou(pr, w["fut"][on], w["efut"][on]).mean()
    idx = np.flatnonzero(sel & w["ongoing"])
    ious = []
    for j in range(0, len(idx), 200):
        b = idx[j:j + 200]
        _, Xh, _ = ongoing_Xs(ctx, w["v"][b], w["qf"][b], w["ok"][b], w["t0"][b], grp[b], zero_prior(ctx), key[b], wd[b], tq, extra)
        p = m_og.predict(Xh.reshape(-1, Xh.shape[-1])).reshape(Xh.shape[:3])
        ious.append(iou(p >= 0.65, w["fut"][b], w["efut"][b]))
    res["ongoing"] = np.concatenate(ious).mean() if ious else np.nan
    return res


def main() -> None:
    t_start = time.time()
    ctxs, tqs, rows = [], {}, {False: [], True: []}
    for i, p in enumerate(T2_PANELS):
        ctx = Ctx(p, i)
        ctx.fitd = np.ones_like(ctx.fitd)
        tqs[p] = load_tq(p)
        for extra in (False, True):
            names_on, on, names_og, og = train_rows(ctx, tqs[p], extra)
            rows[extra].append((on, og, names_on, names_og))
        ctx.z = {"link_ids": ctx.z["link_ids"]}
        ctx.q = ctx.el = None
        ctxs.append(ctx)
        print(f"[{p}] train rows ({time.time() - t_start:.0f}s)", flush=True)
    models = {}
    for extra, rs in rows.items():
        n_on, n_og = rs[0][2], rs[0][3]
        models[extra] = (fit(n_on, [(r[0][0], r[0][1], 1.0) for r in rs], 600, 63),
                         fit(n_og, [(r[1][0], r[1][1], 1.0) for r in rs], 2000, 127))
        imp = models[extra][0].feature_importance("gain")
        top = ", ".join(f"{n_on[j]} {imp[j] / imp.sum():.2f}" for j in np.argsort(-imp)[:6])
        print(f"trained extra={extra} ({time.time() - t_start:.0f}s); onset top gain: {top}", flush=True)
    del rows
    for split in ("validation", "private"):
        res = {e: {"onset": {}, "ongoing": {}} for e in models}
        for ctx in ctxs:
            z = load(ctx.panel, split)
            w = pseudo_windows(ctx, z)
            grp = (z["weekday"][w["d"]] >= 5).astype(int)
            key = np.stack([w["d"], grp], 1)
            wd = z["weekday"][w["d"]].astype(int)
            ev = w["d"] < EVAL_DAYS
            pri_on = table_prior(ctx)
            for extra, (mo, mg) in models.items():
                r = score(ctx, w, ev, grp, key, wd, tqs[ctx.panel], extra, mo, mg, pri_on)
                for k2, val in r.items():
                    if not np.isnan(val):
                        res[extra][k2][ctx.panel] = val
        print(f"\n== {split}: pseudo windows on days < {EVAL_DAYS}, family mean ==")
        for extra, r in res.items():
            a, b = family_mean(r["onset"]), family_mean(r["ongoing"])
            print(f"  robust{' + stress' if extra else '         '}  onset {a:.3f}  ongoing {b:.3f}  S_queue {(a + b) / 2:.3f}", flush=True)
    print(f"total {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
