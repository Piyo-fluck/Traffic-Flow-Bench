"""Task 2: occupancy features. Occupancy is in the official window history, but no T2 model has used it.

Breakdown-probability models describe the chance that a queue forms from the SPEED AND OCCUPANCY of the traffic
arriving at a bottleneck (bivariate Weibull; stochastic capacity). Our T2 features use speed and flow only. Per
window and link, from the history (only cells that are observed and have a finite occupancy):
    occ_last, occ_max3, occ_mean, occ_trend (last-3 mean - first-3 mean), and occ_max3 of links +-1, +-2
The comparison is paired (identical rows and prior-dropout draws):
    onset   : H's configuration (prior dropout, 5 bags; expected-IoU ratio decision, c 1.0), with / without occ
    ongoing : I's configuration (no prior, demand stress; tau 0.65), with / without occ
It is scored on the blank-aware official-like windows (notes 26, 28), days 1-14 and all, validation and private,
family mean.

    py analysis/t2_occupancy.py
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
from t2_m1_models import Ctx, hist_from_truth, nanlast  # noqa: E402
from t2_m1_predict import feats_og, feats_on, table_prior, zero_prior  # noqa: E402
from t2_official_like_eval import ongoing_official  # noqa: E402
from t2_prior_robust import dropout_prior  # noqa: E402
from t2_pseudo_eval import iou  # noqa: E402
from t2_shift_remedies import PARAMS  # noqa: E402
from tfb.data import load  # noqa: E402
from tfb.features import shift  # noqa: E402
from tfb.t1eval import family_mean  # noqa: E402

N_BAGS, MAX_ROWS = 5, 400_000
DAY_SETS = {"d1-14": 14, "all": 31}
OCC_NAMES = ["occ_last", "occ_max3", "occ_mean", "occ_trend", "occ_max3_s+1", "occ_max3_s-1", "occ_max3_s+2", "occ_max3_s-2"]
rng = np.random.default_rng(9)


def fmean(d):
    d = {k: v for k, v in d.items() if np.isfinite(v)}
    return family_mean(d) if d else float("nan")


def occ_feats(occ: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """(N, H, S) occupancy history -> (N, S, len(OCC_NAMES))."""
    good = ok & np.isfinite(occ)
    o = np.where(good, occ, np.nan)
    with np.errstate(all="ignore"):
        last = nanlast(o, good)
        mx3 = np.nanmax(o[:, -3:], 1)
        mean = np.nanmean(o, 1)
        trend = np.nanmean(o[:, -3:], 1) - np.nanmean(o[:, :3], 1)
    cols = [last, mx3, mean, trend] + [shift(mx3, k, 1) for k in (1, -1, 2, -2)]
    return np.stack(cols, -1).astype(np.float32)


def occ_hist(occ_all: np.ndarray, d: np.ndarray, t0: np.ndarray) -> np.ndarray:
    return occ_all[d[:, None], t0[:, None] + np.arange(-H, 0)]


def main() -> None:
    t_start = time.time()
    Xon = {(o, b): [] for o in (False, True) for b in range(N_BAGS)}
    yon, Xog_a, Xog_b, yog, ctxs = [], [], [], [], []
    names_on = names_og = None
    for i, p in enumerate(T2_PANELS):
        ctx = Ctx(p, i)
        ctx.fitd = np.ones_like(ctx.fitd)
        z = ctx.z
        # onset rows
        d, t0 = ctx.eo
        hist = hist_from_truth(ctx, d, t0)
        oc = occ_feats(occ_hist(z["true_occ"], d, t0), hist[2])
        for b in range(N_BAGS):
            pri = dropout_prior(ctx)  # one draw per bag, shared by both variants (same key object)
            names_on, X = feats_on(ctx, False, *hist, t0, ctx.g[d], pri, d, z["weekday"][d])
            Xon[(False, b)].append(X.reshape(-1, X.shape[-1]))
            Xon[(True, b)].append(np.concatenate([X, oc], -1).reshape(-1, X.shape[-1] + oc.shape[-1]))
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
            oc = np.broadcast_to(occ_feats(occ_hist(z["true_occ"], dd, tt), ok)[:, None], Xh.shape[:3] + (len(OCC_NAMES),))
            pa.append(Xh.reshape(-1, Xh.shape[-1])[keep]); pb.append(oc.reshape(-1, len(OCC_NAMES))[keep])
            yl.append(y.reshape(-1)[keep])
        yp = np.concatenate(yl)
        sub = rng.choice(len(yp), MAX_ROWS, replace=False) if len(yp) > MAX_ROWS else np.arange(len(yp))
        Xog_a.append(np.concatenate(pa)[sub]); Xog_b.append(np.concatenate(pb)[sub]); yog.append(yp[sub])
        ctx.z = {k: z[k] for k in ("link_ids", "weekday")}
        ctx.q = ctx.el = None
        ctxs.append(ctx)
        print(f"[{p}] rows ({time.time() - t_start:.0f}s)", flush=True)
    y_on = np.concatenate(yon)
    m_on = {}
    for o in (False, True):
        nm = names_on + (OCC_NAMES if o else [])
        m_on[o] = [lgb.train(dict(PARAMS, num_leaves=63, seed=b), lgb.Dataset(np.concatenate(Xon[(o, b)]), y_on, feature_name=nm), 600)
                   for b in range(N_BAGS)]
        print(f"trained onset occ={o} ({time.time() - t_start:.0f}s)", flush=True)
    del Xon
    y_og = np.concatenate(yog)
    A = np.concatenate(Xog_a)
    m_og = {False: lgb.train(dict(PARAMS, num_leaves=127), lgb.Dataset(A, y_og, feature_name=names_og), 3300)}
    print(f"trained ongoing base ({time.time() - t_start:.0f}s)", flush=True)
    B = np.concatenate([A, np.concatenate(Xog_b)], 1)
    del A, Xog_a, Xog_b
    m_og[True] = lgb.train(dict(PARAMS, num_leaves=127), lgb.Dataset(B, y_og, feature_name=names_og + OCC_NAMES), 3300)
    imp = m_og[True].feature_importance("gain")
    nm = names_og + OCC_NAMES
    print(f"trained ongoing +occ ({time.time() - t_start:.0f}s); occ share of gain "
          f"{imp[-len(OCC_NAMES):].sum() / imp.sum():.3f}; top: "
          + ", ".join(f"{nm[j]} {imp[j] / imp.sum():.3f}" for j in np.argsort(-imp)[:6]), flush=True)
    del B
    res: dict = {}
    for ctx in ctxs:
        for split in ("validation", "private"):
            z = load(ctx.panel, split)
            blank = official_blank(ctx.panel, split, z)
            on = first_onsets(ctx.vf, z, blank=blank)
            if len(on["d"]):
                grp = (z["weekday"][on["d"]] >= 5).astype(int)
                key = np.stack([on["d"], grp], 1)
                _, X = feats_on(ctx, False, on["v"], on["qf"], on["ok"], on["t0"], grp, table_prior(ctx), key,
                                z["weekday"][on["d"]].astype(int))
                oc = occ_feats(occ_hist(z["obs_occ"], on["d"], on["t0"]), on["ok"])
                for o in (False, True):
                    Xo = np.concatenate([X, oc], -1) if o else X
                    p = np.mean([m.predict(Xo.reshape(-1, Xo.shape[-1])) for m in m_on[o]], 0).reshape(Xo.shape[:2])
                    pr = np.zeros(on["fut"].shape, bool)
                    pr[:, F - 1] = np.stack([select(q, "ratio", 1.0) for q in p])
                    s = iou(pr, on["fut"], on["efut"])
                    for ds, dmax in DAY_SETS.items():
                        m = on["d"] < dmax
                        res.setdefault(("onset", split, ds), {}).setdefault(o, {})[ctx.panel] = s[m].mean() if m.any() else np.nan
            og, _ = ongoing_official(ctx, z, blank)
            if len(og["d"]):
                grp = (z["weekday"][og["d"]] >= 5).astype(int)
                key = np.stack([og["d"], grp], 1)
                wdo = z["weekday"][og["d"]].astype(int)
                sa, sb = [], []
                for j in range(0, len(og["d"]), 200):
                    b = slice(j, j + 200)
                    _, Xh, _ = feats_og(ctx, True, og["v"][b], og["qf"][b], og["ok"][b], og["t0"][b], grp[b], zero_prior(ctx), key[b], wdo[b])
                    oc = np.broadcast_to(occ_feats(occ_hist(z["obs_occ"], og["d"][b], og["t0"][b]), og["ok"][b])[:, None],
                                         Xh.shape[:3] + (len(OCC_NAMES),))
                    pa = m_og[False].predict(Xh.reshape(-1, Xh.shape[-1])).reshape(Xh.shape[:3])
                    pb = m_og[True].predict(np.concatenate([Xh, oc], -1).reshape(-1, Xh.shape[-1] + len(OCC_NAMES))).reshape(Xh.shape[:3])
                    sa.append(iou(pa >= 0.65, og["fut"][b], og["efut"][b])); sb.append(iou(pb >= 0.65, og["fut"][b], og["efut"][b]))
                for o, s in ((False, np.concatenate(sa)), (True, np.concatenate(sb))):
                    for ds, dmax in DAY_SETS.items():
                        m = og["d"] < dmax
                        res.setdefault(("ongoing", split, ds), {}).setdefault(o, {})[ctx.panel] = s[m].mean() if m.any() else np.nan
        print(f"  scored {ctx.panel} ({time.time() - t_start:.0f}s)", flush=True)
    for (kind, split, ds), r in res.items():
        a, b = fmean(r[False]), fmean(r[True])
        diffs = [r[True][p] - r[False][p] for p in r[False] if np.isfinite(r[False][p]) and np.isfinite(r[True][p])]
        print(f"== {kind} {split} {ds}: base {a:.3f} -> +occ {b:.3f} ({b - a:+.3f}; better/worse "
              f"{sum(x > 1e-9 for x in diffs)}/{sum(x < -1e-9 for x in diffs)})")
    print(f"total {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
