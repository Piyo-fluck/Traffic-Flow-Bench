"""Task 2 onset as learning-to-rank: LightGBM lambdarank per window vs the binary onset model.

Onset is a per-window top-k selection: which links queue at T+30. Our binary model ranks the true link first in
53% of first-morning onsets and within the top 3 in 85% (note 28 era analysis). LambdaMART optimises the ranking
(NDCG) within each group directly. The comparison is paired: same features as H (feats_on, no stress), same
prior-dropout draws, 5 bags each.
    binary      H's model -> expected-IoU ratio rule (c 1.0)                        [K / M onset without CNN]
    rank        order the links by the lambdarank score, k from the binary rule's choice
    hybrid      order by the mean of the two per-window rank positions, k from the binary rule
It is scored on blank-aware first-morning onset windows, validation and private, days 1-14 and all.
Also reported: top-1 hit and rank <= 3 of the best true link.

    py analysis/t2_onset_rank.py
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
from t2_eda02 import T2_PANELS  # noqa: E402
from t2_eda03 import F  # noqa: E402
from t2_expected_iou import select  # noqa: E402
from t2_first_onset_eval import first_onsets, official_blank  # noqa: E402
from t2_m1_models import Ctx, hist_from_truth  # noqa: E402
from t2_m1_predict import feats_on, table_prior  # noqa: E402
from t2_prior_robust import dropout_prior  # noqa: E402
from t2_pseudo_eval import iou  # noqa: E402
from t2_shift_remedies import PARAMS  # noqa: E402
from tfb.data import load  # noqa: E402
from tfb.t1eval import family_mean  # noqa: E402

N_BAGS = 5
DAY_SETS = {"d1-14": 14, "all": 31}
RANK_PARAMS = dict(objective="lambdarank", metric="ndcg", eval_at=[1, 3, 5], learning_rate=0.05, num_leaves=63,
                   min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                   verbose=-1, num_threads=8, lambdarank_truncation_level=20)


def fmean(d):
    d = {k: v for k, v in d.items() if np.isfinite(v)}
    return family_mean(d) if d else float("nan")


def ranks_desc(x: np.ndarray) -> np.ndarray:
    """Per-row rank positions (0 = largest)."""
    return np.argsort(np.argsort(-x, 1), 1)


def main() -> None:
    t_start = time.time()
    Xb = {b: [] for b in range(N_BAGS)}
    ys, groups, ctxs, names = [], [], [], None
    for i, p in enumerate(T2_PANELS):
        ctx = Ctx(p, i)
        ctx.fitd = np.ones_like(ctx.fitd)
        d, t0 = ctx.eo
        hist = hist_from_truth(ctx, d, t0)
        for b in range(N_BAGS):
            names, X = feats_on(ctx, False, *hist, t0, ctx.g[d], dropout_prior(ctx), d, ctx.z["weekday"][d])
            Xb[b].append(X.reshape(-1, X.shape[-1]))
        ys.append(ctx.q[d, t0 + F].reshape(-1).astype(int))
        groups.append(np.full(len(d), ctx.S))                 # one query group per window, S links each
        ctx.z = {k: ctx.z[k] for k in ("link_ids", "weekday")}
        ctx.q = ctx.el = None
        ctxs.append(ctx)
        print(f"[{p}] onset windows {len(d)} ({time.time() - t_start:.0f}s)", flush=True)
    y, grp = np.concatenate(ys), np.concatenate(groups)
    m_bin, m_rank = [], []
    for b in range(N_BAGS):
        X = np.concatenate(Xb[b])
        m_bin.append(lgb.train(dict(PARAMS, num_leaves=63, seed=b), lgb.Dataset(X, y, feature_name=names), 600))
        m_rank.append(lgb.train(dict(RANK_PARAMS, seed=b), lgb.Dataset(X, y, group=grp, feature_name=names), 600))
        print(f"trained bag {b} ({time.time() - t_start:.0f}s)", flush=True)
    del Xb
    res: dict = {}
    for ctx in ctxs:
        for split in ("validation", "private"):
            z = load(ctx.panel, split)
            on = first_onsets(ctx.vf, z, blank=official_blank(ctx.panel, split, z))
            if not len(on["d"]):
                continue
            grp_w = (z["weekday"][on["d"]] >= 5).astype(int)
            key = np.stack([on["d"], grp_w], 1)
            _, X = feats_on(ctx, False, on["v"], on["qf"], on["ok"], on["t0"], grp_w, table_prior(ctx), key,
                            z["weekday"][on["d"]].astype(int))
            Xf = X.reshape(-1, X.shape[-1])
            pb = np.mean([m.predict(Xf) for m in m_bin], 0).reshape(X.shape[:2])
            ranks_r = np.mean([ranks_desc(m.predict(Xf).reshape(X.shape[:2])) for m in m_rank], 0)  # mean rank position
            sel_bin = np.stack([select(q, "ratio", 1.0) for q in pb])
            k = sel_bin.sum(1)
            true = on["fut"][:, -1] & on["efut"][:, -1]
            orders = {"binary": ranks_desc(pb), "rank": ranks_desc(-ranks_r),
                      "hybrid": ranks_desc(-(ranks_desc(pb) + ranks_r))}
            for name, rk in orders.items():
                pr = np.zeros(on["fut"].shape, bool)
                pr[:, F - 1] = sel_bin if name == "binary" else rk < k[:, None]
                s = iou(pr, on["fut"], on["efut"])
                best = np.array([rk[j][true[j]].min() if true[j].any() else 99 for j in range(len(rk))])
                for ds, dmax in DAY_SETS.items():
                    m = on["d"] < dmax
                    r = res.setdefault((split, ds), {}).setdefault(name, {"iou": {}, "top1": {}, "top3": {}})
                    r["iou"][ctx.panel] = s[m].mean() if m.any() else np.nan
                    r["top1"][ctx.panel] = (best[m] == 0).mean() if m.any() else np.nan
                    r["top3"][ctx.panel] = (best[m] <= 2).mean() if m.any() else np.nan
        print(f"  scored {ctx.panel} ({time.time() - t_start:.0f}s)", flush=True)
    for (split, ds), rr in res.items():
        base = rr["binary"]["iou"]
        print(f"\n== {split} {ds}: onset (first-morning, blank-aware), family mean ==")
        for name, r in rr.items():
            diffs = [r["iou"][p] - base[p] for p in base if np.isfinite(base[p]) and np.isfinite(r["iou"][p])]
            print(f"  {name:<7} IoU {fmean(r['iou']):.3f} ({fmean(r['iou']) - fmean(base):+.3f}; "
                  f"{sum(x > 1e-9 for x in diffs)}/{sum(x < -1e-9 for x in diffs)})  top1 {fmean(r['top1']):.3f}  top3 {fmean(r['top3']):.3f}")
    print(f"total {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
