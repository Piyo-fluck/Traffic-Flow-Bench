"""Task 2 onset: how noisy is one prior-dropout draw, and does averaging several draws (bagging) help?

The same robust configuration scored pseudo onset IoU 0.354 (note 20) and 0.286 (t2_demand_stress) on validation.
The only difference was the random prior-dropout draw. Here N onset models are trained, each on its own dropout
draw with its own LightGBM seed, using the feature columns of one t2_struct_noid variant. The script reports:
    single : mean and sd of onset IoU over the N single-draw models
    bag    : onset IoU of the averaged probability (then k = round(1.2 * sum p), as in production)
Onset windows only (ongoing uses no priors, so bagging leaves it unchanged). Pseudo windows with days < 14.

    py analysis/t2_bag_eval.py noid+all 5
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
from t2_demand_stress import EXTRA_ON, load_tq  # noqa: E402
from t2_eda02 import T2_PANELS  # noqa: E402
from t2_eda03 import F  # noqa: E402
from t2_m1_models import Ctx, hist_from_truth, onset_decide  # noqa: E402
from t2_m1_predict import table_prior  # noqa: E402
from t2_prior_robust import dropout_prior  # noqa: E402
from t2_pseudo_eval import iou, pseudo_windows  # noqa: E402
from t2_shift_remedies import PARAMS  # noqa: E402
from t2_struct_noid import effective_capacity, full_onset, struct_vecs, subsets  # noqa: E402
from tfb.data import load  # noqa: E402
from tfb.t1eval import family_mean  # noqa: E402

EVAL_DAYS = 14


def main() -> None:
    variant = sys.argv[1] if len(sys.argv) > 1 else "noid+all"
    n_bags = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    t_start = time.time()
    ctxs, extra, Xb, ys = [], {}, [[] for _ in range(n_bags)], []
    names = None
    for i, p in enumerate(T2_PANELS):
        ctx = Ctx(p, i)
        ctx.fitd = np.ones_like(ctx.fitd)
        ceff = effective_capacity(ctx)
        extra[p] = (struct_vecs(ctx, ceff), ceff, load_tq(p))
        d, t0 = ctx.eo
        hist = hist_from_truth(ctx, d, t0)
        for b in range(n_bags):
            names, X = full_onset(ctx, *extra[p], *hist, t0, ctx.g[d], dropout_prior(ctx), d, ctx.z["weekday"][d])
            Xb[b].append(X.reshape(-1, X.shape[-1]))
        ys.append(ctx.q[d, t0 + F].reshape(-1))
        ctx.z = {"link_ids": ctx.z["link_ids"]}
        ctx.q = ctx.el = None
        ctxs.append(ctx)
        print(f"[{p}] onset rows x {n_bags} draws ({time.time() - t_start:.0f}s)", flush=True)
    cols = subsets(names, EXTRA_ON)[variant]
    y = np.concatenate(ys)
    models = []
    for b in range(n_bags):  # own dropout draw and own LightGBM seed per bag
        ds = lgb.Dataset(np.concatenate(Xb[b])[:, cols], y, feature_name=[names[j] for j in cols])
        models.append(lgb.train(dict(PARAMS, num_leaves=63, seed=b), ds, 600))
        print(f"trained bag {b} ({time.time() - t_start:.0f}s)", flush=True)
    del Xb
    for split in ("validation", "private"):
        single = [dict() for _ in range(n_bags)]
        bag: dict[str, float] = {}
        for ctx in ctxs:
            z = load(ctx.panel, split)
            w = pseudo_windows(ctx, z)
            on = (w["d"] < EVAL_DAYS) & w["onset"]
            if not on.any():
                continue
            grp = (z["weekday"][w["d"]] >= 5).astype(int)
            key = np.stack([w["d"], grp], 1)
            wd = z["weekday"][w["d"]].astype(int)
            _, X = full_onset(ctx, *extra[ctx.panel], w["v"][on], w["qf"][on], w["ok"][on], w["t0"][on], grp[on],
                              table_prior(ctx), key[on], wd[on])
            Xv = X[..., cols].reshape(-1, len(cols))
            ps = [m.predict(Xv).reshape(X.shape[:2]) for m in models]

            def score(p):
                pr = np.zeros((on.sum(), F, ctx.S), bool)
                pr[:, F - 1] = onset_decide(p, ("sum", 1.2))
                return iou(pr, w["fut"][on], w["efut"][on]).mean()
            for b, p in enumerate(ps):
                single[b][ctx.panel] = score(p)
            bag[ctx.panel] = score(np.mean(ps, 0))
        s = np.array([family_mean(r) for r in single])
        print(f"== {split} ({variant}, {n_bags} draws): single-draw onset {s.mean():.3f} +- {s.std():.3f} "
              f"(min {s.min():.3f}, max {s.max():.3f}) | bagged onset {family_mean(bag):.3f}", flush=True)
    print(f"total {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
