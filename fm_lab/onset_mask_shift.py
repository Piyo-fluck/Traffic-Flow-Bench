"""Is the onset "run shift" really a masking shift?

H and every onset member are trained on TRUTH histories (hist_from_truth: full coverage), and their train-tail
check also reads truth. The scored windows read the RELEASED layer, where Task 1 masks hide ~32% of the eligible
cells (and the official Task 2 blanks more). note 108 attributed H's 0.924 -> 0.79 drop to the run; today's
TabPFN control showed the same drop without any prior feature. Here the two causes are separated:

    train   truth histories  |  released-layer histories of the train split (the same masking process)
    score   train tail (truth) | train tail (released layer) | validation | private

H's recipe (feats_on, prior dropout for training, train priors for the tail, table priors for the scored months),
one LightGBM, fitted on the 2030 days only.

    .venv/Scripts/python.exe fm_lab/onset_mask_shift.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parent
PROJ = LAB.parent
for p in (PROJ, PROJ / "analysis", PROJ / "scripts"):
    sys.path.insert(0, str(p))
import lightgbm as lgb  # noqa: E402

from t2_eda02 import T2_PANELS  # noqa: E402
from t2_eda03 import F, H  # noqa: E402
from t2_expected_iou import select  # noqa: E402
from t2_first_onset_eval import first_onsets, official_blank  # noqa: E402
from t2_m1_models import Ctx, hist_from_truth  # noqa: E402
from t2_m1_predict import feats_on, table_prior, train_prior  # noqa: E402
from t2_prior_robust import dropout_prior  # noqa: E402
from t2_pseudo_eval import iou  # noqa: E402
from tfb.data import load  # noqa: E402
from tfb.t1eval import FIT_END, family_mean  # noqa: E402


def hist_released(ctx, d, t0):
    hi = t0[:, None] + np.arange(-H, 0)
    v, q = ctx.z["obs_speed"][d[:, None], hi], ctx.z["obs_flow"][d[:, None], hi]
    ok = ctx.z["elig"][d[:, None], hi] & np.isfinite(v) & np.isfinite(q)
    return v, q, ok


def onset_iou(p, fut, efut):
    pr = np.zeros(fut.shape, bool)
    pr[:, F - 1] = np.stack([select(q, "ratio", 1.0) for q in p])
    return iou(pr, fut, efut)


def main() -> None:
    t_start = time.time()
    X_tr = {"truth": [], "released": []}
    y_tr = []
    evals = []
    for i, panel in enumerate(T2_PANELS):
        ctx = Ctx(panel, i)
        d, t0 = ctx.eo
        fit = np.asarray(ctx.z["dates"])[d].astype(str) < FIT_END
        e = ctx.el[d[fit], t0[fit] + F]
        for src, hf in (("truth", hist_from_truth), ("released", hist_released)):
            _, X = feats_on(ctx, False, *hf(ctx, d[fit], t0[fit]), t0[fit], ctx.g[d[fit]], dropout_prior(ctx),
                            d[fit], ctx.z["weekday"][d[fit]])
            X_tr[src].append(X[e])
        y_tr.append(ctx.q[d[fit], t0[fit] + F][e])
        fi = t0[~fit, None] + np.arange(1, F + 1)
        fut, efut = ctx.q[d[~fit, None], fi], ctx.el[d[~fit, None], fi]
        for src, hf in (("truth", hist_from_truth), ("released", hist_released)):
            _, X = feats_on(ctx, False, *hf(ctx, d[~fit], t0[~fit]), t0[~fit], ctx.g[d[~fit]], train_prior(ctx),
                            d[~fit], ctx.z["weekday"][d[~fit]])
            evals.append((panel, f"tail ({src} hist)", X, fut, efut))
        for split in ("validation", "private"):
            z = load(panel, split)
            on = first_onsets(ctx.vf, z, blank=official_blank(panel, split, z))
            if not len(on["d"]):
                continue
            grp = (z["weekday"][on["d"]] >= 5).astype(int)
            _, X = feats_on(ctx, False, on["v"], on["qf"], on["ok"], on["t0"], grp, table_prior(ctx),
                            np.stack([on["d"], grp], 1), z["weekday"][on["d"]].astype(int))
            evals.append((panel, split, X, on["fut"], on["efut"]))
        print(f"[{panel}] ({time.time() - t_start:.0f}s)", flush=True)
    y = np.concatenate(y_tr).astype(int)
    par = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=50, feature_fraction=0.8,
               bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=8)
    models = {f"train on {src} hist": lgb.train(par, lgb.Dataset(np.concatenate(X_tr[src]), y), 600)
              for src in ("truth", "released")}
    both = np.concatenate([np.concatenate(X_tr["truth"]), np.concatenate(X_tr["released"])])
    models["train on both"] = lgb.train(par, lgb.Dataset(both, np.r_[y, y]), 600)
    res = {}
    for panel, kind, X, fut, efut in evals:
        for nm, m in models.items():
            p = m.predict(X.reshape(-1, X.shape[-1])).reshape(X.shape[:2])
            res.setdefault((kind, nm), {})[panel] = onset_iou(p, fut, efut).mean()
    kinds = ["tail (truth hist)", "tail (released hist)", "validation", "private"]
    print(f"\nonset family IoU (ratio rule), H recipe, fitted on 2030 days\n")
    print(f"{'model':<24}" + "".join(f"{k:>22}" for k in kinds))
    for nm in models:
        print(f"{nm:<24}" + "".join(f"{family_mean(res[(k, nm)]):>22.4f}" for k in kinds))


if __name__ == "__main__":
    main()
