"""Is there a non-linear, scale-dependent mean bias left in the Task 1 CNN residual?

The diagnosis (note 100) showed the CNN is linearly calibrated - the slope of its predicted deviation on the true
deviation equals the MSE-optimal shrinkage in every group - and that its error is about 3% of the flow level.
That is consistent with two very different worlds:

    multiplicative NOISE        Var(q | X) grows with the level, nothing more to predict
    multiplicative STRUCTURE    E[q | X] itself is mis-estimated in a level-dependent way

Linear calibration cannot tell them apart: it only says E[r * q_hat] = 0, not E[r | q_hat = 800] = 0. This fits a
small model to the residual from features that are all available at scoring time, out of fold by day, and asks
whether the corrected flow RMSE moves. If it does not, a ratio parameterisation has little to gain either.

    py analysis/t1_residual_scale_bias.py [tag]
"""
from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tfb.data import PANELS, load  # noqa: E402
from tfb.features import profile, truth_valid  # noqa: E402
from tfb.t1eval import split_days  # noqa: E402

PRED = ROOT / "work" / "preds"
TAG = sys.argv[1] if len(sys.argv) > 1 else "t1_m3all"
FOLDS = 3
BINS = (0, 400, 800, 1200, 1800, 1e9)
NAMES = ["q_hat", "base_q", "v_hat", "base_v", "volatility", "tod", "link", "lanes", "obs_rate", "occ_near"]
PARAMS = {"objective": "l2", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 500,
          "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "num_threads": 8}


def main() -> None:
    X_all, r_all, q_all, fold_all, lvl_all = [], [], [], [], []
    loff = 0
    for panel in PANELS:
        f = PRED / f"{TAG}_{panel}_test.npz"
        if not f.exists():
            continue
        p = np.load(f)
        z = load(panel, "train")
        fit_d, _ = split_days(z)
        days = p["days"]
        lanes = z["lanes"]
        ts, tq = z["true_speed"][days], z["true_flow"][days]
        ql = tq / lanes
        valid = truth_valid(z)
        bq = profile(z["true_flow"], valid, z["weekday"], fit_d)[days] / lanes
        bv = profile(z["true_speed"], valid, z["weekday"], fit_d)[days]
        tgt = z["target"][days] & np.isfinite(ts) & np.isfinite(tq)
        obs = z["elig"][days] & np.isfinite(ql) & ~z["target"][days]
        qhat, vhat = p["pq"] / lanes, p["pv"]
        # everything below is computable at scoring time: the released layer only
        qo = np.where(obs, ql, np.nan)
        d1 = np.abs(np.diff(qo, axis=1))
        vol = np.nan_to_num(np.nanmean(np.where(np.isfinite(d1), d1, np.nan), axis=1, keepdims=True), nan=0.0)
        vol = np.broadcast_to(vol, ql.shape)
        occ = np.where(obs, np.nan_to_num(z["obs_occ"][days], nan=0.0), np.nan)
        occ_near = np.nan_to_num(np.nanmean(occ, axis=1, keepdims=True), nan=0.0)
        occ_near = np.broadcast_to(occ_near, ql.shape)
        obs_rate = np.broadcast_to(obs.mean(axis=(1, 2))[:, None, None], ql.shape)
        D, T, S = ql.shape
        tod = np.broadcast_to((np.arange(T) / T)[None, :, None], ql.shape)
        link = np.broadcast_to((np.arange(S) + loff)[None, None, :], ql.shape)
        loff += S
        lane_b = np.broadcast_to(lanes[None, None, :], ql.shape)
        fold = np.broadcast_to((np.arange(D) % FOLDS)[:, None, None], ql.shape)
        m = tgt
        X_all.append(np.stack([qhat[m], bq[m], vhat[m], bv[m], vol[m], tod[m], link[m], lane_b[m],
                               obs_rate[m], occ_near[m]], 1).astype(np.float32))
        r_all.append((ql - qhat)[m].astype(np.float32))
        q_all.append(ql[m].astype(np.float32))
        lvl_all.append(qhat[m].astype(np.float32))
        fold_all.append(fold[m].astype(np.int8))
        print(f"  {panel}", flush=True)

    X = np.concatenate(X_all); r = np.concatenate(r_all); fold = np.concatenate(fold_all)
    lvl = np.concatenate(lvl_all)
    print(f"\n{len(r):,} target cells, base flow RMSE {np.sqrt((r ** 2).mean()):.3f}")
    pred = np.zeros_like(r)
    for k in range(FOLDS):
        tr, te = fold != k, fold == k
        gbm = lgb.train(PARAMS, lgb.Dataset(X[tr], label=r[tr], feature_name=NAMES, free_raw_data=False),
                        num_boost_round=300)
        pred[te] = gbm.predict(X[te]).astype(np.float32)
        if k == 0:
            imp = sorted(zip(gbm.feature_importance("gain"), NAMES), reverse=True)[:5]
            tot = sum(g for g, _ in imp) or 1
            print("   top gain: " + ", ".join(f"{n} {g / tot * 100:.0f}%" for g, n in imp))
    print(f"   corrector output: mean {pred.mean():+.3f}, sd {pred.std():.3f}")
    for eta in (0.5, 1.0):
        print(f"\n   eta {eta:.1f}: flow RMSE {np.sqrt(((r - eta * pred) ** 2).mean()):.3f} "
              f"(base {np.sqrt((r ** 2).mean()):.3f})")
    print(f"\n{'flow bin':>16} {'cells':>10} {'RMSE':>8} {'corrected':>10} {'delta':>8} {'mean resid':>11}")
    for i in range(len(BINS) - 1):
        m = (lvl >= BINS[i]) & (lvl < BINS[i + 1])
        if not m.any():
            continue
        b, c = np.sqrt((r[m] ** 2).mean()), np.sqrt(((r - pred)[m] ** 2).mean())
        print(f"{BINS[i]:>7.0f}-{BINS[i + 1]:<8.0f} {int(m.sum()):10,} {b:8.2f} {c:10.2f} {c - b:+8.3f} "
              f"{r[m].mean():+11.3f}")


if __name__ == "__main__":
    main()
