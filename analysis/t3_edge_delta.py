"""Predict the density INCREMENT across a target-visible edge, instead of the density level.

Note 105 found that 70% of S_LWR's residual sits on transitions where exactly one of the two cells is a target.
There the other end is the released truth, so at scoring time k is KNOWN on one side and the quantity the metric
wants is

    dN = L (k[t+1] - k[t])

- an increment over five minutes, not a level near 40 veh/km. The temporal smoothing of note 105 was implicitly
the model dN_hat = 0 and lost; that says nothing about whether dN can be predicted from the visible anchor.

A small LightGBM is fitted on these edges with an MAE loss, out of fold by day, from features available at
scoring time, and compared with what the production CNN implies today. The headline is the numerator of S_LWR on
these edges: the CNN currently contributes 1,979,135 of the 2,825,364 total.

    py analysis/t3_edge_delta.py [tag]
"""
from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tfb.data import PANELS, load  # noqa: E402
from tfb.physics import fd_params, ramp_ok  # noqa: E402

PRED = ROOT / "work" / "preds"
TAG = sys.argv[1] if len(sys.argv) > 1 else "t1_m3A"
FOLDS = 3
NAMES = ["k_vis", "k_cnn_tgt", "d_cnn", "q_cnn", "v_cnn", "k_prev_step", "k_up", "k_down", "has_up", "has_down",
         "tod", "link", "lanes", "length", "regime", "target_is_later", "k_vis_prev", "k_vis_prev2"]
PARAMS = {"objective": "l1", "learning_rate": 0.05, "num_leaves": 127, "min_data_in_leaf": 200,
          "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "num_threads": 8}
ROUNDS = 400


def main() -> None:
    X, Y, CUR, FOLD, LEN = [], [], [], [], []
    loff = 0
    for panel in PANELS:
        f = PRED / f"{TAG}_{panel}_test.npz"
        if not f.exists():
            continue
        p = np.load(f)
        z = load(panel, "train")
        days = p["days"]
        prm = fd_params(panel, z["link_ids"])
        ok = ramp_ok(panel, "train", z)[days]
        L, lanes = prm["length"], prm["lanes"]
        ts, tq = z["true_speed"][days], z["true_flow"][days]
        tgt = z["target"][days] & np.isfinite(ts) & np.isfinite(tq)
        use = z["elig"][days] & np.isfinite(ts) & np.isfinite(tq)
        k_true = tq / np.maximum(ts, 1.0)
        k_cnn = p["pq"] / np.maximum(p["pv"], 1.0)
        k_sub = np.where(tgt, k_cnn, k_true)
        k_obs = np.where(use & ~tgt, k_true, np.nan)          # what is visible at scoring time
        edge = use[:, :-1] & use[:, 1:] & ok[:, :-1] & (tgt[:, :-1] ^ tgt[:, 1:])
        later = tgt[:, 1:] & edge                              # the target is the later cell
        d, t, s = np.nonzero(edge)
        if not len(d):
            continue
        ti = np.where(later[d, t, s], t, t + 1)                # index of the target cell
        vi = np.where(later[d, t, s], t + 1, t) * 0 + np.where(later[d, t, s], t, t + 1)
        vi = np.where(later[d, t, s], t, t + 1)
        vi = np.where(later[d, t, s], t, t + 1)                # visible index: the other one
        tgt_idx = np.where(later[d, t, s], t + 1, t)
        vis_idx = np.where(later[d, t, s], t, t + 1)
        y = L[s] * (k_true[d, t + 1, s] - k_true[d, t, s])
        cur = L[s] * (k_sub[d, t + 1, s] - k_sub[d, t, s])
        prev = np.where(vis_idx >= 1, np.nan_to_num(k_obs[d, np.maximum(vis_idx - 1, 0), s], nan=0.0), 0.0)
        prev2 = np.where(vis_idx >= 2, np.nan_to_num(k_obs[d, np.maximum(vis_idx - 2, 0), s], nan=0.0), 0.0)
        kv = k_true[d, vis_idx, s]
        up = np.where(s > 0, np.nan_to_num(k_obs[d, tgt_idx, np.maximum(s - 1, 0)], nan=0.0), 0.0)
        dn = np.where(s < len(L) - 1, np.nan_to_num(k_obs[d, tgt_idx, np.minimum(s + 1, len(L) - 1)], nan=0.0), 0)
        feats = np.stack([
            kv, k_cnn[d, tgt_idx, s], k_cnn[d, tgt_idx, s] - kv,
            p["pq"][d, tgt_idx, s] / lanes[s], p["pv"][d, tgt_idx, s],
            kv - prev, up, dn, (up > 0).astype(np.float32), (dn > 0).astype(np.float32),
            tgt_idx / k_true.shape[1], s + loff, lanes[s], L[s],
            np.searchsorted(["R1", "R2", "R3"], np.asarray(z["regime"], str)[days][d]).astype(np.float32),
            later[d, t, s].astype(np.float32), prev, prev2], 1).astype(np.float32)
        loff += len(L)
        X.append(feats); Y.append(y.astype(np.float32)); CUR.append(cur.astype(np.float32))
        FOLD.append((d % FOLDS).astype(np.int8)); LEN.append(L[s].astype(np.float32))
        print(f"  {panel}: {len(d):,} one-target edges", flush=True)

    X = np.concatenate(X); Y = np.concatenate(Y); CUR = np.concatenate(CUR)
    FOLD = np.concatenate(FOLD)
    print(f"\n{len(Y):,} edges; current |residual| sum {np.abs(CUR - Y).sum():,.0f}, "
          f"|dN_true| sum {np.abs(Y).sum():,.0f}")
    pred = np.zeros_like(Y)
    for k in range(FOLDS):
        tr, te = FOLD != k, FOLD == k
        gbm = lgb.train(PARAMS, lgb.Dataset(X[tr], label=Y[tr], feature_name=NAMES, free_raw_data=False),
                        num_boost_round=ROUNDS)
        pred[te] = gbm.predict(X[te]).astype(np.float32)
        if k == 0:
            imp = sorted(zip(gbm.feature_importance("gain"), NAMES), reverse=True)[:6]
            tot = sum(g for g, _ in imp) or 1
            print("   top gain: " + ", ".join(f"{n} {g / tot * 100:.0f}%" for g, n in imp))
    print(f"\n{'estimator':<28} {'sum |residual|':>16} {'vs current':>12}")
    base = float(np.abs(CUR - Y).sum())
    print(f"{'CNN (current)':<28} {base:16,.0f} {0.0:12.1%}")
    print(f"{'zero increment':<28} {np.abs(Y).sum():16,.0f} {np.abs(Y).sum() / base - 1:12.1%}")
    for a in (0.25, 0.5, 0.75, 1.0):
        v = float(np.abs((1 - a) * CUR + a * pred - Y).sum())
        print(f"{'blend a=' + format(a, '.2f'):<28} {v:16,.0f} {v / base - 1:12.1%}")


if __name__ == "__main__":
    main()
