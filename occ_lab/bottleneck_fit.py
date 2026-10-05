"""Fit the onset bottleneck classifier on the official-like onset windows of BOTH scored months (note 129) and
save it for scripts/t2_onset_blend_predict.py --bn-model. The cross-month evaluation is occ_lab/bottleneck_cls.py.

    .venv/Scripts/python.exe occ_lab/bottleneck_fit.py
"""
from __future__ import annotations

import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta"))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

import bottleneck_cls as B  # noqa: E402

C = B.C
CAP = "--cap" in sys.argv       # note 145: + the learned discharge-capacity headroom (bottleneck_capacity.py)
PRIOR4 = "--prior4" in sys.argv  # note 145: training-month capacity + the month's last <= 4 earlier days


def main() -> None:
    X, y = [], []
    for i, panel in enumerate(C.T2_PANELS):
        Bd = B.build(i, panel)
        if CAP or PRIOR4:
            import bottleneck_capacity as BK
            ctx, stat, W = C.windows(i, panel)
            vf = np.asarray(ctx.vf, np.float32)
            Pr = BK.prior_stats(panel, vf) if PRIOR4 else None
        for s, D in Bd.items():
            if PRIOR4:
                A = BK.day_arrays(W[s]["z"], vf)
                X.append(np.concatenate([D["X"], BK.prior_features(D, Pr), BK.capacity_features_trunc(D, A)], 1))
            elif CAP:
                X.append(np.concatenate([D["X"], BK.capacity_features(D, BK.day_arrays(W[s]["z"], vf))], 1))
            else:
                X.append(D["X"])
            y.append(D["y"])
        print(f"  {panel}", flush=True)
    X, y = np.concatenate(X), np.concatenate(y)
    par = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 30,
           "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "num_threads": 8}
    m = lgb.train(par, lgb.Dataset(X, y), 200)
    out = LAB / "models" / ("bottleneck_prior4_lgb.txt" if PRIOR4 else "bottleneck_cap_lgb.txt" if CAP else "bottleneck_lgb.txt")
    m.save_model(str(out))
    print(f"saved {out}: {len(y):,} candidate rows, {int(y.sum())} positives", flush=True)


if __name__ == "__main__":
    main()
