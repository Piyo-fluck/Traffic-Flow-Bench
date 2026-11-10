"""Fit the residual corrector on the out-of-fold cache and save it for the official-window run.

Unweighted L2 on r = y - p0 over every eligible cell of the 2031-01/02 windows: `occ_lab/residual_oof.py` shows
the weighted variants (alpha 3, alpha 9) buy rescue at a spoil rate that costs more than it gains, while the
unweighted fit is positive on both scored months at every eta and gate tried.

    python occ_lab/fit_corrector.py --out occ_lab/models/residual_lgb.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta"))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

import common as C  # noqa: E402
from residual_diag import NAMES, TAU  # noqa: E402
from residual_oof import panel_data  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=LAB / "models" / "residual_lgb.txt")
    ap.add_argument("--rounds", type=int, default=400)
    args = ap.parse_args()
    d = {p: panel_data(p, "oof") for p in C.T2_PANELS}
    d = {p: v for p, v in d.items() if v is not None}
    X = np.concatenate([d[p][0][d[p][2]] for p in d])
    y = np.concatenate([d[p][1][d[p][2]].astype(np.float32) for p in d])
    p0 = np.concatenate([d[p][3][d[p][2]] for p in d])
    print(f"{len(y):,} cells, blend wrong on {(((p0 >= TAU) != (y >= 0.5)).mean() * 100):.2f}%", flush=True)
    gbm = lgb.train({"objective": "l2", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 200,
                     "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1,
                     "num_threads": 8},
                    lgb.Dataset(X, label=y - p0, feature_name=NAMES, free_raw_data=False),
                    num_boost_round=args.rounds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    gbm.save_model(str(args.out))
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
