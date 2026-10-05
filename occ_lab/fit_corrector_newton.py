"""Fit the Newton / LogitBoost corrector of note 88 on the whole held-out tail and save it.

AI's 15 features and AI's training data (the tail's masked-layer windows, `cache_cov.py` env obs); target
z = (y - p0) / (p0 (1 - p0) + 1e-3) clipped to +-4, weights p0 (1 - p0) + 1e-3, 400 trees. Applied as
p1 = sigmoid(logit p0 + eta * z_hat) by `scripts/t2_residual_predict.py --update newton`.

    python occ_lab/fit_corrector_newton.py --out occ_lab/models/residual_lgb_newton.txt
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
from logit_update import EPS, IDX, ZMAX  # noqa: E402
from residual_cov import CACHE, PARAMS, load_env  # noqa: E402
from residual_diag import NAMES  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=LAB / "models" / "residual_lgb_newton.txt")
    args = ap.parse_args()
    Xs, ys, ps = [], [], []
    for p in C.T2_PANELS:
        lab = np.load(CACHE / f"cov_labels_{p}.npz")
        f, base, _ = load_env("obs", p)
        m = lab["e"]
        Xs.append(f[m][:, IDX]); ys.append(lab["y"][m].astype(np.float32)); ps.append(base[m])
    X, y, p0 = np.concatenate(Xs), np.concatenate(ys), np.concatenate(ps)
    w = p0 * (1 - p0)
    z = np.clip((y - p0) / (w + EPS), -ZMAX, ZMAX)
    gbm = lgb.train(PARAMS, lgb.Dataset(X, label=z, weight=w + EPS, feature_name=NAMES, free_raw_data=False),
                    num_boost_round=400)
    gbm.save_model(str(args.out))
    print(f"{len(y):,} cells; saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
