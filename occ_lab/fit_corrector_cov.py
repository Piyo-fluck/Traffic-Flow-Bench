"""Fit the `all` corrector of note 87 on the whole held-out tail and save it.

All five coverage environments of `cache_cov.py` (obs, c50, c60, c70, full), the 15 features of candidate AI plus
the window's coverage, unweighted L2 on r = y - p0, 400 trees.

    python occ_lab/fit_corrector_cov.py --out occ_lab/models/residual_lgb_cov.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta"))
import numpy as np  # noqa: E402

import common as C  # noqa: E402
from residual_cov import CACHE, ENVS, FNAMES, fit, load_env  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=LAB / "models" / "residual_lgb_cov.txt")
    args = ap.parse_args()
    Xs, ys, ps = [], [], []
    for p in C.T2_PANELS:
        lab = np.load(CACHE / f"cov_labels_{p}.npz")
        for v in ENVS:
            f, base, _ = load_env(v, p)
            m = lab["e"]
            Xs.append(f[m]); ys.append(lab["y"][m].astype(np.float32)); ps.append(base[m])
    X, y, p0 = np.concatenate(Xs), np.concatenate(ys), np.concatenate(ps)
    print(f"{len(y):,} cells over {len(ENVS)} environments", flush=True)
    gbm, _ = fit(X, y, p0, FNAMES)
    gbm.save_model(str(args.out))
    print(f"saved {args.out}; features {gbm.feature_name()}", flush=True)


if __name__ == "__main__":
    main()
