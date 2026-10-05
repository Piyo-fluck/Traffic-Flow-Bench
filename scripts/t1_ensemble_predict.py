"""Average the validation/private predictions of several Task 1 production runs into one state file per split.

Each run directory holds pred_<split>_<panel>.npz (pv, pq) written by scripts/t1_m3_predict.py.

    py scripts/t1_ensemble_predict.py --runs work/sub_m3v1 work/sub_m3v1_s1 work/sub_m3w128 --out-dir work/sub_m3ens3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tfb.data import PANELS, load  # noqa: E402
from tfb.submit import append_csv, check_against_template, state_frame  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("validation", "private"):
        out = args.out_dir / f"state_{split}.csv"
        for i, panel in enumerate(PANELS):
            z = load(panel, split)
            preds = [np.load(r / f"pred_{split}_{panel}.npz") for r in args.runs]
            pv = np.mean([p["pv"] for p in preds], axis=0)
            pq = np.mean([p["pq"] for p in preds], axis=0)
            frame = state_frame(panel, z, pv, pq)
            check_against_template(panel, split, frame)
            append_csv(frame, out, first=(i == 0))
            spread = np.nanmean(np.std([p["pv"][z["target"]] for p in preds], axis=0))
            print(f"[{split} {panel}] {len(frame):,} rows, mean speed spread across runs {spread:.2f} km/h", flush=True)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
