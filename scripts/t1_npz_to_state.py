"""Assemble Task 1 state CSVs from pred_<split>_<panel>.npz files (torch-free), checking each panel against its template.

    py scripts/t1_npz_to_state.py --pred-dir work/sub_m3ens3_month
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
    ap.add_argument("--pred-dir", type=Path, required=True)
    args = ap.parse_args()
    for split in ("validation", "private"):
        out = args.pred_dir / f"state_{split}.csv"
        for i, panel in enumerate(PANELS):
            with np.load(args.pred_dir / f"pred_{split}_{panel}.npz") as f:
                pv, pq = f["pv"], f["pq"]
            z = load(panel, split)
            frame = state_frame(panel, z, pv, pq)
            check_against_template(panel, split, frame)
            append_csv(frame, out, first=(i == 0))
            print(f"[{split} {panel}] {len(frame):,} rows", flush=True)
        print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
