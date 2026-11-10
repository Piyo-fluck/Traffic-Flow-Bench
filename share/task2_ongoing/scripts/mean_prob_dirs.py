"""Average the official-window probabilities of several seed directories (prob_<split>_<panel>.npz) into one.

Used for the 4-seed p_PS (sub_t2pseudo_ssl_s20..s23 -> sub_t2pseudo_ssl4) and the 8-seed adapted transformers
(sub_t2tfmps_s3..s10 -> sub_t2tfmps8). Window order is checked to be identical across the seeds.

    python scripts/mean_prob_dirs.py --out work/sub_t2pseudo_ssl4 work/sub_t2pseudo_ssl_s20 ... work/sub_t2pseudo_ssl_s23
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("dirs", type=Path, nargs="+")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    names = sorted(p.name for p in args.dirs[0].glob("prob_*.npz"))
    assert names, f"no prob_*.npz in {args.dirs[0]}"
    for name in names:
        zs = [np.load(d / name) for d in args.dirs]
        assert all((z["window_id"] == zs[0]["window_id"]).all() for z in zs), f"window order differs: {name}"
        np.savez(args.out / name, prob=np.mean([z["prob"] for z in zs], 0), is_onset=zs[0]["is_onset"],
                 window_id=zs[0]["window_id"], t0=zs[0]["t0"])
    print(f"{args.out}: {len(names)} files, mean of {len(args.dirs)} seeds")


if __name__ == "__main__":
    main()
