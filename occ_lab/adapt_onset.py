"""Idea 9b on top of the single-horizon onset models, on the official-like windows, so it can be scored locally.

`scripts/t2_adapt_official.py` does this for the real official windows; this does the same thing for the
blank-aware pseudo windows the local onset evaluation uses, so the adapted single-horizon model can be compared
with the adapted production model (`ref["on_ad9b"]`) that candidate AI ships.

Per panel and split, over the days that carry an onset window, in ascending order:
    1. predict that day's onset windows with the current weights (T+30 map)
    2. adapt the trunk and the reconstruction head on history windows drawn from STRICTLY EARLIER days of the
       released layer - masked reconstruction only, no Task 2 label (30 steps, lr 1e-4)
The weights carry forward, so nothing is ever predicted with weights that saw its own day.

Writes occ_lab/cache/onset_adapt_<tag>_<split>_<panel>.npz with prob (N, S) in the window order of
`common.windows`.

    python occ_lab/adapt_onset.py --ckpt occ_lab/models/onset30rec_cnn_s20.pt --tag onset30_s20
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ["T2CNN_ARCH"] = "cnn"
os.environ.setdefault("T2CNN_REC", "0.3")
LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta"))
import numpy as np  # noqa: E402

import common as C  # noqa: E402

CACHE = LAB / "cache"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=16)
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for i, panel in enumerate(C.T2_PANELS):
        ctx, stat, W = C.windows(i, panel)
        for s in C.SPLITS:
            on, z = W[s]["on"], W[s]["z"]
            if not len(on["d"]):
                continue
            net, params = C.build_net(args.ckpt, "full")
            rng = np.random.default_rng(20 + i)
            prob = np.zeros((len(on["d"]), ctx.S), np.float32)
            for day in np.unique(on["d"]):
                idx = np.flatnonzero(on["d"] == day)
                x = C.make_x(ctx, stat, on, idx, z)
                prob[idx] = C.cnn.predict(net, x)[:, C.F - 1]          # before adapting on this day's past
                pool = np.arange(day)                                   # strictly earlier days only
                if len(pool):
                    C.adapt(net, params, lambda: C.past_batch(ctx, stat, z, pool, args.bs, rng),
                            args.steps, args.lr)
            np.savez(CACHE / f"onset_adapt_{args.tag}_{s}_{panel}.npz", prob=prob, d=on["d"])
        print(f"  {panel} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
