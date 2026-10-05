"""Add the STID probabilities to the ongoing cache, in the same window order as occ_lab/cache_occ.py.

Writes occ_lab/cache/stid_<split>_<panel>.npz with one array per checkpoint; occ_lab/sweep_occ.py picks them up.

    python occ_lab/cache_stid.py --ckpt occ_lab/models/stid_s0.pt occ_lab/models/stid_s1.pt
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ["T2CNN_ARCH"] = "cnn"
LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import common as C  # noqa: E402
from train_occ import occ_channels  # noqa: E402
from train_stid import STID, predict  # noqa: E402

CACHE = LAB / "cache"
BLOCK = 384


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, nargs="+", required=True)
    args = ap.parse_args()
    C.cnn.REC = 0.0
    CACHE.mkdir(parents=True, exist_ok=True)
    nets = {}
    for p in args.ckpt:
        d = torch.load(p, map_location=C.DEV)
        net = STID(d["cin"], d["n_links"], d["n_slots"]).to(C.DEV)
        net.load_state_dict(d["state"])
        nets[p.stem] = net.eval()
    profs = np.load(LAB / "models" / "occ_profiles.npz")
    loff = 0
    for i, panel in enumerate(C.T2_PANELS):
        ctx, stat, W = C.windows(i, panel)
        lid = np.arange(ctx.S) + loff                       # the offsets of train_stid, in the same panel order
        loff += ctx.S
        for s in C.SPLITS:
            og, z = W[s]["og"], W[s]["z"]
            if not len(og["d"]):
                continue
            dow = z["weekday"][og["d"]].astype(np.int64)
            P = {k: np.zeros(og["fut"].shape, np.float16) for k in nets}
            for a in range(0, len(og["d"]), BLOCK):
                m = np.arange(a, min(a + BLOCK, len(og["d"])))
                hi = og["t0"][m][:, None] + np.arange(-C.H, 0)
                occ = z["obs_occ"][og["d"][m][:, None], hi]
                x = np.concatenate([C.make_x(ctx, stat, og, m, z), occ_channels(occ, og["ok"][m], profs[panel])], 1)
                for k, n in nets.items():
                    P[k][m] = predict(n, x, lid, og["t0"][m].astype(np.int64), dow[m]).astype(np.float16)
            np.savez(CACHE / f"stid_{s}_{panel}.npz", **P)
        print(f"  {panel} cached (links {ctx.S})", flush=True)


if __name__ == "__main__":
    main()
