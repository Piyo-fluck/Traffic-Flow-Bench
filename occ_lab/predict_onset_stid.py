"""Onset STID probabilities on the official-like onset windows, in the layout `eval_onset.py` reads.

    python occ_lab/predict_onset_stid.py --ckpt occ_lab/models/onset_stid_s*.pt --tag stid30
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
from graph_ident import adj as chain_adj, lappe  # noqa: E402
from train_occ import occ_channels  # noqa: E402
from train_onset_stid import OnsetSTID, predict  # noqa: E402

CACHE = LAB / "cache"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, nargs="+", required=True)
    ap.add_argument("--tag", default="stid30")
    args = ap.parse_args()
    C.cnn.REC = 0.0
    CACHE.mkdir(parents=True, exist_ok=True)
    nets, ident, k_lap = [], "id", 12
    for p in args.ckpt:
        d = torch.load(p, map_location=C.DEV)
        ident, k_lap = d.get("ident", "id"), d.get("k_lap", 12)
        net = OnsetSTID(d["cin"], d["n_links"], d["n_slots"], d=d.get("d", 96), ds=d.get("ds", 32), blocks=d.get("blocks", 3),
                        ident=ident, k_lap=k_lap).to(C.DEV)
        net.load_state_dict(d["state"])
        nets.append(net.eval())
    print(f"{len(nets)} checkpoints, identity '{ident}'", flush=True)
    profs = np.load(LAB / "models" / "occ_profiles.npz")
    loff = 0
    for i, panel in enumerate(C.T2_PANELS):
        ctx, stat, W = C.windows(i, panel)
        lid = np.arange(ctx.S) + loff
        loff += ctx.S
        lap = lappe(ctx, k_lap) if ident == "lappe" else None
        adjm = chain_adj(ctx) if ident == "ab" else None
        for s in C.SPLITS:
            on, z = W[s]["on"], W[s]["z"]
            if not len(on["d"]):
                continue
            idx = np.arange(len(on["d"]))
            hi = on["t0"][:, None] + np.arange(-C.H, 0)
            occ = z["obs_occ"][on["d"][:, None], hi]
            x = np.concatenate([C.make_x(ctx, stat, on, idx, z), occ_channels(occ, on["ok"], profs[panel])], 1)
            dow = z["weekday"][on["d"]].astype(np.int64)
            p = np.mean([predict(n, x, lid, on["t0"].astype(np.int64), dow, lap=lap, adj=adjm)
                         for n in nets], 0)
            np.savez(CACHE / f"onset_adapt_{args.tag}_{s}_{panel}.npz", prob=p.astype(np.float32), d=on["d"])
        print(f"  {panel}", flush=True)


if __name__ == "__main__":
    main()
