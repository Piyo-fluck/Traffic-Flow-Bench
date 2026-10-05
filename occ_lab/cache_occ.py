"""Cache the ongoing probabilities so blend weights can be searched without touching the GPU again.

Writes occ_lab/cache/og_<split>_<panel>.npz with the labels, the eligibility mask, the three production member
probabilities and one array per occupancy checkpoint (occ_cnn_s0, occ_tfm_s1, ...). Everything is float16, which
is well inside the resolution that a 0.50 threshold needs.

    python occ_lab/cache_occ.py --cnn occ_lab/models/occ_cnn_s*.pt --tfm occ_lab/models/occ_tfm_s*.pt
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

CACHE = LAB / "cache"
BLOCK = 384


def load(p):
    arch = "tfm" if "_tfm_" in p.name else "cnn"
    sd = torch.load(p, map_location=C.DEV)
    cin = sd["gru.weight_ih_l0"].shape[1] if arch == "tfm" else sd["inp.weight"].shape[1]
    net = (C.cnn.T2Tfm if arch == "tfm" else C.cnn.T2Net)(cin, C.cnn.WIDTH)
    net.load_state_dict(sd)
    return net.to(C.DEV).eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnn", type=Path, nargs="*", default=[])
    ap.add_argument("--tfm", type=Path, nargs="*", default=[])
    args = ap.parse_args()
    C.cnn.REC = 0.0
    CACHE.mkdir(parents=True, exist_ok=True)
    nets = {p.stem: load(p) for p in list(args.cnn) + list(args.tfm)}
    profs = np.load(LAB / "models" / "occ_profiles.npz")
    for i, panel in enumerate(C.T2_PANELS):
        ctx, stat, W = C.windows(i, panel)
        for s in C.SPLITS:
            og, ref, z = W[s]["og"], W[s]["ref"], W[s]["z"]
            if not len(og["d"]):
                continue
            out = {"y": og["fut"], "e": og["efut"], "gs1": ref["og_gs1"].astype(np.float16),
                   "tfm8": ref["og_tfm8"].astype(np.float16), "cnn3": ref["og_cnn3"].astype(np.float16)}
            P = {k: np.zeros(og["fut"].shape, np.float16) for k in nets}
            for a in range(0, len(og["d"]), BLOCK):
                m = np.arange(a, min(a + BLOCK, len(og["d"])))
                hi = og["t0"][m][:, None] + np.arange(-C.H, 0)
                occ = z["obs_occ"][og["d"][m][:, None], hi]
                x = np.concatenate([C.make_x(ctx, stat, og, m, z), occ_channels(occ, og["ok"][m], profs[panel])], 1)
                for k, n in nets.items():
                    P[k][m] = C.cnn.predict(n, x).astype(np.float16)
            np.savez(CACHE / f"og_{s}_{panel}.npz", **out, **P)
        print(f"  {panel} cached", flush=True)


if __name__ == "__main__":
    main()
