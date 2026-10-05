"""Score the single replacement model against the twelve it is meant to replace.

The bar, from `occ_lab/member_alone.py` on the official-like pool of both scored months:

    tfm8 alone                      0.8191          old (12 models) alone      0.8295
    0.4 tfm8  + 0.6 p_PS  0.8437    0.4 old + 0.6 p_PS         0.8424

so one model has to reach 0.8191 alone and 0.8437 blended to retire eight seeds, and 0.8295 / 0.8424 to retire
all twelve.

    T2CNN_WIDTH=128 python occ_lab/eval_one.py --ckpt occ_lab/models/one_tfm_w128_s0.pt
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("T2CNN_ARCH", "tfm")
os.environ.setdefault("T2CNN_LOVASZ", "0.5")
LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta"))
sys.path.insert(0, str(LAB.parent / "analysis"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import common as C  # noqa: E402
from online_adapter import load_split  # noqa: E402
from residual_diag import TAU  # noqa: E402
from train_occ import occ_channels  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--profiles", type=Path, default=LAB / "models" / "occ_profiles_one.npz")
    ap.add_argument("--tag", default="s20")
    ap.add_argument("--w-ps", type=float, nargs="+", default=[0.4, 0.5, 0.6, 0.7])
    args = ap.parse_args()
    profs = np.load(args.profiles)
    sd = torch.load(args.ckpt, map_location=C.DEV)
    cin = sd["gru.weight_ih_l0"].shape[1] if "gru.weight_ih_l0" in sd else sd["inp.weight"].shape[1]
    net = (C.cnn.T2Tfm if C.cnn.ARCH == "tfm" else C.cnn.T2Net)(cin, C.cnn.WIDTH).to(C.DEV)
    net.load_state_dict(sd)
    net.eval()
    print(f"{args.ckpt.name}: {C.cnn.ARCH} width {C.cnn.WIDTH}, {cin} channels", flush=True)

    fam = {}
    for i, panel in enumerate(C.T2_PANELS):
        ctx, stat, Wd = C.windows(i, panel)
        sp = load_split(i, panel, args.tag)
        for s, D in sp.items():
            og, z = Wd[s]["og"], Wd[s]["z"]
            m = np.arange(len(og["d"]))
            x = C.make_x(ctx, stat, og, m, z)
            if cin > x.shape[1]:                             # the checkpoint wants the occupancy channels
                hi = og["t0"][:, None] + np.arange(-C.H, 0)
                x = np.concatenate([x, occ_channels(z["obs_occ"][og["d"][:, None], hi], og["ok"],
                                                    profs[panel])], 1)
            p1 = C.cnn.predict(net, x)
            y, e, ps = D["y"], D["e"], D["ps"]
            fam.setdefault((s, "one alone"), {})[panel] = C.iou(p1 >= TAU, y, e).mean()
            fam.setdefault((s, "tfm8 alone"), {})[panel] = C.iou(D["P"]["tfm8"] >= TAU, y, e).mean()
            fam.setdefault((s, "old alone"), {})[panel] = C.iou(D["old"] >= TAU, y, e).mean()
            for w in args.w_ps:
                for nm, q in (("one", p1), ("tfm8", D["P"]["tfm8"]), ("old", D["old"])):
                    fam.setdefault((s, f"{nm} + p_PS {w:g}"), {})[panel] = C.iou(
                        ((1 - w) * q + w * ps) >= TAU, y, e).mean()
        ctx.z = ctx.q = ctx.el = None
        print(f"  {panel}", flush=True)

    names = ["one alone", "tfm8 alone", "old alone"] + \
        [f"{nm} + p_PS {w:g}" for w in args.w_ps for nm in ("one", "tfm8", "old")]
    print(f"\nongoing family IoU at tau 0.50, official-like pool of the scored months\n")
    print(f"{'setting':<22}" + "".join(f"{s:>13}" for s in C.SPLITS) + f"{'mean':>10}")
    for nm in names:
        v = [C.family_mean(fam[(s, nm)]) for s in C.SPLITS]
        print(f"{nm:<22}" + "".join(f"{x:>13.4f}" for x in v) + f"{float(np.mean(v)):>10.4f}")


if __name__ == "__main__":
    main()
