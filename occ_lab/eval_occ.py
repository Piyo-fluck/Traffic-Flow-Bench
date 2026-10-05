"""Score the occupancy-fed members, with the user's criterion as the headline.

For every candidate model m the headline is not its own IoU but

    D_m = P(m is right | the production blend is wrong)      over the eligible ongoing cells,

next to the cell-level error correlation with the current members and what the blend does when m replaces (or
joins) its family. Note 83's reference numbers for a new architecture: 0.13-0.14% of cells rescued against
1.76-1.86% spoiled, correlations 0.43-0.57.

    python occ_lab/eval_occ.py --cnn occ_lab/models/occ_cnn_s0.pt occ_lab/models/occ_cnn_s1.pt \
                               --tfm occ_lab/models/occ_tfm_s0.pt occ_lab/models/occ_tfm_s1.pt
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

TAU = 0.50
BLOCK = 384


def load(p, arch):
    sd = torch.load(p, map_location=C.DEV)
    cin = sd["gru.weight_ih_l0"].shape[1] if arch == "tfm" else sd["inp.weight"].shape[1]
    net = (C.cnn.T2Tfm if arch == "tfm" else C.cnn.T2Net)(cin, C.cnn.WIDTH)
    net.load_state_dict(sd)
    return net.to(C.DEV).eval()


def occ_predict(nets, ctx, stat, og, z, prof):
    """Mean probability of an ensemble, in blocks so the 15-channel input never exists in full."""
    out = np.zeros(og["fut"].shape, np.float32)
    for a in range(0, len(og["d"]), BLOCK):
        m = np.arange(a, min(a + BLOCK, len(og["d"])))
        hi = og["t0"][m][:, None] + np.arange(-C.H, 0)
        occ = z["obs_occ"][og["d"][m][:, None], hi]           # the released layer, blanks and all
        x = np.concatenate([C.make_x(ctx, stat, og, m, z), occ_channels(occ, og["ok"][m], prof)], 1)
        out[m] = np.mean([C.cnn.predict(n, x) for n in nets], 0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnn", type=Path, nargs="*", default=[])
    ap.add_argument("--tfm", type=Path, nargs="*", default=[])
    args = ap.parse_args()
    C.cnn.REC = 0.0
    nets = {"occ_cnn": [load(p, "cnn") for p in args.cnn], "occ_tfm": [load(p, "tfm") for p in args.tfm]}
    nets = {k: v for k, v in nets.items() if v}
    profs = np.load(LAB / "models" / "occ_profiles.npz")
    fam, stats = {}, {}
    for i, panel in enumerate(C.T2_PANELS):
        ctx, stat, W = C.windows(i, panel)
        for s in C.SPLITS:
            og, ref, z = W[s]["og"], W[s]["ref"], W[s]["z"]
            if not len(og["d"]):
                continue
            P = {k: occ_predict(v, ctx, stat, og, z, profs[panel]) for k, v in nets.items()}
            bl = 0.2 * ref["og_gs1"] + 0.6 * ref["og_tfm8"] + 0.2 * ref["og_cnn3"]
            y, e = og["fut"], og["efut"]
            variants = {"blend (T)": bl, "gs1": ref["og_gs1"], "tfm8": ref["og_tfm8"], "cnn3": ref["og_cnn3"]}
            for k, p in P.items():
                variants[f"{k} alone"] = p
                for w in (0.15, 0.30):
                    variants[f"blend + {k} {w:.2f}"] = (1 - w) * bl + w * p
            if "occ_cnn" in P:
                variants["cnn3 slot -> occ_cnn"] = 0.2 * ref["og_gs1"] + 0.6 * ref["og_tfm8"] + 0.2 * P["occ_cnn"]
            if "occ_tfm" in P:
                variants["tfm8 slot -> occ_tfm"] = 0.2 * ref["og_gs1"] + 0.6 * P["occ_tfm"] + 0.2 * ref["og_cnn3"]
            if len(P) == 2:
                variants["both slots -> occ"] = 0.2 * ref["og_gs1"] + 0.6 * P["occ_tfm"] + 0.2 * P["occ_cnn"]
                variants["blend + both 0.30"] = 0.7 * bl + 0.15 * P["occ_tfm"] + 0.15 * P["occ_cnn"]
            for n, p in variants.items():
                fam.setdefault((s, n), {})[panel] = C.iou(p >= TAU, y, e).mean()
            bl_wrong = ((bl >= TAU) != y)[e]
            for k, p in list(P.items()) + [("cnn3 (reference)", ref["og_cnn3"])]:
                wrong = ((p >= TAU) != y)[e]
                a = stats.setdefault((s, k), [0, 0, 0, 0, []])
                a[0] += int((bl_wrong & ~wrong).sum()); a[1] += int(bl_wrong.sum())
                a[2] += int((~bl_wrong & wrong).sum()); a[3] += int((~bl_wrong).sum())
                a[4].append(float(np.corrcoef(bl_wrong, wrong)[0, 1]))
        print(f"  {panel}", flush=True)
    for s in C.SPLITS:
        b = C.family_mean(fam[(s, "blend (T)")])
        print(f"\n=== ongoing {s}: blend family {b:.4f} ===")
        for (ss, n), d in fam.items():
            if ss == s and n != "blend (T)":
                print(f"   {n:<28} {C.family_mean(d):.4f} ({C.family_mean(d) - b:+.4f})")
        for (ss, k), a in stats.items():
            if ss == s:
                print(f"   D_m {k:<18} right where the blend is wrong {a[0] / max(a[1], 1):.3f} "
                      f"({a[0]:,}/{a[1]:,}); wrong where the blend is right {a[2] / max(a[3], 1):.4f}; "
                      f"error corr {np.mean(a[4]):.2f}")


if __name__ == "__main__":
    main()
