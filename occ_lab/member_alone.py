"""What each ongoing member scores on its own, to size the single model that has to replace twelve of them."""
from __future__ import annotations

import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta"))
sys.path.insert(0, str(LAB.parent / "analysis"))
import numpy as np  # noqa: E402

import common as C  # noqa: E402
from online_adapter import load_split  # noqa: E402
from residual_diag import TAU  # noqa: E402

fam = {}
for i, panel in enumerate(C.T2_PANELS):
    for s, D in load_split(i, panel, "s20").items():
        P = dict(D["P"])
        P["p_PS"] = D["ps"]
        P["old (12 models)"] = D["old"]
        P["0.2 gs1 + 0.6 occ_tfm + 0.2 occ_cnn"] = 0.2 * P["gs1"] + 0.6 * P["occ_tfm"] + 0.2 * P["occ_cnn"]
        for k, v in P.items():
            fam.setdefault((s, k), {})[panel] = C.iou(v >= TAU, D["y"], D["e"]).mean()
            fam.setdefault((s, k + "  + p_PS 0.6"), {})[panel] = C.iou(
                (0.4 * v + 0.6 * D["ps"]) >= TAU, D["y"], D["e"]).mean()
    print(f"  {panel}", flush=True)

names = ["gs1", "tfm8", "cnn3", "occ_cnn", "occ_tfm", "stid", "p_PS",
         "0.2 gs1 + 0.6 occ_tfm + 0.2 occ_cnn", "old (12 models)"]
print(f"\nongoing family IoU at tau 0.50\n")
print(f"{'member':<38}{'alone':>9}{'blended with p_PS 0.6':>24}")
for nm in names:
    a = np.mean([C.family_mean(fam[(s, nm)]) for s in C.SPLITS])
    b = np.mean([C.family_mean(fam[(s, nm + "  + p_PS 0.6")]) for s in C.SPLITS])
    print(f"{nm:<38}{a:>9.4f}{b:>24.4f}")
