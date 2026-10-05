"""4-seed vs 8-seed onset STID at the weights candidate AN ships, and a couple of neighbours."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["T2CNN_ARCH"] = "cnn"
LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta"))
import numpy as np  # noqa: E402

import common as C  # noqa: E402

CACHE = LAB / "cache"
W = [(0.70, 0.30, 0.00), (0.35, 0.35, 0.30), (0.45, 0.40, 0.15), (0.30, 0.35, 0.35), (0.50, 0.40, 0.10)]


def main() -> None:
    fam = {}
    for i, panel in enumerate(C.T2_PANELS):
        _, _, Wd = C.windows(i, panel)
        for s in C.SPLITS:
            on, ref = Wd[s]["on"], Wd[s]["ref"]
            if not len(on["d"]):
                continue
            H, ad = ref["on_H"], ref["on_ad9b"]
            for tag in ("stid30", "stid30x8"):
                f = CACHE / f"onset_adapt_{tag}_{s}_{panel}.npz"
                if not f.exists():
                    continue
                st = np.load(f)["prob"]
                for wh, wa, ws in W:
                    fam.setdefault((s, tag, (wh, wa, ws)), {})[panel] = \
                        C.onset_iou(wh * H + wa * ad + ws * st, on).mean()
        print(f"  {panel}", flush=True)
    for s in C.SPLITS:
        b = C.family_mean(fam[(s, "stid30", (0.70, 0.30, 0.00))])
        print(f"\n=== onset {s}: production {b:.4f} ===")
        for (ss, tag, w), d in fam.items():
            if ss == s and w != (0.70, 0.30, 0.00):
                n = "4 seeds" if tag == "stid30" else "8 seeds"
                print(f"   H {w[0]:.2f} ad9b {w[1]:.2f} STID {w[2]:.2f}  {n}  "
                      f"{C.family_mean(d):.4f} ({C.family_mean(d) - b:+.4f})")


if __name__ == "__main__":
    main()
