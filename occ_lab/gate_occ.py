"""Where the blend is uncertain, is the occupancy CNN actually better - and does overriding there pay?

The sweep says a flat weight buys +0.0004/+0.0008 family IoU, which is nothing, and the reason is arithmetic: the
blend is already right on 98.5% of the cells, so a member that spoils 0.9-1.1% of those cancels its 28-32% rescue
rate on the 1.5% it gets wrong. A member only pays flat if spoil < rescue x 0.015.

That leaves the gated form: use the occupancy model ONLY on the cells where the blend is near its threshold, where
the base rate is not 98.5% but closer to a coin flip (note 80: the disputed cells are 70-73% right). This measures,
for several definitions of "near", the accuracy of the blend, of the occupancy CNN, and of the mixture there, and
the family IoU that a gated override produces.

    python occ_lab/gate_occ.py
"""
from __future__ import annotations

import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta"))
import numpy as np  # noqa: E402

import common as C  # noqa: E402

CACHE = LAB / "cache"
TAU = 0.50
BANDS = (0.05, 0.10, 0.15, 0.20, 0.30)
MIXW = (0.25, 0.50, 0.75, 1.00)


def main() -> None:
    fam, acc = {}, {}
    for panel in C.T2_PANELS:
        for s in C.SPLITS:
            f = CACHE / f"og_{s}_{panel}.npz"
            if not f.exists():
                continue
            z = np.load(f)
            y, e = z["y"], z["e"]
            occ = np.mean([z[k].astype(np.float32) for k in z.files if k.startswith("occ_cnn")], 0)
            occt = np.mean([z[k].astype(np.float32) for k in z.files if k.startswith("occ_tfm")], 0)
            gs1, tfm8, cnn3 = (z[k].astype(np.float32) for k in ("gs1", "tfm8", "cnn3"))
            base = 0.2 * gs1 + 0.6 * tfm8 + 0.2 * cnn3
            fam.setdefault((s, "blend (T)"), {})[panel] = C.iou(base >= TAU, y, e).mean()
            for d in BANDS:
                near = np.abs(base - TAU) < d
                for w in MIXW:
                    for nm, q in (("occ_cnn", occ), ("occ_both", 0.5 * occ + 0.5 * occt)):
                        p = np.where(near, (1 - w) * base + w * q, base)
                        fam.setdefault((s, f"gate |p-.5|<{d:.2f} w={w:.2f} {nm}"), {})[panel] = \
                            C.iou(p >= TAU, y, e).mean()
                m = near & e
                a = acc.setdefault((s, d), [0, 0, 0, 0, 0])
                a[0] += int(m.sum())
                a[1] += int((((base >= TAU) == y) & m).sum())
                a[2] += int((((occ >= TAU) == y) & m).sum())
                a[3] += int(((((0.5 * base + 0.5 * occ) >= TAU) == y) & m).sum())
                a[4] += int((y & m).sum())
    for s in C.SPLITS:
        b = C.family_mean(fam[(s, "blend (T)")])
        print(f"\n=== ongoing {s}: blend family {b:.4f} ===")
        for v, n in sorted(((C.family_mean(d), n) for (ss, n), d in fam.items() if ss == s), reverse=True)[:12]:
            print(f"   {n:<40} {v:.4f} ({v - b:+.4f})")
        print("   accuracy on the near-threshold cells (blend / occ_cnn / half-and-half), share queued:")
        for (ss, d), a in acc.items():
            if ss == s and a[0]:
                print(f"     |p-.5| < {d:.2f}: n={a[0]:>9,} ({a[0] / 1e6:.2f}M)  "
                      f"{a[1] / a[0]:.3f} / {a[2] / a[0]:.3f} / {a[3] / a[0]:.3f}   queued {a[4] / a[0]:.3f}")


if __name__ == "__main__":
    main()
