"""Search ongoing blend weights over the cached probabilities, with D_m reported for every member.

The production blend is 0.20 gs1 + 0.60 tfm8 + 0.20 cnn3 at tau 0.50. The question this answers is not whether a
new member is good on its own - it is trained on a few seeds against three and eight - but whether a slot is
better spent on a mixture, and by how much. Reads occ_lab/cache/og_*.npz (cache_occ.py) and, when present,
occ_lab/cache/stid_*.npz (cache_stid.py).

    python occ_lab/sweep_occ.py
"""
from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta"))
import numpy as np  # noqa: E402

import common as C  # noqa: E402

CACHE = LAB / "cache"
TAU = 0.50


def mean_of(z, prefix):
    ks = [k for k in z.files if k.startswith(prefix)]
    return np.mean([z[k].astype(np.float32) for k in ks], 0) if ks else None


def main() -> None:
    fam, stat = {}, {}
    for panel in C.T2_PANELS:
        for s in C.SPLITS:
            f = CACHE / f"og_{s}_{panel}.npz"
            if not f.exists():
                continue
            z = np.load(f)
            y, e = z["y"], z["e"]
            occ_cnn, occ_tfm = mean_of(z, "occ_cnn"), mean_of(z, "occ_tfm")
            gs1, tfm8, cnn3 = (z[k].astype(np.float32) for k in ("gs1", "tfm8", "cnn3"))
            fs = CACHE / f"stid_{s}_{panel}.npz"
            stid = mean_of(np.load(fs), "stid") if fs.exists() else None
            base = 0.2 * gs1 + 0.6 * tfm8 + 0.2 * cnn3
            V = {"blend (T)": base}
            for a in (0.05, 0.10, 0.15, 0.20):
                V[f"cnn slot {0.20 - a:.2f} cnn3 + {a:.2f} occ_cnn"] = base + a * (occ_cnn - cnn3)
                V[f"tfm slot {0.60 - a:.2f} tfm8 + {a:.2f} occ_tfm"] = base + a * (occ_tfm - tfm8)
            for a, b in product((0.05, 0.10), (0.05, 0.10)):
                V[f"cnn +{a:.2f} occ_cnn / tfm +{b:.2f} occ_tfm"] = base + a * (occ_cnn - cnn3) + b * (occ_tfm - tfm8)
            if stid is not None:
                for a in (0.05, 0.10, 0.15, 0.20, 0.30):
                    V[f"blend + stid {a:.2f} (flat)"] = (1 - a) * base + a * stid
                    V[f"cnn slot {0.20 - a:.2f} cnn3 + {a:.2f} stid"] = base + min(a, 0.2) * (stid - cnn3)
                    V[f"tfm slot {0.60 - a:.2f} tfm8 + {a:.2f} stid"] = base + a * (stid - tfm8)
                    V[f"gs1 slot {0.20 - a:.2f} + {a:.2f} stid"] = base + min(a, 0.2) * (stid - gs1)
                for a, b in product((0.05, 0.10), (0.05, 0.10)):
                    V[f"stid {a:.2f} (from cnn3) / occ_cnn {b:.2f} (from tfm8)"] = (
                        base + a * (stid - cnn3) + b * (occ_cnn - tfm8))
                    V[f"stid {a:.2f} + occ_tfm {b:.2f} (both from tfm8)"] = (
                        base + a * (stid - tfm8) + b * (occ_tfm - tfm8))
            for n, p in V.items():
                fam.setdefault((s, n), {})[panel] = C.iou(p >= TAU, y, e).mean()
            bw = ((base >= TAU) != y)[e]
            members = [("occ_cnn", occ_cnn), ("occ_tfm", occ_tfm), ("cnn3", cnn3), ("tfm8", tfm8), ("gs1", gs1)]
            if stid is not None:
                members.insert(0, ("stid", stid))
            for n, p in members:
                w = ((p >= TAU) != y)[e]
                a = stat.setdefault((s, n), [0, 0, 0, 0, [], 0, 0])
                a[0] += int((bw & ~w).sum()); a[1] += int(bw.sum())
                a[2] += int((~bw & w).sum()); a[3] += int((~bw).sum())
                a[4].append(float(np.corrcoef(bw, w)[0, 1]))
                a[5] += int((~w).sum()); a[6] += int(e.sum())
    for s in C.SPLITS:
        b = C.family_mean(fam[(s, "blend (T)")])
        print(f"\n=== ongoing {s}: blend family {b:.4f} ===")
        for v, n in sorted(((C.family_mean(d), n) for (ss, n), d in fam.items() if ss == s), reverse=True)[:14]:
            print(f"   {n:<48} {v:.4f} ({v - b:+.4f})")
        print("   (worst 3)")
        for v, n in sorted(((C.family_mean(d), n) for (ss, n), d in fam.items() if ss == s))[:3]:
            print(f"   {n:<48} {v:.4f} ({v - b:+.4f})")
        for (ss, n), a in stat.items():
            if ss == s:
                print(f"   D_m {n:<9} rescue {a[0] / max(a[1], 1):.3f}  spoil {a[2] / max(a[3], 1):.4f}  "
                      f"corr {np.mean(a[4]):.2f}  own cell accuracy {a[5] / max(a[6], 1):.4f}")


if __name__ == "__main__":
    main()
