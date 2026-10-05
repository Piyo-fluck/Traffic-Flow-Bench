"""STID is the most orthogonal member measured so far (corr 0.48-0.50 against the blend, where cnn3 sits at
0.66-0.68) and it rescues 0.27-0.31 of the blend's errors - but it spoils 0.025, six times the 0.0042 a flat
mixture can afford. Two ways a noisy-but-orthogonal member can still pay, both cheap:

  small weights   below the 0.05 the coarse sweep started at, where the spoil term shrinks faster than the rescue
                  term if the disagreements are concentrated on genuinely ambiguous cells;
  confidence gate let STID move the blend only where STID itself is far from 0.5 AND the blend is near 0.5, so its
                  noise cannot reach the cells the blend already has.

    python occ_lab/sweep_stid_fine.py
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
    fam, acc = {}, {}
    for panel in C.T2_PANELS:
        for s in C.SPLITS:
            f, fs = CACHE / f"og_{s}_{panel}.npz", CACHE / f"stid_{s}_{panel}.npz"
            if not (f.exists() and fs.exists()):
                continue
            z = np.load(f)
            y, e = z["y"], z["e"]
            stid = mean_of(np.load(fs), "stid")
            occ_cnn, occ_tfm = mean_of(z, "occ_cnn"), mean_of(z, "occ_tfm")
            gs1, tfm8, cnn3 = (z[k].astype(np.float32) for k in ("gs1", "tfm8", "cnn3"))
            base = 0.2 * gs1 + 0.6 * tfm8 + 0.2 * cnn3
            best5 = base + 0.05 * (occ_cnn - cnn3) + 0.05 * (occ_tfm - tfm8)      # the 5-member candidate AH uses
            V = {"blend (T)": base, "AH (occ 0.05 + 0.05)": best5}
            for a in (0.01, 0.02, 0.03, 0.04, 0.05):
                V[f"stid {a:.2f} from tfm8"] = base + a * (stid - tfm8)
                V[f"stid {a:.2f} from cnn3"] = base + a * (stid - cnn3)
                V[f"AH + stid {a:.2f} from tfm8"] = best5 + a * (stid - tfm8)
            for d, k in product((0.10, 0.20), (0.30, 0.40, 0.45)):
                near = np.abs(base - TAU) < d
                conf = np.abs(stid - TAU) > k
                for w in (0.5, 1.0):
                    V[f"gate blend<{d:.2f} stid>{k:.2f} w={w:.1f}"] = np.where(
                        near & conf, (1 - w) * base + w * stid, base)
                    V[f"gate on AH <{d:.2f} stid>{k:.2f} w={w:.1f}"] = np.where(
                        near & conf, (1 - w) * best5 + w * stid, best5)
            for n, p in V.items():
                fam.setdefault((s, n), {})[panel] = C.iou(p >= TAU, y, e).mean()
            for d, k in product((0.10, 0.20), (0.30, 0.40, 0.45)):
                m = e & (np.abs(base - TAU) < d) & (np.abs(stid - TAU) > k)
                a = acc.setdefault((s, d, k), [0, 0, 0])
                a[0] += int(m.sum())
                a[1] += int((((base >= TAU) == y) & m).sum())
                a[2] += int((((stid >= TAU) == y) & m).sum())
    for s in C.SPLITS:
        b = C.family_mean(fam[(s, "blend (T)")])
        print(f"\n=== ongoing {s}: blend family {b:.4f} ===")
        for v, n in sorted(((C.family_mean(d), n) for (ss, n), d in fam.items() if ss == s), reverse=True)[:12]:
            print(f"   {n:<40} {v:.4f} ({v - b:+.4f})")
        print("   where the blend is unsure and STID is confident (blend / STID correct):")
        for (ss, d, k), a in acc.items():
            if ss == s and a[0]:
                print(f"     blend<{d:.2f}, stid>{k:.2f}: n={a[0]:>8,}  {a[1] / a[0]:.3f} / {a[2] / a[0]:.3f}")


if __name__ == "__main__":
    main()
