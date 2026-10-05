"""Four-member onset blend: H / adapted CNN / free-identity STID / neighbourhood-identity STID.

Both identity variants beat production on their own (note 95), so the question is whether they are complementary
or the same thing twice. Simplex at 0.05 over the four members, expected-IoU ratio rule as usual.

    python occ_lab/onset_four_sweep.py
"""
from __future__ import annotations

import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta"))
import numpy as np  # noqa: E402

import common as C  # noqa: E402

CACHE = LAB / "cache"
STEP = 0.05


def main() -> None:
    data = {}
    for i, panel in enumerate(C.T2_PANELS):
        _, _, W = C.windows(i, panel)
        for s in C.SPLITS:
            on, ref = W[s]["on"], W[s]["ref"]
            fa = CACHE / f"onset_adapt_stid30_{s}_{panel}.npz"
            fb = CACHE / f"onset_adapt_stid_ab_{s}_{panel}.npz"
            if not len(on["d"]) or not (fa.exists() and fb.exists()):
                continue
            data[(s, panel)] = (ref["on_H"], ref["on_ad9b"], np.load(fa)["prob"], np.load(fb)["prob"], on)
        print(f"  {panel} loaded", flush=True)
    g = np.round(np.arange(0, 1 + 1e-9, STEP), 2)
    combos = [(a, b, c, round(1 - a - b - c, 2)) for a in g for b in g for c in g if a + b + c <= 1 + 1e-9]
    fam = {}
    for (s, panel), (Hq, ad, s_id, s_ab, on) in data.items():
        for w in combos:
            p = w[0] * Hq + w[1] * ad + w[2] * s_id + w[3] * s_ab
            fam.setdefault((s, w), {})[panel] = C.onset_iou(p, on).mean()
    F = {k: C.family_mean(v) for k, v in fam.items()}
    base = {s: F[(s, (0.7, 0.3, 0.0, 0.0))] for s in C.SPLITS}
    print("\nproduction .7H + .3 ad9b: " + "  ".join(f"{s} {base[s]:.4f}" for s in C.SPLITS))
    for name, w in (("AN (shipped)", (0.35, 0.35, 0.30, 0.0)),
                    ("best free-ID only", (0.30, 0.35, 0.35, 0.0)),
                    ("best neighbourhood only", (0.20, 0.40, 0.0, 0.40))):
        print(f"{name:<24} " + "  ".join(f"{s} {F[(s, w)] - base[s]:+.4f}" for s in C.SPLITS))
    both = sorted(((min(F[(s, w)] - base[s] for s in C.SPLITS), w) for w in combos), reverse=True)
    n = sum(1 for v, _ in both if v > 0)
    print(f"\n{n} of {len(combos)} mixtures beat production on both months; best 12 by the worse month:")
    for v, w in both[:12]:
        print(f"   H {w[0]:.2f} ad9b {w[1]:.2f} ID {w[2]:.2f} NBR {w[3]:.2f}   "
              + "  ".join(f"{s} {F[(s, w)] - base[s]:+.4f}" for s in C.SPLITS))


if __name__ == "__main__":
    main()
