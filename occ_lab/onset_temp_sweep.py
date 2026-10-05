"""Temperature on the onset STID before blending, over the weight simplex.

Two results say the onset score depends on the SCALE of the member probabilities, not only on their ranking:
averaging eight seeds instead of four flattens the STID map and loses (note 93), and a residual correction that
moves the predicted set by 0.04 links per window costs 0.019 (note 94). Both act through the expected-IoU top-k
rule, which reads the probability values.

So the member's sharpness is a free parameter:

    p_stid(T) = sigmoid(logit(p_stid) / T)      T < 1 sharpens, T > 1 flattens

This sweeps T against the H / ad9b / STID simplex for the 4-seed and the 8-seed STID, and prints the mean number
of links each window ends up predicting, so the mechanism is visible.

    python occ_lab/onset_temp_sweep.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta"))
import numpy as np  # noqa: E402

import common as C  # noqa: E402
from residual_diag import lg  # noqa: E402
from t2_expected_iou import select  # noqa: E402

CACHE = LAB / "cache"
TEMPS = (0.40, 0.55, 0.70, 0.85, 1.00, 1.30, 1.70)
AN = (0.35, 0.35, 0.30)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=float, default=0.05)
    args = ap.parse_args()
    data = {}
    for i, panel in enumerate(C.T2_PANELS):
        _, _, W = C.windows(i, panel)
        for s in C.SPLITS:
            on, ref = W[s]["on"], W[s]["ref"]
            if not len(on["d"]):
                continue
            st = {}
            for tag, name in (("stid30", "4 seeds"), ("stid30x8", "8 seeds")):
                f = CACHE / f"onset_adapt_{tag}_{s}_{panel}.npz"
                if f.exists():
                    st[name] = np.load(f)["prob"]
            data[(s, panel)] = (ref["on_H"], ref["on_ad9b"], st, on)
        print(f"  {panel} loaded", flush=True)
    g = np.round(np.arange(0, 1 + 1e-9, args.step), 2)
    combos = [(a, round(b, 2), round(1 - a - b, 2)) for a in g for b in g if a + b <= 1 + 1e-9]
    fam, ksz = {}, {}
    for (s, panel), (Hq, ad, st, on) in data.items():
        for name, q in st.items():
            for T in TEMPS:
                qt = 1 / (1 + np.exp(-lg(q) / T))
                for wh, wa, ws in combos:
                    p = wh * Hq + wa * ad + ws * qt
                    fam.setdefault((s, name, T, (wh, wa, ws)), {})[panel] = C.onset_iou(p, on).mean()
                    if (wh, wa, ws) == AN:
                        ksz.setdefault((s, name, T), []).append(
                            np.array([select(v, "ratio", 1.0).sum() for v in p], float))
    base = {s: C.family_mean(fam[(s, "4 seeds", 1.00, (0.7, 0.3, 0.0))]) for s in C.SPLITS}
    an = {s: C.family_mean(fam[(s, "4 seeds", 1.00, AN)]) for s in C.SPLITS}
    print("\nproduction .7H+.3ad9b: " + "  ".join(f"{s} {base[s]:.4f}" for s in C.SPLITS))
    print("candidate AN (4 seeds, T=1, .35/.35/.30): "
          + "  ".join(f"{s} {an[s]:.4f} ({an[s] - base[s]:+.4f})" for s in C.SPLITS))
    for name in ("4 seeds", "8 seeds"):
        print(f"\n=== {name} ===")
        print("  at AN's weights, by temperature:")
        for T in TEMPS:
            if (C.SPLITS[0], name, T, AN) not in fam:
                continue
            k = np.concatenate(ksz[(C.SPLITS[0], name, T)]).mean()
            print(f"    T {T:.2f}: " + "  ".join(
                f"{s} {C.family_mean(fam[(s, name, T, AN)]) - base[s]:+.4f}" for s in C.SPLITS)
                + f"   links/window(val) {k:.2f}")
        best = []
        for T in TEMPS:
            for w in combos:
                if (C.SPLITS[0], name, T, w) not in fam:
                    continue
                v = [C.family_mean(fam[(s, name, T, w)]) - base[s] for s in C.SPLITS]
                best.append((min(v), v, T, w))
        best.sort(reverse=True)
        print("  best mixtures by the worse month:")
        for mn, v, T, w in best[:8]:
            print(f"    T {T:.2f}  H {w[0]:.2f} ad9b {w[1]:.2f} STID {w[2]:.2f}   "
                  + "  ".join(f"{s} {x:+.4f}" for s, x in zip(C.SPLITS, v)))


if __name__ == "__main__":
    main()
