"""Weight grid for the onset blend H / adapted CNN / onset STID, on the cached probabilities.

The coarse grid of `eval_onset.py` put two mixtures ahead of production on BOTH months; this refines it on a
simplex grid of 0.05 so the chosen point is not a knife edge. Prints every combination that beats production on
both months, and the neighbourhood of the best.

    python occ_lab/onset_weight_sweep.py --tag stid30
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

import common as C  # noqa: E402

CACHE = LAB / "cache"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="stid30")
    ap.add_argument("--step", type=float, default=0.05)
    args = ap.parse_args()
    data = {}
    for i, panel in enumerate(C.T2_PANELS):
        _, _, W = C.windows(i, panel)
        for s in C.SPLITS:
            on, ref = W[s]["on"], W[s]["ref"]
            f = CACHE / f"onset_adapt_{args.tag}_{s}_{panel}.npz"
            if not len(on["d"]) or not f.exists():
                continue
            data[(s, panel)] = (ref["on_H"], ref["on_ad9b"], np.load(f)["prob"], on)
        print(f"  {panel} loaded", flush=True)
    grid = np.round(np.arange(0, 1 + 1e-9, args.step), 2)
    res = {}
    for wh in grid:
        for wa in grid:
            ws = round(1 - wh - wa, 2)
            if ws < -1e-9 or ws > 0.6:
                continue
            for s in C.SPLITS:
                fam = {}
                for (ss, panel), (H, ad, st, on) in data.items():
                    if ss != s:
                        continue
                    fam[panel] = C.onset_iou(wh * H + wa * ad + ws * st, on).mean()
                res[(round(wh, 2), round(wa, 2), ws, s)] = C.family_mean(fam)
    base = {s: res[(0.7, 0.3, 0.0, s)] for s in C.SPLITS}
    print(f"\nproduction .7H + .3 ad9b: " + "  ".join(f"{s} {base[s]:.4f}" for s in C.SPLITS))
    print("\nwithout STID (the two production members re-weighted):")
    for wa in np.round(np.arange(0.0, 0.85, 0.1), 2):
        wh = round(1 - wa, 2)
        if (wh, wa, 0.0, C.SPLITS[0]) in res:
            print(f"   H {wh:.2f}  ad9b {wa:.2f}   "
                  + "  ".join(f"{s} {res[(wh, wa, 0.0, s)] - base[s]:+.4f}" for s in C.SPLITS))
    print("\nbest mixture for each STID weight (by the worse month):")
    for ws in np.round(np.arange(0.0, 0.65, 0.05), 2):
        cand = [(min(res[(wh, wa, w2, s)] - base[s] for s in C.SPLITS), wh, wa)
                for (wh, wa, w2, s) in res if s == C.SPLITS[0] and abs(w2 - ws) < 1e-9]
        if cand:
            mn, wh, wa = max(cand)
            print(f"   STID {ws:.2f}: worse month {mn:+.4f} at H {wh:.2f} ad9b {wa:.2f}   "
                  + "  ".join(f"{s} {res[(wh, wa, round(ws, 2), s)] - base[s]:+.4f}" for s in C.SPLITS))
    both = []
    for (wh, wa, ws, s) in list(res):
        if s != C.SPLITS[0]:
            continue
        g = [res[(wh, wa, ws, x)] - base[x] for x in C.SPLITS]
        if all(v > 0 for v in g):
            both.append((min(g), g, (wh, wa, ws)))
    both.sort(reverse=True)
    print(f"{len(both)} of {len(res) // len(C.SPLITS)} mixtures beat production on both months; "
          f"best by the worse month:")
    for mn, g, w in both[:15]:
        print(f"   H {w[0]:.2f}  ad9b {w[1]:.2f}  STID {w[2]:.2f}   "
              + "  ".join(f"{s} {v:+.4f}" for s, v in zip(C.SPLITS, g)))


if __name__ == "__main__":
    main()
