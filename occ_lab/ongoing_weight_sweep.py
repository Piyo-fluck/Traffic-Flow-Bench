"""Full simplex over the ongoing members, the way the onset sweep was run (note 93).

Note 85 tried the ongoing STID only as an addition to the frozen production blend (0.05 .. 0.30 taken
proportionally) and as a slot replacement, and everything lost. The onset result showed why that may have been
the wrong search: the mixture that won there needed the strongest member cut from 0.70 to 0.35, which a
proportional addition can never reach.

So: every mixture of gs1 / tfm8 / cnn3 / stid on a 0.05 simplex, thresholded at 0.50, scored on both months.
Production is 0.20 / 0.60 / 0.20 / 0. Optionally the occupancy members replace cnn3 / tfm8 (--use-occ).

    python occ_lab/ongoing_weight_sweep.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta"))
import numpy as np  # noqa: E402

import common as C  # noqa: E402

CACHE = LAB / "cache"
TAU = 0.50


def mean_of(z, prefix):
    ks = [k for k in z.files if k.startswith(prefix)]
    return np.mean([z[k].astype(np.float32) for k in ks], 0) if ks else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--use-occ", action="store_true", help="swap cnn3 -> occ_cnn and tfm8 -> occ_tfm")
    args = ap.parse_args()
    data = {}
    for panel in C.T2_PANELS:
        for s in C.SPLITS:
            f, fs = CACHE / f"og_{s}_{panel}.npz", CACHE / f"stid_{s}_{panel}.npz"
            if not (f.exists() and fs.exists()):
                continue
            z = np.load(f)
            P = [z["gs1"].astype(np.float32),
                 (mean_of(z, "occ_tfm") if args.use_occ else z["tfm8"].astype(np.float32)),
                 (mean_of(z, "occ_cnn") if args.use_occ else z["cnn3"].astype(np.float32)),
                 mean_of(np.load(fs), "stid")]
            data[(s, panel)] = (np.stack(P), z["y"], z["e"])
    names = ["gs1", "occ_tfm" if args.use_occ else "tfm8", "occ_cnn" if args.use_occ else "cnn3", "stid"]
    g = np.round(np.arange(0, 1 + 1e-9, args.step), 2)
    combos = [(a, b, c, round(1 - a - b - c, 2)) for a in g for b in g for c in g
              if -1e-9 <= 1 - a - b - c <= 1 + 1e-9]
    print(f"{len(combos)} mixtures over {names}", flush=True)
    t0 = time.time()
    res = {}
    for (s, panel), (P, y, e) in data.items():
        for w in combos:
            p = (w[0] * P[0] + w[1] * P[1] + w[2] * P[2] + w[3] * P[3]) >= TAU
            res.setdefault((w, s), {})[panel] = C.iou(p, y, e).mean()
        print(f"  {panel} {s} ({time.time() - t0:.0f}s)", flush=True)
    fam = {k: C.family_mean(v) for k, v in res.items()}
    base = {s: fam[((0.2, 0.6, 0.2, 0.0), s)] for s in C.SPLITS}
    print("\nproduction 0.20 gs1 + 0.60 tfm8 + 0.20 cnn3: "
          + "  ".join(f"{s} {base[s]:.4f}" for s in C.SPLITS))
    print("\nbest mixture for each stid weight (by the worse month):")
    for ws in g:
        cand = [(min(fam[(w, s)] - base[s] for s in C.SPLITS), w) for w in combos if abs(w[3] - ws) < 1e-9]
        if cand:
            mn, w = max(cand)
            print(f"   stid {ws:.2f}: worse month {mn:+.4f} at " + " ".join(f"{n} {v:.2f}" for n, v in zip(names, w))
                  + "   " + "  ".join(f"{s} {fam[(w, s)] - base[s]:+.4f}" for s in C.SPLITS), flush=True)
    both = sorted(((min(fam[(w, s)] - base[s] for s in C.SPLITS), w) for w in combos), reverse=True)
    n_both = sum(1 for v, _ in both if v > 0)
    print(f"\n{n_both} of {len(combos)} mixtures beat production on both months; best 12:")
    for v, w in both[:12]:
        print("   " + " ".join(f"{n} {x:.2f}" for n, x in zip(names, w))
              + "   " + "  ".join(f"{s} {fam[(w, s)] - base[s]:+.4f}" for s in C.SPLITS))


if __name__ == "__main__":
    main()
