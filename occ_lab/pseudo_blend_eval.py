"""Does the pseudo-supervised adaptation survive inside the production blends?

`pseudo_adapt.py --save` writes the adapted model's probabilities on the official-like windows. Alone it beats
the production 9b adaptation on both months and both halves (note 106). This puts it where it would actually be
used:

    onset    candidate AN ships 0.35 H + 0.35 ad9b + 0.30 STID; the new model replaces the ad9b slot
    ongoing  production is 0.20 gs1 + 0.60 tfm8 + 0.20 cnn3 at tau 0.50; the new model takes the cnn3 slot, or
             joins as a fourth member over a simplex

    python occ_lab/pseudo_blend_eval.py --tag s20
"""
from __future__ import annotations

import argparse
import sys
from itertools import product
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
    ap.add_argument("--tag", default="s20")
    args = ap.parse_args()
    fam_on, fam_og = {}, {}
    for i, panel in enumerate(C.T2_PANELS):
        f = CACHE / f"pseudo_{args.tag}_{C.SPLITS[0]}_{panel}.npz"
        if not f.exists():
            continue
        ctx, stat, W = C.windows(i, panel)
        for s in C.SPLITS:
            on, og, ref = W[s]["on"], W[s]["og"], W[s]["ref"]
            new = np.load(CACHE / f"pseudo_{args.tag}_{s}_{panel}.npz")
            # ---- onset
            H, ad = ref["on_H"], ref["on_ad9b"]
            fs = CACHE / f"onset_adapt_stid30_{s}_{panel}.npz"
            if fs.exists() and len(on["d"]):
                st = np.load(fs)["prob"]
                v = {"AN (.35H .35ad9b .30STID)": 0.35 * H + 0.35 * ad + 0.30 * st,
                     "AN with the new model in the ad9b slot": 0.35 * H + 0.35 * new["on"] + 0.30 * st,
                     ".35H + .175ad9b + .175new + .30STID": 0.35 * H + 0.175 * ad + 0.175 * new["on"] + 0.30 * st,
                     "old production .7H + .3ad9b": 0.7 * H + 0.3 * ad,
                     "old production, new model": 0.7 * H + 0.3 * new["on"]}
                for n, p in v.items():
                    fam_on.setdefault((s, n), {})[panel] = C.onset_iou(p, on).mean()
            # ---- ongoing
            fo = CACHE / f"og_{s}_{panel}.npz"
            if fo.exists() and len(og["d"]):
                z = np.load(fo)
                gs1, tfm8, cnn3 = (z[k].astype(np.float32) for k in ("gs1", "tfm8", "cnn3"))
                nw = new["og"].astype(np.float32)
                base = 0.2 * gs1 + 0.6 * tfm8 + 0.2 * cnn3
                v = {"production .2/.6/.2": base, "cnn3 slot -> new": 0.2 * gs1 + 0.6 * tfm8 + 0.2 * nw,
                     "new alone": nw, "cnn3 alone": cnn3}
                for a in (0.1, 0.2, 0.3):
                    v[f"base + new {a:.1f}"] = (1 - a) * base + a * nw
                for wg, wt, wn in product((0.10, 0.20), (0.40, 0.50, 0.60), (0.10, 0.20, 0.30, 0.40)):
                    wc = round(1 - wg - wt - wn, 2)
                    if wc < -1e-9:
                        continue
                    v[f"gs1 {wg:.2f} tfm8 {wt:.2f} cnn3 {wc:.2f} new {wn:.2f}"] = (
                        wg * gs1 + wt * tfm8 + wc * cnn3 + wn * nw)
                y, e = z["y"], z["e"]
                for n, p in v.items():
                    fam_og.setdefault((s, n), {})[panel] = C.iou(p >= TAU, y, e).mean()
        print(f"  {panel}", flush=True)

    for s in C.SPLITS:
        print(f"\n=== onset {s} ===")
        b = C.family_mean(fam_on[(s, "AN (.35H .35ad9b .30STID)")])
        for (ss, n), d in fam_on.items():
            if ss == s:
                print(f"   {n:<42} {C.family_mean(d):.4f} ({C.family_mean(d) - b:+.4f})")
        print(f"=== ongoing {s} ===")
        b = C.family_mean(fam_og[(s, "production .2/.6/.2")])
        rows = sorted(((C.family_mean(d), n) for (ss, n), d in fam_og.items() if ss == s), reverse=True)
        for v, n in rows[:10]:
            print(f"   {n:<42} {v:.4f} ({v - b:+.4f})")


if __name__ == "__main__":
    main()
