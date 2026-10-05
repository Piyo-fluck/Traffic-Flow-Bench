"""With the new adapted member in the ongoing blend, is AI's residual corrector still worth applying?

The corrector was fitted on the residual of the OLD blend (0.20 gs1 + 0.60 tfm8 + 0.20 cnn3). If the blend
changes, its input distribution changes with it. Three states are compared on the official-like windows:

    AI            old blend + corrector (what candidate AN ships for ongoing)
    new blend     old blend + 0.30 of the pseudo-supervised adapted model, no corrector
    new + corr    the same, with the corrector applied anyway

    python occ_lab/pseudo_corrector_check.py --tag s20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta"))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

import common as C  # noqa: E402
from residual_diag import NAMES, TAU, lg  # noqa: E402

CACHE = LAB / "cache"
ETA, GATE = 0.75, 0.10


def mean_of(z, prefix):
    ks = [k for k in z.files if k.startswith(prefix)]
    return np.mean([z[k].astype(np.float32) for k in ks], 0) if ks else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="s20")
    ap.add_argument("--w-new", type=float, default=0.30)
    args = ap.parse_args()
    gbm = lgb.Booster(model_file=str(LAB / "models" / "residual_lgb.txt"))
    fam = {}
    for panel in C.T2_PANELS:
        for s in C.SPLITS:
            fo = CACHE / f"og_{s}_{panel}.npz"
            fp = CACHE / f"pseudo_{args.tag}_{s}_{panel}.npz"
            fs = CACHE / f"stid_{s}_{panel}.npz"
            if not (fo.exists() and fp.exists() and fs.exists()):
                continue
            z = np.load(fo)
            y, e = z["y"], z["e"]
            gs1, tfm8, cnn3 = (z[k].astype(np.float32) for k in ("gs1", "tfm8", "cnn3"))
            occ_cnn, occ_tfm = mean_of(z, "occ_cnn"), mean_of(z, "occ_tfm")
            stid = mean_of(np.load(fs), "stid")
            nw = np.load(fp)["og"].astype(np.float32)
            old = 0.2 * gs1 + 0.6 * tfm8 + 0.2 * cnn3
            new = (1 - args.w_new) * old + args.w_new * nw

            def corrected(p0):
                N, F, S = p0.shape
                hor = np.broadcast_to(np.arange(F, dtype=np.float32)[None, :, None], p0.shape)
                pos = np.broadcast_to((np.arange(S, dtype=np.float32) / max(S - 1, 1))[None, None, :], p0.shape)
                mem = np.stack([gs1, tfm8, cnn3, occ_cnn, occ_tfm, stid])
                f = np.stack([p0, lg(p0), np.abs(p0 - 0.5), mem.std(0), mem.max(0) - mem.min(0),
                              gs1, tfm8, cnn3, occ_cnn, occ_tfm, stid,
                              lg(stid) - lg(p0), lg(occ_cnn) - lg(p0), hor, pos], -1)
                r = np.zeros(p0.shape, np.float32)
                r[e] = gbm.predict(f[e]).astype(np.float32)
                return np.clip(p0 + ETA * np.where(np.abs(r) >= GATE, r, 0.0), 0, 1)

            for n, p in (("old blend", old), ("AI = old + corrector", corrected(old)),
                         (f"new blend (+{args.w_new:.2f})", new), ("new + corrector", corrected(new))):
                fam.setdefault((s, n), {})[panel] = C.iou(p >= TAU, y, e).mean()
        print(f"  {panel}", flush=True)
    for s in C.SPLITS:
        b = C.family_mean(fam[(s, "AI = old + corrector")])
        print(f"\n=== ongoing {s} (against AI, which candidate AN ships) ===")
        for (ss, n), d in fam.items():
            if ss == s:
                print(f"   {n:<28} {C.family_mean(d):.4f} ({C.family_mean(d) - b:+.4f})")


if __name__ == "__main__":
    main()
