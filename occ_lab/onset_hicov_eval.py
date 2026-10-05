"""An onset evaluation whose history coverage matches the official windows - and its retrospective test.

Note 76: every local onset comparison we have ever made used pseudo windows whose histories are 52% observed
against the official 76%, because they inherit the Task 1 target mask. That is the standing suspect for onset
decisions that looked good locally and lost on the leaderboard (AN +0.023 local -> +0.0025; AO +0.033 -> -0.0046).

`cache_onset_oof.py --truth-hist` fixes it: on the held-out tail of the train month (2031-01/02, out of fold for
every member) the histories come from the TRUE layer restricted to eligibility, which is 0.763 observed against
the official 0.761.

This scores the mixtures whose leaderboard outcome we already know. The evaluation is only worth using if it
reproduces the known ordering:

    0.70 H + 0.30 ad9b            candidates T..AM
    0.35 H + 0.35 ad9b + 0.30 ST  candidate AN, +0.00251 over AI on the leaderboard  -> must beat the above
    0.20 H + 0.40 ad9b + 0.40 ST  candidate AO, -0.00460 against AN                  -> must lose to AN

    python occ_lab/onset_hicov_eval.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta"))
sys.path.insert(0, str(LAB.parent / "analysis"))
import numpy as np  # noqa: E402

import common as C  # noqa: E402
from t2_expected_iou import select  # noqa: E402

CACHE = LAB / "cache"

MIX = {                                        # (w_H, w_ad9b, w_stid)
    "T..AM  .70H .30ad": (0.70, 0.30, 0.00),
    "AN     .35H .35ad .30st": (0.35, 0.35, 0.30),
    "AO     .20H .40ad .40st": (0.20, 0.40, 0.40),
    "AP     .35H .35ad .30st": (0.35, 0.35, 0.30),      # AP's onset is AN's with ad9b split; same family here
}


def iou(pred, y):
    i = (pred & y).sum(-1)
    u = (pred | y).sum(-1)
    return np.where(u > 0, i / np.maximum(u, 1), 1.0)


def score(p, y, e, c):
    out = np.zeros(len(p))
    for j in range(len(p)):
        m = e[j]
        if not m.any():
            out[j] = 1.0
            continue
        pr = np.zeros(m.sum(), bool)
        pr[:] = select(p[j][m], "ratio", c)
        out[j] = iou(pr, y[j][m])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cs", type=float, nargs="*", default=[1.0])
    args = ap.parse_args()
    fam, covs, nwin = {}, [], 0
    for panel in C.T2_PANELS:
        f = CACHE / f"onset_hicov_{panel}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        covs.append(float(z["cov"])); nwin += len(z["y"])
        for name, (wh, wa, ws) in MIX.items():
            p = wh * z["H"] + wa * z["ad9b"] + ws * z["stid"]
            for c in args.cs:
                fam.setdefault((name, c), {})[panel] = score(p, z["y"], z["e"], c).mean()
    print(f"{nwin} onset windows over {len(covs)} panels, history coverage {np.mean(covs):.3f} "
          f"(official 0.761, the old pseudo windows 0.524)\n")
    base = None
    print(f"{'mixture':<28} {'c':>4} {'onset IoU':>10} {'vs T..AM':>10}   known LB")
    known = {"AN     .35H .35ad .30st": "+0.00251 over AI", "AO     .20H .40ad .40st": "-0.00460 vs AN",
             "T..AM  .70H .30ad": "reference"}
    for (name, c), d in fam.items():
        v = C.family_mean(d)
        if base is None:
            base = v
        print(f"{name:<28} {c:>4.2f} {v:>10.4f} {v - base:>+10.4f}   {known.get(name, '')}")


if __name__ == "__main__":
    main()
