"""Item 2 of the user's Task 2 plan: let the pseudo-supervised member's blend weight depend on the horizon.

The ongoing blend is a single scalar,

    base = (1 - w_PS) * (0.20 gs1 + 0.60 tfm8 + 0.20 cnn3) + w_PS * p_PS,    w_PS = 0.60 since candidate AR

but the two sides do not fail the same way along the horizon. p_PS is supervised on the run's own released
queues, which pins the near horizon; the frozen ensemble carries the train months' dynamics, which is what a
30-minute extrapolation needs. If that is right the weight should be LOWER late, not constant - and the
prediction is directional, so it is falsifiable: adopt only if `w_late < w_early` reproduces in both directions.
(The user wrote the test as `w_late > w_early`; either sign reproducing in both directions is the finding, a
split verdict is not.)

Two parameterisations, both with the corrector's features regenerated from the new base, because five of its
fifteen inputs are functions of it:

    step      w[h] = w_early for T+5..T+15, w_late for T+20..T+30
    sigmoid   w[h] = sigmoid(a + b h), fitted by the same grid over its two endpoint values

Fitted on one month and read on the other, both directions, and reported twice: over the whole month and over
the windows with at most `--early` earlier days, which is the regime the official windows are in (note 120).

    python occ_lab/w_ps_by_horizon.py --tag s20
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta"))
sys.path.insert(0, str(LAB.parent / "analysis"))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

import common as C  # noqa: E402
from online_adapter import ETA, GATE, global_features, load_split  # noqa: E402
from residual_diag import TAU  # noqa: E402


def weights(w_early, w_late, Fh, mode):
    if mode == "step":
        h = np.arange(Fh)
        return np.where(h < Fh // 2, w_early, w_late).astype(np.float32)
    lo, hi = np.log(w_early / (1 - w_early)), np.log(w_late / (1 - w_late))
    z = lo + (hi - lo) * np.arange(Fh) / max(Fh - 1, 1)
    return (1 / (1 + np.exp(-z))).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="s20")
    ap.add_argument("--mode", default="step", choices=("step", "sigmoid"))
    ap.add_argument("--grid", type=float, nargs="+", default=[0.4, 0.5, 0.6, 0.7, 0.8])
    ap.add_argument("--early", type=int, default=4)
    ap.add_argument("--corrector", type=Path, default=LAB / "models" / "residual_lgb.txt")
    args = ap.parse_args()
    gbm = lgb.Booster(model_file=str(args.corrector))
    grid = [(a, b) for a in args.grid for b in args.grid]
    fam, fam_e = {}, {}
    t0_ = time.time()
    for i, panel in enumerate(C.T2_PANELS):
        sp = load_split(i, panel, args.tag)
        for s, D in sp.items():
            old, ps, y, e, dd = D["old"], D["ps"], D["y"], D["e"], D["d"]
            days = np.unique(dd)
            early = np.isin(dd, days[1:args.early + 1])
            Fh = old.shape[1]
            for a, b in grid:
                w = weights(a, b, Fh, args.mode)[None, :, None]
                base = ((1 - w) * old + w * ps).astype(np.float32)
                F = global_features(D["P"], base)
                rh = gbm.predict(F.reshape(-1, F.shape[-1])).reshape(base.shape).astype(np.float32)
                pg = np.clip(base + ETA * np.where(np.abs(rh) >= GATE, rh, 0.0), 0, 1)
                v_ = C.iou(pg >= TAU, y, e)
                fam.setdefault((s, (a, b)), {})[panel] = v_.mean()
                fam_e.setdefault((s, (a, b)), {})[panel] = v_[early].mean()
        print(f"  {panel} ({time.time() - t0_:.0f}s)", flush=True)

    for tag, F_ in (("whole month", fam), (f"at most {args.early} earlier days", fam_e)):
        M = {s: {g: C.family_mean(F_[(s, g)]) for g in grid} for s in C.SPLITS}
        flat = M[C.SPLITS[0]][(0.6, 0.6)] if (0.6, 0.6) in M[C.SPLITS[0]] else None
        print(f"\nongoing family IoU, w_PS by horizon ('{args.mode}'), {tag}"
              + (f"; AR's flat 0.60 on {C.SPLITS[0]} = {flat:.4f}" if flat is not None else "") + "\n")
        print(f"{'w_early':>8}" + "".join(f"{f'late {b:g}':>12}" for b in args.grid))
        for s in C.SPLITS:
            print(f"[{s}]")
            for a in args.grid:
                print(f"{a:>8.2f}" + "".join(f"{M[s][(a, b)]:>12.4f}" for b in args.grid))
        for fit, test in ((C.SPLITS[0], C.SPLITS[1]), (C.SPLITS[1], C.SPLITS[0])):
            best = max(grid, key=lambda g: M[fit][g])
            print(f"  fit on {fit}: best (early {best[0]:g}, late {best[1]:g}) -> {test} "
                  f"{M[test][best]:.4f} against flat 0.60 {M[test][(0.6, 0.6)]:.4f} "
                  f"({M[test][best] - M[test][(0.6, 0.6)]:+.4f}), late {'<' if best[1] < best[0] else '>' if best[1] > best[0] else '='} early")


if __name__ == "__main__":
    main()
