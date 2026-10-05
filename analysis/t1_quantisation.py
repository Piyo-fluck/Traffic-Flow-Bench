"""Does the released data live on a lattice? Reverse-engineering the measurement process.

The user's point (note 103): Kaggle solutions that move ranks often come from working out how the generator
produced the numbers, not from a better model. Here the natural suspicion is counting: if flow is a count of
vehicles over a 5-minute bin converted to veh/h, it lives on a lattice of 12 veh/h per vehicle (60/5), and
occupancy may be quantised by the sampling of the detector.

This checks, on the TRUE layer of the train split:

    * how many distinct values each channel takes, and the smallest gaps between sorted distinct values
    * the residual to the nearest multiple of a candidate step g, for a grid of g
    * whether the lattice is per link total or per lane, and whether it depends on the number of lanes

    py analysis/t1_quantisation.py [panels...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tfb.data import PANELS, load  # noqa: E402

PANEL_ARGS = sys.argv[1:] or ["D7_I10_E", "D12_I5_N"]
STEPS = (1.0, 2.0, 3.0, 4.0, 6.0, 12.0, 15.0, 20.0, 30.0, 60.0)


def lattice_report(name, v, steps=STEPS):
    v = v[np.isfinite(v)]
    v = v[v > 0]
    u = np.unique(v)
    print(f"\n{name}: {len(v):,} values, {len(u):,} distinct "
          f"({len(u) / len(v) * 100:.2f}% of the sample), range {u[0]:.4f} .. {u[-1]:.2f}")
    if len(u) > 1:
        d = np.diff(u)
        d = d[d > 1e-9]
        print(f"   gaps between distinct values: min {d.min():.6f}, 1st pct {np.percentile(d, 1):.6f}, "
              f"median {np.median(d):.6f}")
    samp = v if len(v) < 4_000_000 else np.random.default_rng(0).choice(v, 4_000_000, replace=False)
    best = []
    for g in steps:
        r = np.abs(samp / g - np.round(samp / g))
        best.append((float(r.mean()), g, float((r < 1e-3).mean())))
    best.sort()
    for m, g, share in best[:4]:
        print(f"   step {g:6.2f}: mean |offset| {m:.4f} of a step, exactly on the lattice {share:.2%}")


def main() -> None:
    for panel in PANEL_ARGS:
        z = load(panel, "train")
        lanes = z["lanes"]
        print(f"\n================ {panel} (lanes {np.unique(lanes)})")
        ok = z["elig"] & np.isfinite(z["true_speed"]) & np.isfinite(z["true_flow"])
        lattice_report("true_flow (per link)", z["true_flow"][ok])
        lattice_report("true_flow / lanes", (z["true_flow"] / lanes)[ok])
        lattice_report("true_speed", z["true_speed"][ok])
        lattice_report("true_occ", z["true_occ"][ok], steps=(0.001, 0.002, 0.005, 0.01, 0.02, 0.05))
        one = lanes == lanes.min()
        if one.any():
            m = ok & one[None, None, :]
            lattice_report(f"true_flow on {int(lanes.min())}-lane links", z["true_flow"][m])
        obs = z["elig"] & np.isfinite(z["obs_flow"])
        same = obs & ok
        if same.any():
            d = np.abs(z["obs_flow"][same] - z["true_flow"][same])
            print(f"\n   observed vs true where both exist: mean |diff| {d.mean():.6f}, "
                  f"max {d.max():.6f} ({int(same.sum()):,} cells)")


if __name__ == "__main__":
    main()
