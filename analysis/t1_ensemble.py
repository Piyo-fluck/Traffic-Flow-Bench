"""Task 1: score saved test-day predictions of two methods and their per-channel blends.

Reads work/preds/t1_<method>_<panel>_test.npz (days, pv, pq) written by the
method scripts, all on the same holdout days.

    py analysis/t1_ensemble.py m2 m3 D12_I5_N D7_I210_W
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tfb.data import load  # noqa: E402
from tfb.metrics import fmt, state_score  # noqa: E402
from tfb.t1eval import family_mean  # noqa: E402

GRID = np.round(np.arange(0, 1.01, 0.1), 2)


def full(z: dict, pred: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
    out = np.zeros(z["target"].shape, np.float32)
    out[pred["days"]] = pred[key]
    return out


def main() -> None:
    a, b, *panels = sys.argv[1:]
    data = {}
    for panel in panels:
        z = load(panel, "train")
        pa = np.load(ROOT / "work" / "preds" / f"t1_{a}_{panel}_test.npz")
        pb = np.load(ROOT / "work" / "preds" / f"t1_{b}_{panel}_test.npz")
        if not np.array_equal(pa["days"], pb["days"]):
            raise ValueError(f"{panel}: {a} and {b} were scored on different days")
        cells = np.zeros(len(z["dates"]), bool)
        cells[pa["days"]] = True
        data[panel] = (z, full(z, pa, "pv"), full(z, pa, "pq"), full(z, pb, "pv"), full(z, pb, "pq"),
                       cells[:, None, None])
        for tag, pv, pq in ((a, data[panel][1], data[panel][2]), (b, data[panel][3], data[panel][4])):
            print(f"[{panel}] {tag:<4} {fmt(state_score(z, pv, pq, cells=data[panel][5]))}")

    # Speed and flow enter S_state separately, so their weights can be chosen separately.
    grid = {}
    for wv in GRID:
        for wq in GRID:
            s = {}
            for panel, (z, av, aq, bv, bq, cells) in data.items():
                s[panel] = state_score(z, wv * av + (1 - wv) * bv, wq * aq + (1 - wq) * bq, cells=cells)["mean"]["S"]
            grid[(wv, wq)] = family_mean(s)
    print(f"\nfamily mean, weight on {a} (speed w_v, flow w_q):")
    print("  w_v\\w_q " + " ".join(f"{w:>6.1f}" for w in GRID))
    for wv in GRID:
        print(f"  {wv:>6.1f}  " + " ".join(f"{grid[(wv, wq)]:.4f}" for wq in GRID))
    best = max(grid, key=grid.get)
    print(f"\n{a} alone {grid[(1.0, 1.0)]:.4f} | {b} alone {grid[(0.0, 0.0)]:.4f} | "
          f"50/50 {grid[(0.5, 0.5)]:.4f} | best w_v={best[0]}, w_q={best[1]}: {grid[best]:.4f}")


if __name__ == "__main__":
    main()
