"""Task 2 audit (measurement only, NOT used for any submission): do released ramp flows
react to the mainline queue state?

Ramp files are not blanked over the Task 2 horizon. If a ramp's flow drops
(or jumps) when its mainline link is queued, the horizon ramp flows would
reveal the forecast target. On train truth, for every valid ramp cell
attached to a mainline station, compare the ramp flow relative to its own
weekday-type x time-of-day mean, when the attached link is queued vs not.

    py analysis/t2_ramp_audit.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
from t2_eda02 import T2_PANELS  # noqa: E402
from t2_eda03 import labels  # noqa: E402
from tfb.data import load  # noqa: E402
from tfb.physics import load_ramps  # noqa: E402


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score_pos > score_neg) by ranks; 0.5 = no signal."""
    x = np.concatenate([pos, neg])
    r = x.argsort().argsort().astype(float) + 1
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main() -> None:
    rng = np.random.default_rng(0)
    print(f"{'panel':<11} {'type':<4} {'n_queued':>9} {'ratio queued':>13} {'ratio free':>11} {'AUC(low flow -> queued)':>24}")
    for panel in T2_PANELS:
        z = load(panel, "train")
        q, el = labels(z)
        r = load_ramps(panel, "train", z)
        wt = z["weekday"] >= 5
        for kind in ("on", "off"):
            flow, valid = r[f"{kind}_q"], r[f"{kind}_v"]
            prof = np.zeros_like(flow)
            for w in (False, True):
                m = valid[wt == w]
                prof[wt == w] = np.where(m, flow[wt == w], 0).sum(0) / np.maximum(m.sum(0), 1)
            ratio = flow / np.maximum(prof, 1.0)
            ok = valid & el & (prof > 20)
            a, b = ratio[ok & q], ratio[ok & ~q]
            if len(a) < 50:
                print(f"{panel:<11} {kind:<4} {len(a):>9}  (too few)")
                continue
            b = rng.choice(b, min(len(b), 200_000), replace=False)
            print(f"{panel:<11} {kind:<4} {len(a):>9,} {np.median(a):>13.3f} {np.median(b):>11.3f} {auc(-a, -b):>24.3f}")


if __name__ == "__main__":
    main()
