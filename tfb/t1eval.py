"""Task 1 local evaluation protocol: fit on Jun–Dec 2030 of train, test on Jan–Feb 2031.

The validation and private months come after train, so the test block is the
last stretch of train. Anything learned from truth (profiles, weights, models)
must be fitted on fit days only.
"""
from __future__ import annotations

import numpy as np

FIT_END = "2031-01-01"


def split_days(z: dict) -> tuple[np.ndarray, np.ndarray]:
    fit = z["dates"] < FIT_END
    return fit, ~fit


def family_mean(panel_scores: dict[str, float]) -> float:
    """Mean over directions within a family, then over families (the leaderboard's rule)."""
    fam: dict[str, list[float]] = {}
    for panel, s in panel_scores.items():
        fam.setdefault(panel.rsplit("_", 1)[0], []).append(s)
    return float(np.mean([np.mean(v) for v in fam.values()]))
