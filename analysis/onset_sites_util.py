"""Note 169: shared helper - earliest-onset windows of one day and their site (downstream-most queued link at T+30)."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tta_lab" / "tta"))
import numpy as np
from t2_eda03 import earliest_onset
F = 6


def day_sites(V, el, vf, rank):
    """list of (t, site) earliest-onset windows of one day; site = downstream-most queued link at t+F"""
    q = ((np.nan_to_num(V, nan=1e9) <= 0.6 * vf) & el)[None]
    out = []
    for t in np.flatnonzero(earliest_onset(q)[0]):
        y = np.flatnonzero(q[0, t + F]); y = y[rank[y] >= 0]
        if len(y):
            out.append((int(t), int(y[np.argmax(rank[y])])))
    return out
