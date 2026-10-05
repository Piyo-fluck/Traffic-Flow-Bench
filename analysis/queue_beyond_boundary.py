"""Note 169: share of queued time at the panel's upstream-most sensored links (travel order 0-2) by hour, truth; and
the share of evening (15-21 h) queue episodes whose tail reaches the upstream-most link (queue spilling beyond)."""
import sys
import warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "analysis")); sys.path.insert(0, str(ROOT / "tta_lab" / "tta"))
import numpy as np
import common as C
from dlsim_gen2 import corridor
from tfb.data import load
warnings.simplefilter("ignore")
for panel in C.T2_PANELS:
    z = load(panel, "train")
    order, *_ = corridor(panel, z)
    vf = np.asarray(z["free_speed"], float)[order]
    v = z["true_speed"][:, :, order]
    cg = np.isfinite(v) & (v <= 0.6 * vf)
    up3 = cg[:, :, :3].any(2)
    ev = slice(180, 252)
    any_q = cg[:, ev].any(2)
    reach = (up3[:, ev] & any_q).sum() / max(any_q.sum(), 1)
    by = " ".join(f"{a:02d}-{b:02d}h {up3[:, a * 12:b * 12].mean():.3f}" for a, b in ((13, 15), (15, 17), (17, 19), (19, 21)))
    print(f"{panel:<10} upstream-end links queued: {by} | evening slots with any queue in which the upstream end is queued: {reach:.2f}", flush=True)
