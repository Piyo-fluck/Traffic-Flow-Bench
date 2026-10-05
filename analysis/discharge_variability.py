"""Note 169: is the capacity random per day? For every (head link h, day) with a queue at h and h+1 free for >= 4
slots, the median discharge flow at h+1 (downstream of the head) that day. Per head with >= 5 days: coefficient of
variation of the daily discharge across days, and its within-day CV (slot to slot) for comparison. Real train
truth vs CTM-generated runs (deterministic capacity)."""
import sys
import warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "analysis")); sys.path.insert(0, str(ROOT / "tta_lab" / "tta"))
import numpy as np
import pandas as pd
from dlsim_gen2 import corridor
from tfb.data import RELEASE, load
import common as C
warnings.simplefilter("ignore")


def cv(z, order, vf, cap):
    v, q = z["true_speed"][:, :, order], z["true_flow"][:, :, order]
    vfo = vf[order]
    cg = np.isfinite(v) & (v <= 0.6 * vfo); fr = np.isfinite(v) & (v > 0.9 * vfo)
    between, within, level = [], [], []
    for h in range(v.shape[2] - 1):
        head = cg[:, :, h] & fr[:, :, h + 1]
        daily, wv = [], []
        for d in range(v.shape[0]):
            m = head[d]
            if m.sum() >= 4:
                x = q[d, m, h + 1]
                daily.append(np.median(x)); wv.append(np.std(x) / max(np.mean(x), 1))
        if len(daily) >= 5:
            daily = np.array(daily)
            between.append(np.std(daily) / np.mean(daily)); within.append(np.mean(wv))
            level.append(np.median(daily) / cap[order][h + 1])
    return np.median(between) if between else np.nan, np.median(within) if within else np.nan, len(between), \
        np.median(level) if level else np.nan


for panel in C.T2_PANELS:
    z = load(panel, "train")
    order, *_ = corridor(panel, z)
    ids = [str(x) for x in z["link_ids"]]
    cap = pd.read_csv(RELEASE / "corridors" / panel / "network" / "fd_parameters.csv").set_index("link_id").loc[ids].capacity_vph.values
    vf = np.asarray(z["free_speed"], float)
    r = cv(z, order, vf, cap)
    gs = [cv(dict(np.load(f, allow_pickle=True)), order, vf, cap) for f in sorted((ROOT / "work" / "genruns_ctm").glob(f"{panel}_r*.npz"))[:3]]
    g = np.nanmean(np.array(gs, float), 0)
    print(f"{panel:<10} real: day-to-day CV of discharge {r[0]:.3f} (within-day {r[1]:.3f}, heads {r[2]}, level {r[3]:.2f} cap_down) | "
          f"CTM: {g[0]:.3f} (within {g[1]:.3f}, heads {g[2]:.0f}, level {g[3]:.2f})", flush=True)
