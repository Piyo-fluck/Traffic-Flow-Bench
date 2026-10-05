"""Note 169: at which capacity does the TRUE data break down? Train truth, travel order (path_set). A breakdown event
= link h becomes the head of a queue (h congested, next downstream sensored link free) after 3 slots with h and
its upstream neighbour free. Printed per panel: flow at h just before / released cap of h and of the next
downstream link, the discharge while the queue lasts / cap, and whether a zone entry lies in the gap just
downstream of h (a merge bottleneck) or an exit (diverge)."""
import sys
import warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "analysis")); sys.path.insert(0, str(ROOT / "tta_lab" / "tta"))
import numpy as np
import pandas as pd
import common as C
from dlsim_gen2 import corridor
from tfb.data import RELEASE, load
warnings.simplefilter("ignore")
for panel in sys.argv[1:] or C.T2_PANELS:
    z = load(panel, "train")
    order, spos, passing, attr, prod = corridor(panel, z)
    ids = [str(x) for x in z["link_ids"]]
    fd = pd.read_csv(RELEASE / "corridors" / panel / "network" / "fd_parameters.csv").set_index("link_id").loc[ids]
    cap = fd.capacity_vph.values[order]; vf = fd.free_speed_kmh.values[order]; lanes = fd.lanes.values[order]
    v, q = z["true_speed"][:, :, order], z["true_flow"][:, :, order]
    cg = np.isfinite(v) & (v <= 0.6 * vf)
    fr = np.isfinite(v) & (v > 0.9 * vf)
    rows = []
    D, T, n = v.shape
    for d in range(D):
        for h in range(1, n - 1):
            hd = cg[d, :, h] & fr[d, :, h + 1]
            ts = np.flatnonzero(hd[1:] & ~hd[:-1]) + 1
            for t in ts:
                if t < 4 or not (fr[d, t - 3:t, h].all() and fr[d, t - 3:t, h - 1].all()):
                    continue
                dur = 0
                while t + dur < T and cg[d, t + dur, h]:
                    dur += 1
                dis = np.nanmedian(q[d, t:t + max(dur, 1), h])
                gap_ent = prod[spos[h] + 1:spos[h + 1] + 1].sum() > 0
                gap_ex = attr[spos[h]:spos[h + 1]].sum() > 0
                rows.append((q[d, t - 1, h] / cap[h], q[d, t - 1, h] / cap[h + 1], dis / cap[h], dis / cap[h + 1],
                             cap[h + 1] / cap[h], lanes[h + 1] - lanes[h], gap_ent, gap_ex, dur))
    r = np.array(rows, float)
    if not len(r):
        print(panel, "no events"); continue
    lane_drop = r[:, 5] < 0
    print(f"{panel:<10} events {len(r)}: pre-breakdown q/cap_h median {np.median(r[:, 0]):.2f} q/cap_down {np.median(r[:, 1]):.2f} | "
          f"discharge/cap_h {np.median(r[:, 2]):.2f} /cap_down {np.median(r[:, 3]):.2f} | cap_down/cap_h median {np.median(r[:, 4]):.2f} "
          f"lane drop downstream {lane_drop.mean():.2f} | entry in gap {r[:, 6].mean():.2f} exit {r[:, 7].mean():.2f} | duration median {np.median(r[:, 8]):.0f} slots", flush=True)
