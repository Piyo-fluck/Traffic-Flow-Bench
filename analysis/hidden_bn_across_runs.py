"""Note 169: do the hidden bottlenecks carry over between runs? For every recurrent head identified on train
(>= 5 queue-days, discharge < 0.95 x next sensored capacity), the median daily discharge in train / validation /
private (observed layer for the scored runs; evaluation use only), and the number of queue-days per run."""
import sys
import warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "analysis"))
import numpy as np
import pandas as pd
from dlsim_gen2 import corridor
from tfb.data import RELEASE, load
warnings.simplefilter("ignore")


def heads(z, order, vf, vkey, qkey, cap):
    v, q = z[vkey][:, :, order], z[qkey][:, :, order]
    vfo = vf[order]
    cg = np.isfinite(v) & (v <= 0.6 * vfo); fr = np.isfinite(v) & (v > 0.9 * vfo)
    out = {}
    for h in range(v.shape[2] - 1):
        hd = cg[:, :, h] & fr[:, :, h + 1]
        daily = [np.median(q[d, hd[d], h + 1]) for d in range(v.shape[0]) if hd[d].sum() >= 4]
        out[h] = (len(daily), np.median(daily) / cap[order][h + 1] if daily else np.nan)
    return out


for panel in sys.argv[1:] or ["D12_I5_S", "D7_I405_N", "D12_I5_N"]:
    zt = load(panel, "train")
    order, *_ = corridor(panel, zt)
    ids = [str(x) for x in zt["link_ids"]]
    cap = pd.read_csv(RELEASE / "corridors" / panel / "network" / "fd_parameters.csv").set_index("link_id").loc[ids].capacity_vph.values
    vf = np.asarray(zt["free_speed"], float)
    ht = heads(zt, order, vf, "true_speed", "true_flow", cap)
    hs = {s: heads(load(panel, s), order, vf, "obs_speed", "obs_flow", cap) for s in ("validation", "private")}
    rows = [(h, *ht[h], *hs["validation"][h], *hs["private"][h]) for h in ht if ht[h][0] >= 5]
    print(f"{panel}: train recurrent heads {len(rows)} (hidden = discharge < 0.95: {sum(r[2] < 0.95 for r in rows)})")
    print("   head | train days level | val days level | pri days level")
    for h, a, b, c, d, e, f in rows:
        flag = " hidden" if b < 0.95 else ""
        print(f"   {h:4d} | {a:3d} {b:.2f} | {c:3d} {d:.2f} | {e:3d} {f:.2f}{flag}")
    # heads active in scored runs but not recurrent on train
    for s in ("validation", "private"):
        new = [(h, n, l) for h, (n, l) in hs[s].items() if n >= 3 and ht[h][0] < 5]
        print(f"   {s}: recurrent heads NOT recurrent on train: {len(new)} {[(h, n, round(l, 2)) for h, n, l in new][:8]}")
