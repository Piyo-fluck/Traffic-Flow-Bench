"""Note 169: congestion share by 2-hour block, truth vs CTM with released(+hidden) capacities vs fitted capacities,
demand unbiased (CONSERVE_FILL, multipliers 1), held-out train days."""
import os
import sys
import warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "analysis")); sys.path.insert(0, str(ROOT / "tta_lab" / "tta"))
import numpy as np
import common as C
from ctm_gen import CTMGen
from tfb.data import load
warnings.simplefilter("ignore")
BL = [(5, 7), (7, 9), (9, 11), (11, 13), (13, 15), (15, 17), (17, 19), (19, 21)]
for panel in (sys.argv[1:] or C.T2_PANELS):
    z = load(panel, "train")
    days = np.arange(1, len(z["dates"]), 6)[:30]
    rows = {}
    for tag in os.environ.get("TAGS", "released,fitted").split(","):
        g = CTMGen(panel, z); g.fit_profiles(z)
        if tag == "fitted":
            c = np.load(ROOT / "work" / f"ctm_caps_{panel}.npy"); cells = g.sens_cell[:-1] + 1
            g.cell_cap[cells] = c; g.cell_w[cells] = c / np.maximum(g.cell_kj[cells] - c / g.cell_vf[cells], 1e-6)
        Q = []
        for d in days:
            up, ent = g.censored_demand(z["true_flow"][d], z["true_speed"][d], int(z["weekday"][d]))
            V, _ = g.simulate(up, ent)
            Q.append(np.nan_to_num(V[:, g.order], nan=1e9) <= 0.6 * g.vf)
        rows[tag] = np.mean(Q, (0, 2))
    vt = z["true_speed"][days][:, :, g.order]
    rows["truth"] = np.mean(np.nan_to_num(vt, nan=1e9) <= 0.6 * g.vf, (0, 2))
    print(panel)
    for tag in ["truth"] + os.environ.get("TAGS", "released,fitted").split(","):
        print(f"   {tag:<9} " + " ".join(f"{a:02d}-{b:02d}h {rows[tag][a * 12:b * 12].mean():.3f}" for a, b in BL), flush=True)
