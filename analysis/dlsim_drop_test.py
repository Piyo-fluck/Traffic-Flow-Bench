"""Note 169: released FD capacities (no fitting) + capacity drop at active bottlenecks, replication on train."""
import sys
import warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "analysis"))
import numpy as np
from dlsim_gen2 import Gen
from dlsim_gen3 import Gen3
from dlsim_fitcap import occupancy
from tfb.data import load
warnings.simplefilter("ignore")
for panel in sys.argv[1:] or ["D12_I5_S"]:
    z = load(panel, "train"); vf = np.asarray(z["free_speed"], float)
    ev = np.arange(3, len(z["dates"]), 6)[:12]
    for tag, G, drop in (("v3 fd caps, no drop", Gen, None), ("v3 fd caps, drop 0.7", Gen, 0.7),
                         ("v4 fd caps, drop 0.7", Gen3, 0.7), ("v4 fd caps, drop 0.8", Gen3, 0.8)):
        g = G(panel, z)
        qs, qt, iou = [], [], []
        for d in ev:
            up, ent = g.demand(z["true_flow"][d])
            V, _ = g.simulate(up, ent, drop=drop)
            a = occupancy(V, vf, g.order)[2]; b = occupancy(z["true_speed"][d], vf, g.order)[2]
            qs.append(a.mean(0)); qt.append(b.mean(0)); iou.append((a & b).sum() / max((a | b).sum(), 1))
        Ms, Mt = np.mean(qs, 0), np.mean(qt, 0)
        print(f"{panel:<10} [{tag}] share {Ms.mean():.3f} / true {Mt.mean():.3f}  map corr {np.corrcoef(Ms, Mt)[0, 1]:.2f}  daily IoU {np.mean(iou):.3f}", flush=True)
