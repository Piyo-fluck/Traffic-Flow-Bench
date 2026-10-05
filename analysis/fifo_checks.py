"""Note 169, FIFO generator: replication (share, map corr, daily IoU) on held-out train days for given panels
under the current environment switches."""
import sys
import warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "analysis"))
import numpy as np
from ctm_gen import CTMGen
from dlsim_fitcap import occupancy
from tfb.data import load
warnings.simplefilter("ignore")
tag = sys.argv[1]
for p in sys.argv[2:]:
    z = load(p, "train"); vf = np.asarray(z["free_speed"], float); g = CTMGen(p, z); g.fit_profiles(z); o = g.order
    qs_, qt_, iou = [], [], []
    for d in np.arange(1, len(z["dates"]), 6)[:30]:
        up, ent = g.censored_demand(z["true_flow"][d], z["true_speed"][d], int(z["weekday"][d]))
        V, _ = g.simulate(up, ent)
        A = occupancy(V, vf, o)[2]; B = occupancy(z["true_speed"][d], vf, o)[2]
        qs_.append(A.mean(0)); qt_.append(B.mean(0)); iou.append((A & B).sum() / max((A | B).sum(), 1))
    Ms, Mt = np.mean(qs_, 0), np.mean(qt_, 0)
    print(f"{p:<10} [{tag}] hidden {g.hidden_bottlenecks} share {Ms.mean():.3f}/{Mt.mean():.3f} map corr {np.corrcoef(Ms, Mt)[0, 1]:.2f} daily IoU {np.mean(iou):.3f}", flush=True)
