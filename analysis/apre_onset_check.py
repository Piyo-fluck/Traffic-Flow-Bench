"""Note 169: pre-breakdown capacity factor a (causal capacity drop in the CTM) vs onset realism on train-day
simulations (v9 generator, no run draws): onset site overlap with the truth, spurious share, median onset time."""
import os
import sys
import warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "analysis"))
import numpy as np
from ctm_gen import CTMGen
from onset_site_distribution import sites
from tfb.data import load
warnings.simplefilter("ignore")
for panel in sys.argv[1:]:
    z = load(panel, "train")
    rs, rt = sites(z, CTMGen(panel, z).order) if False else (None, None)
    for a in (1.0,):
        os.environ["A_PRE"] = str(a)
        g = CTMGen(panel, z, a_pre=a); g.fit_profiles(z)
        ch = np.load(ROOT / "work" / f"ctm_hb_choice_{panel}.npz")
        moved = g.apply_choice(ch["heads"], ch["caps"], ch["sub"])
        o, S = g.order, g.S
        if rs is None:
            rs, rt = sites(z, o)
        V = []
        for d in range(0, len(z["dates"]), 2):
            up, ent = g.censored_demand(z["true_flow"][d], z["true_speed"][d], int(z["weekday"][d]))
            V.append(g.simulate(up, ent)[0])
        zs = dict(true_speed=np.array(V, np.float32), elig=np.isfinite(np.array(V)), free_speed=z["free_speed"], dates=z["dates"][::2], weekday=z["weekday"][::2])
        ts, tt = sites(zs, o)
        hr, ht = np.bincount(rs, minlength=S) / len(rs), np.bincount(ts, minlength=S) / max(len(ts), 1)
        print(f"{panel:<10} a {a:.2f}: onset site overlap {np.minimum(hr, ht).sum():.2f}, spurious {ht[hr < 0.01].sum():.2f}, "
              f"entries moved {moved} | onset time median real {np.median(rt) / 12:.1f} h / sim {np.median(tt) / 12:.1f} h, sim onsets {len(ts)}", flush=True)
