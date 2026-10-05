"""Note 169 (evaluation only): do the per-split Task 4 weak priors carry each run's demand draw?
Per entry gap (dlsim_gen2 structure): zone production under each split's weak prior (sum of path flows starting in
the gap) vs the entry demand estimated from that split's released flows (free-flow slots, 15-19 h, weekdays).
Correlations of log(val / train) and log(pri / train) between prior and demand, and of the priors' own ratios."""
import sys
import warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "analysis")); sys.path.insert(0, str(ROOT / "tta_lab" / "tta"))
import numpy as np
import pandas as pd
import common as C
from dlsim_gen2 import Gen, corridor
from tfb.data import RELEASE, load
warnings.simplefilter("ignore")
PM = slice(180, 228)


def prior_prod(panel, split, main_pos, spos, ent):
    ps = pd.read_csv(RELEASE / "corridors" / panel / "network" / "path_set.csv")
    w = pd.read_csv(RELEASE / "task4" / panel / split / "synthetic_weak_prior.csv").merge(ps[["path_id", "link_seq"]], on="path_id")
    first = np.array([main_pos[s.split(";")[0]] for s in w.link_seq])
    prod = np.zeros(max(main_pos.values()) + 1); np.add.at(prod, first, w.path_flow.to_numpy(float))
    return np.array([prod[spos[b - 1] + 1:spos[b] + 1].sum() for b in ent])


for panel in C.T2_PANELS:
    zt = load(panel, "train")
    g = Gen(panel, zt)
    ps = pd.read_csv(RELEASE / "corridors" / panel / "network" / "path_set.csv")
    main = max(ps.link_seq.str.split(";"), key=len)
    pos = {l: i for i, l in enumerate(main)}
    _, spos, *_ = corridor(panel, zt)
    pr = {s: prior_prod(panel, s, pos, spos, g.ent) for s in ("train", "validation", "private")}
    dm = {}
    for s in ("train", "validation", "private"):
        z = zt if s == "train" else load(panel, s)
        g.fit_profiles(z, vkey="true_speed" if s == "train" else "obs_speed", qkey="true_flow" if s == "train" else "obs_flow")
        dm[s] = g.P[:5, PM].mean((0, 1))
    out = []
    for s in ("validation", "private"):
        m = (pr[s] > 0) & (pr["train"] > 0) & (dm[s] > 30) & (dm["train"] > 30)
        rp, rd = np.log(pr[s][m] / pr["train"][m]), np.log(dm[s][m] / dm["train"][m])
        out.append(f"{s[:3]}: corr(prior ratio, demand ratio) {np.corrcoef(rp, rd)[0, 1]:+.2f} (n {m.sum()}, prior sd {rp.std():.3f}, demand sd {rd.std():.3f})")
    m = (pr["validation"] > 0) & (pr["private"] > 0) & (pr["train"] > 0)
    cvp = np.corrcoef(np.log(pr["validation"][m] / pr["train"][m]), np.log(pr["private"][m] / pr["train"][m]))[0, 1]
    ml = (pr["train"] > 0) & (dm["train"] > 30)
    lvl = np.corrcoef(np.log(pr["train"][ml]), np.log(dm["train"][ml]))[0, 1]
    print(f"{panel:<10} " + " | ".join(out) + f" | priors val-vs-pri ratio corr {cvp:+.2f} | train level corr(prior, demand) {lvl:+.2f}", flush=True)
