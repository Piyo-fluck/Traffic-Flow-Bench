"""Note 169: at the spurious onset sites, how does the queue start - locally (downstream neighbour free: a local
breakdown) or by spillback (downstream neighbour already queued)? And the time of the first queue of the day at the
site, truth vs train-day simulation (v9 generator)."""
import sys
import warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "analysis"))
import numpy as np
from ctm_gen import CTMGen
from tfb.data import load
warnings.simplefilter("ignore")
SITES = {"D12_I5_S": [97, 11], "D12_I5_N": [239, 114], "D7_I210_W": [80, 47]}


def starts(v, vf, h):
    """per day: first slot the site is queued, and whether its downstream neighbour was queued then (spillback)"""
    cg = np.nan_to_num(v, nan=1e9) <= 0.6 * vf
    out = []
    for d in range(v.shape[0]):
        t = np.flatnonzero(cg[d, 60:252, h])
        if len(t):
            t = t[0] + 60
            spill = bool(cg[d, t, h + 1]) if h + 1 < v.shape[2] else False
            out.append((t, spill))
    return out


for panel, hs in SITES.items():
    z = load(panel, "train")
    g = CTMGen(panel, z); g.fit_profiles(z)
    ch = np.load(ROOT / "work" / f"ctm_hb_choice_{panel}.npz")
    for h, c in zip(ch["heads"], ch["caps"]):
        cell = g.sens_cell[int(h)] + 1; g.cell_cap[cell] = c; g.cell_w[cell] = c / max(g.cell_kj[cell] - c / g.cell_vf[cell], 1e-6)
    o, vf = g.order, g.vf
    days = np.arange(0, len(z["dates"]), 3)
    vt = z["true_speed"][days][:, :, o]
    vs = []
    for d in days:
        up, ent = g.censored_demand(z["true_flow"][d], z["true_speed"][d], int(z["weekday"][d]))
        vs.append(g.simulate(up, ent)[0][:, o])
    vs = np.array(vs)
    for h in hs:
        for tag, v in (("truth", vt), ("sim", vs)):
            s = starts(v, vf, h)
            if not s:
                print(f"{panel} site {h} {tag}: never queued"); continue
            t = np.array([x[0] for x in s]); sp = np.array([x[1] for x in s])
            print(f"{panel:<10} site {h:3d} {tag:<5}: queued on {len(s)}/{len(days)} days, first queue median {np.median(t) / 12:.1f} h "
                  f"[q10 {np.quantile(t, .1) / 12:.1f}, q90 {np.quantile(t, .9) / 12:.1f}], spillback (downstream already queued) {sp.mean():.2f}", flush=True)
