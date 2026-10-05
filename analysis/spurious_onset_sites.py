"""Note 169: anatomy of spurious onset sites in the v9 generator. Per panel: onset site shares (most downstream
queued link at T+30 of earliest-onset windows) in the truth, in the generated runs (genruns_v9) and in train-day
simulations (the v9 generator on the train days' own demand, no run draws); for the top generated sites that the truth
(< 1 %) does not use: the truth's queued share at the site (any time), a hidden bottleneck at the site (its capacity
/ released), the released capacity ratio to the next link, the entry merging in the gap and its morning demand."""
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
    g = CTMGen(panel, z); g.fit_profiles(z)
    ch = np.load(ROOT / "work" / f"ctm_hb_choice_{panel}.npz")
    for h, c in zip(ch["heads"], ch["caps"]):
        cell = g.sens_cell[int(h)] + 1; g.cell_cap[cell] = c; g.cell_w[cell] = c / max(g.cell_kj[cell] - c / g.cell_vf[cell], 1e-6)
    o = g.order; S = g.S
    rank = np.full(S, -1); rank[o] = np.arange(g.n)
    rs, _ = sites(z, o)
    gs = np.concatenate([sites(dict(np.load(f, allow_pickle=True)), o)[0] for f in sorted((ROOT / "work" / "genruns_v9").glob(f"{panel}_r*.npz"))])
    # train-day simulations
    V = []
    for d in range(0, len(z["dates"]), 2):
        up, ent = g.censored_demand(z["true_flow"][d], z["true_speed"][d], int(z["weekday"][d]))
        V.append(g.simulate(up, ent)[0])
    zs = dict(true_speed=np.array(V, np.float32), elig=np.isfinite(np.array(V)), free_speed=z["free_speed"], dates=z["dates"][::2], weekday=z["weekday"][::2])
    ts, _ = sites(zs, o)
    hr, hg, ht = (np.bincount(x, minlength=S) / max(len(x), 1) for x in (rs, gs, ts))
    vt = z["true_speed"]; qd_true = (np.nan_to_num(vt, nan=1e9) <= 0.6 * np.asarray(z["free_speed"], float)).mean((0, 1))
    hb = {int(h): (float(c), bool(sb)) for h, c, sb in zip(ch["heads"], ch["caps"], ch["sub"])}
    ent_at = {int(b): i for i, b in enumerate(g.ent)}
    print(f"{panel}: spurious share (sites < 1 % of real): generated {hg[hr < 0.01].sum():.2f}, train-day sims {ht[hr < 0.01].sum():.2f}")
    for sidx in np.argsort(-(hg * (hr < 0.01)))[:5]:
        if hg[sidx] < 0.03:
            break
        r_ = rank[sidx]
        hbs = hb.get(int(r_))
        i_e = ent_at.get(int(r_) + 1)
        print(f"   site z {int(sidx)} (travel {int(r_)}): gen {hg[sidx]:.2f} train-day sim {ht[sidx]:.2f} real onset {hr[sidx]:.3f} | truth queued share {qd_true[sidx]:.3f} | "
              f"hidden bn {('%.0f (%.2f x released, %s)' % (hbs[0], hbs[0] / g.cap[min(r_ + 1, g.n - 1)], 'minus entry' if hbs[1] else 'raw')) if hbs else 'none'} | "
              f"released cap next/this {g.cap[min(r_ + 1, g.n - 1)] / g.cap[r_]:.2f} | entry in gap: {('demand AM %.0f' % g.P[1, 84:120, i_e].mean()) if i_e is not None else 'none'}", flush=True)
