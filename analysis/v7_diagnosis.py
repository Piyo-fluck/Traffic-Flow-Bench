"""Note 169: v7 diagnosis on train days (no run draws). (1) upstream-end onsets: hidden bottlenecks near the upstream
end, morning upstream demand vs the first links' capacity, queued share of the first 3 links (truth vs sim);
(2) morning (06-11 h) flow at the sensored links: sim with multipliers 1 vs the fitted multipliers vs truth."""
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
AM = slice(72, 132)
for panel in C.T2_PANELS:
    z = load(panel, "train")
    g = CTMGen(panel, z); g.fit_profiles(z)
    th = np.load(ROOT / "work" / f"dlsim_theta_{panel}.npz")
    o, vf = g.order, g.vf
    days = np.arange(1, len(z["dates"]), 6)[:25]
    q_true = np.nanmean(z["true_flow"][days][:, AM][:, :, o], (0, 1))
    v_true = z["true_speed"][days][:, :, o]
    out = {}
    for nm, (t1, t2) in (("theta 1", (None, None)), ("fitted", (th["theta"], th["theta_up"]))):
        Q, V = [], []
        for d in days:
            up, ent = g.censored_demand(z["true_flow"][d], z["true_speed"][d], int(z["weekday"][d]), t1, t2)
            v_, q_ = g.simulate(up, ent)
            Q.append(q_[AM][:, o]); V.append(v_[:, o])
        out[nm] = (np.nanmean(np.array(Q), (0, 1)), np.array(V))
    r1 = out["theta 1"][0] / np.maximum(q_true, 1); r2 = out["fitted"][0] / np.maximum(q_true, 1)
    qd = lambda V: (np.nan_to_num(V, nan=1e9) <= 0.6 * vf)[:, 72:240, :3].mean((0, 1))  # noqa: E731
    hb = [(h, round(float(g.cell_cap[g.sens_cell[h] + 1] / g.cap[min(h + 1, g.n - 1)]), 2)) for h in range(g.n - 1)
          if g.cell_cap[g.sens_cell[h] + 1] < 0.95 * g.cap[min(h + 1, g.n - 1)]]
    up_am = np.nanmean(z["true_flow"][days][:, AM][:, :, o[0]])
    print(f"{panel:<10} morning flow sim/true: theta 1 median {np.nanmedian(r1):.3f}, fitted {np.nanmedian(r2):.3f} | "
          f"hidden bottlenecks (travel idx, cap/released) {hb[:6]} | upstream morning flow {up_am:.0f} vs cap link0 {g.cap[0]:.0f}, link1 {g.cap[1]:.0f} | "
          f"queued share of links 0-2: true {np.round(qd(v_true), 3).tolist()} sim-theta1 {np.round(qd(out['theta 1'][1]), 3).tolist()} sim-fitted {np.round(qd(out['fitted'][1]), 3).tolist()}", flush=True)
