"""Note 169: which gaps create the free-flow flow surplus? Per gap (sensors a -> b, travel order), morning free
slots (truth and sim both free at a and b), mean absolute errors err_a = sim - true at a, err_b at b; the gap's own
surplus D = err_b - (1 - F) err_a (veh/h). Related to the gap's properties: unsensored cells, whether the entry
merges into an unsensored cell or into b itself, F, entry demand level, hidden bottleneck in the gap."""
import sys
import warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "analysis")); sys.path.insert(0, str(ROOT / "tta_lab" / "tta"))
import numpy as np
from ctm_gen import CTMGen
from tfb.data import load
warnings.simplefilter("ignore")
AM = slice(72, 132)
for panel in sys.argv[1:] or ["D7_I405_N", "D12_I5_N"]:
    z = load(panel, "train"); g = CTMGen(panel, z); g.fit_profiles(z); o = g.order
    E, FR = [], []
    for d in np.arange(1, len(z["dates"]), 6)[:25]:
        up, ent = g.censored_demand(z["true_flow"][d], z["true_speed"][d], int(z["weekday"][d]))
        V, Q = g.simulate(up, ent)
        qt, vt, qs, vs = z["true_flow"][d][:, o], z["true_speed"][d][:, o], Q[:, o], V[:, o]
        free = (vt > 0.9 * g.vf) & (vs > 0.9 * g.vf)
        E.append(np.where(free, qs - qt, np.nan)[AM]); FR.append(free[AM])
    E = np.concatenate(E)
    err = np.nanmean(E, 0)
    D = err[1:] - (1 - g.f_exit[:-1]) * err[:-1]
    nvirt = np.diff(g.sens_cell) - 1
    ent_gap = {int(b): i for i, b in enumerate(g.ent)}
    ent_into_b = np.zeros(g.n - 1, bool); has_ent = np.zeros(g.n - 1, bool); edem = np.zeros(g.n - 1)
    for b, i in ent_gap.items():
        if 1 <= b < g.n:
            has_ent[b - 1] = True
            ent_into_b[b - 1] = g.ent_cell[i] == g.sens_cell[b]
            edem[b - 1] = g.P[1, AM, i].mean()
    ok = np.isfinite(D)
    print(f"{panel}: gaps {ok.sum()}, mean per-gap surplus {np.nanmean(D):+.1f} veh/h (sum {np.nansum(D):+.0f}); "
          f"mean flow {np.nanmean(z['true_flow'][:, AM]):.0f}")
    for nm, m in (("no unsensored cells", nvirt == 0), (">= 1 unsensored cell", nvirt >= 1), ("entry merges into b itself", has_ent & ent_into_b),
                  ("entry merges into an unsensored cell", has_ent & ~ent_into_b), ("no entry", ~has_ent), ("F > 0.1", g.f_exit[:-1] > 0.1),
                  ("F <= 0.1", g.f_exit[:-1] <= 0.1)):
        mm = m & ok
        print(f"   {nm:<38} n {mm.sum():4d}  mean surplus {np.nanmean(D[mm]) if mm.any() else np.nan:+.1f}")
    print(f"   corr(surplus, entry demand) {np.corrcoef(D[ok & has_ent], edem[ok & has_ent])[0, 1]:+.2f}; corr(surplus, F) {np.corrcoef(D[ok], g.f_exit[:-1][ok])[0, 1]:+.2f}")
