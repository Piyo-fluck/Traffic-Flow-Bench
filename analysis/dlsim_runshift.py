"""Note 169, the generator's key test: do demand differences alone reproduce how a scored run's queue map differs
from the train run's? (The scored months' released layer is used to evaluate the method, not to fit a model.)

Capacities fitted on train (work/dlsim_caps_<panel>.npy). Demand per day from each run's own observed flows (day
mode: free and queued slots alike, interpolated along the corridor where missing). Maps = per-link queued share,
06-20 h. Reported: corr(true scored, true train) [how different the runs are], corr(sim with scored demand, true
scored), and corr of the CHANGES (sim scored - sim train) vs (true scored - true train).

    .venv/Scripts/python.exe analysis/dlsim_runshift.py [panels]
"""
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "analysis"))
import numpy as np  # noqa: E402

import os  # noqa: E402

from dlsim_gen2 import Gen  # noqa: E402
from dlsim_gen4 import Gen4  # noqa: E402

V5 = os.environ.get("DLSIM_V5", "") == "1" or os.environ.get("CTM", "") == "1"
USE_CTM = os.environ.get("CTM", "") == "1"
from tfb.data import load  # noqa: E402

A, B = 72, 240


def maps(g, z, vkey, qkey, vf, days):
    ms, mt = [], []
    for d in days:
        q = z[qkey][d]
        if V5:
            up, ent = g.censored_demand(q, z[vkey][d], int(z["weekday"][d]), g.theta, g.theta_up)
        else:
            up, ent = g.demand(q)
        V, _ = g.simulate(up, ent)
        ms.append((np.nan_to_num(V, nan=1e9) <= 0.6 * vf)[A:B])
        v = z[vkey][d][A:B]
        ok = np.isfinite(v)
        mt.append(np.where(ok, (np.nan_to_num(v, nan=1e9) <= 0.6 * vf), np.nan))
    return np.mean(ms, (0, 1)), np.nanmean(np.array(mt, float), (0, 1))


def main():
    warnings.simplefilter("ignore")
    for panel in sys.argv[1:] or ["D12_I5_S"]:
        zt = load(panel, "train")
        vf = np.asarray(zt["free_speed"], float)
        if V5:
            if USE_CTM:
                from ctm_gen import CTMGen
                g = CTMGen(panel, zt)
            else:
                g = Gen4(panel, zt)
            g.fit_profiles(zt)
            t = np.load(ROOT / "work" / f"dlsim_theta_{panel}.npz")
            g.theta, g.theta_up = t["theta"], t["theta_up"]
        else:
            g = Gen(panel, zt)
            cp = ROOT / "work" / f"dlsim_caps_{panel}.npy"
            if cp.exists():
                g.cap_all[:g.n] = np.load(cp)
        St, Tt = maps(g, zt, "true_speed", "true_flow", vf, np.arange(0, len(zt["dates"]), 5)[:40])
        for split in ("validation", "private"):
            zs = load(panel, split)
            Ss, Ts = maps(g, zs, "obs_speed", "obs_flow", vf, np.arange(len(zs["dates"])))
            m = np.isfinite(Ts) & np.isfinite(Tt)
            c_runs = np.corrcoef(Ts[m], Tt[m])[0, 1]
            c_sim = np.corrcoef(Ss[m], Ts[m])[0, 1]
            c_simtr = np.corrcoef(St[m], Ts[m])[0, 1]
            c_delta = np.corrcoef((Ss - St)[m], (Ts - Tt)[m])[0, 1]
            print(f"{panel:<10} {split[:3]}: true run vs train {c_runs:.2f} | sim(train demand) vs true run {c_simtr:.2f} | "
                  f"sim(run demand) vs true run {c_sim:.2f} | corr of changes {c_delta:+.2f} | shares true {np.nanmean(Ts):.3f} "
                  f"sim {Ss.mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
