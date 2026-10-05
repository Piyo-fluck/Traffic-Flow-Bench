"""Note 169: generate new runs of a panel with the fitted DLSim generator.

A run draws one per-entry demand factor exp(sigma * eta), eta iid N(0,1) (sigma = the panel's run-to-run log sd
of entry demand relative to train, analysis/dlsim_demand_draws.py), a panel scale exp(0.02 N) and an upstream
scale exp(0.015 N), and Poisson(days / 30) incidents (random sensored link, 06-19 h start, 20-90 min, capacity
x U(0.3, 0.7)). Each day = a train day's own demand (dlsim_gen2 day mode) times the run factors, simulated with the
train-fitted capacities; availability (elig) is the source day's. Saved as work/genruns/<panel>_r<k>.npz with the
keys the Task 2 code reads (true_speed, true_flow, elig, weekday, dates, link_ids, free_speed) plus src_day.

    .venv/Scripts/python.exe analysis/dlsim_make_runs.py --runs 8 --days 60 [--panels ...]
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "analysis")); sys.path.insert(0, str(ROOT / "tta_lab" / "tta"))
import numpy as np  # noqa: E402

import common as C  # noqa: E402
import os  # noqa: E402

from dlsim_gen2 import SUB, T5, Gen  # noqa: E402
from dlsim_gen4 import Gen4  # noqa: E402

V5 = os.environ.get("DLSIM_V5", "") == "1" or os.environ.get("CTM", "") == "1"
USE_CTM = os.environ.get("CTM", "") == "1"
SIG_SCALE = float(os.environ.get("SIG_SCALE", 1.0))
# CTM: per-panel draw scale so that generated runs differ about as much as the real runs do (note 169)
SCALE_CTM = {"D12_I5_S": 2.0, "D7_I405_S": 1.0, "D12_I5_N": 2.5, "D7_I10_E": 2.5}
from tfb.data import load  # noqa: E402

SIGMA = {"D7_I10_E": 0.086, "D7_I10_W": 0.064, "D7_I210_E": 0.035, "D7_I210_W": 0.099, "D7_I405_N": 0.072,
         "D7_I405_S": 0.117, "D12_I5_N": 0.038, "D12_I5_S": 0.108}          # mean of the val/pri sds (the ratio's sd)


def main():
    warnings.simplefilter("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--panels", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = ROOT / "work" / (os.environ.get("GEN_OUT") or ("genruns_ctm" if USE_CTM else "genruns5" if V5 else "genruns"))
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    for pi, panel in enumerate(a.panels or C.T2_PANELS):
        z = load(panel, "train")
        if V5:
            if USE_CTM:
                from ctm_gen import CTMGen
                g = CTMGen(panel, z)
            else:
                g = Gen4(panel, z)
            g.fit_profiles(z)
            if os.environ.get("RULE", "") == "1":             # truth-only capacity type + pre-breakdown factors (note 169)
                ch = np.load(ROOT / "work" / os.environ.get("RULE_OUT", "ctm_rule0_{}.npz").format(panel))
                g.apply_choice(ch["heads"], ch["caps"], ch["sub"])
                g.ap[ch["ap_cells"]] = ch["ap"]; g.out_ap = float(ch["out_ap"])
                g.theta, g.theta_up = None, None
            elif os.environ.get("HB_CHOICE", "") == "1":       # physics-only version: per-head merge-position choice
                ch = np.load(ROOT / "work" / f"ctm_hb_choice_{panel}.npz")
                g.apply_choice(ch["heads"], ch["caps"], ch["sub"])
                g.theta, g.theta_up = None, None
            elif os.environ.get("CTM_CAPS", "") == "1":        # flow-consistent version: fitted caps, multipliers 1
                c = np.load(ROOT / "work" / f"ctm_caps_{panel}.npy")
                cells = g.sens_cell[:-1] + 1
                g.cell_cap[cells] = c
                g.cell_w[cells] = c / np.maximum(g.cell_kj[cells] - c / g.cell_vf[cells], 1e-6)
                g.theta, g.theta_up = None, None
            else:
                t = np.load(ROOT / "work" / f"dlsim_theta_{panel}.npz")
                g.theta, g.theta_up = t["theta"], t["theta_up"]
        else:
            g = Gen(panel, z)
            g.cap_all[:g.n] = np.load(ROOT / "work" / f"dlsim_caps_{panel}.npy")
        vf = np.asarray(z["free_speed"], float)
        nd = len(z["dates"])
        maps = []
        for k in range(a.runs):
            rng = np.random.default_rng(1000 * a.seed + 100 * pi + k)
            scale = (SCALE_CTM.get(panel, 1.5) if os.environ.get("OLD_SCALES", "") == "1" else 1.0) if USE_CTM and SIG_SCALE == 1.0 else SIG_SCALE
            fe = np.exp(scale * SIGMA[panel] * rng.standard_normal(len(g.ent)))          # the ratio's sd = one draw vs train
            sc, su = np.exp(0.02 * rng.standard_normal()), np.exp(0.015 * rng.standard_normal())
            days = np.sort(rng.choice(nd, a.days, replace=False))
            n_inc = rng.poisson(a.days / 30)
            inc = [(int(rng.integers(a.days)), int(rng.integers(g.n)), int(rng.integers(72, 228)),
                    int(rng.integers(4, 19)), float(rng.uniform(0.3, 0.7))) for _ in range(n_inc)]
            V = np.full((a.days, T5, g.S), np.nan, np.float32); Q = np.full_like(V, np.nan)
            for i, d in enumerate(days):
                if V5:
                    up, ent = g.censored_demand(z["true_flow"][d], z["true_speed"][d], int(z["weekday"][d]),
                                                g.theta, g.theta_up)
                else:
                    up, ent = g.demand(z["true_flow"][d])
                mult = None
                for (di, l, t0, dur, r) in inc:
                    if di == i:
                        mult = np.ones((T5, g.n)) if mult is None else mult
                        mult[t0:t0 + dur, l] *= r
                V[i], Q[i] = g.simulate(up * sc * su, ent * sc * fe[None, :], qmax_mult=mult)
            el = z["elig"][days] & np.isfinite(V)
            np.savez_compressed(out / f"{panel}_r{k}.npz", true_speed=V, true_flow=Q, elig=el,
                                weekday=z["weekday"][days], dates=z["dates"][days], link_ids=z["link_ids"],
                                free_speed=z["free_speed"], src_day=days, incidents=np.array(inc, float).reshape(-1, 5))
            qd = (np.nan_to_num(V, nan=1e9) <= 0.6 * vf) & el
            maps.append(qd[:, 72:240].mean((0, 1)))
            print(f"  {panel} r{k}: queued share {qd[:, 72:240].mean():.3f}, incidents {n_inc} "
                  f"({time.time() - t_start:.0f}s)", flush=True)
        vt = z["true_speed"]
        qt = (np.nan_to_num(vt, nan=1e9) <= 0.6 * vf) & np.isfinite(vt)
        mt = qt[:, 72:240].mean((0, 1))
        M = np.corrcoef(np.array(maps))
        cw = M[np.triu_indices(len(maps), 1)]
        ct = [np.corrcoef(m, mt)[0, 1] for m in maps]
        print(f"{panel}: map corr between generated runs {np.mean(cw):.2f} [{cw.min():.2f}, {cw.max():.2f}], "
              f"generated vs train truth {np.mean(ct):.2f} [{min(ct):.2f}, {max(ct):.2f}]", flush=True)


if __name__ == "__main__":
    main()
