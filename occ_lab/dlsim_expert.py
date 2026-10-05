"""Note 170: the fitted DLSim generator as a physics expert at prediction time (Task 2, online).

For a window (day d, origin t0) of a scored month: demand for every slot before t0 comes from that day's released
flows (dlsim_gen2 day mode: entries = q_b - (1 - f) q_a, interpolated along the corridor where unobserved); slot t0
and later are never read. From t0 on, each entry (and the upstream boundary) keeps its level of the last three
slots scaled by the train weekday profile's shape, ratio clipped to [0.5, 2]. The day is simulated from 00:00 to
t0 + F with the train-fitted capacities (work/dlsim_caps_<panel>.npy). Score per cell = sigmoid((0.6 vf - v) /
(0.05 vf)) of the simulated 5-min speed.

Saves occ_lab/cache/dlsim_<split>_<panel>.npz: og (N_og, F, S) over T+5..T+30 and on (N_on, S) at T+30, in the
order of common.windows.

    .venv/Scripts/python.exe occ_lab/dlsim_expert.py
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

LAB = Path(__file__).resolve().parent
ROOT = LAB.parent
sys.path.insert(0, str(LAB)); sys.path.insert(0, str(ROOT / "tta_lab" / "tta")); sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402

import common as C  # noqa: E402
from dlsim_gen2 import Gen  # noqa: E402
from tfb.data import load  # noqa: E402

F = C.F


def forecast(g, q_day, t0, wd):
    q = q_day.copy()
    q[t0:] = np.nan                                              # causal: slot t0 and later unseen
    up, ent = g.demand(q)
    up, ent = np.array(up, float), np.array(ent, float)
    n = t0 + F + 1
    last = slice(max(0, t0 - 3), t0)
    lvl_e = ent[last].mean(0); lvl_u = up[last].mean()
    pe = g.P[wd]; pu = g.U[wd]
    base_e = np.maximum(pe[last].mean(0), 1.0); base_u = max(pu[last].mean(), 1.0)
    for t in range(t0, n):
        ent[t] = lvl_e * np.clip(pe[t] / base_e, 0.5, 2.0)
        up[t] = lvl_u * np.clip(pu[t] / base_u, 0.5, 2.0)
    V, _ = g.simulate(up, ent, n_slots=n)
    return V


def score(V, vf):
    return 1.0 / (1.0 + np.exp(-(0.6 * vf - np.nan_to_num(V, nan=1e9)) / (0.05 * vf)))


def main():
    warnings.simplefilter("ignore")
    t_start = time.time()
    for i, panel in enumerate(C.T2_PANELS):
        zt = load(panel, "train")
        g = Gen(panel, zt)
        g.cap_all[:g.n] = np.load(ROOT / "work" / f"dlsim_caps_{panel}.npy")
        g.fit_profiles(zt)
        vf = np.asarray(zt["free_speed"], float)
        ctx, stat, W = C.windows(i, panel)
        for s in C.SPLITS:
            z, og, on = W[s]["z"], W[s]["og"], W[s]["on"]
            P_og = np.zeros((len(og["d"]), F, g.S), np.float32)
            P_on = np.zeros((len(on["d"]), g.S), np.float32)
            R_og = np.zeros((len(og["d"]), g.S), np.float32); R_on = np.zeros((len(on["d"]), g.S), np.float32)
            for n in range(len(og["d"])):
                d, t0 = int(og["d"][n]), int(og["t0"][n])
                V = forecast(g, z["obs_flow"][d], t0, int(z["weekday"][d]))
                P_og[n] = score(V[t0 + 1:t0 + F + 1], vf); R_og[n] = score(V[t0 - 1], vf)
            for n in range(len(on["d"])):
                d, t0 = int(on["d"][n]), int(on["t0"][n])
                V = forecast(g, z["obs_flow"][d], t0, int(z["weekday"][d]))
                P_on[n] = score(V[t0 + F], vf); R_on[n] = score(V[t0 - 1], vf)
            np.savez(LAB / "cache" / f"dlsim_{s}_{panel}.npz", og=P_og, on=P_on, og_prev=R_og, on_prev=R_on)
            sg = C.ongoing_iou(P_og, og, 0.5).mean() if len(og["d"]) else np.nan
            so = C.onset_iou(P_on, on).mean() if len(on["d"]) else np.nan
            print(f"  {panel} {s}: og {len(og['d'])} windows IoU {sg:.3f} | on {len(on['d'])} IoU {so:.3f} "
                  f"({time.time() - t_start:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
