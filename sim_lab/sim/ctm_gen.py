"""Synthetic simulation runs for Task 2 pretraining (plan 1, see ../SIM_EXPERIMENT.md).

Why: the train split is ONE simulation run, so every Task 2 model learns that run's habitual bottlenecks. The
scored months are other runs, with other demand draws and incidents, and the misses are exactly the windows where
another bottleneck breaks. This script creates many runs of the same corridor with our own cell-transmission
model, so a model can learn "which bottleneck breaks given this state" across runs, not the location prior of one.

Physics (reused from analysis/t2_ctm_sim.py, note 21):
  triangular FD per link from fd_parameters (vf, cap, kjam), Godunov/CTM with 6-s steps, Daganzo merge with
  mainline priority lanes/(lanes+1), fixed off-ramp split fractions and weekday x tod on-ramp profiles estimated on
  train, point queues at the upstream boundary and on the on-ramps.
Capacity (--cap-mode):
  fd         the released capacity
  discharge  (default) the 90th percentile of the flow at train queue heads (link s free, s-1 queued) where at
             least 50 such cells exist; note 21 found discharge rates transfer across runs within 0.5%. The
             released capacity elsewhere
Randomisation (all from train-derived profiles; nothing from validation/private):
  run level  global demand scale, per-on-ramp scale, a peak time shift, capacity jitter (0.5%)
  day level  global scale, per-on-ramp scale, AR(1) noise over time per ramp and at the upstream boundary
  incidents  Poisson number per run; a random link loses a random share of its capacity for 15-60 minutes
Sensor layer: per 5-min slot the mean FD speed and the mean link inflow, with multiplicative noise; the
eligibility mask of a random train day.

Output: <out>/<panel>/run_<k>.npz with the keys Task 2 code reads from a split (dates, weekday, true_speed,
true_flow, elig, free_speed, link_ids, lanes) plus meta_* keys with the run parameters.

    python sim/ctm_gen.py --out sim_lab/data --runs 40 --days 30            # all 8 panels
    python sim/ctm_gen.py --out sim_lab/data --runs 2 --days 7 --panels D7_I10_E   # smoke test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parents[1]
PROJ = LAB.parent
for p in (PROJ, PROJ / "analysis", PROJ / "scripts"):
    sys.path.insert(0, str(p))
import t2_ctm_sim  # noqa: E402
from t2_eda02 import T2_PANELS  # noqa: E402
from tfb.data import T, load  # noqa: E402

t2_ctm_sim.CACHE = LAB / "assets"          # the {panel}_t1tables.npz profiles ship with the kit
DT_H, SUB = t2_ctm_sim.DT_H, t2_ctm_sim.SUB


def discharge_capacity(z, vf, min_n=50, pct=90.0):
    """The pct-th percentile of the flow at link s while s-1 is queued and s is free (train truth), NaN where fewer
    than min_n such cells. A high percentile, because after a queue dissolves the flow drops below capacity."""
    v, q, el = z["true_speed"], z["true_flow"], z["elig"]
    qd = el & np.isfinite(v) & (v <= 0.6 * vf)
    fr = el & np.isfinite(v) & (v > 0.8 * vf)
    out = np.full(len(vf), np.nan)
    for s in range(1, len(vf)):
        m = qd[:, :, s - 1] & fr[:, :, s] & np.isfinite(q[:, :, s])
        if m.sum() >= min_n:
            out[s] = np.percentile(q[:, :, s][m], pct)
    return out


class Gen(t2_ctm_sim.Net):
    def __init__(self, panel, cap_mode="discharge", min_n=50, pct=90.0):
        super().__init__(panel)
        z = load(panel, "train")
        self.panel = panel
        self.free_speed, self.lanes_z = z["free_speed"], z["lanes"]
        self.train_elig = z["elig"]
        self.cap_fd = self.cap.copy()
        if cap_mode == "discharge":
            dc = discharge_capacity(z, z["free_speed"], min_n, pct)
            use = np.isfinite(dc) & (dc > 0.3 * self.cap_fd)
            self.cap = np.where(use, dc, self.cap_fd)
            self.n_discharge = int(use.sum())
        else:
            self.n_discharge = 0
        self.w = self.cap / np.maximum(self.kj - self.cap / np.maximum(self.vf_fd, 1), 1e-6)

    # ------------------------------------------------------------------ one run = D days, simulated together
    def run(self, D, rng, a):
        S = len(self.vf_fd)
        wd = (rng.integers(7) + np.arange(D)) % 7
        # run-level draws
        g_run = np.exp(rng.normal(0, a.run_scale))
        ramp_run = np.exp(rng.normal(0, a.run_ramp, S))
        shift = int(np.round(rng.uniform(-a.shift, a.shift) / 5))             # peak shift in slots
        cap = self.cap * np.exp(rng.normal(0, a.cap_jitter, S))
        # incidents: (day, link, start slot, n slots, remaining capacity share)
        n_inc = rng.poisson(a.incidents_per_day * D)
        inc = [(rng.integers(D), rng.integers(S), rng.integers(60, 240), rng.integers(3, 13),
                1 - rng.uniform(a.inc_min, a.inc_max)) for _ in range(n_inc)]
        # day-level draws and AR(1) noise
        g_day = g_run * np.exp(rng.normal(0, a.day_scale, D))
        ramp_day = ramp_run[None] * np.exp(rng.normal(0, a.day_ramp, (D, S)))
        rho = np.exp(-5 / a.ar_minutes)
        eps_r = np.zeros((D, S))
        eps_u = np.zeros(D)
        slot = np.clip(np.arange(T) - shift, 0, T - 1)
        # state
        kc0 = self.tq[wd, slot[0]] / np.maximum(self.tv[wd, slot[0]], 1.0)
        k = np.clip(np.nan_to_num(kc0), 0, 0.9 * self.kj)
        rq = np.zeros((D, S))
        uq = np.zeros(D)
        vf, kj, L, f = self.vf_fd, self.kj, self.L, self.f_off
        lanes_prev = np.concatenate([self.lanes[:1], self.lanes[:-1]])
        p = lanes_prev / (lanes_prev + 1.0)
        keep = 1.0 - np.concatenate([[0.0], f[:-1]])
        K = np.zeros((D, T, S), np.float32)
        Q = np.zeros((D, T, S), np.float32)
        for j in range(T):
            eps_r = rho * eps_r + np.sqrt(1 - rho ** 2) * rng.normal(0, a.ar_sigma, (D, S))
            eps_u = rho * eps_u + np.sqrt(1 - rho ** 2) * rng.normal(0, a.ar_sigma, D)
            up = g_day * self.tq[wd, slot[j], 0] * np.exp(eps_u)
            ramp = g_day[:, None] * ramp_day * self.ramp_prof[wd, slot[j]] * np.exp(eps_r)
            capj = np.broadcast_to(cap, (D, S)).copy()
            for (dd, ss, t0, n, share) in inc:
                if t0 <= j < t0 + n:
                    capj[dd, ss] *= share
            w = capj / np.maximum(kj - capj / np.maximum(vf, 1), 1e-6)
            ksum = np.zeros((D, S))
            qsum = np.zeros((D, S))
            for _ in range(SUB):
                Dm_ = np.minimum(vf * k, capj)
                Sp = np.minimum(capj, w * np.maximum(kj - k, 0))
                Dm = np.concatenate([(up + uq / DT_H)[:, None], Dm_[:, :-1] * keep[None, 1:]], 1)
                Dr = ramp + rq / DT_H
                free = Dm + Dr <= Sp
                fm = np.where(free, Dm, np.minimum(t2_ctm_sim.mid(Dm, Sp - Dr, p * Sp), Dm))
                fm = np.maximum(fm, 0)
                fr = np.where(free, Dr, np.clip(Sp - fm, 0, Dr))
                out = np.concatenate([fm[:, 1:] / keep[None, 1:], Dm_[:, -1:]], 1)
                k = np.clip(k + DT_H / L * (fm + fr - out), 0, kj)
                rq = np.maximum(rq + (ramp - fr) * DT_H, 0)
                uq = np.maximum(uq + (up - fm[:, 0]) * DT_H, 0)
                ksum += k
                qsum += fm + fr
            K[:, j], Q[:, j] = ksum / SUB, qsum / SUB
        with np.errstate(divide="ignore", invalid="ignore"):
            wbar = cap / np.maximum(kj - cap / np.maximum(vf, 1), 1e-6)
            v = np.minimum(vf, np.where(K > 1e-3, wbar * (kj / np.maximum(K, 1e-3) - 1), vf))
        v = np.clip(v, 1.0, None) * np.exp(rng.normal(0, a.noise_v, v.shape))
        q = np.clip(Q, 0, None) * np.exp(rng.normal(0, a.noise_q, Q.shape))
        elig = self.train_elig[rng.integers(len(self.train_elig), size=D)]
        meta = dict(g_run=float(g_run), shift_slots=shift, n_incidents=len(inc), incidents=[list(map(float, x)) for x in inc])
        return dict(weekday=wd.astype(np.int8), true_speed=v.astype(np.float32), true_flow=q.astype(np.float32),
                    elig=elig, free_speed=self.free_speed, link_ids=self.link_ids, lanes=self.lanes_z,
                    dates=np.array([f"sim-{i:03d}" for i in range(D)])), meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--panels", nargs="*", default=None)
    ap.add_argument("--runs", type=int, default=40)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cap-mode", default="discharge", choices=("discharge", "fd"))
    ap.add_argument("--discharge-pct", type=float, default=90.0)
    ap.add_argument("--discharge-min-n", type=int, default=50)
    ap.add_argument("--run-scale", type=float, default=0.04, help="sd of the log global demand scale per run")
    ap.add_argument("--run-ramp", type=float, default=0.08, help="sd of the log per-on-ramp scale per run")
    ap.add_argument("--shift", type=float, default=15.0, help="max peak shift per run, minutes")
    ap.add_argument("--cap-jitter", type=float, default=0.005)
    ap.add_argument("--day-scale", type=float, default=0.03)
    ap.add_argument("--day-ramp", type=float, default=0.05)
    ap.add_argument("--ar-sigma", type=float, default=0.05)
    ap.add_argument("--ar-minutes", type=float, default=30.0)
    ap.add_argument("--incidents-per-day", type=float, default=0.05)
    ap.add_argument("--inc-min", type=float, default=0.3)
    ap.add_argument("--inc-max", type=float, default=0.6)
    ap.add_argument("--noise-v", type=float, default=0.02)
    ap.add_argument("--noise-q", type=float, default=0.03)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "gen_args.json").write_text(json.dumps({k: str(v) for k, v in vars(a).items()}, indent=1))
    t_all = time.time()
    for pi, panel in enumerate(T2_PANELS):
        if a.panels and panel not in a.panels:
            continue
        g = Gen(panel, a.cap_mode, a.discharge_min_n, a.discharge_pct)
        (a.out / panel).mkdir(exist_ok=True)
        print(f"[{panel}] {len(g.vf_fd)} links, discharge-calibrated capacities: {g.n_discharge}", flush=True)
        for r in range(a.runs):
            rng = np.random.default_rng(10_000 * a.seed + 100 * pi + r)
            z, meta = g.run(a.days, rng, a)
            np.savez_compressed(a.out / panel / f"run_{r:03d}.npz", **z,
                                meta=np.array(json.dumps(meta)))
            qshare = float(((z["true_speed"] <= 0.6 * z["free_speed"]) & z["elig"]).mean())
            print(f"  run {r}: queued share {qshare:.3f}, incidents {meta['n_incidents']} ({time.time() - t_all:.0f}s)",
                  flush=True)


if __name__ == "__main__":
    main()
