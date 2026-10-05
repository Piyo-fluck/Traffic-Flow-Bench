"""Generating runs with the organizers' own physics (DLSim4GMNS, Newell merge loading), note 169.

A panel becomes a DLSim4GMNS merge network: the mainline chain in corridor order (upstream -> downstream), one
origin link per mainline link with an attached on-ramp (merging with lane-proportional capacity split, 1 lane
against the mainline lanes), and an off-ramp sink per link with an attached off-ramp (fixed off_frac, no
spillback). Mainline FD from fd_parameters (triangular: vf, capacity, k_jam total, w = cap / (kj - kc)).

Demand of a day: upstream boundary = the first mainline link's flow, on-ramps = the attached on-ramp flow, both per
5-min slot; off_frac per link = off-ramp flow / mainline flow over the train run. Output: 5-min speed and flow per
mainline link, in z's link order.

    .venv/Scripts/python.exe analysis/dlsim_gen.py D12_I5_S        # replicate train days, compare queue maps
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tta_lab" / "tta")); sys.path.insert(0, str(ROOT / "occ_lab"))
from dlsim4gmns import native  # noqa: E402

from tfb.data import RELEASE, load  # noqa: E402
from tfb.physics import load_ramps  # noqa: E402

TICK, SUB, T5 = 6.0, 50, 288
UP_HIGHER = {"D7_I10_E"}


class Net:
    def __init__(self, panel, z, cap_mult=1.0, kj_lane=None, ramp_cap=2200.0):
        fd = pd.read_csv(RELEASE / "corridors" / panel / "network" / "fd_parameters.csv").set_index("link_id")
        ids = [str(x) for x in z["link_ids"]]
        f = fd.loc[ids]
        S = len(ids)
        self.S, self.panel = S, panel
        self.order = np.arange(S)[::-1] if panel in UP_HIGHER else np.arange(S)   # upstream -> downstream
        self.len = f.length_km.to_numpy(float)
        self.lanes = f.lanes.to_numpy(float)
        self.vf = f.free_speed_kmh.to_numpy(float)
        self.cap = f.capacity_vph.to_numpy(float) * cap_mult
        self.kj = (self.lanes * kj_lane) if kj_lane else f.k_jam.to_numpy(float)
        self.kc = self.cap / self.vf
        self.w = self.cap / np.maximum(self.kj - self.kc, 1e-6)
        self.len = np.maximum(self.len, 1.05 * self.vf * TICK / 3600)             # one free-flow tick at least
        r = load_ramps(panel, "train", z)
        self.on_att, self.off_att = r["on_att"].astype(bool), r["off_att"].astype(bool)
        with np.errstate(all="ignore"):
            offv = r["off_v"] & np.isfinite(z["true_flow"])
            fr = np.where(offv, r["off_q"], 0).sum((0, 1)) / np.where(offv, z["true_flow"], 0).sum((0, 1))
        self.off_frac = np.clip(np.nan_to_num(fr), 0, 0.6) * self.off_att
        # full network: mainline 0..S-1 (z index), on-ramp origins, off-ramp sinks
        self.on_links = np.flatnonzero(self.on_att)
        self.off_links = np.flatnonzero(self.off_frac > 0)
        n_on, n_off = len(self.on_links), len(self.off_links)
        L = S + n_on + n_off
        down = np.full(L, -1)
        pos = np.empty(S, int); pos[self.order] = np.arange(S)
        for k in range(S - 1):
            down[self.order[k]] = self.order[k + 1]
        for i, s in enumerate(self.on_links):
            down[S + i] = s
        off_down = np.full(L, -1); off_frac = np.zeros(L)
        for i, s in enumerate(self.off_links):
            off_down[s] = S + n_on + i; off_frac[s] = self.off_frac[s]
        self.L, self.down, self.off_down, self.off_frac_all = L, down, off_down, off_frac
        ramp_len = np.full(n_on + n_off, 0.3)
        self.length = np.concatenate([self.len, ramp_len])
        self.lanes_all = np.concatenate([self.lanes, np.ones(n_on + n_off)])
        self.vf_all = np.concatenate([self.vf, np.full(n_on + n_off, 70.0)])
        cap_r = np.full(n_on + n_off, ramp_cap)
        self.cap_all = np.concatenate([self.cap, cap_r])
        kj_r = np.full(n_on + n_off, 150.0)
        kj_all = np.concatenate([self.kj, kj_r])
        kc_all = self.cap_all / self.vf_all
        self.w_all = self.cap_all / np.maximum(kj_all - kc_all, 1e-6)
        self.kj_lane_all = kj_all / self.lanes_all
        self.head = self.order[0]

    def simulate(self, up, on, qmax_mult=None):
        """up (T5,) veh/h into the most upstream link; on (T5, S) veh/h entering at each mainline link (only
        on_att links used). qmax_mult optional (T5, S) capacity multiplier (incidents).
        Returns speed, flow (T5, S) per mainline link."""
        Tt = T5 * SUB
        inflow = np.zeros((self.L, Tt))
        inflow[self.head] = np.repeat(np.nan_to_num(up), SUB)
        for i, s in enumerate(self.on_links):
            inflow[self.S + i] = np.repeat(np.nan_to_num(on[:, s]), SUB)
        qmax = np.repeat(self.cap_all[:, None], Tt, 1)
        if qmax_mult is not None:
            qmax[:self.S] *= np.repeat(qmax_mult.T, SUB, 1)
        A, D = native.newell_merge(self.length, self.lanes_all, self.vf_all, self.w_all, self.kj_lane_all, qmax,
                                   self.down, inflow, TICK, off_down=self.off_down, off_frac=self.off_frac_all)
        A, D = np.asarray(A)[:self.S], np.asarray(D)[:self.S]
        b = np.arange(0, Tt, SUB)
        e = np.minimum(b + SUB, Tt - 1)
        hrs = (e - b) * TICK / 3600
        flow = (D[:, e] - D[:, b]) / np.maximum(hrs, 1e-9)                        # (S, T5)
        on_link = (A - D)
        n_avg = np.add.reduceat(on_link, b, axis=1) / np.diff(np.append(b, Tt))
        dens = n_avg / self.len[:, None]
        spd = np.where(dens > 1e-6, flow / np.maximum(dens, 1e-9), self.vf[:, None])
        spd = np.minimum(spd, self.vf[:, None])
        return spd.T, flow.T


def day_demand(net, z, r, d):
    q = z["true_flow"][d]                                                     # (T5, S)
    up = q[:, net.head]
    on = np.where(r["on_v"][d], r["on_q"][d], np.nan)
    # missing on-ramp slots: the train weekday profile at that link
    return up, on


def main():
    warnings.simplefilter("ignore")
    panel = sys.argv[1] if len(sys.argv) > 1 else "D12_I5_S"
    z = load(panel, "train")
    r = load_ramps(panel, "train", z)
    vf = np.asarray(z["free_speed"], float)
    with np.errstate(all="ignore"):
        prof = np.stack([np.nanmean(np.where(r["on_v"], r["on_q"], np.nan)[z["weekday"] == k], 0) for k in range(7)])
    days = np.arange(0, len(z["dates"]), 9)[:12]
    for tag, kw in (("fd", {}), ("fd cap x1.2", {"cap_mult": 1.2}), ("kj 130/lane", {"kj_lane": 130.0})):
        net = Net(panel, z, **kw)
        qs, qr, share_s, share_r = [], [], [], []
        for d in days:
            up, on = day_demand(net, z, r, d)
            on = np.where(np.isfinite(on), on, prof[z["weekday"][d]])
            sp, fl = net.simulate(np.where(np.isfinite(up), up, np.nanmean(up)), on)
            v = z["true_speed"][d]
            ok = np.isfinite(v)
            qs.append((sp <= 0.6 * vf)[72:240]); qr.append(((np.nan_to_num(v, nan=1e9) <= 0.6 * vf) & ok)[72:240])
        Ms, Mr = np.mean(qs, (0, 1)), np.mean(qr, (0, 1))
        cell = np.mean([(a & b).sum() / max((a | b).sum(), 1) for a, b in zip(qs, qr)])
        print(f"{panel} [{tag}] queued share sim {Ms.mean():.3f} / true {Mr.mean():.3f}  map corr "
              f"{np.corrcoef(Ms, Mr)[0, 1]:.2f}  daily space-time IoU {cell:.3f}", flush=True)


if __name__ == "__main__":
    main()
