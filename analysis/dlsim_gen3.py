"""Generator v4 (note 169): the full corridor, unsensored links included.

v2/v3 chained only the sensored links, so the storage between two sensors (k_jam x length x lanes of the 1-4
unsensored corridor links in between) was missing and queues spread over too many sensored links. Here every
corridor link between the first and the last sensored one is a DLSim link: sensored links keep their FD row;
unsensored links get the median sensored length and the FD of the nearest upstream sensored link. Entries attach at
the corridor position of their zone (the first link of its paths; per gap the zone with the largest production),
exits leave after the corridor position of their zone with off_frac = the zone's attraction / passing volume under
the weak prior. Demand estimation per gap is unchanged (dlsim_gen2.Gen.demand).

Array layout as in v2 so the fitting and generation code applies unchanged: [0, n) sensored links in travel order,
[n, n + ne) entry origins, [n + ne, n + ne + nx) exit sinks, then the unsensored links.

    .venv/Scripts/python.exe analysis/dlsim_gen3.py [panels]      # replication with v2's fitted caps as a start
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "analysis"))
from dlsim_gen2 import TICK, Gen, corridor  # noqa: E402

from tfb.data import RELEASE, load  # noqa: E402


class Gen3(Gen):
    def __init__(self, panel, z, entry_lanes=1.0, entry_cap=4000.0):
        super().__init__(panel, z, entry_lanes=entry_lanes, entry_cap=entry_cap)      # sensored arrays, f_exit, ent
        order, spos, passing, attr, prod = corridor(panel, z)
        n = self.n
        med_len = float(np.median(self.len))
        cpos = np.arange(spos[0], spos[-1] + 1)                                 # corridor positions in the chain
        sens = {int(p): k for k, p in enumerate(spos)}
        virt = [int(p) for p in cpos if int(p) not in sens]
        # entries: one origin per gap with an entry (as in v2), attached at its zone's corridor position
        ent_pos = []
        for b in self.ent:
            lo, hi = spos[b - 1] + 1, spos[b] + 1
            cand = np.arange(lo, hi)
            ent_pos.append(int(cand[np.argmax(prod[lo:hi])]) if len(cand) else int(spos[b]))
        exits = [int(p) for p in cpos[:-1] if attr[p] > 0]
        ne, nx, nv = len(self.ent), len(exits), len(virt)
        L = n + ne + nx + nv
        idx = dict(sens)
        for j, p in enumerate(virt):
            idx[p] = n + ne + nx + j
        down = np.full(L, -1)
        for a, b in zip(cpos[:-1], cpos[1:]):
            down[idx[int(a)]] = idx[int(b)]
        for i, p in enumerate(ent_pos):
            down[n + i] = idx[p]
        off_down = np.full(L, -1); off_frac = np.zeros(L)
        for i, p in enumerate(exits):
            off_down[idx[p]] = n + ne + i
            off_frac[idx[p]] = np.clip(attr[p] / max(passing[p], 1e-9), 0, 0.8)
        # per-link physics
        up_sens = np.array([max(k for pp, k in sens.items() if pp <= p) for p in virt], int) if nv else np.zeros(0, int)
        length = np.concatenate([self.len, np.full(ne + nx, 0.3), np.full(nv, med_len)])
        lanes = np.concatenate([self.lanes, np.full(ne, entry_lanes), np.ones(nx), self.lanes[up_sens]])
        vf = np.concatenate([self.vf, np.full(ne + nx, 80.0), self.vf[up_sens]])
        cap = np.concatenate([self.cap_all[:n], np.full(ne, entry_cap), np.full(nx, 1e5), self.cap[up_sens] * 1.05])
        kj_s = self.kj_lane * self.lanes
        kj = np.concatenate([kj_s, np.full(ne, 150.0 * entry_lanes), np.full(nx, 1e4), kj_s[up_sens]])
        kc = cap / vf
        self.L, self.down, self.off_down, self.off_frac = L, down, off_down, off_frac
        self.length, self.lanes_all, self.vf_all, self.cap_all = length, lanes, vf, cap
        self.w_all = cap / np.maximum(kj - kc, 1e-6)
        self.kjl_all = kj / lanes
        self.virt_up = up_sens
        self.n_virt = nv


def main():
    warnings.simplefilter("ignore")
    from dlsim_fitcap import score
    for panel in sys.argv[1:] or ["D7_I210_W"]:
        z = load(panel, "train")
        vf = np.asarray(z["free_speed"], float)
        nd = len(z["dates"])
        ev = np.arange(3, nd, 6)[:20]
        for tag, G in (("v3 sensored only", Gen), ("v4 full corridor", Gen3)):
            g = G(panel, z)
            g.fit_profiles(z)
            g.cap_all[:g.n] = np.load(ROOT / "work" / f"dlsim_caps_{panel}.npy")
            a, b, c, d = score(g, z, ev, vf)
            print(f"{panel:<10} [{tag}] held-out share {a:.3f} / true {b:.3f}  map corr {c:.2f}  daily IoU {d:.3f}"
                  + (f"  (+{g.n_virt} unsensored links)" if hasattr(g, "n_virt") else ""), flush=True)


if __name__ == "__main__":
    main()
