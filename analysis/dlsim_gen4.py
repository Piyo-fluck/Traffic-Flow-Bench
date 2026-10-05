"""Generator v5 (note 169): receiving-capacity gates, released FD, no fitted capacities.

analysis/dlsim_breakdown_anatomy.py showed how the TRUE data breaks down: the head h of a new queue discharges at
0.96-0.99 x the capacity of the next DOWNSTREAM link (0.58-0.78 x its own), where capacity drops by about 20 %;
no capacity drop after breakdown. That is receiving (supply) capacity as in a Godunov / CTM node: the inflow into a
link is capped by the link's capacity, so the queue sits upstream of the lower-capacity link. DLSim's Newell link
model caps inflow only by storage, which puts the queue inside the bottleneck link instead.

Here every corridor link c (sensored or not, full corridor as in v4) is preceded by a short gate link with c's
lanes, FD and capacity; the mainline from upstream and the zone entry merging at c (lane-proportional) both enter
the gate, whose outflow is capped at c's capacity. Exits leave the corridor link before the next gate.
Layout: [0, n) sensored links in travel order, [n, n + ne) entries, [n + ne, n + ne + nx) exit sinks, then the
unsensored corridor links, then the gates.

    .venv/Scripts/python.exe analysis/dlsim_gen4.py [panels]
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

from tfb.data import load  # noqa: E402


class Gen4(Gen):
    def __init__(self, panel, z, entry_lanes=1.0, entry_cap=4000.0, gate_len=None, virt_from="down"):
        super().__init__(panel, z, entry_lanes=entry_lanes, entry_cap=entry_cap)
        order, spos, passing, attr, prod = corridor(panel, z)
        n = self.n
        med_len = float(np.median(self.len)) * float(__import__("os").environ.get("VIRT_LEN", 1.0))
        med_len = max(med_len, 1.05 * float(np.max(self.vf)) * TICK / 3600)
        cpos = [int(p) for p in np.arange(spos[0], spos[-1] + 1)]
        sens = {int(p): k for k, p in enumerate(spos)}
        virt = [p for p in cpos if p not in sens]
        ent_pos = []
        for b in self.ent:
            lo, hi = spos[b - 1] + 1, spos[b] + 1
            ent_pos.append(int(lo + np.argmax(prod[lo:hi])) if hi > lo else int(spos[b]))
        exits = [p for p in cpos[:-1] if attr[p] > 0]
        ne, nx, nv, ng = len(self.ent), len(exits), len(virt), len(cpos) - 1
        L = n + ne + nx + nv + ng
        idx = dict(sens)
        for j, p in enumerate(virt):
            idx[p] = n + ne + nx + j
        gate = {p: n + ne + nx + nv + j for j, p in enumerate(cpos[1:])}      # gate in front of corridor link p
        if virt_from == "down":        # the capacity change sits right after the upstream sensor (the true head)
            up_sens = np.array([min(k for pp, k in sens.items() if pp >= p) for p in virt], int) if nv else np.zeros(0, int)
        else:
            up_sens = np.array([max(k for pp, k in sens.items() if pp <= p) for p in virt], int) if nv else np.zeros(0, int)
        # per corridor position physics (sensored: own row; unsensored: nearest upstream sensored, median length)
        def phys(p):
            k = sens.get(p)
            if k is None:
                k = int(up_sens[virt.index(p)])
                return med_len, self.lanes[k], self.vf[k], self.cap[k], self.kj_lane[k]
            return self.len[k], self.lanes[k], self.vf[k], self.cap[k], self.kj_lane[k]
        length = np.zeros(L); lanes = np.ones(L); vf = np.full(L, 80.0); cap = np.full(L, 1e5); kjl = np.full(L, 1e4)
        for p in cpos:
            ln, la, v_, c_, kl = phys(p)
            i = idx[p]
            length[i], lanes[i], vf[i], cap[i], kjl[i] = ln, la, v_, c_ * 1.02, kl         # the gate binds, not the link
            if p in gate:
                g = gate[p]
                length[g] = gate_len or 1.05 * v_ * TICK / 3600
                lanes[g], vf[g], cap[g], kjl[g] = la, v_, c_, kl
        length[n:n + ne] = 0.3; lanes[n:n + ne] = entry_lanes; vf[n:n + ne] = 80.0
        cap[n:n + ne] = entry_cap; kjl[n:n + ne] = 150.0
        length[n + ne:n + ne + nx] = 0.3
        down = np.full(L, -1)
        for a, b in zip(cpos[:-1], cpos[1:]):
            down[idx[a]] = gate[b]
            down[gate[b]] = idx[b]
        for i, p in enumerate(ent_pos):
            down[n + i] = gate[p] if p in gate else idx[p]
        off_down = np.full(L, -1); off_frac = np.zeros(L)
        for i, p in enumerate(exits):
            off_down[idx[p]] = n + ne + i
            off_frac[idx[p]] = np.clip(attr[p] / max(passing[p], 1e-9), 0, 0.8)
        kc = cap / vf
        kj = kjl * lanes
        self.L, self.down, self.off_down, self.off_frac = L, down, off_down, off_frac
        self.length, self.lanes_all, self.vf_all, self.cap_all = length, lanes, vf, cap
        self.w_all = cap / np.maximum(kj - kc, 1e-6)
        self.kjl_all = kjl
        self.n_virt = nv
        # only gates constrain: a corridor link's own outflow is left free (w, k_crit keep the true capacity)
        self.cap_all = cap.copy()
        corr_ix = np.array([idx[p] for p in cpos])
        self.cap_all[corr_ix] = cap[corr_ix] * float(__import__("os").environ.get("LINK_CAP", 1.02)) / 1.02
        self.corr_ix = corr_ix
        self.cpos, self.idx_map, self.ent_pos, self.exits_pos = cpos, idx, ent_pos, exits
        self.kj_arr = kj
        self.gate_up = np.array([idx[a] for a in cpos[:-1]])                   # corridor link feeding each gate
        self.gate_ix = np.array([gate[b] for b in cpos[1:]])
        self.mu = cap.copy()                                                    # released capacity = queue discharge
        # storage geometry per gap (a = b-1 -> b): link b's length + the unsensored links between (median length)
        nvirt_gap = np.diff(spos) - 1
        self.gap_len_b = self.len[1:]
        self.gap_len_v = nvirt_gap * med_len

    def simulate_cd(self, up, ent, a=1.08, iters=5, n_slots=288, qmax_mult=None):
        """Two capacities (note 169): a gate passes up to a x the released capacity while its upstream link is free
        (pre-breakdown), and the released capacity (queue discharge) while the upstream link is congested. Solved
        as a fixed point of the congestion schedule."""
        from dlsim_gen2 import SUB
        up, ent = up[:n_slots], ent[:n_slots]
        Tt = n_slots * SUB
        kc_tot = self.mu / self.vf_all
        crit = 1.05 * kc_tot * self.length
        # pre-breakdown both the sending (corridor link) and receiving (gate) capacities are a x released
        extra = np.ones((self.L, Tt)); extra[self.gate_ix] = a; extra[self.corr_ix] = a
        up_of_link = {int(gt): int(u) for u, gt in zip(self.gate_up, self.gate_ix)}
        for it in range(iters):
            A, D = self._run(up, ent, qmax_mult, n_slots, extra)
            content = A - D
            cong_c = content[self.corr_ix] > crit[self.corr_ix][:, None]           # each corridor link congested
            cong_g = content[self.gate_up] > crit[self.gate_up][:, None]           # link feeding each gate
            new = np.ones((self.L, Tt))
            new[self.gate_ix] = np.where(cong_g, 1.0, a)
            new[self.corr_ix] = np.where(cong_c, 1.0, a)
            if np.array_equal(new, extra):
                break
            extra = new
        return self._post(A, D, n_slots)

    def conservation_demand(self, q_day, v_day):
        """Entry demand by conservation: e_g = q_b - (1 - f) q_a + dN_g/dt, N_g = vehicles stored between the two
        sensors (density k = q / v, exact on the released FD, times length). The storage term brings back the
        demand that a queue absorbs; in free flow it is ~0 and this is the day mode of v2."""
        q = pd.DataFrame(q_day[:, self.order]).interpolate(axis=1, limit_direction="both").ffill().bfill().to_numpy()
        v = pd.DataFrame(v_day[:, self.order]).interpolate(axis=1, limit_direction="both").ffill().bfill().to_numpy()
        k = q / np.maximum(v, 1.0)
        b = self.ent
        N = k[:, b] * self.gap_len_b[b - 1] + 0.5 * (k[:, b] + k[:, b - 1]) * self.gap_len_v[b - 1]
        dN = np.gradient(N, axis=0) * 12.0                                     # veh per hour (5-min slots)
        sm = int(__import__("os").environ.get("DN_SMOOTH", 1))
        if sm > 1:                                                              # symmetric noise -> no upward bias
            ker = np.ones(sm) / sm
            dN = np.apply_along_axis(lambda x: np.convolve(x, ker, "same"), 0, dN)
        e = q[:, b] - (1 - self.f_exit[b - 1]) * q[:, b - 1] + dN
        return q[:, 0], np.clip(e, 0, None)

    BLOCKS = np.array([0] * 72 + [1] * 36 + [2] * 72 + [3] * 48 + [4] * 60)     # 0-6, 6-9, 9-15, 15-19, 19-24 h

    def censored_demand(self, q_day, v_day, wd, theta=None, theta_up=None):
        """Observed allocations are LOWER bounds of demand while a queue holds them back: demand = max(conservation
        estimate, weekday profile x the day's level), the upstream boundary likewise. theta (n_entries, 5) and
        theta_up (5,) scale the profile part in the slots where it is the binding (censored) estimate."""
        up_c, e_c = self.conservation_demand(q_day, v_day)
        up_p, e_p = self.mixed_demand2(q_day, v_day, wd)
        if theta is not None:
            e_p = e_p * theta[:, self.BLOCKS].T
        if theta_up is not None:
            up_p = up_p * theta_up[self.BLOCKS]
        if __import__("os").environ.get("UNBIASED_DEM", "1") == "1":
            # free slots (both bracketing sensors free): the observed demand as is - taking the max with a noisy
            # second estimate there only adds upward bias (note 169); max() only where a queue censors the flows
            v, q = v_day[:, self.order], q_day[:, self.order]
            free = np.isfinite(v) & (v > 0.85 * self.vf) & np.isfinite(q)
            okE = free[:, self.ent] & free[:, self.ent - 1]
            e_obs = np.clip(q[:, self.ent] - (1 - self.f_exit[self.ent - 1]) * q[:, self.ent - 1], 0, None)
            self.f_t = None
            if len(getattr(self, "cls_ent", [])) and getattr(self, "f_bg", None) is not None:
                # ramp-to-ramp classes (note 169): the exit after h takes F_bg (q_h - W) + W, not F q_h; the entry
                # of the same gap follows from that exit volume (else it is clipped to 0 during every surge)
                from scipy.ndimage import uniform_filter1d
                ent_ix = {int(b): i for i, b in enumerate(self.ent)}
                bf = getattr(self, "cls_betaf", None)
                for i in range(len(self.cls_ent)):
                    j_e = self.cls_ent[i]
                    h = int(np.flatnonzero(self.sens_cell == self.cls_exit[i])[0])
                    i_x = ent_ix.get(h + 1)
                    if i_x is None:
                        continue
                    e_j = np.nan_to_num(e_obs[:, j_e])
                    if bf is None:
                        W = self.cls_beta[i] * e_j
                    else:
                        eb = uniform_filter1d(e_j, 13, mode="nearest")
                        W = np.clip(self.cls_beta[i] * eb + bf[i] * (e_j - eb), 0, e_j)
                    Fb = self.f_bg[:q.shape[0], self.cls_exit[i]]
                    X = Fb * np.maximum(q[:, h] - W, 0) + W
                    e_obs[:, i_x] = np.clip(q[:, h + 1] - (q[:, h] - X), 0, None)
            if __import__("os").environ.get("EXIT_TV", "0") == "1":
                # time-varying exit share (note 169): a surge that enters and leaves within a few gaps (ramp-to-ramp
                # paths) makes q_b - (1 - F) q_a fall below the gap's usual entry; with a fixed F the model clips the
                # entry and sends the surge down the mainline. There, keep the entry at rho x its weekday profile
                # and let the gap's exit share absorb the rest: F_t = 1 - (q_b - e) / q_a
                rho = float(__import__("os").environ.get("EXIT_TV_RHO", 0.5))
                h = self.ent - 1
                e_raw = q[:, self.ent] - (1 - self.f_exit[h]) * q[:, h]
                floor = rho * self.P[wd]
                tv = okE & (e_raw < floor) & (self.f_exit[h] > 0)[None, :] & np.isfinite(e_raw)
                F = np.tile(self.f_exit[:self.n - 1].astype(float), (q.shape[0], 1))
                with np.errstate(all="ignore"):
                    Ft = np.clip(1 - (q[:, self.ent] - floor) / np.maximum(q[:, h], 1.0), 0, 0.95)
                for i in range(len(self.ent)):
                    m = tv[:, i]
                    F[m, h[i]] = np.maximum(Ft[m, i], self.f_exit[h[i]])
                e_obs = np.where(tv, floor, e_obs)
                self.f_t = F
            ent = np.where(okE, np.nan_to_num(e_obs), np.maximum(e_c, e_p))
            up = np.where(free[:, 0], np.nan_to_num(q[:, 0]), np.maximum(up_c, up_p))
            cf = __import__("os").environ.get("CONSERVE_FILL", "1")
            if cf == "1":
                # the fill must move demand EARLIER, not add vehicles (note 169): within every censored run of
                # slots (plus the recovery slots until the next free slot), rescale the filled demand so its total
                # equals the observed allocation over the same run
                alloc_e = np.nan_to_num(np.maximum(e_c, 0)); alloc_u = np.nan_to_num(np.maximum(up_c, 0))
                ent = ent.copy(); up = up.copy()

                def conserve(x, alloc, free_mask):
                    T = len(x); t = 0
                    while t < T:
                        if free_mask[t]:
                            t += 1; continue
                        a = t
                        while t < T and not free_mask[t]:
                            t += 1
                        b = min(T, t + 3)
                        tot_obs, tot_fill = alloc[a:b].sum(), x[a:b].sum()
                        if tot_fill > 0:
                            x[a:b] *= tot_obs / tot_fill
                    return x
                for i in range(ent.shape[1]):
                    ent[:, i] = conserve(ent[:, i], alloc_e[:, i], okE[:, i])
                up = conserve(up, alloc_u, free[:, 0])
            elif cf == "2":
                # arrival-curve reconstruction (note 169): a censored run plus the release period after it (observed
                # demand above the day's level = the backlog draining, up to 12 slots) forms one episode; arrivals
                # follow the day-level profile shape and their total equals the observed allocation over the episode
                wd_ = wd
                P_ = self.P[wd_] if theta is None else self.P[wd_] * theta[:, self.BLOCKS].T
                U_ = self.U[wd_] if theta_up is None else self.U[wd_] * theta_up[self.BLOCKS]
                e_all = np.where(okE, np.nan_to_num(e_obs), np.nan_to_num(np.maximum(e_c, 0)))
                u_all = np.where(free[:, 0], np.nan_to_num(q[:, 0]), np.nan_to_num(np.maximum(up_c, 0)))

                def rebuild(alloc, shape, free_mask):
                    x = alloc.copy(); T = len(x)
                    with np.errstate(all="ignore"):
                        rr = np.nanmedian(np.where(free_mask[60:252] & (shape[60:252] > 50), alloc[60:252] / np.maximum(shape[60:252], 1), np.nan))
                    lvl = shape * (np.clip(rr, 0.5, 2.0) if np.isfinite(rr) else 1.0)
                    t = 0
                    while t < T:
                        if free_mask[t]:
                            t += 1; continue
                        a = t
                        while t < T and not free_mask[t]:
                            t += 1
                        b = t
                        while b < T and b < t + 12 and alloc[b] > 1.05 * lvl[b]:
                            b += 1
                        tot, sh = alloc[a:b].sum(), lvl[a:b].sum()
                        if sh > 0:
                            x[a:b] = lvl[a:b] * tot / sh
                        t = b
                    return x
                ent = np.stack([rebuild(e_all[:, i], P_[:, i], okE[:, i]) for i in range(e_all.shape[1])], 1)
                up = rebuild(u_all, U_, free[:, 0])
            return up, ent
        return np.maximum(up_c, up_p), np.maximum(e_c, e_p)


VF = __import__("os").environ.get("VIRT_FROM", "down")
DEM = __import__("os").environ.get("DEM", "censored")
CD_A = float(__import__("os").environ.get("CD_A", 0))


def main():
    warnings.simplefilter("ignore")
    from dlsim_fitcap import occupancy
    for panel in sys.argv[1:] or ["D12_I5_S", "D7_I210_W"]:
        z = load(panel, "train")
        vf = np.asarray(z["free_speed"], float)
        ev = np.arange(3, len(z["dates"]), 6)[:12]
        g = Gen4(panel, z, virt_from=VF)
        g.fit_profiles(z)
        qs, qt, iou = [], [], []
        for d in ev:
            up, ent = (g.censored_demand(z["true_flow"][d], z["true_speed"][d], int(z["weekday"][d])) if DEM == "censored"
                       else g.conservation_demand(z["true_flow"][d], z["true_speed"][d]))
            V, _ = g.simulate_cd(up, ent, a=CD_A) if CD_A > 0 else g.simulate(up, ent)
            a = occupancy(V, vf, g.order)[2]; b = occupancy(z["true_speed"][d], vf, g.order)[2]
            qs.append(a.mean(0)); qt.append(b.mean(0)); iou.append((a & b).sum() / max((a | b).sum(), 1))
        Ms, Mt = np.mean(qs, 0), np.mean(qt, 0)
        print(f"{panel:<10} [v5 gates, {DEM} demand, cap-drop a={CD_A}] share {Ms.mean():.3f} / true {Mt.mean():.3f}  "
              f"map corr {np.corrcoef(Ms, Mt)[0, 1]:.2f}  daily IoU {np.mean(iou):.3f}", flush=True)


if __name__ == "__main__":
    main()
