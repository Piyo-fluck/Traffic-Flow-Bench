"""Generator v2 (note 169): the organizers' physics (DLSim4GMNS Newell merge) on the panel's sensored links, with
the real entries/exits of path_set.csv.

Structure: the corridor is the longest path of path_set; the sensored links (z) lie on it in travel order. Every
OD zone is an interchange with an entry and an exit, attached in the gap between two sensored links. Per gap g
(upstream sensor a -> downstream sensor b): exit fraction f_g = the zone's attraction / the volume passing it
under the Task 4 weak prior (a ratio, so the prior's scale does not matter), and the entry demand
e_g(t) = max(0, q_b(t) - (1 - f_g) q_a(t)) from the day's flows (train truth). Mainline FD per link from
fd_parameters (cap_mult, kj override optional), w = cap / (kj - kc). Entries merge into b with `entry_lanes`
lanes against b's upstream lanes (DLSim lane-proportional merge).

    .venv/Scripts/python.exe analysis/dlsim_gen2.py [panels...]
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tta_lab" / "tta"))
from dlsim4gmns import native  # noqa: E402

from tfb.data import RELEASE, load  # noqa: E402

import os as _os
TICK = float(_os.environ.get("DLSIM_TICK", 6.0))
SUB, T5 = int(round(300 / TICK)), 288


def corridor(panel, z):
    net = RELEASE / "corridors" / panel / "network"
    ps = pd.read_csv(net / "path_set.csv")
    seqs = ps.link_seq.str.split(";")
    main = max(seqs, key=len)
    pos = {l: i for i, l in enumerate(main)}
    ids = [str(x) for x in z["link_ids"]]
    zpos = np.array([pos.get(l, -1) for l in ids])
    order = np.array([i for i in np.argsort(zpos) if zpos[i] >= 0])            # z indices, upstream -> downstream
    spos = zpos[order]
    # zone attraction / passing volume under the weak prior (train)
    w = pd.read_csv(RELEASE / "task4" / panel / "train" / "synthetic_weak_prior.csv")
    m = w.merge(ps[["path_id", "link_seq"]], on="path_id")
    first = np.array([pos[s.split(";")[0]] for s in m.link_seq])
    last = np.array([pos[s.split(";")[-1]] for s in m.link_seq])
    f_ = m.path_flow.to_numpy(float)
    nC = len(main)
    passing = np.zeros(nC + 1)                                                  # volume on corridor link c
    np.add.at(passing, first, f_); np.add.at(passing, last + 1, -f_)
    passing = np.cumsum(passing)[:nC]
    attr = np.zeros(nC); np.add.at(attr, last, f_)
    prod = np.zeros(nC); np.add.at(prod, first, f_)
    return order, spos, passing, attr, prod


class Gen:
    def __init__(self, panel, z, cap_mult=1.0, kj_lane=None, entry_lanes=1.0, entry_cap=4000.0, w_fix=None):
        fd = pd.read_csv(RELEASE / "corridors" / panel / "network" / "fd_parameters.csv").set_index("link_id")
        ids = [str(x) for x in z["link_ids"]]
        order, spos, passing, attr, prod = corridor(panel, z)
        f = fd.loc[[ids[i] for i in order]]
        n = len(order)
        self.order, self.n, self.S = order, n, len(ids)
        self.len = np.maximum(f.length_km.to_numpy(float), 1.05 * f.free_speed_kmh.to_numpy(float) * TICK / 3600)
        self.lanes = f.lanes.to_numpy(float)
        self.vf = f.free_speed_kmh.to_numpy(float)
        self.cap = f.capacity_vph.to_numpy(float) * cap_mult
        kj = (self.lanes * kj_lane) if kj_lane else f.k_jam.to_numpy(float)
        kc = self.cap / self.vf
        self.w = np.full(n, w_fix) if w_fix else self.cap / np.maximum(kj - kc, 1e-6)
        self.kj_lane = kj / self.lanes
        # exits: fraction of the flow of sensored link a leaving between a and b
        self.f_exit = np.zeros(n)
        for g in range(n - 1):
            lo, hi = spos[g], spos[g + 1]
            a = attr[lo:hi].sum()
            self.f_exit[g] = np.clip(a / max(passing[lo], 1e-9), 0, 0.8)
        self.has_entry = np.array([prod[spos[g - 1] + 1:spos[g] + 1].sum() > 0 if g > 0 else True for g in range(n)])
        ent = np.flatnonzero(self.has_entry[1:]) + 1                           # entry origins merge into link g
        self.ent = ent
        ne = len(ent)
        L = n + ne + (n - 1)                                                    # mainline, entries, exit sinks
        down = np.full(L, -1)
        down[:n - 1] = np.arange(1, n)
        down[n:n + ne] = ent
        off_down = np.full(L, -1); off_frac = np.zeros(L)
        for g in range(n - 1):
            if self.f_exit[g] > 0:
                off_down[g] = n + ne + g; off_frac[g] = self.f_exit[g]
        self.L, self.down, self.off_down, self.off_frac = L, down, off_down, off_frac
        k_extra = ne + n - 1
        self.length = np.concatenate([self.len, np.full(k_extra, 0.3)])
        self.lanes_all = np.concatenate([self.lanes, np.full(ne, entry_lanes), np.ones(n - 1)])
        self.vf_all = np.concatenate([self.vf, np.full(k_extra, 80.0)])
        self.cap_all = np.concatenate([self.cap, np.full(ne, entry_cap), np.full(n - 1, 1e5)])
        kj_x = np.concatenate([np.full(ne, 150.0 * entry_lanes), np.full(n - 1, 1e4)])
        kc_x = self.cap_all[n:] / self.vf_all[n:]
        self.w_all = np.concatenate([self.w, self.cap_all[n:] / np.maximum(kj_x - kc_x, 1e-6)])
        self.kjl_all = np.concatenate([self.kj_lane, kj_x / np.concatenate([np.full(ne, entry_lanes), np.ones(n - 1)])])

    def demand(self, q):
        """q (T5, S) day flows (z order, NaN allowed) -> upstream inflow (T5,), entries (T5, n_entries)."""
        qs = q[:, self.order]
        qs = pd.DataFrame(qs).interpolate(axis=1, limit_direction="both").ffill().bfill().to_numpy()
        up = qs[:, 0]
        e = qs[:, self.ent] - (1 - self.f_exit[self.ent - 1]) * qs[:, self.ent - 1]
        return up, np.clip(e, 0, None)

    def fit_profiles(self, z, ff=0.85, vkey="true_speed", qkey="true_flow"):
        """Entry demand profiles from FREE-FLOW slots only (a and b both > ff vf): weekday x slot mean, gaps in
        time filled by linear interpolation; the upstream boundary likewise (its own free-flow slots)."""
        v, q = z[vkey][:, :, self.order], z[qkey][:, :, self.order]
        vf = self.vf
        free = np.isfinite(v) & (v > ff * vf) & np.isfinite(q)
        e = q[:, :, self.ent] - (1 - self.f_exit[self.ent - 1]) * q[:, :, self.ent - 1]
        okE = free[:, :, self.ent] & free[:, :, self.ent - 1]
        wd = z["weekday"]
        P = np.full((7, T5, len(self.ent)), np.nan); U = np.full((7, T5), np.nan)
        for k in range(7):
            m = wd == k
            with np.errstate(all="ignore"):
                P[k] = np.nanmean(np.where(okE[m], e[m], np.nan), 0)
                U[k] = np.nanmean(np.where(free[m, :, 0], q[m, :, 0], np.nan), 0)
        fill = lambda a: pd.DataFrame(a).interpolate(axis=0, limit_direction="both").to_numpy()  # noqa: E731
        self.P = np.clip(np.stack([np.nan_to_num(fill(P[k])) for k in range(7)]), 0, None)
        self.U = np.stack([np.nan_to_num(fill(U[k][:, None])[:, 0]) for k in range(7)])

    def discharge_caps(self, z, pct=90, min_n=30):
        """Capacity of sensored link b = the pct-th percentile of its flow while it is free and its upstream
        neighbour is queued (a queue head), where >= min_n such cells exist; the released capacity elsewhere."""
        v, q = z["true_speed"][:, :, self.order], z["true_flow"][:, :, self.order]
        vf = self.vf
        qd = np.isfinite(v) & (v <= 0.6 * vf); fr = np.isfinite(v) & (v > 0.8 * vf)
        for b in range(1, self.n):
            m = qd[:, :, b - 1] & fr[:, :, b] & np.isfinite(q[:, :, b - 1])
            if m.sum() >= min_n:                                # the queue's last link discharges at the bottleneck
                c = np.percentile(q[:, :, b - 1][m], pct)
                self.cap_all[b - 1] = min(self.cap_all[b - 1], c)

    def mixed_demand2(self, q_day, v_day, wd, ff=0.85):
        """The day's own entry demand where both bracketing sensors are free-flow; elsewhere the weekday profile
        scaled by the day's level (median ratio day / profile over the entry's free slots of that day)."""
        v, q = v_day[:, self.order], q_day[:, self.order]
        free = np.isfinite(v) & (v > ff * self.vf) & np.isfinite(q)
        e = q[:, self.ent] - (1 - self.f_exit[self.ent - 1]) * q[:, self.ent - 1]
        okE = free[:, self.ent] & free[:, self.ent - 1]
        P = self.P[wd]
        with np.errstate(all="ignore"):
            r = np.nanmedian(np.where(okE & (P > 50), np.clip(e, 0, None) / np.maximum(P, 1), np.nan)[60:252], 0)
        r = np.clip(np.nan_to_num(r, nan=1.0), 0.5, 2.0)
        ent = np.where(okE, np.clip(e, 0, None), P * r[None, :])
        U = self.U[wd]
        with np.errstate(all="ignore"):
            ru = np.nanmedian(np.where(free[:, 0] & (U > 50), q[:, 0] / np.maximum(U, 1), np.nan)[60:252])
        ru = float(np.clip(np.nan_to_num(ru, nan=1.0), 0.5, 2.0))
        up = np.where(free[:, 0], q[:, 0], U * ru)
        return np.nan_to_num(up), np.nan_to_num(ent)

    def mixed_demand(self, z, d, ff=0.85):
        """The day's own entry demand where both bracketing sensors are free-flow, else the weekday profile."""
        v, q = z["true_speed"][d][:, self.order], z["true_flow"][d][:, self.order]
        free = np.isfinite(v) & (v > ff * self.vf) & np.isfinite(q)
        e = q[:, self.ent] - (1 - self.f_exit[self.ent - 1]) * q[:, self.ent - 1]
        okE = free[:, self.ent] & free[:, self.ent - 1]
        wd = z["weekday"][d]
        ent = np.where(okE, np.clip(e, 0, None), self.P[wd])
        up = np.where(free[:, 0], q[:, 0], self.U[wd])
        return up, ent

    def profile_demand(self, wd, scale=1.0):
        return self.U[wd] * scale, self.P[wd] * scale

    def mainline_idx(self):
        """indices of the corridor (mainline) links in the DLSim arrays: sensored, plus unsensored in v4"""
        nv = getattr(self, "n_virt", 0)
        return np.concatenate([np.arange(self.n), np.arange(self.L - nv, self.L)])

    def simulate(self, up, ent, qmax_mult=None, n_slots=T5, drop=None, iters=4):
        """drop: capacity drop factor at active bottlenecks (a congested link whose downstream link is free
        discharges at drop x capacity while it stays congested), solved as a fixed point over `iters` runs."""
        up, ent = up[:n_slots], ent[:n_slots]
        if qmax_mult is not None:
            qmax_mult = qmax_mult[:n_slots]
        if drop is None:
            return self._simulate(up, ent, qmax_mult, n_slots)
        Tt = n_slots * SUB
        ml = self.mainline_idx()
        kc_tot = self.cap_all / self.vf_all
        crit = 1.05 * kc_tot * self.length                              # content above which a link is congested
        extra = np.ones((self.L, Tt))
        for it in range(iters):
            A, D = self._run(up, ent, qmax_mult, n_slots, extra)
            on = A - D
            cong = on > crit[:, None]
            dn = self.down
            head = np.zeros_like(cong)
            has = dn >= 0
            head[has] = cong[has] & ~cong[dn[has]]
            new = np.ones((self.L, Tt))
            new[ml] = np.where(head[ml], drop, 1.0)
            if np.array_equal(new, extra):
                break
            extra = new
        return self._post(A, D, n_slots)

    def _run(self, up, ent, qmax_mult, n_slots, extra=None):
        Tt = n_slots * SUB
        inflow = np.zeros((self.L, Tt))
        inflow[0] = np.repeat(up, SUB)
        inflow[self.n:self.n + len(self.ent)] = np.repeat(ent.T, SUB, 1)
        qmax = np.repeat(self.cap_all[:, None], Tt, 1)
        if qmax_mult is not None:
            qmax[:self.n] *= np.repeat(qmax_mult.T, SUB, 1)
        if extra is not None:
            qmax *= extra
        A, D = native.newell_merge(self.length, self.lanes_all, self.vf_all, self.w_all, self.kjl_all, qmax,
                                   self.down, inflow, TICK, off_down=self.off_down, off_frac=self.off_frac)
        return np.asarray(A), np.asarray(D)

    def _simulate(self, up, ent, qmax_mult, n_slots):
        A, D = self._run(up, ent, qmax_mult, n_slots)
        return self._post(A, D, n_slots)

    def _post(self, A, D, n_slots):
        Tt = n_slots * SUB
        A, D = A[:self.n], D[:self.n]
        b = np.arange(0, Tt, SUB)
        e = np.minimum(b + SUB, Tt - 1)
        flow = (D[:, e] - D[:, b]) / ((e - b) * TICK / 3600)
        n_avg = np.add.reduceat(A - D, b, axis=1) / np.diff(np.append(b, Tt))
        dens = n_avg / self.len[:, None]
        spd = np.minimum(np.where(dens > 1e-6, flow / np.maximum(dens, 1e-9), self.vf[:, None]), self.vf[:, None])
        V = np.full((n_slots, self.S), np.nan); Q = np.full((n_slots, self.S), np.nan)
        V[:, self.order] = spd.T; Q[:, self.order] = flow.T
        return V, Q

    def _unused_old_simulate(self, up, ent, qmax_mult=None, n_slots=T5):
        Tt = n_slots * SUB
        inflow = np.zeros((self.L, Tt))
        inflow[0] = np.repeat(up, SUB)
        inflow[self.n:self.n + len(self.ent)] = np.repeat(ent.T, SUB, 1)
        qmax = np.repeat(self.cap_all[:, None], Tt, 1)
        if qmax_mult is not None:
            qmax[:self.n] *= np.repeat(qmax_mult.T, SUB, 1)
        A, D = native.newell_merge(self.length, self.lanes_all, self.vf_all, self.w_all, self.kjl_all, qmax,
                                   self.down, inflow, TICK, off_down=self.off_down, off_frac=self.off_frac)
        A, D = np.asarray(A)[:self.n], np.asarray(D)[:self.n]
        b = np.arange(0, Tt, SUB)
        e = np.minimum(b + SUB, Tt - 1)
        flow = (D[:, e] - D[:, b]) / ((e - b) * TICK / 3600)
        n_avg = np.add.reduceat(A - D, b, axis=1) / np.diff(np.append(b, Tt))
        dens = n_avg / self.len[:, None]
        spd = np.minimum(np.where(dens > 1e-6, flow / np.maximum(dens, 1e-9), self.vf[:, None]), self.vf[:, None])
        V = np.full((n_slots, self.S), np.nan); Q = np.full((n_slots, self.S), np.nan)
        V[:, self.order] = spd.T; Q[:, self.order] = flow.T
        return V, Q


def replicate(panel, variants, days_step=9, n_days=12):
    z = load(panel, "train")
    vf = np.asarray(z["free_speed"], float)
    days = np.arange(0, len(z["dates"]), days_step)[:n_days]
    for tag, kw in variants:
        g = Gen(panel, z, **{k: v for k, v in kw.items() if not k.startswith("_")})
        qs, qr = [], []
        g.fit_profiles(z)
        if kw.get("_dis", False):
            g.discharge_caps(z, pct=kw.get("_pct", 90))
        for d in days:
            mode = kw.get("_dem", "mixed")
            up, ent = (g.mixed_demand(z, d) if mode == "mixed" else g.profile_demand(z["weekday"][d]) if mode == "profile"
                       else g.demand(z["true_flow"][d]))
            sc = kw.get("_scale", 1.0)
            V, _ = g.simulate(up * sc, ent * sc)
            v = z["true_speed"][d]
            qs.append((np.nan_to_num(V, nan=1e9) <= 0.6 * vf)[72:240])
            qr.append(((np.nan_to_num(v, nan=1e9) <= 0.6 * vf))[72:240])
        Ms, Mr = np.mean(qs, (0, 1)), np.mean(qr, (0, 1))
        iou = np.mean([(a & b).sum() / max((a | b).sum(), 1) for a, b in zip(qs, qr)])
        print(f"{panel:<10} [{tag}] queued share sim {Ms.mean():.3f} / true {Mr.mean():.3f}  map corr "
              f"{np.corrcoef(Ms, Mr)[0, 1]:.2f}  daily space-time IoU {iou:.3f}", flush=True)


def main():
    warnings.simplefilter("ignore")
    panels = sys.argv[1:] or ["D12_I5_S"]
    variants = [("mixed, queue-tail discharge caps p90", {"_dis": True}),
                ("mixed, p50", {"_dis": True, "_pct": 50}),
                ("day flows, p90", {"_dis": True, "_dem": "day"}),
                ("day flows, p50", {"_dis": True, "_dem": "day", "_pct": 50})]
    for p in panels:
        replicate(p, variants)


if __name__ == "__main__":
    main()
