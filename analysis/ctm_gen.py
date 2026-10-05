"""Note 169: a Godunov / cell-transmission generator on the full corridor (replaces the DLSim Newell link model).

Why: the truth breaks down like a CTM node (analysis/dlsim_breakdown_anatomy.py: the head discharges at the
downstream link's capacity, the data sits exactly on the released triangular FD), and the DLSim Newell link model
produced congestion unrelated to capacity (its storage-based inflow on short links). Cells = corridor links in
travel order (sensored + unsensored, geometry / entries / exits / demand from dlsim_gen4.Gen4). Per step:
    sending   S_c = min(vf k, a_c(t) cap_c)     receiving R_c = min(b_c(t) cap_c, w (kj - k))
    node c -> c+1: mainline (1 - f_c) S_c and the zone entry merging at c+1 (origin queue), lane-proportional split
    of R_{c+1} when they exceed it (water filling); exits f_c S_c leave freely (no off-ramp spillback)
    capacity drop (optional, causal): b_{c+1} = 1 while cell c is congested (k > 1.05 kc), else a_pre
Output: 5-min mean speed and flow per sensored link, z order.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from numba import njit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "analysis"))
from dlsim_gen4 import Gen4  # noqa: E402


@njit(cache=True)
def _ctm(L, vf, cap, kj, w, lanes, f, ent_cell, e_in, up_in, dt, sub, ap, ent_lanes, ent_cap, capm, k0, out_cap, fifo, mcap, out_ap, merge_rule,
         cls_ent, cls_exit, wsh):
    M = L.shape[0]; T5 = up_in.shape[0]; NE = ent_cell.shape[0]; K = cls_ent.shape[0]
    k = k0.copy(); oq = 0.0; eq = np.zeros(NE)
    # origin-tagged classes (note 169): class i = vehicles of entry cls_ent[i] that all leave at the exit of cell
    # cls_exit[i] and at no exit before it; they ride with the mainline, the cell's exit share mixes by composition
    kw = np.zeros((K, M)); phi = np.zeros(K); win = np.zeros((K, M)); wout = np.zeros((K, M))
    kc = cap / vf
    ent_of = -np.ones(M, np.int64)
    for j in range(NE):
        ent_of[ent_cell[j]] = j
    q_out = np.zeros((T5, M)); k_out = np.zeros((T5, M))
    S = np.zeros(M); R = np.zeros(M); flow = np.zeros(M); inflow = np.zeros(M)
    for t in range(T5):
        for s in range(sub):
            for c in range(M):
                cm = capm[t, c]                                              # incidents: capacity multiplier
                S[c] = min(vf[c] * k[c], max(ap[c], 1.0) * cap[c] * cm)
                bc = ap[c]
                if c > 0 and k[c - 1] > 1.05 * kc[c - 1]:
                    bc = 1.0
                if bc > 1.0:
                    # pre-breakdown capacity above the discharge: the receiving FD's apex moves up with it (with
                    # the discharge FD, w (kj - k) caps R at cap for k >= kc and the factor had no effect, note 169)
                    cb = bc * cap[c] * cm
                    wb = cb / max(kj[c] - cb / vf[c], 1e-6)
                    R[c] = min(cb, wb * max(kj[c] - k[c], 0.0))
                else:
                    R[c] = min(bc * cap[c] * cm, w[c] * max(kj[c] - k[c], 0.0))
            for c in range(M):
                inflow[c] = 0.0; flow[c] = 0.0
                for i in range(K):
                    win[i, c] = 0.0; wout[i, c] = 0.0
            # upstream boundary
            d0 = up_in[t] + oq / dt
            x0 = min(d0, R[0])
            oq = max(oq + (up_in[t] - x0) * dt, 0.0)
            inflow[0] += x0
            for c in range(M - 1):
                fc = f[t, c]
                if K > 0 and k[c] > 1e-9:
                    bg = 1.0; fx = 0.0
                    for i in range(K):
                        phi[i] = min(kw[i, c] / k[c], 1.0); bg -= phi[i]
                        if cls_exit[i] == c:
                            fx += phi[i]
                    fc = min(max(bg, 0.0) * fc + fx, 0.999)
                elif K > 0:
                    for i in range(K):
                        phi[i] = 0.0
                m_dem = (1.0 - fc) * S[c]
                j = ent_of[c + 1]
                e_dem = 0.0
                if j >= 0:
                    e_dem = min(e_in[t, j] + eq[j] / dt, ent_cap)
                Rn = R[c + 1]
                if m_dem + e_dem <= Rn:
                    am, ae = m_dem, e_dem
                else:
                    wm = lanes[c] / (lanes[c] + ent_lanes); we = 1.0 - wm
                    if e_dem == 0.0:
                        am, ae = min(m_dem, Rn), 0.0
                    elif merge_rule == 1:                                           # demand-proportional
                        am = Rn * m_dem / (m_dem + e_dem); ae = Rn - am
                    elif merge_rule == 2:                                           # entry first
                        ae = min(e_dem, Rn); am = min(m_dem, Rn - ae)
                    else:
                        am = min(m_dem, max(Rn * wm, Rn - e_dem))
                        ae = min(e_dem, Rn - am)
                mc = mcap[c + 1] * (ap[c + 1] if k[c] <= 1.05 * kc[c] else 1.0)
                if am > mc:                                                         # mainline-only bottleneck
                    am = mc
                    ae = min(e_dem, Rn - am)
                if fifo and fc > 0:
                    # FIFO diverge: when the mainline movement is held back, the exiting vehicles queue with it
                    flow[c] = am / max(1.0 - fc, 1e-9)
                else:
                    flow[c] = am + fc * S[c]
                inflow[c + 1] += am + ae
                for i in range(K):
                    if phi[i] > 0.0:
                        if cls_exit[i] == c:
                            wout[i, c] += (flow[c] - am) * phi[i] / max(fc, 1e-9)
                        else:
                            x = am * phi[i] / max(1.0 - fc, 1e-9)
                            wout[i, c] += x; win[i, c + 1] += x
                    if j >= 0 and cls_ent[i] == j and e_dem > 0.0:
                        win[i, c + 1] += ae * wsh[t, i]
                if j >= 0:
                    eq[j] = max(eq[j] + (e_in[t, j] - ae) * dt, 0.0)
            oc = out_cap * (out_ap if k[M - 1] <= 1.05 * kc[M - 1] else 1.0)
            flow[M - 1] = min(S[M - 1], oc)                               # downstream boundary (outside the panel)
            for i in range(K):
                if k[M - 1] > 1e-9:
                    wout[i, M - 1] += flow[M - 1] * min(kw[i, M - 1] / k[M - 1], 1.0)
            for c in range(M):
                k[c] = min(max(k[c] + dt / L[c] * (inflow[c] - flow[c]), 0.0), kj[c])
                for i in range(K):
                    kw[i, c] = min(max(kw[i, c] + dt / L[c] * (win[i, c] - wout[i, c]), 0.0), k[c])
                q_out[t, c] += flow[c] / sub
                k_out[t, c] += k[c] / sub
    return q_out, k_out


class CTMGen(Gen4):
    def __init__(self, panel, z, dt=2.0, a_pre=float(__import__("os").environ.get("A_PRE", 1.0)), ent_lanes=1.0, ent_cap=4000.0):
        super().__init__(panel, z)
        cp = self.cpos
        ix = np.array([self.idx_map[p] for p in cp])
        self.cell_L = self.length[ix].astype(np.float64)
        self.cell_vf = self.vf_all[ix]; self.cell_cap = self.mu[ix]
        self.cell_kj = (self.kjl_all * self.lanes_all)[ix]
        self.cell_w = self.cell_cap / np.maximum(self.cell_kj - self.cell_cap / self.cell_vf, 1e-6)
        self.cell_lanes = self.lanes_all[ix]
        pos_cell = {p: c for c, p in enumerate(cp)}
        self.cell_f = np.zeros(len(cp))
        for i, p in enumerate(self.exits_pos):
            self.cell_f[pos_cell[p]] = self.off_frac[self.idx_map[p]]
        self.ent_cell = np.array([pos_cell[p] for p in self.ent_pos], np.int64)
        if __import__("os").environ.get("GAP_EXITS", "1") == "1":
            # exits exactly as the demand estimate assumes (note 169): the gap's whole exit share F_g leaves the
            # sensored link at the gap's upstream end, before any entry of that gap merges
            self.cell_f = np.zeros(len(cp))
        self.sens_cell = np.array([pos_cell[p] for p in np.asarray(cp)[np.isin(cp, [p for p in cp if self.idx_map[p] < self.n])]])
        order_by_idx = {self.idx_map[p]: pos_cell[p] for p in cp if self.idx_map[p] < self.n}
        self.sens_cell = np.array([order_by_idx[i] for i in range(self.n)], np.int64)
        if __import__("os").environ.get("GAP_EXITS", "1") == "1":
            for h in range(self.n - 1):
                self.cell_f[self.sens_cell[h]] = self.f_exit[h]
        self.dt, self.a_pre, self.ent_lanes_, self.ent_cap_ = dt, a_pre, ent_lanes, ent_cap
        L_min = self.cell_vf * dt / 3600 * 1.05
        self.cell_L = np.maximum(self.cell_L, L_min)
        self.cap_released_cell = self.cell_cap.copy()
        self.hidden_bottlenecks = 0
        self.out_cap = 1e9
        self.mcap = np.full(len(self.cell_L), 1e9)
        # per-cell breakdown factor (note 169): while the cell upstream is uncongested the cell receives up to
        # ap x cap (the flow a head passes before it breaks down), after breakdown cap (the queue discharge)
        self.ap = np.full(len(self.cell_L), float(a_pre)); self.out_ap = 1.0
        self.cls_ent = np.zeros(0, np.int64); self.cls_exit = np.zeros(0, np.int64); self.cls_beta = np.zeros(0)
        self.f_bg = None                                                  # background exit shares with classes
        if not hasattr(self, "P"):
            self.fit_profiles(z)
        if __import__("os").environ.get("OUT_BOUNDARY", "1") == "1":
            self._downstream_boundary(z)
        # hidden bottlenecks per panel, chosen by held-out TRAIN replication (note 169): they hurt D7_I10_E and D7_I210_E
        if __import__("os").environ.get("HIDDEN_BN", "1") == "1" and (panel not in ("D7_I10_E", "D7_I210_E") or __import__("os").environ.get("HB_ALL", "") == "1"):
            self._hidden_bottlenecks(z)

    def apply_choice(self, heads, caps, subs):
        """set the hidden capacities chosen per head; where the entry was subtracted (it merges downstream of the
        bottleneck), move its merge cell past the bottleneck cell so the structure matches the assumption (note 169)"""
        ent_at = {int(b): i for i, b in enumerate(self.ent)}
        moved = 0
        self.head_cell = {}                                               # where each head's capacity acts
        for h, c, sb in zip(heads, caps, subs):
            h = int(h); cell = self.sens_cell[h] + 1
            self.head_cell[h] = cell
            self.cell_cap[cell] = c
            self.cell_w[cell] = c / max(self.cell_kj[cell] - c / self.cell_vf[cell], 1e-6)
            i_e = ent_at.get(h + 1)
            if (not sb and i_e is not None and __import__("os").environ.get("HB_RAW_AT_MERGE", "0") == "1"
                    and self.ent_cell[i_e] > cell):
                # 'raw' = discharge measured at the next sensor, which counts the entry: the bottleneck is after the
                # merge, so it caps the merge cell's receiving side, not the mainline-only cell before it (note 169)
                self.cell_cap[cell] = self.cap_released_cell[cell]
                self.cell_w[cell] = self.cell_cap[cell] / max(self.cell_kj[cell] - self.cell_cap[cell] / self.cell_vf[cell], 1e-6)
                cm = self.ent_cell[i_e]
                self.head_cell[h] = cm
                self.cell_cap[cm] = min(c, self.cell_cap[cm])
                self.cell_w[cm] = self.cell_cap[cm] / max(self.cell_kj[cm] - self.cell_cap[cm] / self.cell_vf[cm], 1e-6)
                moved += 1
                continue
            if sb and __import__("os").environ.get("MAINLINE_CAP", "1") == "1":
                # the entry merges after the bottleneck: cap the mainline movement into the cell, not the cell
                self.mcap[cell] = c
                self.cell_cap[cell] = self.cap_released_cell[cell]
                self.cell_w[cell] = self.cell_cap[cell] / max(self.cell_kj[cell] - self.cell_cap[cell] / self.cell_vf[cell], 1e-6)
                moved += 1
        return moved

    def _downstream_boundary(self, z, min_days=5, min_slots=4):
        """A queue entering from outside the panel: the downstream-most link queued while its upstream neighbour is
        free. Its median daily discharge then limits the corridor's outflow (D7_I405_N, note 169)."""
        o = self.order
        v, q = z["true_speed"][:, :, o], z["true_flow"][:, :, o]
        cg = np.isfinite(v) & (v <= 0.6 * self.vf)
        m = cg[:, :, -1] & ~cg[:, :, -2]
        daily = [np.median(q[d, m[d], -1]) for d in range(v.shape[0]) if m[d].sum() >= min_slots]
        if len(daily) >= max(min_days, 0.15 * v.shape[0]):
            self.out_cap = float(np.median(daily))
            if __import__("os").environ.get("OUT_CAP_MODE", "") == "ffp95":
                # the end passes more than the queued discharge before the outside queue arrives (note 169)
                ff = np.isfinite(v[:, :, -1]) & (v[:, :, -1] > 0.9 * self.vf[-1])
                self.out_cap = max(self.out_cap, float(np.nanquantile(q[:, :, -1][ff], 0.95)))

    def _hidden_bottlenecks(self, z, min_days=5, min_slots=4, min_share=float(__import__("os").environ.get("HB_MIN_SHARE", 0.15))):
        """Recurrent queue heads discharge below the released capacity of the next sensored link (0.64-0.79 x on 6
        panels): the binding capacity sits in the unsensored gap. Its value is observed directly - the median daily
        discharge at the next sensor while the head is queued and that sensor free. Assign it to the first cell
        after the head (an unsensored cell when the gap has one, else the next sensored cell's receiving side)."""
        o = self.order
        v, q = z["true_speed"][:, :, o], z["true_flow"][:, :, o]
        vf = self.vf
        cg = np.isfinite(v) & (v <= 0.6 * vf); fr = np.isfinite(v) & (v > float(__import__("os").environ.get("HB_HEAD_FREE", 0.9)) * vf)   # head: h+1 not queued (note 169)
        pos_cell = {p: c for c, p in enumerate(self.cpos)}
        spos = np.array(self.cpos)[self.sens_cell] if False else None
        sens_pos = sorted(p for p in self.cpos if self.idx_map[p] < self.n)
        sub_entry = __import__("os").environ.get("HB_SUB_ENTRY", "1") == "1"
        ent_at = {int(b): i for i, b in enumerate(self.ent)}             # entry merging in the gap h -> h+1 (b = h+1)
        wdays = z["weekday"]
        for h in range(self.n - 1):
            head = cg[:, :, h] & fr[:, :, h + 1]
            i_e = ent_at.get(h + 1) if sub_entry else None
            rule = __import__("os").environ.get("HB_RULE", "geom")
            if i_e is not None and rule == "geom":
                # geometry: the bottleneck is the first cell after the head; subtract the entry only if it merges
                # further downstream (then the next sensor counts it but the bottleneck does not)
                if not (self.ent_cell[i_e] > self.sens_cell[h] + 1):
                    i_e = None
            daily = []
            for d in range(v.shape[0]):
                if head[d].sum() < min_slots:
                    continue
                x = q[d, head[d], h + 1]
                if i_e is not None and hasattr(self, "P"):
                    # the sensor downstream also counts the entry merging after the bottleneck (note 169)
                    x = x - self.P[wdays[d]][head[d], i_e]
                daily.append(np.median(x))
            if len(daily) < max(min_days, min_share * v.shape[0]):     # rare heads = incidents, not capacity
                continue
            c_obs = float(np.median(daily))
            if i_e is not None and rule == "choice":                 # keep both options for a per-head choice
                raw = float(np.median([np.median(q[d, head[d], h + 1]) for d in range(v.shape[0]) if head[d].sum() >= min_slots]))
                self.hb_options = getattr(self, "hb_options", {})
                self.hb_options[h] = (raw, c_obs)
            cell_after = self.sens_cell[h] + 1                                  # first cell downstream of the head
            if c_obs < 0.95 * self.cell_cap[cell_after]:
                self.cell_cap[cell_after] = c_obs
                kc = c_obs / self.cell_vf[cell_after]
                self.cell_w[cell_after] = c_obs / max(self.cell_kj[cell_after] - kc, 1e-6)
                self.hidden_bottlenecks += 1

    def simulate(self, up, ent, qmax_mult=None, n_slots=288, k0=None, **kw):
        """qmax_mult: optional (T5, n_sensored) capacity multiplier on the sensored links (incidents)."""
        sub = int(round(300 / self.dt))
        capm = np.ones((n_slots, len(self.cell_L)))
        if qmax_mult is not None:
            capm[:, self.sens_cell] = qmax_mult[:n_slots]
        f2 = np.tile(self.cell_f.astype(np.float64), (n_slots, 1))
        if self.f_bg is not None:                                         # origin-tagged classes: background shares
            f2 = np.ascontiguousarray(self.f_bg[:n_slots])
        if getattr(self, "f_t", None) is not None:                     # time-varying exit shares (EXIT_TV)
            f2[:, self.sens_cell[:self.n - 1]] = self.f_t[:n_slots, :self.n - 1]
        q, k = _ctm(self.cell_L, self.cell_vf, self.cell_cap, self.cell_kj, self.cell_w, self.cell_lanes, f2,
                    self.ent_cell, np.ascontiguousarray(ent[:n_slots], np.float64),
                    np.ascontiguousarray(up[:n_slots], np.float64), self.dt / 3600, sub, self.ap,
                    self.ent_lanes_, self.ent_cap_, capm, np.zeros(len(self.cell_L)) if k0 is None else k0, self.out_cap,
                    __import__("os").environ.get("FIFO", "0") == "1", self.mcap, self.out_ap,
                    int(__import__("os").environ.get("MERGE_RULE", 0)),
                    self.cls_ent, self.cls_exit, np.ascontiguousarray(self.cls_wsh(n_slots, ent)))
        qs, ks = q[:, self.sens_cell], k[:, self.sens_cell]
        vfs = self.cell_vf[self.sens_cell]
        v = np.minimum(np.where(ks > 1e-6, qs / np.maximum(ks, 1e-9), vfs), vfs)
        V = np.full((n_slots, self.S), np.nan); Q = np.full((n_slots, self.S), np.nan)
        V[:, self.order] = v; Q[:, self.order] = qs
        self.last_k = k                                                   # (n_slots, cells) mean density
        return V, Q

    def cls_wsh(self, n_slots, ent):
        """per-slot class share of the entry's flow. Constant beta, or (cls_betaf set) the ramp-to-ramp share of
        the entry's FLUCTUATION: W = beta0 x ebar + betaf x (e - ebar), ebar = the day's 1 h centred mean of the
        entry (truth D12_I5_N 236->240: large surges and dips of entry 236 pass 240 at 0.1-0.3, note 169)"""
        K = len(self.cls_ent)
        out = np.tile(self.cls_beta.astype(np.float64), (n_slots, 1))
        bf = getattr(self, "cls_betaf", None)
        if K == 0 or bf is None:
            return out
        from scipy.ndimage import uniform_filter1d
        for i in range(K):
            e = np.asarray(ent[:n_slots, self.cls_ent[i]], float)
            eb = uniform_filter1d(e, 13, mode="nearest")
            W = np.clip(self.cls_beta[i] * eb + bf[i] * (e - eb), 0, e)
            with np.errstate(all="ignore"):
                out[:, i] = np.where(e > 1, W / np.maximum(e, 1), self.cls_beta[i])
        return out

    def set_od_pairs(self, z, pairs):
        """pairs: [(entry sensored index j (gap j-1 -> j), exit sensored index h (gap h -> h+1), beta)]: a share beta
        of entry j's vehicles leaves at the exit after h and at no exit in between (ramp-to-ramp paths, note 169).
        Background exit shares per slot keep the weekday-mean exit volumes: at h (F q - W) / (q - W), at the exits
        in between F q / (q - W), W = beta x the entry's weekday profile; q = weekday-mean sensored flow."""
        ent_at = {int(b): i for i, b in enumerate(self.ent)}
        o = self.order
        qm = np.nanmean(np.where(np.isfinite(z["true_flow"]), z["true_flow"], np.nan)[:, :, o], 0)   # (T, n)
        T = qm.shape[0]
        fb = np.tile(self.cell_f.astype(np.float64), (T, 1))
        Pm = np.mean(np.stack([self.P[w] for w in range(7)]), 0)                                  # (T, n_ent)
        ce, cx, cb = [], [], []
        for j, h, beta in pairs:
            i_e = ent_at[int(j)]
            W = beta * Pm[:, i_e]
            ce.append(i_e); cx.append(int(self.sens_cell[h])); cb.append(beta)
            for m in range(j, h + 1):
                F = self.f_exit[m]
                if F <= 0:
                    continue
                c = self.sens_cell[m]; q = np.maximum(qm[:, m], 1.0)
                with np.errstate(all="ignore"):
                    if m == h:
                        fb[:, c] = np.clip((F * q - W) / np.maximum(q - W, 1.0), 0, 0.95)
                    else:
                        fb[:, c] = np.clip(F * q / np.maximum(q - W, 1.0), 0, 0.95)
        self.cls_ent = np.array(ce, np.int64); self.cls_exit = np.array(cx, np.int64); self.cls_beta = np.array(cb, float)
        self.f_bg = np.nan_to_num(fb)

    def state_from_obs(self, v_last, q_last):
        """cell densities from the last observed slot (z order arrays; k = q / v exact on the released FD),
        interpolated along the corridor over missing sensors and the unsensored cells"""
        k_s = np.where(np.isfinite(v_last) & np.isfinite(q_last), q_last / np.maximum(v_last, 1.0), np.nan)[self.order]
        pos_s = self.sens_cell.astype(float)
        ok = np.isfinite(k_s)
        if not ok.any():
            return np.zeros(len(self.cell_L))
        k = np.interp(np.arange(len(self.cell_L)), pos_s[ok], k_s[ok])
        return np.clip(k, 0, self.cell_kj * 0.98)
