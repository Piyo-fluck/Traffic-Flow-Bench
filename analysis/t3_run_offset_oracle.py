"""The ceiling of target-run constant-offset decoding: give every run its ORACLE offset.

The decision unit for S_LWR is not the cell but the run of consecutive target slots. Adding one constant c to the
density over a run leaves every internal target-target difference untouched - so the 846,229 of residual that
note 105 attributes to both-target transitions is frozen - and moves only the run's two visible<->target
boundaries, which carry the other 1,979,135 (70%).

At a boundary the neighbour is the truth, so with e = k_pred - k_true the boundary residual is |e + c| L, and the
run's whole contribution to the score is

    0.10/(3 Rhs_r) * [ Lleft |e_a + c| + Lright |e_b + c| ]  +  the state cost of moving m cells by c

Density goes back onto (q, v) by the minimum-move projection under the metric's own curvature:

    min w_q (q' - q_A)^2 + w_v (v' - v_A)^2  s.t.  q'/v' = k'  ->  v' = (w_q k' q_A + w_v v_A)/(w_q k'^2 + w_v)

This gives each run the c that an oracle knowing e_a and e_b would choose, then scores the result end to end with
`tfb.metrics.state_score` and `tfb.physics.physics_scores`. It is the ceiling of the whole branch: c-only, (c, d)
and any empirical-Bayes approximation of them all sit below it.

    .venv/Scripts/python.exe analysis/t3_run_offset_oracle.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tfb.data import PANELS, load  # noqa: E402
from tfb.metrics import REGIMES, state_score  # noqa: E402
from tfb.physics import fd_params, physics_scores, ramp_ok  # noqa: E402

PRED = ROOT / "work" / "preds"
TAG = sys.argv[1] if len(sys.argv) > 1 else "t1_m3all"
GRID = np.round(np.arange(-3.0, 3.0001, 0.05), 4)


def project(k, qA, vA, wq, wv):
    """Minimum-move (q', v') with q'/v' = k, under the metric's curvature."""
    v = (wq * k * qA + wv * vA) / np.maximum(wq * k * k + wv, 1e-12)
    return k * v, v


def main() -> None:
    t0 = time.time()
    acc = {}
    for panel in PANELS:
        f = PRED / f"{TAG}_{panel}_test.npz"
        if not f.exists():
            continue
        p = np.load(f)
        z = load(panel, "train")
        days = p["days"]
        prm = fd_params(panel, z["link_ids"])
        L = prm["length"].astype(np.float64)
        lanes = np.broadcast_to(z["lanes"], p["pv"].shape).astype(np.float64)
        ok = ramp_ok(panel, "train", z)[days]
        ts, tq = z["true_speed"][days].astype(np.float64), z["true_flow"][days].astype(np.float64)
        tgt = z["target"][days] & np.isfinite(ts) & np.isfinite(tq)
        use = z["elig"][days] & np.isfinite(ts) & np.isfinite(tq)
        reg = np.asarray(z["regime"], str)[days]
        vA, qA = p["pv"].astype(np.float64), p["pq"].astype(np.float64)
        k_true = tq / np.maximum(ts, 1.0)
        k_A = np.where(tgt, qA / np.maximum(vA, 1.0), k_true)
        e = np.where(tgt, k_A - k_true, 0.0)
        trans = use[:, :-1] & use[:, 1:] & ok[:, :-1]

        # ---- the metric's local curvature, per regime, from the current solution
        wv_r, wq_r, rhs_r = {}, {}, {}
        for r in REGIMES:
            m = tgt & (reg == r)[:, None, None]
            n = max(int(m.sum()), 1)
            rv = float(np.sqrt(np.mean((vA[m] - ts[m]) ** 2)))
            rq = float(np.sqrt(np.mean(((qA[m] - tq[m]) / lanes[m]) ** 2)))
            wv_r[r] = 0.35 / 3 * 0.54 / 25 / (2 * n * max(rv, 1e-9))
            wq_r[r] = 0.35 / 3 * 0.46 / 600 / (2 * n * max(rq, 1e-9))
            dm = (reg == r)[:, None, None] & trans
            rhs_r[r] = max(float((np.abs(k_true[:, 1:] - k_true[:, :-1]) * L)[dm].sum()), 1e-9)

        # ---- runs of consecutive target slots, per (day, link)
        D, T, S = tgt.shape
        run_id = np.cumsum(~tgt, axis=1)
        lab = np.where(tgt, run_id + (np.arange(D)[:, None, None] * 10_000
                                      + np.arange(S)[None, None, :] * 10_000_000), -1)
        flat = lab.transpose(0, 2, 1).ravel()
        sel = flat >= 0
        _, inv = np.unique(flat[sel], return_inverse=True)
        R = int(inv.max()) + 1
        di = np.broadcast_to(np.arange(D)[:, None, None], tgt.shape).transpose(0, 2, 1).ravel()[sel]
        si = np.broadcast_to(np.arange(S)[None, None, :], tgt.shape).transpose(0, 2, 1).ravel()[sel]
        ti = np.broadcast_to(np.arange(T)[None, :, None], tgt.shape).transpose(0, 2, 1).ravel()[sel]
        order = np.lexsort((ti, inv))
        inv_s, di_s, si_s, ti_s = inv[order], di[order], si[order], ti[order]
        bnd = np.flatnonzero(np.diff(inv_s)) + 1
        starts = np.concatenate([[0], bnd])
        ends = np.concatenate([bnd, [len(inv_s)]]) - 1
        da, sa, ta = di_s[starts], si_s[starts], ti_s[starts]
        tb = ti_s[ends]

        okL = (ta > 0) & trans[da, np.maximum(ta - 1, 0), sa]
        okR = (tb < T - 1) & trans[da, np.minimum(tb, T - 2), sa]
        eL, eR = e[da, ta, sa], e[da, tb, sa]
        Lr = L[sa]
        ridx = {r: i for i, r in enumerate(REGIMES)}
        rrun = np.array([ridx[x] for x in reg[da]])
        wl = np.array([0.10 / 3 / rhs_r[REGIMES[i]] for i in rrun])

        kc, qc, vc = k_A[di_s, ti_s, si_s], qA[di_s, ti_s, si_s], vA[di_s, ti_s, si_s]
        tvc, tqc, lc = ts[di_s, ti_s, si_s], tq[di_s, ti_s, si_s], lanes[di_s, ti_s, si_s]
        rcell = np.array([ridx[x] for x in reg[di_s]])
        wvc = np.array([wv_r[REGIMES[i]] for i in rcell])
        wqc = np.array([wq_r[REGIMES[i]] for i in rcell])
        base_v = (vc - tvc) ** 2
        base_q = ((qc - tqc) / lc) ** 2
        best = np.zeros(R)
        best_cost = np.full(R, np.inf)
        for c in GRID:
            q2, v2 = project(kc + c, qc, vc, wqc, wvc)
            dcell = wvc * ((v2 - tvc) ** 2 - base_v) + wqc * (((q2 - tqc) / lc) ** 2 - base_q)
            state = np.bincount(inv_s, weights=dcell, minlength=R)
            lwr = wl * (okL * Lr * (np.abs(eL + c) - np.abs(eL))
                        + okR * Lr * (np.abs(eR + c) - np.abs(eR)))
            tc = state + lwr
            better = tc < best_cost
            best_cost[better] = tc[better]
            best[better] = c

        cfull = best[inv_s]
        q2, v2 = project(kc + cfull, qc, vc, wqc, wvc)
        sv, sq = vA.copy(), qA.copy()
        sv[di_s, ti_s, si_s] = v2
        sq[di_s, ti_s, si_s] = q2
        zd = dict(target=z["target"][days], lanes=z["lanes"], true_speed=ts, true_flow=tq,
                  regime=reg, elig=z["elig"][days], dates=z["dates"][days])
        for nm, pv_, pq_ in (("base", vA, qA), ("oracle", sv, sq)):
            st = state_score(zd, pv_, pq_)["mean"]["S"]
            ph = physics_scores(zd, pv_, pq_, prm, ok)["mean"]
            acc.setdefault(nm, []).append((st, ph["S_FD"], ph["S_LWR"]))
        print(f"  {panel}: {R:,} runs, mean |c| {float(np.abs(best).mean()):.3f} veh/km, "
              f"moved {float((best != 0).mean()) * 100:.1f}% ({time.time() - t0:.0f}s)", flush=True)
        del z

    print(f"\n{'':<10} {'S_state':>9} {'S_FD':>8} {'S_LWR':>8} {'0.35s+0.05f+0.10l':>18} {'delta':>9}")
    b = None
    for nm in ("base", "oracle"):
        st, fd, lw = (float(np.mean([x[i] for x in acc[nm]])) for i in range(3))
        tt = 0.35 * st + 0.05 * fd + 0.10 * lw
        if b is None:
            b = tt
        print(f"{nm:<10} {st:>9.4f} {fd:>8.4f} {lw:>8.4f} {tt:>18.5f} {tt - b:>+9.5f}")


if __name__ == "__main__":
    main()
