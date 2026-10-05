"""Target-run constant-offset decoding with an OBSERVABLE proxy for the boundary error.

`t3_run_offset_oracle.py` put the ceiling of this branch at +0.01678 (S_LWR 0.6486 -> 0.8180, S_state -0.0003)
by giving every run the offset an oracle knowing e_a and e_b would choose. That oracle is over a quantity we have
proven unpredictable (note 103: kNN residual correlation +0.010), so conditioning on coarse buckets would put
c* at 0 and pay nothing.

But the run boundaries have an observable proxy. The cell just outside a run is VISIBLE, hence the truth, so

    ehat_L = k_A[a] - k_true[a-1]        ehat_R = k_A[b] - k_true[b+1]

are computable, and they equal the true boundary error plus one slot of the truth's own change. The offset then
solves the same per-run problem as the oracle with alpha * ehat in place of e:

    c* = argmin_c  0.10/(3 Rhs_r) [ Lleft |alpha ehat_L + c| + Lright |alpha ehat_R + c| ] + state_cost(c)

alpha is the single free parameter - how much of the proxy to believe - and it is calibrated on one month and
read off on the other. Unlike the uniform temporal smoothing of note 105 (+0.00004), a run offset leaves every
internal target-target difference untouched, so the 30% of the residual that lives there cannot be damaged.

    .venv/Scripts/python.exe analysis/t3_run_offset_feasible.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from t3_run_offset_oracle import project  # noqa: E402
from tfb.data import PANELS, load  # noqa: E402
from tfb.metrics import REGIMES, state_score  # noqa: E402
from tfb.physics import fd_params, physics_scores, ramp_ok  # noqa: E402

PRED = ROOT / "work" / "preds"
TAG = sys.argv[1] if len(sys.argv) > 1 else "t1_m3all"
CGRID = np.round(np.arange(-3.0, 3.0001, 0.1), 4)
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
HALVES = ("2031-01", "2031-02")


def main() -> None:
    t0 = time.time()
    acc = {}                                    # (alpha, half) -> list of (S_state, S_FD, S_LWR)
    for panel in PANELS:
        f = PRED / f"{TAG}_{panel}_test.npz"
        if not f.exists():
            continue
        p = np.load(f)
        z = load(panel, "train")
        days = p["days"]
        dates = np.asarray(z["dates"], str)[days]
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
        trans = use[:, :-1] & use[:, 1:] & ok[:, :-1]

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
        eL = np.where(okL, k_A[da, ta, sa] - k_true[da, np.maximum(ta - 1, 0), sa], 0.0)   # the proxy
        eR = np.where(okR, k_A[da, tb, sa] - k_true[da, np.minimum(tb + 1, T - 1), sa], 0.0)
        Lr = L[sa]
        ridx = {r: i for i, r in enumerate(REGIMES)}
        wl = np.array([0.10 / 3 / rhs_r[REGIMES[ridx[x]]] for x in reg[da]])
        run_month = np.array([d[:7] for d in dates[da]])

        kc, qc, vc = k_A[di_s, ti_s, si_s], qA[di_s, ti_s, si_s], vA[di_s, ti_s, si_s]
        tvc, tqc, lc = ts[di_s, ti_s, si_s], tq[di_s, ti_s, si_s], lanes[di_s, ti_s, si_s]
        rcell = np.array([ridx[x] for x in reg[di_s]])
        wvc = np.array([wv_r[REGIMES[i]] for i in rcell])
        wqc = np.array([wq_r[REGIMES[i]] for i in rcell])
        base_v = (vc - tvc) ** 2
        base_q = ((qc - tqc) / lc) ** 2

        # the state cost does not depend on alpha, so pay for it once
        # The decoder may NOT see the truth, so the state cost is the displacement from the CNN, which for an
        # unbiased estimator is the expected increase in squared error. An earlier version of this script used
        # the realised change against the truth and so had an oracle inside its baseline (note 111).
        SC = np.empty((R, len(CGRID)), np.float32)
        for j, c in enumerate(CGRID):
            q2, v2 = project(kc + c, qc, vc, wqc, wvc)
            SC[:, j] = np.bincount(inv_s, weights=wvc * (v2 - vc) ** 2 + wqc * ((q2 - qc) / lc) ** 2,
                                   minlength=R)
        _ = (base_v, base_q)
        aL, aR = (okL * Lr * wl)[:, None], (okR * Lr * wl)[:, None]
        Cg = CGRID[None, :]

        for alpha in ALPHAS:
            hL, hR = (alpha * eL)[:, None], (alpha * eR)[:, None]
            cost = SC + aL * (np.abs(hL + Cg) - np.abs(hL)) + aR * (np.abs(hR + Cg) - np.abs(hR))
            best = CGRID[np.argmin(cost, 1)]
            cfull = best[inv_s]
            q2, v2 = project(kc + cfull, qc, vc, wqc, wvc)
            sv, sq = vA.copy(), qA.copy()
            sv[di_s, ti_s, si_s] = v2
            sq[di_s, ti_s, si_s] = q2
            for h in HALVES:
                dm = np.char.startswith(dates, h)
                if not dm.any():
                    continue
                zd = dict(target=z["target"][days][dm], lanes=z["lanes"], true_speed=ts[dm], true_flow=tq[dm],
                          regime=reg[dm], elig=z["elig"][days][dm], dates=dates[dm])
                st = state_score(zd, sv[dm], sq[dm])["mean"]["S"]
                ph = physics_scores(zd, sv[dm], sq[dm], prm, ok[dm])["mean"]
                acc.setdefault((alpha, h), []).append((st, ph["S_FD"], ph["S_LWR"]))
            _ = run_month
        print(f"  {panel}: {R:,} runs ({time.time() - t0:.0f}s)", flush=True)
        del z, SC

    def tot(a, h):
        v = acc[(a, h)]
        st, fd, lw = (float(np.mean([x[i] for x in v])) for i in range(3))
        return 0.35 * st + 0.05 * fd + 0.10 * lw, st, fd, lw

    print(f"\n{'alpha':>6} " + " ".join(f"{h + ' total':>15}" for h in HALVES)
          + f" {'S_LWR 01':>9} {'S_LWR 02':>9} {'S_state 01':>11}")
    for a in ALPHAS:
        t1, s1, _, l1 = tot(a, HALVES[0])
        t2, s2, _, l2 = tot(a, HALVES[1])
        print(f"{a:>6.2f} {t1:>15.5f} {t2:>15.5f} {l1:>9.4f} {l2:>9.4f} {s1:>11.4f}")
    print("\nheld-out: calibrate alpha on one month, read the other")
    for cal, ev in ((HALVES[0], HALVES[1]), (HALVES[1], HALVES[0])):
        a = max(ALPHAS, key=lambda x: tot(x, cal)[0])
        print(f"  calibrated on {cal} -> alpha {a:.2f}; on {ev}: "
              f"{tot(a, ev)[0]:.5f} against {tot(0.0, ev)[0]:.5f} ({tot(a, ev)[0] - tot(0.0, ev)[0]:+.5f})")


if __name__ == "__main__":
    main()
