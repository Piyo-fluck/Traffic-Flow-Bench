"""Empirical-Bayes target-run offsets: one trust level alpha per (regime, run length, boundary count).

`t3_run_offset_feasible.py` used a single global alpha on the observable boundary proxy and held out +0.00060 on
both months, both picking alpha = 0.20, against a +0.01678 oracle. A single scalar is the crudest possible
posterior: how much the proxy should be believed plainly depends on how long the run is and on whether one or
both of its boundaries are scored.

Calibration may use the truth, so on one month the realised cost of each candidate alpha is accumulated per
group, the best alpha per group is read off, and the other month is decoded with it and scored end to end. The
groups are coarse on purpose (note 103: nothing finer has ever carried signal):

    regime (R1/R2/R3) x run length (1, 2-3, 4+) x whether both boundaries are scored

    .venv/Scripts/python.exe analysis/t3_run_offset_eb.py
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
TAG = "t1_m3all"
CGRID = np.round(np.arange(-3.0, 3.0001, 0.1), 4)
ALPHAS = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.55, 0.70])
HALVES = ("2031-01", "2031-02")
NG = 3 * 3 * 2


def prep(panel):
    """Everything the two passes need for one panel."""
    f = PRED / f"{TAG}_{panel}_test.npz"
    if not f.exists():
        return None
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
    ridx = {r: i for i, r in enumerate(REGIMES)}

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
    starts, ends = np.concatenate([[0], bnd]), np.concatenate([bnd, [len(inv_s)]]) - 1
    da, sa, ta, tb = di_s[starts], si_s[starts], ti_s[starts], ti_s[ends]
    m_len = (ends - starts + 1)

    okL = (ta > 0) & trans[da, np.maximum(ta - 1, 0), sa]
    okR = (tb < T - 1) & trans[da, np.minimum(tb, T - 2), sa]
    hatL = np.where(okL, k_A[da, ta, sa] - k_true[da, np.maximum(ta - 1, 0), sa], 0.0)
    hatR = np.where(okR, k_A[da, tb, sa] - k_true[da, np.minimum(tb + 1, T - 1), sa], 0.0)
    trueL = np.where(okL, k_A[da, ta, sa] - k_true[da, ta, sa], 0.0)
    trueR = np.where(okR, k_A[da, tb, sa] - k_true[da, tb, sa], 0.0)
    Lr = L[sa]
    wl = np.array([0.10 / 3 / rhs_r[REGIMES[ridx[x]]] for x in reg[da]])
    lb = np.where(m_len == 1, 0, np.where(m_len <= 3, 1, 2))
    grp = (np.array([ridx[x] for x in reg[da]]) * 6 + lb * 2 + (okL & okR).astype(int))

    kc, qc, vc = k_A[di_s, ti_s, si_s], qA[di_s, ti_s, si_s], vA[di_s, ti_s, si_s]
    tvc, tqc, lc = ts[di_s, ti_s, si_s], tq[di_s, ti_s, si_s], lanes[di_s, ti_s, si_s]
    rcell = np.array([ridx[x] for x in reg[di_s]])
    wvc = np.array([wv_r[REGIMES[i]] for i in rcell])
    wqc = np.array([wq_r[REGIMES[i]] for i in rcell])
    # two state costs. SCF is what the DECODER may use: the displacement from the CNN, which for an unbiased
    # estimator is the expected increase in squared error. SCT is the realised change against the truth, which
    # only calibration may look at.
    SCF = np.empty((R, len(CGRID)), np.float32)
    SCT = np.empty((R, len(CGRID)), np.float32)
    bv, bq = (vc - tvc) ** 2, ((qc - tqc) / lc) ** 2
    for j, c in enumerate(CGRID):
        q2, v2 = project(kc + c, qc, vc, wqc, wvc)
        SCF[:, j] = np.bincount(inv_s, weights=wvc * (v2 - vc) ** 2 + wqc * ((q2 - qc) / lc) ** 2,
                                minlength=R)
        SCT[:, j] = np.bincount(inv_s, weights=wvc * ((v2 - tvc) ** 2 - bv)
                                + wqc * (((q2 - tqc) / lc) ** 2 - bq), minlength=R)
    aL, aR = (okL * Lr * wl), (okR * Lr * wl)
    month = np.array([d[:7] for d in dates[da]])
    return dict(R=R, SCF=SCF, SCT=SCT, aL=aL, aR=aR, hatL=hatL, hatR=hatR, trueL=trueL, trueR=trueR, grp=grp,
                month=month, inv_s=inv_s, di_s=di_s, ti_s=ti_s, si_s=si_s, kc=kc, qc=qc, vc=vc,
                wvc=wvc, wqc=wqc, vA=vA, qA=qA, ts=ts, tq=tq, prm=prm, ok=ok, dates=dates,
                zd=dict(target=z["target"][days], lanes=z["lanes"], true_speed=ts, true_flow=tq,
                        regime=reg, elig=z["elig"][days], dates=dates))


def choose_c(d, alpha_of_run):
    """The offset each run takes at its own alpha."""
    Cg = CGRID[None, :]
    hL = (alpha_of_run * d["hatL"])[:, None]
    hR = (alpha_of_run * d["hatR"])[:, None]
    cost = d["SCF"] + d["aL"][:, None] * (np.abs(hL + Cg) - np.abs(hL)) \
        + d["aR"][:, None] * (np.abs(hR + Cg) - np.abs(hR))
    return CGRID[np.argmin(cost, 1)]


def realised(d, c):
    """The true cost of taking offset c, per run - the calibration objective."""
    j = np.searchsorted(CGRID, np.round(c, 4))
    st = d["SCT"][np.arange(len(c)), np.clip(j, 0, len(CGRID) - 1)]
    tl, tr = d["trueL"], d["trueR"]
    return st + d["aL"] * (np.abs(tl + c) - np.abs(tl)) + d["aR"] * (np.abs(tr + c) - np.abs(tr))


def score(d, c):
    q2, v2 = project(d["kc"] + c[d["inv_s"]], d["qc"], d["vc"], d["wqc"], d["wvc"])
    sv, sq = d["vA"].copy(), d["qA"].copy()
    sv[d["di_s"], d["ti_s"], d["si_s"]] = v2
    sq[d["di_s"], d["ti_s"], d["si_s"]] = q2
    return sv, sq


def main() -> None:
    t0 = time.time()
    cost = {h: np.zeros((NG, len(ALPHAS))) for h in HALVES}
    cache = {}
    for panel in PANELS:
        d = prep(panel)
        if d is None:
            continue
        for ai, a in enumerate(ALPHAS):
            c = choose_c(d, np.full(d["R"], a))
            rc = realised(d, c)
            for h in HALVES:
                m = d["month"] == h
                np.add.at(cost[h][:, ai], d["grp"][m], rc[m])
        cache[panel] = d
        print(f"  {panel} ({time.time() - t0:.0f}s)", flush=True)

    acc = {}
    for cal, ev in ((HALVES[0], HALVES[1]), (HALVES[1], HALVES[0])):
        pick = ALPHAS[np.argmin(cost[cal], 1)]
        print(f"\ncalibrated on {cal}: alpha by (regime, run length, both boundaries)")
        for gi in range(NG):
            r, lb, bb = gi // 6, (gi % 6) // 2, gi % 2
            lbl = ("1", "2-3", "4+")[lb]
            print(f"   {REGIMES[r]} len {lbl:<3} both={bb}  alpha {pick[gi]:.2f}")
        for panel, d in cache.items():
            for nm, c in (("base", np.zeros(d["R"])), ("global 0.20", choose_c(d, np.full(d["R"], 0.20))),
                          ("empirical Bayes", choose_c(d, pick[d["grp"]]))):
                sv, sq = score(d, c)
                m = d["dates"] != ""
                m = np.char.startswith(d["dates"], ev)
                zd = {k: (v[m] if isinstance(v, np.ndarray) and v.ndim and len(v) == len(m) else v)
                      for k, v in d["zd"].items()}
                st = state_score(zd, sv[m], sq[m])["mean"]["S"]
                ph = physics_scores(zd, sv[m], sq[m], d["prm"], d["ok"][m])["mean"]
                acc.setdefault((ev, nm), []).append((st, ph["S_FD"], ph["S_LWR"]))

    print(f"\n{'evaluated on':<14} {'decoder':<18} {'S_state':>9} {'S_LWR':>8} {'total part':>12} {'delta':>9}")
    for h in HALVES:
        b = None
        for nm in ("base", "global 0.20", "empirical Bayes"):
            v = acc[(h, nm)]
            st, fd, lw = (float(np.mean([x[i] for x in v])) for i in range(3))
            tt = 0.35 * st + 0.05 * fd + 0.10 * lw
            if b is None:
                b = tt
            print(f"{h:<14} {nm:<18} {st:>9.4f} {lw:>8.4f} {tt:>12.5f} {tt - b:>+9.5f}")


if __name__ == "__main__":
    main()
