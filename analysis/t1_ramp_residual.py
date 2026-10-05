"""Do the released ramp flows say anything about the Task 1 residual?

Feeding the ramp channels to the network (M3_V2=1, no dN loss, width 64, 10 panels) made S_LWR *worse* by 1.7
sd. That test is the least favourable one possible, though: capacity is the binding constraint (note 115), so
seven extra channels can dilute even if they carry something. This measures the question directly, with no model
and no training in the way.

`r_on` and `r_off` appear in the very equation S_LWR scores,

    N(t+dt) - N(t) = dt * (q_in + r_on - q_out - r_off)

so if the network is failing to use them, its density residual at a target cell should line up with them. Two
levels are tested:

    e        the relative density residual at the cell, against the ramp flows and their time differences
    de       the residual's own time difference, which is what the conservation equation constrains

Controls: the same regressors with the days shuffled, which must give zero.

    .venv/Scripts/python.exe analysis/t1_ramp_residual.py [tag]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tfb.data import PANELS, load  # noqa: E402
from tfb.physics import load_ramps  # noqa: E402

PRED = ROOT / "work" / "preds"
TAG = sys.argv[1] if len(sys.argv) > 1 else "t1_m3all"
NAMES = ("r_on", "r_off", "r_net", "d r_on", "d r_off", "d r_net")


def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 1000:
        return np.nan
    x, y = a[m] - a[m].mean(), b[m] - b[m].mean()
    s = np.sqrt((x @ x) * (y @ y))
    return float(x @ y / s) if s > 0 else np.nan


def r2_of(y, X):
    m = np.isfinite(y) & np.isfinite(X).all(1)
    y, X = y[m], X[m]
    if len(y) < 1000:
        return np.nan
    X = np.column_stack([X - X.mean(0), np.ones(len(y))])
    y = y - y.mean()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(1 - ((y - X @ beta) ** 2).sum() / (y @ y))


def main() -> None:
    acc, acc_s, r2s, r2d, n_tot = [], [], [], [], 0
    for panel in PANELS:
        f = PRED / f"{TAG}_{panel}_test.npz"
        if not f.exists():
            continue
        p = np.load(f)
        z = load(panel, "train")
        days = p["days"]
        r = load_ramps(panel, "train", z)
        lanes = np.broadcast_to(z["lanes"], p["pv"].shape).astype(np.float64)
        on = np.where(r["on_v"][days], r["on_q"][days], np.nan).astype(np.float64) / lanes
        off = np.where(r["off_v"][days], r["off_q"][days], np.nan).astype(np.float64) / lanes
        att = r["on_att"] | r["off_att"]
        ts, tq = z["true_speed"][days].astype(np.float64), z["true_flow"][days].astype(np.float64)
        tgt = z["target"][days] & np.isfinite(ts) & np.isfinite(tq) & att[None, None, :]
        k_t = tq / np.maximum(ts, 1.0)
        k_p = p["pq"].astype(np.float64) / np.maximum(p["pv"].astype(np.float64), 1.0)
        e = np.where(tgt, (k_p - k_t) / np.maximum(k_t, 1e-9), np.nan)

        def dt_(a):
            d = np.full_like(a, np.nan)
            d[:, 1:] = a[:, 1:] - a[:, :-1]
            return d

        net = on - off
        reg = [on, off, net, dt_(on), dt_(off), dt_(net)]
        de = dt_(e)                                   # non-nan only where both slots are targets
        ef, def_ = e.ravel(), de.ravel()
        acc.append([corr(ef, x.ravel()) for x in reg])
        X = np.column_stack([x.ravel() for x in reg])
        r2s.append(r2_of(ef, X))
        r2d.append(r2_of(def_, X))
        sh = np.roll(e, 7, axis=0)                    # control: the residual from a week-shifted day
        acc_s.append([corr(sh.ravel(), x.ravel()) for x in reg])
        n_tot += int(np.isfinite(ef).sum())
        print(f"  {panel}: {int(tgt.sum()):,} target cells on ramp links", flush=True)
        del z

    A, S = np.array(acc, float), np.array(acc_s, float)
    print(f"\n{n_tot:,} target cells on ramp-attached links, {len(A)} panels\n")
    print(f"{'regressor':<10} {'corr with e':>13} {'day-shifted':>13}")
    for j, nm in enumerate(NAMES):
        print(f"{nm:<10} {np.nanmean(A[:, j]):>+13.4f} {np.nanmean(S[:, j]):>+13.4f}")
    print(f"\nlinear fit of the residual on all six, R^2:")
    print(f"   level     e   {np.nanmean(r2s):.5f}   (max over panels {np.nanmax(r2s):.5f})")
    print(f"   increment de  {np.nanmean(r2d):.5f}   (max over panels {np.nanmax(r2d):.5f})")


if __name__ == "__main__":
    main()
