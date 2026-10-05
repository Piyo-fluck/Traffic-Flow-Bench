"""Is the Task 1 residual correlated ACROSS corridors at the same moment?

Every structure test so far has been inside one panel: mask geometry and time neighbours (note 96), the lattice,
the CTM and nearest-neighbour retrieval (note 103), and the upstream advection scan over +-160 links (note 107).
The ten corridors are different roads but the SAME simulation run, so if the generator has a shared stochastic
driver - a demand multiplier per five minutes, say - and our model cannot see it, the residual will correlate
across panels at the same (day, timestamp). That would be information available at test time, because the other
corridors' visible cells are released.

For each panel and each (day, slot) of the local test block:

    res      the mean signed residual over that moment's target cells, as a relative density error
    truth    the mean flow per lane over that moment's eligible cells, as a level check

Both are de-meaned BY SLOT within the panel, so a shared time-of-day shape cannot manufacture a correlation;
what is left is the day-specific anomaly. The control pairs panel A's day d with panel B's day d+1, which must
come out at zero.

    .venv/Scripts/python.exe analysis/t1_cross_panel.py [tag]
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tfb.data import PANELS, load  # noqa: E402

PRED = ROOT / "work" / "preds"
TAG = sys.argv[1] if len(sys.argv) > 1 else "t1_m3all"


def deslot(a):
    """Remove each slot's across-day mean, leaving the day-specific anomaly."""
    m = np.nanmean(a, axis=0, keepdims=True)
    return a - m


def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 500:
        return np.nan
    x, y = a[m] - a[m].mean(), b[m] - b[m].mean()
    return float(x @ y / np.sqrt((x @ x) * (y @ y)))


def main() -> None:
    res, lvl, days_ref = {}, {}, None
    for panel in PANELS:
        f = PRED / f"{TAG}_{panel}_test.npz"
        if not f.exists():
            continue
        p = np.load(f)
        z = load(panel, "train")
        days = p["days"]
        days_ref = days if days_ref is None else days_ref
        assert (days == days_ref).all(), "panels were evaluated on different days"
        ts, tq = z["true_speed"][days].astype(np.float64), z["true_flow"][days].astype(np.float64)
        lanes = np.broadcast_to(z["lanes"], ts.shape).astype(np.float64)
        tgt = z["target"][days] & np.isfinite(ts) & np.isfinite(tq)
        elig = z["elig"][days] & np.isfinite(ts) & np.isfinite(tq)
        k_t = tq / np.maximum(ts, 1.0)
        k_p = p["pq"].astype(np.float64) / np.maximum(p["pv"].astype(np.float64), 1.0)
        e = np.where(tgt, (k_p - k_t) / np.maximum(k_t, 1e-9), np.nan)
        with np.errstate(invalid="ignore"):
            res[panel] = deslot(np.nanmean(e, axis=2))                       # (days, slots)
            lvl[panel] = deslot(np.nanmean(np.where(elig, tq / lanes, np.nan), axis=2))
        print(f"  {panel}: {int(tgt.sum()):,} targets, "
              f"{np.isfinite(res[panel]).sum():,} (day, slot) cells", flush=True)
        del z

    names = list(res)
    print(f"\n{len(names)} panels, {len(days_ref)} days x 288 slots, de-meaned by slot\n")
    for label, tab in (("residual", res), ("true flow level", lvl)):
        same = [corr(tab[a].ravel(), tab[b].ravel()) for a, b in combinations(names, 2)]
        shift = [corr(tab[a][:-1].ravel(), tab[b][1:].ravel()) for a, b in combinations(names, 2)]
        same, shift = np.array(same, float), np.array(shift, float)
        print(f"{label:<16} same moment: mean {np.nanmean(same):+.4f}  median {np.nanmedian(same):+.4f}  "
              f"min {np.nanmin(same):+.4f}  max {np.nanmax(same):+.4f}")
        print(f"{'':<16} day-shifted : mean {np.nanmean(shift):+.4f}  "
              f"max |.| {np.nanmax(np.abs(shift)):.4f}   (control, expect 0)")

    # how much of one panel's residual anomaly the other nine explain together
    A = np.stack([res[n].ravel() for n in names])
    ok = np.isfinite(A).all(0)
    A = A[:, ok]
    A = A - A.mean(1, keepdims=True)
    r2 = []
    for i in range(len(names)):
        y = A[i]
        X = np.delete(A, i, 0).T
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        r2.append(1 - ((y - X @ beta) ** 2).sum() / (y ** 2).sum())
    print(f"\nleave-one-out linear fit of a panel's residual anomaly on the other nine:")
    print(f"  R^2 mean {np.mean(r2):.4f}  min {np.min(r2):.4f}  max {np.max(r2):.4f}  "
          f"({A.shape[1]:,} moments)")


if __name__ == "__main__":
    main()
