"""Stop measuring the residual and look at it. Print the actual numbers around the cells we get most wrong.

Everything so far has been aggregate: correlations, variances, R^2. This dumps individual target cells with
their neighbourhood as a readable table - what the detector actually recorded on either side, what the truth was,
and what we said - so the cases can be read rather than summarised.

    .venv/Scripts/python.exe analysis/t1_look_at_cases.py [n_worst] [tag]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tfb.data import PANELS, load  # noqa: E402

PRED = ROOT / "work" / "preds"
N_WORST = int(sys.argv[1]) if len(sys.argv) > 1 else 6
TAG = sys.argv[2] if len(sys.argv) > 2 else "t1_m3all"
W = 5                                   # slots either side


def show(panel, z, days, pv, pq, d, t, s, note):
    lanes = float(z["lanes"][s])
    ts, tq = z["true_speed"][days[d]], z["true_flow"][days[d]]
    ov, oq = z["obs_speed"][days[d]], z["obs_flow"][days[d]]
    tgt = z["target"][days[d]]
    lo, hi = max(t - W, 0), min(t + W + 1, ts.shape[0])
    print(f"\n{panel}  link {z['link_ids'][s]}  {z['dates'][days[d]]} ({z['regime'][days[d]]})  "
          f"{lanes:.0f} lanes   {note}")
    print(f"  {'time':>6} {'state':>9} | {'true q':>8} {'pred q':>8} {'err':>7} | "
          f"{'true v':>7} {'pred v':>7} {'err':>6}")
    for u in range(lo, hi):
        hh, mm = divmod(u * 5, 60)
        vis = np.isfinite(ov[u, s]) and np.isfinite(oq[u, s])
        state = "target" if tgt[u, s] else ("visible" if vis else "missing")
        qt = tq[u, s] / lanes
        vt = ts[u, s]
        if tgt[u, s]:
            qp, vp = pq[u, s] / lanes, pv[u, s]
            mark = " <<<" if u == t else ""
            print(f"  {hh:02d}:{mm:02d}  {state:>9} | {qt / 12:8.1f} {qp / 12:8.1f} {(qp - qt) / 12:+7.1f} | "
                  f"{vt:7.1f} {vp:7.1f} {vp - vt:+6.1f}{mark}")
        else:
            print(f"  {hh:02d}:{mm:02d}  {state:>9} | {qt / 12:8.1f} {'-':>8} {'-':>7} | "
                  f"{vt:7.1f} {'-':>7} {'-':>6}")
    print("   (flow in vehicles per 5 min per lane, speed in km/h)")


def main() -> None:
    picks = []
    store = {}
    for panel in PANELS:
        f = PRED / f"{TAG}_{panel}_test.npz"
        if not f.exists():
            continue
        p = np.load(f)
        z = load(panel, "train")
        days = p["days"]
        ts, tq = z["true_speed"][days], z["true_flow"][days]
        lanes = np.broadcast_to(z["lanes"], ts.shape)
        m = z["target"][days] & np.isfinite(ts) & np.isfinite(tq)
        err = np.where(m, np.abs(p["pq"] - tq) / lanes / 12, 0.0)     # vehicles per 5 min per lane
        flat = err.ravel()
        idx = np.argpartition(-flat, N_WORST)[:N_WORST]
        for i in idx:
            picks.append((float(flat[i]), panel, *np.unravel_index(i, err.shape)))
        # one ordinary case for contrast: the cell closest to the median error
        med = np.median(err[m])
        j = np.argmin(np.where(m.ravel(), np.abs(flat - med), 1e9))
        picks.append((-1.0, panel, *np.unravel_index(j, err.shape)))
        store[panel] = (z, days, p)
    picks.sort(key=lambda r: -r[0])
    print(f"{N_WORST} worst cells per panel by flow error, plus one median cell per panel\n"
          f"{'=' * 96}")
    for e, panel, d, t, s in picks[:N_WORST + 2]:
        z, days, p = store[panel]
        note = f"error {e:+.1f} vehicles" if e > 0 else "a MEDIAN cell, for contrast"
        show(panel, z, days, p["pv"][d], p["pq"][d], d, t, s, note)
    for e, panel, d, t, s in [r for r in picks if r[0] < 0][:2]:
        z, days, p = store[panel]
        show(panel, z, days, p["pv"][d], p["pq"][d], d, t, s, "a MEDIAN cell, for contrast")


if __name__ == "__main__":
    main()
