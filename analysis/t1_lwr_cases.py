"""Read the S_LWR residual one transition at a time.

Note 117 found that 82.6% of the numerator sits away from congestion transitions, in ordinary cells. That is a
statement about where the mass is, not about what those cells look like. This ranks the individual transitions
by their own contribution to the numerator, |e[t+1] - e[t]| * L, and prints the neighbourhood: the truth's
density, ours, the error at each end, and what the conservation term actually asks for.

    .venv/Scripts/python.exe analysis/t1_lwr_cases.py [n_per_panel] [tag]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tfb.data import PANELS, load  # noqa: E402
from tfb.physics import fd_params, ramp_ok  # noqa: E402

PRED = ROOT / "work" / "preds"
N_PER = int(sys.argv[1]) if len(sys.argv) > 1 else 2
TAG = sys.argv[2] if len(sys.argv) > 2 else "t1_m3all"
W = 4


def show(panel, z, days, p, L, d, t, s, note):
    ts, tq = z["true_speed"][days[d]], z["true_flow"][days[d]]
    tgt = z["target"][days[d]]
    k_t = tq / np.maximum(ts, 1.0)
    k_p = np.where(tgt, p["pq"][d] / np.maximum(p["pv"][d], 1.0), k_t)
    lo, hi = max(t - W, 0), min(t + W + 2, ts.shape[0])
    print(f"\n{panel}  link {z['link_ids'][s]}  {z['dates'][days[d]]} ({z['regime'][days[d]]})  "
          f"L = {L[s]:.3f} km   {note}")
    print(f"  {'time':>6} {'state':>8} | {'true k':>8} {'pred k':>8} {'e':>7} || "
          f"{'dN true':>8} {'dN pred':>8} {'|resid|':>8}")
    for u in range(lo, hi):
        hh, mm = divmod(u * 5, 60)
        st = "target" if tgt[u, s] else ("vis" if np.isfinite(ts[u, s]) else "missing")
        e = k_p[u, s] - k_t[u, s] if tgt[u, s] else 0.0
        kp = f"{k_p[u, s]:8.2f}" if tgt[u, s] else f"{'-':>8}"
        row = f"  {hh:02d}:{mm:02d} {st:>8} | {k_t[u, s]:8.2f} {kp} {e:+7.2f} ||"
        if u + 1 < ts.shape[0] and np.isfinite(ts[u + 1, s]) and np.isfinite(ts[u, s]):
            dn_t = (k_t[u + 1, s] - k_t[u, s]) * L[s]
            dn_p = (k_p[u + 1, s] - k_p[u, s]) * L[s]
            mark = " <<<" if u == t else ""
            row += f" {dn_t:8.3f} {dn_p:8.3f} {abs(dn_p - dn_t):8.3f}{mark}"
        print(row)
    print("   (k in vehicles per km, dN in vehicles; the residual is |dN_pred - dN_true|)")


def main() -> None:
    picks, store = [], {}
    for panel in PANELS:
        f = PRED / f"{TAG}_{panel}_test.npz"
        if not f.exists():
            continue
        p = np.load(f)
        z = load(panel, "train")
        days = p["days"]
        L = fd_params(panel, z["link_ids"])["length"].astype(np.float64)
        ok = ramp_ok(panel, "train", z)[days]
        ts, tq = z["true_speed"][days].astype(np.float64), z["true_flow"][days].astype(np.float64)
        tgt = z["target"][days] & np.isfinite(ts) & np.isfinite(tq)
        use = z["elig"][days] & np.isfinite(ts) & np.isfinite(tq)
        k_t = tq / np.maximum(ts, 1.0)
        k_p = np.where(tgt, p["pq"].astype(np.float64) / np.maximum(p["pv"].astype(np.float64), 1.0), k_t)
        e = np.where(tgt, k_p - k_t, 0.0)
        trans = use[:, :-1] & use[:, 1:] & ok[:, :-1]
        res = np.where(trans, np.abs(e[:, 1:] - e[:, :-1]) * L, 0.0)
        flat = res.ravel()
        idx = np.argpartition(-flat, N_PER)[:N_PER]
        for i in idx:
            picks.append((float(flat[i]), panel, *np.unravel_index(i, res.shape)))
        med = np.median(res[trans])
        j = int(np.argmin(np.where(trans.ravel(), np.abs(flat - med), 1e18)))
        picks.append((-float(flat[j]), panel, *np.unravel_index(j, res.shape)))
        store[panel] = (z, days, p, L)
        print(f"  {panel}: numerator {res[trans].sum():,.0f}, median transition {med:.4f}", flush=True)
        del z
    picks.sort(key=lambda r: -r[0])
    print(f"\n{'=' * 96}\nWORST individual transitions")
    for c, panel, d, t, s in picks[:N_PER + 3]:
        z, days, p, L = store[panel]
        show(panel, z, days, p, L, d, t, s, f"contributes {c:.3f} vehicles to the numerator")
    print(f"\n{'=' * 96}\nTYPICAL transitions (the median contributor of each panel)")
    for c, panel, d, t, s in [r for r in picks if r[0] < 0][:3]:
        z, days, p, L = store[panel]
        show(panel, z, days, p, L, d, t, s, f"contributes {-c:.4f} vehicles - a MEDIAN transition")


if __name__ == "__main__":
    main()
