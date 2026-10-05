"""Note 169: fit the generator's bottleneck capacities on train by matching queue-head occupancy.

Head occupancy of sensored link l (travel order) = share of (day, 06-20 h slot) with l queued (v <= 0.6 vf) and
the next sensored link free: the bottleneck discharging l. Each round simulates the fit days with their own flows
as demand (analysis/dlsim_gen2.py, day mode) and updates c_l <- c_l * exp(-eta * (H_true - H_sim) / s), clipped to
[lo, 1.05] x the released capacity. Evaluated on held-out train days (map corr, daily space-time IoU).

    .venv/Scripts/python.exe analysis/dlsim_fitcap.py D12_I5_S D7_I405_N
"""
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "analysis"))
import numpy as np  # noqa: E402

import os  # noqa: E402

from dlsim_gen2 import Gen  # noqa: E402
from dlsim_gen3 import Gen3  # noqa: E402

V4 = os.environ.get("DLSIM_V4", "") == "1"                     # full corridor (unsensored links included)
CAPS = "dlsim_caps3" if V4 else "dlsim_caps"
from tfb.data import load  # noqa: E402

A, B = 72, 240


def occupancy(V, vf, order):
    qd = (np.nan_to_num(V, nan=1e9) <= 0.6 * vf)[A:B][:, order]
    head = np.zeros(qd.shape[1]); head[:-1] = (qd[:, :-1] & ~qd[:, 1:]).mean(0)
    return head, qd.mean(0), qd


def run(g, z, days, dem="day"):
    Vs = []
    for d in days:
        up, ent = g.demand(z["true_flow"][d]) if dem == "day" else g.mixed_demand(z, d)
        V, _ = g.simulate(up, ent)
        Vs.append(V)
    return Vs


def score(g, z, days, vf):
    Vs = run(g, z, days)
    hs, qs, iou = [], [], []
    for V, d in zip(Vs, days):
        h, q, a = occupancy(V, vf, g.order)
        _, qt, b = occupancy(z["true_speed"][d], vf, g.order)
        iou.append((a & b).sum() / max((a | b).sum(), 1)); qs.append(q); hs.append(qt)
    Ms, Mt = np.mean(qs, 0), np.mean(hs, 0)
    return Ms.mean(), Mt.mean(), np.corrcoef(Ms, Mt)[0, 1], np.mean(iou)


def main():
    warnings.simplefilter("ignore")
    for panel in sys.argv[1:] or ["D12_I5_S"]:
        z = load(panel, "train")
        vf = np.asarray(z["free_speed"], float)
        g = Gen3(panel, z) if V4 else Gen(panel, z)
        g.fit_profiles(z)
        nd = len(z["dates"])
        fit_days, ev_days = np.arange(0, nd, 6)[:30], np.arange(3, nd, 6)[:20]
        base = g.cap_all[:g.n].copy()
        Ht = np.mean([occupancy(z["true_speed"][d], vf, g.order)[0] for d in fit_days], 0)
        print(f"{panel} start: sim share {score(g, z, ev_days, vf)[0]:.3f} / true {score(g, z, ev_days, vf)[1]:.3f}",
              flush=True)
        Qt = np.mean([occupancy(z["true_speed"][d], vf, g.order)[1] for d in fit_days], 0)
        lo = 0.35
        best, best_cap = -1.0, g.cap_all[:g.n].copy()
        for it in range(40):
            eta = 1.0 / (1 + it / 8)
            occ = [occupancy(V, vf, g.order) for V in run(g, z, fit_days)]
            Hs, Qs = np.mean([o[0] for o in occ], 0), np.mean([o[1] for o in occ], 0)
            qt_fit = [occupancy(z["true_speed"][d], vf, g.order)[2] for d in fit_days]
            fit_iou = np.mean([(o[2] & b).sum() / max((o[2] | b).sum(), 1) for o, b in zip(occ, qt_fit)])
            if fit_iou > best:
                best, best_cap = fit_iou, g.cap_all[:g.n].copy()
            s = max(Ht.max(), 1e-3)
            # head mismatch moves the bottleneck; queued-share mismatch just downstream of a queue sets its extent
            dq = np.zeros(g.n); dq[:-1] = (Qt - Qs)[:-1] - (Qt - Qs)[1:]
            step = (Ht - Hs) / s + 0.5 * np.clip(dq, -1, 1) / max(Qt.max(), 1e-3)
            g.cap_all[:g.n] = np.clip(g.cap_all[:g.n] * np.exp(-eta * 0.5 * step), lo * base, 1.05 * base)
            if it % 10 == 9:
                a, b, c, d = score(g, z, ev_days, vf)
                print(f"  it {it + 1}: held-out sim share {a:.3f} / true {b:.3f}  map corr {c:.2f}  daily IoU {d:.3f}  "
                      f"head L1 {np.abs(Ht - Hs).sum():.3f}  caps<0.9 base {(g.cap_all[:g.n] < 0.9 * base).sum()}",
                      flush=True)
        g.cap_all[:g.n] = best_cap
        a, b, c, d = score(g, z, ev_days, vf)
        print(f"  BEST (fit IoU {best:.3f}): held-out sim share {a:.3f} / true {b:.3f}  map corr {c:.2f}  daily IoU {d:.3f}",
              flush=True)
        # reference: another train day's truth as the prediction of this day (day-to-day variation of one run)
        ref = []
        for d_ in ev_days:
            a_ = occupancy(z["true_speed"][d_], vf, g.order)[2]
            b_ = occupancy(z["true_speed"][(d_ + 7) % nd], vf, g.order)[2]
            ref.append((a_ & b_).sum() / max((a_ | b_).sum(), 1))
        print(f"  reference: same weekday next week's truth, daily IoU {np.mean(ref):.3f}", flush=True)
        np.save(ROOT / "work" / f"{CAPS}_{panel}.npy", g.cap_all[:g.n])


if __name__ == "__main__":
    main()
