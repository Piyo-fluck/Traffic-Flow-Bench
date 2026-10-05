"""Where does the Task 1 error actually sit? Decomposition by regime and by temporal isolation.

MASK_SPEC settles the geometry: a Task 1 target is drawn i.i.d. per (day, timestamp, link) by hashing
`panel|regime|date|timestamp|link_id`, at 0.20 / 0.30 / 0.50 for R1 / R2 / R3, and one day carries one regime.
There are no blocks by construction - but at R3 a target's two temporal neighbours are BOTH masked about 25% of
the time, and those cells are where a local interpolator has nothing to lean on. This measures how much of the
loss lives there, using the production CNN's saved test predictions (train days 2031-01/02, the real official
mask, the real regimes).

Reported per cell group: share of targets, speed RMSE, per-lane flow RMSE, and each group's contribution to the
score gap (the weighted error mass it carries).

    py analysis/t1_gap_anatomy.py [tag]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tfb.data import PANELS, load  # noqa: E402
from tfb.metrics import FLOW_NORM, FLOW_W, SPEED_NORM, SPEED_W  # noqa: E402

TAG = sys.argv[1] if len(sys.argv) > 1 else "t1_m3all"
PRED = ROOT / "work" / "preds"


def runs_of_masked(t_mask):
    """For each masked cell, how many of its two temporal neighbours are also masked (0, 1 or 2)."""
    left = np.zeros_like(t_mask)
    right = np.zeros_like(t_mask)
    left[:, 1:] = t_mask[:, :-1]
    right[:, :-1] = t_mask[:, 1:]
    left[:, 0] = t_mask[:, 0]                      # day edges count as unavailable
    right[:, -1] = t_mask[:, -1]
    return left.astype(np.int8) + right.astype(np.int8)


def main() -> None:
    acc = {}
    for panel in PANELS:
        f = PRED / f"{TAG}_{panel}_test.npz"
        if not f.exists():
            continue
        p = np.load(f)
        z = load(panel, "train")
        days, pv, pq = p["days"], p["pv"], p["pq"]
        ts, tq = z["true_speed"][days], z["true_flow"][days]
        lanes = z["lanes"]
        tgt = z["target"][days] & np.isfinite(ts) & np.isfinite(tq)
        reg = np.asarray(z["regime"], str)[days]
        nb = runs_of_masked(z["target"][days])     # 2 = both temporal neighbours masked
        dv = (pv[days if pv.shape[0] == len(z["dates"]) else np.arange(len(days))] - ts)
        dq = (pq[days if pq.shape[0] == len(z["dates"]) else np.arange(len(days))] - tq) / lanes
        for r in ("R1", "R2", "R3"):
            for k in (0, 1, 2):
                m = tgt & (reg[:, None, None] == r) & (nb == k)
                if not m.any():
                    continue
                a = acc.setdefault((r, k), [0.0, 0.0, 0])
                a[0] += float((dv[m] ** 2).sum()); a[1] += float((dq[m] ** 2).sum()); a[2] += int(m.sum())
        vf = z["free_speed"]
        state = np.where(np.nan_to_num(ts, nan=1e9) <= 0.6 * vf, 2,
                         np.where(np.nan_to_num(ts, nan=1e9) <= 0.85 * vf, 1, 0))
        for st in (0, 1, 2):
            m = tgt & (state == st)
            if m.any():
                b = acc.setdefault(("state", st), [0.0, 0.0, 0])
                b[0] += float((dv[m] ** 2).sum()); b[1] += float((dq[m] ** 2).sum()); b[2] += int(m.sum())
        print(f"  {panel}", flush=True)
    st_acc = {k[1]: v for k, v in acc.items() if k[0] == "state"}
    acc = {k: v for k, v in acc.items() if k[0] != "state"}
    n_all = sum(a[2] for a in acc.values())
    sv = sum(a[0] for a in acc.values())
    sq = sum(a[1] for a in acc.values())
    print(f"\ntargets {n_all:,}; overall speed RMSE {np.sqrt(sv / n_all):.3f} km/h, "
          f"flow RMSE {np.sqrt(sq / n_all):.2f} veh/h/lane")
    print(f"{'group':<12} {'share':>7} {'v RMSE':>8} {'q RMSE':>8} {'v err mass':>11} {'q err mass':>11}")
    for (r, k), a in sorted(acc.items()):
        name = {0: "isolated", 1: "one side", 2: "both gone"}[k]
        print(f"{r} {name:<9} {a[2] / n_all:7.1%} {np.sqrt(a[0] / a[2]):8.3f} {np.sqrt(a[1] / a[2]):8.2f} "
              f"{a[0] / sv:11.1%} {a[1] / sq:11.1%}")
    for k in (0, 1, 2):
        g = [a for (r, kk), a in acc.items() if kk == k]
        n = sum(a[2] for a in g)
        if n:
            print(f"all regimes, {['isolated', 'one side', 'both gone'][k]:<9} {n / n_all:7.1%} "
                  f"{np.sqrt(sum(a[0] for a in g) / n):8.3f} {np.sqrt(sum(a[1] for a in g) / n):8.2f} "
                  f"{sum(a[0] for a in g) / sv:11.1%} {sum(a[1] for a in g) / sq:11.1%}")
    print("\nby traffic state of the true cell:")
    for st in sorted(st_acc):
        a = st_acc[st]
        nm = {0: "free flow", 1: "transition", 2: "queued"}[st]
        print(f"   {nm:<11} {a[2] / n_all:7.1%} {np.sqrt(a[0] / a[2]):8.3f} {np.sqrt(a[1] / a[2]):8.2f} "
              f"{a[0] / sv:11.1%} {a[1] / sq:11.1%}")
    print(f"\n(score weights: speed {SPEED_W} / norm {SPEED_NORM}, flow {FLOW_W} / norm {FLOW_NORM})")


if __name__ == "__main__":
    main()
