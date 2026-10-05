"""Compare Task 1 families on the local test block, alone and blended.

Reads the saved test predictions in work/preds (`t1_<tag>_<panel>_test.npz`) and reports, on the real official
targets of 2031-01/02: S_state (the leaderboard's family mean), speed and per-lane flow RMSE, the same split by
traffic state, and the blend x = (1 - a) x_A + a x_B for a grid of a.

    py analysis/t1_blend_eval.py m3all m4
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tfb.data import PANELS, load  # noqa: E402
from tfb.metrics import state_score  # noqa: E402
from tfb.t1eval import family_mean  # noqa: E402

PRED = ROOT / "work" / "preds"
ALPHAS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)


def main() -> None:
    tag_a, tag_b = (sys.argv[1:3] + ["m3all", "m4"])[:2]
    panels = [p for p in PANELS if (PRED / f"t1_{tag_a}_{p}_test.npz").exists()
              and (PRED / f"t1_{tag_b}_{p}_test.npz").exists()]
    print(f"{len(panels)} panels with both {tag_a} and {tag_b}: {', '.join(panels)}")
    acc = {a: {} for a in ALPHAS}
    err = {a: {} for a in ALPHAS}
    for panel in panels:
        A = np.load(PRED / f"t1_{tag_a}_{panel}_test.npz")
        B = np.load(PRED / f"t1_{tag_b}_{panel}_test.npz")
        assert (A["days"] == B["days"]).all(), f"{panel}: different test days"
        days = A["days"]
        z = load(panel, "train")
        ts, tq, lanes, vf = z["true_speed"], z["true_flow"], z["lanes"], z["free_speed"]
        tgt = z["target"][days] & np.isfinite(ts[days]) & np.isfinite(tq[days])
        cells = np.zeros(len(z["dates"]), bool)
        cells[days] = True
        for a in ALPHAS:
            pv = np.zeros_like(ts)
            pq = np.zeros_like(tq)
            pv[days] = (1 - a) * A["pv"] + a * B["pv"]
            pq[days] = (1 - a) * A["pq"] + a * B["pq"]
            acc[a][panel] = state_score(z, pv, pq, cells=cells[:, None, None])["mean"]["S"]
            dv, dq = pv[days] - ts[days], (pq[days] - tq[days]) / lanes
            free = tgt & (np.nan_to_num(ts[days], nan=0.0) > 0.85 * vf)
            for nm, m in (("all", tgt), ("free", free), ("queued", tgt & (ts[days] <= 0.6 * vf))):
                e = err[a].setdefault(nm, [0.0, 0.0, 0])
                e[0] += float((dv[m] ** 2).sum()); e[1] += float((dq[m] ** 2).sum()); e[2] += int(m.sum())
        print(f"  {panel}", flush=True)
    print(f"\n{'alpha':>6} {'S_state':>9} {'v RMSE':>8} {'q RMSE':>8} {'q RMSE free':>12} {'q RMSE queued':>14}")
    for a in ALPHAS:
        e, f, q = err[a]["all"], err[a]["free"], err[a]["queued"]
        print(f"{a:6.2f} {family_mean(acc[a]):9.4f} {np.sqrt(e[0] / e[2]):8.3f} {np.sqrt(e[1] / e[2]):8.2f} "
              f"{np.sqrt(f[1] / f[2]):12.2f} {np.sqrt(q[1] / q[2]):14.2f}")
    best = max(ALPHAS, key=lambda a: family_mean(acc[a]))
    print(f"\nbest alpha {best:.2f}: S_state {family_mean(acc[best]):.4f} "
          f"(alpha 0 = {tag_a}: {family_mean(acc[0.0]):.4f}, alpha 1 = {tag_b}: {family_mean(acc[1.0]):.4f})")
    for p in panels:
        print(f"   {p:<12} {tag_a} {acc[0.0][p]:.4f}  {tag_b} {acc[1.0][p]:.4f}  best-alpha {acc[best][p]:.4f}")


if __name__ == "__main__":
    main()
