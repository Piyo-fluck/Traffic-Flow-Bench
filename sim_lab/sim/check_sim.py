"""Calibration gate for the synthetic runs (step 1 of ../SIM_EXPERIMENT.md): do they look like the train run?

Per panel, the synthetic runs against the real train split:
  queued share         share of eligible cells with v <= 0.6 vf
  bottleneck map corr  correlation of the per-link queued share (the mean over runs, and the median of runs)
  top-8 overlap        overlap of the 8 most-queued links
  run diversity        mean pairwise correlation of the per-link maps between runs (1.0 = every run the same,
                       the failure mode we want to avoid; the train run cannot show this)
  onset tod            median hour of the first morning onset (earliest-onset origins, 04:00-14:00)
  onset windows/day    earliest-onset origins per day (the Task 2 training signal)
  flow level           mean flow, synthetic / train
Targets are in SIM_EXPERIMENT.md section 3.

    python sim/check_sim.py --data sim_lab/data
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parents[1]
PROJ = LAB.parent
for p in (PROJ, PROJ / "analysis", PROJ / "scripts"):
    sys.path.insert(0, str(p))
from t2_eda02 import T2_PANELS  # noqa: E402
from t2_eda03 import earliest_onset, labels  # noqa: E402
from tfb.data import load  # noqa: E402


def stats(z):
    q, el = labels(z)
    m = q.mean((0, 1))
    eo = earliest_onset(q)
    d, t = np.nonzero(eo[:, 48:168])
    tod = np.median(t + 48) / 12 if len(t) else np.nan
    return dict(share=q.sum() / max(el.sum(), 1), map=m, tod=tod, on_per_day=len(d) / len(q),
                flow=np.nanmean(np.where(el, z["true_flow"], np.nan)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--panels", nargs="*", default=None)
    args = ap.parse_args()
    print(f"{'panel':<11} {'share tr/sim':>14} {'map corr mean/med':>18} {'top8':>5} {'diversity':>9} "
          f"{'onset h tr/sim':>15} {'onsets/day tr/sim':>18} {'flow ratio':>10}")
    for panel in T2_PANELS:
        if args.panels and panel not in args.panels:
            continue
        files = sorted((args.data / panel).glob("run_*.npz"))
        if not files:
            continue
        tr = stats(load(panel, "train"))
        ss = [stats({k: np.load(f)[k] for k in ("true_speed", "true_flow", "elig", "free_speed")}) for f in files]
        maps = np.stack([s["map"] for s in ss])
        mean_map = maps.mean(0)
        corr = lambda a, b: float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")  # noqa: E731
        c_mean = corr(mean_map, tr["map"])
        c_med = float(np.nanmedian([corr(m, tr["map"]) for m in maps]))
        top8 = len(set(np.argsort(-mean_map)[:8]) & set(np.argsort(-tr["map"])[:8]))
        pair = [corr(maps[i], maps[j]) for i in range(len(maps)) for j in range(i + 1, len(maps))]
        div = float(np.nanmean(pair)) if pair else float("nan")
        print(f"{panel:<11} {tr['share']:.3f}/{np.mean([s['share'] for s in ss]):.3f}      "
              f"{c_mean:>6.2f}/{c_med:<6.2f}      {top8:>3}  {div:>8.2f}   "
              f"{tr['tod']:>5.1f}/{np.nanmedian([s['tod'] for s in ss]):<5.1f}     "
              f"{tr['on_per_day']:>6.2f}/{np.mean([s['on_per_day'] for s in ss]):<6.2f}     "
              f"{np.mean([s['flow'] for s in ss]) / tr['flow']:>6.2f}", flush=True)


if __name__ == "__main__":
    main()
