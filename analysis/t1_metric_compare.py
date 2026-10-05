"""A vs the metric-aligned losses: per-channel RMSE, S_state by month and by panel."""
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
TAGS = sys.argv[1:] or ["t1_m3A", "t1_m3D", "t1_m3E"]


def main() -> None:
    rm, per, mon = {}, {}, {}
    for panel in PANELS:
        fs = {t: PRED / f"{t}_{panel}_test.npz" for t in TAGS}
        if not all(f.exists() for f in fs.values()):
            continue
        z = load(panel, "train")
        P = {t: np.load(f) for t, f in fs.items()}
        days = P[TAGS[0]]["days"]
        lanes = z["lanes"]
        ts, tq = z["true_speed"], z["true_flow"]
        tgt = z["target"][days] & np.isfinite(ts[days]) & np.isfinite(tq[days])
        month = np.array([d[5:7] for d in z["dates"][days].astype(str)])
        cells = np.zeros(len(z["dates"]), bool); cells[days] = True
        for t in TAGS:
            pv = np.zeros_like(ts); pq = np.zeros_like(tq)
            pv[days], pq[days] = P[t]["pv"], P[t]["pq"]
            per.setdefault(t, {})[panel] = state_score(z, pv, pq, cells=cells[:, None, None])["mean"]["S"]
            a = rm.setdefault(t, [0.0, 0.0, 0])
            a[0] += float(((P[t]["pv"] - ts[days])[tgt] ** 2).sum())
            a[1] += float((((P[t]["pq"] - tq[days]) / lanes)[tgt] ** 2).sum())
            a[2] += int(tgt.sum())
            for mo in ("01", "02"):
                c2 = np.zeros(len(z["dates"]), bool); c2[days[month == mo]] = True
                mon.setdefault((t, mo), {})[panel] = state_score(z, pv, pq, cells=c2[:, None, None])["mean"]["S"]
        print(f"  {panel}", flush=True)
    print(f"\n{'tag':<10} {'S_state':>9} {'speed RMSE':>11} {'flow RMSE':>10} {'S 2031-01':>10} {'S 2031-02':>10}")
    for t in TAGS:
        a = rm[t]
        print(f"{t:<10} {family_mean(per[t]):9.4f} {np.sqrt(a[0] / a[2]):11.3f} {np.sqrt(a[1] / a[2]):10.2f} "
              f"{family_mean(mon[(t, '01')]):10.4f} {family_mean(mon[(t, '02')]):10.4f}")
    print(f"\n{'panel':<12} " + " ".join(f"{t:>10}" for t in TAGS))
    for panel in per[TAGS[0]]:
        print(f"{panel:<12} " + " ".join(f"{per[t][panel]:10.4f}" for t in TAGS))


if __name__ == "__main__":
    main()
