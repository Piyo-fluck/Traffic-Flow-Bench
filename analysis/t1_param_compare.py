"""A vs B vs C: flow RMSE by level bin and by month, and the blend with A (the stop-criterion table)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tfb.data import PANELS, load  # noqa: E402

PRED = ROOT / "work" / "preds"
TAGS = sys.argv[1:] or ["t1_m3A", "t1_m3B", "t1_m3C"]
BINS = (0, 400, 800, 1200, 1800, 1e9)


def main() -> None:
    acc, blend = {}, {}
    for panel in PANELS:
        fs = {t: PRED / f"{t}_{panel}_test.npz" for t in TAGS}
        if not all(f.exists() for f in fs.values()):
            continue
        z = load(panel, "train")
        P = {t: np.load(f) for t, f in fs.items()}
        days = P[TAGS[0]]["days"]
        lanes = z["lanes"]
        ts, tq = z["true_speed"][days], z["true_flow"][days]
        ql = tq / lanes
        tgt = z["target"][days] & np.isfinite(ts) & np.isfinite(tq)
        month = np.array([d[5:7] for d in z["dates"][days].astype(str)])
        for t in TAGS:
            q = P[t]["pq"] / lanes
            for i in range(len(BINS) - 1):
                m = tgt & (ql >= BINS[i]) & (ql < BINS[i + 1])
                a = acc.setdefault((t, f"{BINS[i]:.0f}-{BINS[i+1]:.0f}"), [0.0, 0])
                a[0] += float(((q - ql)[m] ** 2).sum()); a[1] += int(m.sum())
            for mo in ("01", "02"):
                m = tgt & (month == mo)[:, None, None]
                a = acc.setdefault((t, f"2031-{mo}"), [0.0, 0])
                a[0] += float(((q - ql)[m] ** 2).sum()); a[1] += int(m.sum())
            a = acc.setdefault((t, "all"), [0.0, 0])
            a[0] += float(((q - ql)[tgt] ** 2).sum()); a[1] += int(tgt.sum())
        for t in TAGS[1:]:
            for al in (0.0, 0.2, 0.3, 0.5, 0.7, 1.0):
                q = (1 - al) * P[TAGS[0]]["pq"] / lanes + al * P[t]["pq"] / lanes
                b = blend.setdefault((t, al), [0.0, 0])
                b[0] += float(((q - ql)[tgt] ** 2).sum()); b[1] += int(tgt.sum())
        print(f"  {panel}", flush=True)
    keys = ["all", "2031-01", "2031-02"] + [f"{BINS[i]:.0f}-{BINS[i+1]:.0f}" for i in range(len(BINS) - 1)]
    print(f"\n{'group':>16} " + " ".join(f"{t:>10}" for t in TAGS) + "   " +
          " ".join(f"{'d ' + t[-1]:>8}" for t in TAGS[1:]))
    for k in keys:
        r = {t: np.sqrt(acc[(t, k)][0] / max(acc[(t, k)][1], 1)) for t in TAGS}
        print(f"{k:>16} " + " ".join(f"{r[t]:10.3f}" for t in TAGS) + "   " +
              " ".join(f"{r[t] - r[TAGS[0]]:+8.3f}" for t in TAGS[1:]))
    print(f"\nblend with {TAGS[0]} (flow RMSE):")
    for t in TAGS[1:]:
        row = {al: np.sqrt(b[0] / b[1]) for (tt, al), b in blend.items() if tt == t}
        print(f"   {t}: " + "  ".join(f"a={al:.1f} {v:.3f}" for al, v in sorted(row.items())))


if __name__ == "__main__":
    main()
