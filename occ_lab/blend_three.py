"""Mixtures of the adapted transformer, the adapted new CNN and the existing p_PS. No GPU."""
from __future__ import annotations
import sys
from pathlib import Path
LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB)); sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta"))
sys.path.insert(0, str(LAB.parent / "analysis"))
import numpy as np  # noqa: E402
import common as C  # noqa: E402
from online_adapter import load_split  # noqa: E402
from residual_diag import TAU  # noqa: E402

CACHE = LAB / "cache"
D = {}
for i, panel in enumerate(C.T2_PANELS):
    for s, d in load_split(i, panel, "s20").items():
        t = np.load(CACHE / f"onead_one90_{s}_{panel}.npz")["og"].astype(np.float32)
        c = np.load(CACHE / f"onead_cnn90_{s}_{panel}.npz")["og"].astype(np.float32)
        D[(s, panel)] = (t, c, d["ps"].astype(np.float32), d["y"], d["e"])
    print(f"  {panel}", flush=True)

grid = [(a, b, round(1 - a - b, 2)) for a in np.arange(0, 1.01, 0.1)
        for b in np.arange(0, 1.01 - a + 1e-9, 0.1) if a + b <= 1.0 + 1e-9]
rows = []
for wt, wc, wp in grid:
    fam = {}
    for (s, panel), (t, c, ps, y, e) in D.items():
        p = wt * t + wc * c + wp * ps
        fam.setdefault(s, {})[panel] = C.iou(p >= TAU, y, e).mean()
    v = [C.family_mean(fam[s]) for s in C.SPLITS]
    rows.append((float(np.mean(v)), wt, wc, wp, v))
rows.sort(reverse=True)
print(f"\n{'tfm90':>7}{'cnn90':>7}{'p_PS':>7}" + "".join(f"{s:>13}" for s in C.SPLITS) + f"{'mean':>10}")
for m, wt, wc, wp, v in rows[:14]:
    n = sum(1 for w in (wt, wc, wp) if w > 0)
    print(f"{wt:>7.1f}{wc:>7.1f}{wp:>7.1f}" + "".join(f"{x:>13.4f}" for x in v) + f"{m:>10.4f}   {n} models")
