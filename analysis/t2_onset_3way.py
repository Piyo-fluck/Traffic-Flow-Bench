"""Task 2 onset: three-way blend of LightGBM (H), the CNN and the transformer, from caches only (seconds).

p = wH * H + wC * CNN_avg + wT * TFM_avg at T+30, then the expected-IoU ratio rule (c 1.0), as in M (wH 0.7, wC 0.3).
The weights are FITTED on one run (all days, family mean) and SCORED on the other, both ways, against M.
The transformer tags come from the command line (default: the 3 production seeds).

    .venv/Scripts/python.exe analysis/t2_onset_3way.py [tfm tags...]
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "scripts"))
from t2_eda02 import T2_PANELS  # noqa: E402
from t2_eda03 import F  # noqa: E402
from t2_expected_iou import select  # noqa: E402
from t2_pseudo_eval import iou  # noqa: E402
from tfb.data import CACHE  # noqa: E402
from tfb.t1eval import family_mean  # noqa: E402

CNN = ["t2_cnn_lov", "t2_cnn_lov_s1", "t2_cnn_lov_s2"]
DAY_SETS = {"d1-14": 14, "all": 31}
GRID = [(h, c, round(1 - h - c, 2)) for h, c in itertools.product((0.4, 0.5, 0.6, 0.7, 0.8, 0.9), (0.0, 0.1, 0.2, 0.3, 0.4))
        if 1 - h - c >= -1e-9]


def fmean(d):
    d = {k: v for k, v in d.items() if np.isfinite(v)}
    return family_mean(d) if d else float("nan")


def main() -> None:
    tfm = sys.argv[1:] or ["t2_cnn_lov_tfm", "t2_cnn_lov_tfm_s1", "t2_cnn_lov_tfm_s2"]
    data = {}
    for panel in T2_PANELS:
        for split in ("validation", "private"):
            c = np.load(CACHE / f"t2probs_{split}_{panel}.npz")
            if not len(c["on_d"]):
                continue
            ld = lambda tags: np.mean([np.load(CACHE / f"t2cnn_{t}_{split}_{panel}.npz")["on_p"] for t in tags], 0)  # noqa: E731
            data[(panel, split)] = (c["on_p"], ld(CNN), ld(tfm), c["on_fut"], c["on_efut"], c["on_d"])

    def score(split, w):
        out = {ds: {} for ds in DAY_SETS}
        for (panel, sp), (h, cn, tf, fut, efut, dd) in data.items():
            if sp != split:
                continue
            p = w[0] * h + w[1] * cn + w[2] * tf
            pr = np.zeros(fut.shape, bool)
            pr[:, F - 1] = np.stack([select(q, "ratio", 1.0) for q in p])
            s = iou(pr, fut, efut)
            for ds, dmax in DAY_SETS.items():
                m = dd < dmax
                out[ds][panel] = s[m].mean() if m.any() else np.nan
        return out

    cache = {(sp, w): score(sp, w) for sp in ("validation", "private") for w in GRID}
    M = (0.7, 0.3, 0.0)
    print(f"tfm tags: {tfm}")
    for fit, ev in (("validation", "private"), ("private", "validation")):
        best = max(GRID, key=lambda w: fmean(cache[(fit, w)]["all"]))
        r, b = cache[(ev, best)], cache[(ev, M)]
        cells = []
        for ds in DAY_SETS:
            diffs = [r[ds][p] - b[ds][p] for p in r[ds] if np.isfinite(r[ds][p]) and np.isfinite(b[ds].get(p, np.nan))]
            cells.append(f"{ds} {fmean(r[ds]):.3f} ({fmean(r[ds]) - fmean(b[ds]):+.3f}; {sum(x > 1e-9 for x in diffs)}/{sum(x < -1e-9 for x in diffs)})")
        print(f"fit {fit} -> best (wH, wC, wT) = {best}; scored on {ev}: " + "  ".join(cells))
    print("\nfull grid, mean of the 4 conditions (vs M), top 12:")
    rows = []
    for w in GRID:
        d = [fmean(cache[(sp, w)][ds]) - fmean(cache[(sp, M)][ds]) for sp in ("validation", "private") for ds in DAY_SETS]
        rows.append((np.mean(d), min(d), w, d))
    for mean_d, min_d, w, d in sorted(rows, key=lambda r: -r[0])[:12]:
        print(f"  {w}: mean {mean_d:+.4f} min {min_d:+.4f} | " + " ".join(f"{x:+.3f}" for x in d))


if __name__ == "__main__":
    main()
