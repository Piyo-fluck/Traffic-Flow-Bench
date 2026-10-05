"""Task 2: does the global-state LightGBM gain (note 43) survive the blend with the CNN? Caches only, seconds.

Inputs on the same blank-aware official-like windows:
    work/cache/t2probs_*   labels, and I's ongoing probs (M's LightGBM)
    work/cache/t2cnn_*     the 3 Lovász CNN seeds (averaged)
    work/cache/t2gs_*      the paired re-trained ongoing LightGBM: g0 = without, g1 = with the global features
Ongoing variants, scored against M's rule (0.5 I + 0.5 CNN, tau 0.60):
    M tau            M's blend at another tau
    g0 w/tau         the re-trained base blended with the CNN (paired control)
    g1 w/tau         the global-feature model blended with the CNN
    g1 alone tau
Days 1-14 and all, validation and private; family mean; panels better/worse vs M.

    .venv/Scripts/python.exe analysis/t2_gs_blend.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "scripts"))
from t2_eda02 import T2_PANELS  # noqa: E402
from t2_pseudo_eval import iou  # noqa: E402
from tfb.data import CACHE  # noqa: E402
from tfb.t1eval import family_mean  # noqa: E402

PROD_TAGS = ["t2_cnn_lov", "t2_cnn_lov_s1", "t2_cnn_lov_s2"]
TAGS = list(PROD_TAGS)
COND = [("validation", "d1-14", 14), ("validation", "all", 31), ("private", "d1-14", 14), ("private", "all", 31)]


def fmean(d):
    d = {k: v for k, v in d.items() if np.isfinite(v)}
    return family_mean(d) if d else float("nan")


def main() -> None:
    global TAGS
    if len(sys.argv) > 1:  # CNN tags to average in the g0/g1 blends (M's own blend keeps the 3 production seeds)
        TAGS = sys.argv[1:]
    res: dict = {}
    for panel in T2_PANELS:
        for split in ("validation", "private"):
            c = np.load(CACHE / f"t2probs_{split}_{panel}.npz")
            if not len(c["og_d"]):
                continue
            g = np.load(CACHE / f"t2gs_{split}_{panel}.npz")
            assert (g["og_d"] == c["og_d"]).all(), "window sets differ"
            ld = lambda tags: np.mean([np.load(CACHE / f"t2cnn_{t}_{split}_{panel}.npz")["og_p"] for t in tags], 0)  # noqa: E731
            cm = ld(PROD_TAGS)  # M's CNN: always the 3 production seeds
            cn = ld(TAGS)       # the CNN set under test
            v = {}
            for tau in (0.50, 0.55, 0.60):
                v[f"M t{tau}"] = (0.5 * c["og_p"] + 0.5 * cm) >= tau
                for w in (0.4, 0.5, 0.6, 0.7):
                    v[f"g0 w{w} t{tau}"] = (w * g["og_p0"] + (1 - w) * cn) >= tau
                    v[f"g1 w{w} t{tau}"] = (w * g["og_p1"] + (1 - w) * cn) >= tau
            for tau in (0.50, 0.60, 0.65):
                v[f"g1 alone t{tau}"] = g["og_p1"] >= tau
            for name, pr in v.items():
                s = iou(pr, c["og_fut"], c["og_efut"])
                for sp, ds, dmax in COND:
                    if sp == split:
                        m = c["og_d"] < dmax
                        res.setdefault(name, {}).setdefault((sp, ds), {})[panel] = s[m].mean() if m.any() else np.nan
    base = res["M t0.6"]
    rows = []
    for name, r in res.items():
        cells, deltas = [], []
        for sp, ds, _ in COND:
            d, b = r[(sp, ds)], base[(sp, ds)]
            diffs = [d[p] - b[p] for p in d if np.isfinite(d[p]) and np.isfinite(b.get(p, np.nan))]
            delta = fmean(d) - fmean(b)
            deltas.append(delta)
            cells.append(f"{fmean(d):.3f} ({delta:+.3f} {sum(x > 1e-9 for x in diffs)}/{sum(x < -1e-9 for x in diffs)})")
        rows.append((np.mean(deltas), min(deltas), name, cells))
    print("ongoing, vs M (0.5 I + 0.5 CNN, tau 0.60).  columns: " + " | ".join(f"{sp} {ds}" for sp, ds, _ in COND))
    for mean_d, min_d, name, cells in sorted(rows, key=lambda r: -r[0]):
        print(f"  {name:<16} mean {mean_d:+.4f} min {min_d:+.4f} | " + " | ".join(cells))


if __name__ == "__main__":
    main()
