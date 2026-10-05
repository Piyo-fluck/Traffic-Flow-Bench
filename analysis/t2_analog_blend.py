"""Task 2 onset: simple LightGBM (+) analog-map blends, offline from the caches of analysis/t2_analog_prior.py (note 47).

Per history kind (masked / filled), split and panel, work/cache/t2analog_<hist>_<split>_<panel>.npz holds, on the
blank-aware official-like onset windows: the truth (fut, efut), the day index, the base LightGBM probability (mean of
its 3 seeds), the LightGBM+analog probabilities, the two analog maps (cand, pca), and the neighbour distances.
Variants (expected-IoU ratio rule at T+30): p = w * LGBM_base + (1 - w) * map, for both keys, w in 0.3 .. 1.0; and a
geometric (logit) blend. Also prints the mean neighbour distance per split (train leave-day-out vs val vs priv).

    .venv/Scripts/python.exe analysis/t2_analog_blend.py
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
from t2_eda03 import F  # noqa: E402
from t2_expected_iou import select  # noqa: E402
from t2_pseudo_eval import iou  # noqa: E402
from tfb.data import CACHE  # noqa: E402
from tfb.t1eval import family_mean  # noqa: E402

DAY_SETS = {"d1-14": 14, "all": 31}
WS = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3)


def fmean(d):
    d = {k: v for k, v in d.items() if np.isfinite(v)}
    return family_mean(d) if d else float("nan")


def lg(p):
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def main() -> None:
    for hist in ("masked", "filled"):
        res: dict = {}
        dists: dict = {}
        for panel in T2_PANELS:
            for split in ("validation", "private"):
                f = CACHE / f"t2analog_{hist}_{split}_{panel}.npz"
                if not f.exists():
                    continue
                c = np.load(f)
                for key in ("cand", "pca"):
                    dists.setdefault((split, key), []).append(np.nanmean(c[f"dist_{key}"]))
                    if f"dist_train_{key}" in c.files:
                        dists.setdefault(("train LOO", key), []).append(float(c[f"dist_train_{key}"]))
                variants = {}
                for key in ("cand", "pca"):
                    for w in WS:
                        variants[f"lin {key} w{w}"] = w * c["p_base"] + (1 - w) * c[f"map_{key}"]
                    for w in (0.7, 0.5):
                        z = w * lg(c["p_base"]) + (1 - w) * lg(c[f"map_{key}"])
                        variants[f"logit {key} w{w}"] = 1 / (1 + np.exp(-z))
                    variants[f"lgbm+{key} feats"] = c[f"p_{key}"]
                for name, p in variants.items():
                    pr = np.zeros(c["fut"].shape, bool)
                    pr[:, F - 1] = np.stack([select(q, "ratio", 1.0) for q in p])
                    s = iou(pr, c["fut"], c["efut"])
                    for ds, dmax in DAY_SETS.items():
                        m = c["d"] < dmax
                        res.setdefault(name, {}).setdefault((split, ds), {})[panel] = s[m].mean() if m.any() else np.nan
        if not res:
            continue
        print(f"\n== {hist} histories: onset IoU (val d1-14 | val all | priv d1-14 | priv all), delta vs LightGBM base (lin w1.0) ==")
        base = res["lin pca w1.0"]
        conds = [("validation", "d1-14"), ("validation", "all"), ("private", "d1-14"), ("private", "all")]
        rows = []
        for name, r in res.items():
            d = [fmean(r[c]) - fmean(base[c]) for c in conds]
            rows.append((np.mean(d), min(d), name, [fmean(r[c]) for c in conds], d))
        for mean_d, min_d, name, vals, d in sorted(rows, key=lambda x: -x[0]):
            print(f"  {name:<18} " + " | ".join(f"{v:.3f} ({x:+.3f})" for v, x in zip(vals, d)) + f"  mean {mean_d:+.4f} min {min_d:+.4f}")
        print("  mean neighbour distance: " + "  ".join(f"{k[0]} {k[1]} {np.mean(v):.2f}" for k, v in sorted(dists.items())))


if __name__ == "__main__":
    main()
