"""Task 2: blend LightGBM and CNN probabilities offline, from caches only (no data loading, seconds).

Inputs per panel and split, on the same blank-aware official-like windows:
    work/cache/t2probs_<split>_<panel>.npz        LightGBM probs (onset = H, ongoing = I) plus labels fut/efut
    work/cache/t2cnn_<TAG>_<split>_<panel>.npz    CNN probs (analysis/t2_cnn.py)
CNN tags are given on the command line and averaged (e.g. several Lovász seeds). Variants:
    ongoing  lgbm (tau 0.65) | cnn_avg | w*lgbm + (1-w)*cnn_avg for w in 0.3..0.7, tau 0.60 / 0.65
    onset    lgbm (expected-IoU ratio, c 1.0) | w*lgbm + (1-w)*cnn_avg for w in 0.7 / 0.85
Days 1-14 and all, validation and private, family mean, panels better/worse vs lgbm.

    py analysis/t2_blend_eval.py t2_cnn_lov t2_cnn_lov_s1 t2_cnn_lov_s2
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
W_OG = (0.3, 0.4, 0.5, 0.6, 0.7)
TAUS = (0.60, 0.65)
W_ON = (0.7, 0.85)


def fmean(d):
    d = {k: v for k, v in d.items() if np.isfinite(v)}
    return family_mean(d) if d else float("nan")


def main() -> None:
    tags = sys.argv[1:] or ["t2_cnn_lov"]
    res: dict = {}
    for panel in T2_PANELS:
        for split in ("validation", "private"):
            c = np.load(CACHE / f"t2probs_{split}_{panel}.npz")
            cn = [np.load(CACHE / f"t2cnn_{t}_{split}_{panel}.npz") for t in tags]
            for x in cn:
                assert (x["on_d"] == c["on_d"]).all() and (x["og_d"] == c["og_d"]).all(), "window sets differ"
            variants = {}
            if len(c["og_d"]):
                cg = np.mean([x["og_p"] for x in cn], 0)
                variants[("ongoing", "lgbm")] = c["og_p"] >= 0.65
                variants[("ongoing", "cnn_avg")] = cg >= 0.65
                for w in W_OG:
                    for tau in TAUS:
                        variants[("ongoing", f"w{w}/t{tau}")] = (w * c["og_p"] + (1 - w) * cg) >= tau
            if len(c["on_d"]):
                co = np.mean([x["on_p"] for x in cn], 0)
                for name, p in [("lgbm", c["on_p"])] + [(f"w{w}", w * c["on_p"] + (1 - w) * co) for w in W_ON]:
                    pr = np.zeros(c["on_fut"].shape, bool)
                    pr[:, F - 1] = np.stack([select(q, "ratio", 1.0) for q in p])
                    variants[("onset", name)] = pr
            for (kind, name), pr in variants.items():
                fut, efut, dd = (c["og_fut"], c["og_efut"], c["og_d"]) if kind == "ongoing" else (c["on_fut"], c["on_efut"], c["on_d"])
                s = iou(pr, fut, efut)
                for ds, dmax in DAY_SETS.items():
                    m = dd < dmax
                    res.setdefault((kind, split, ds), {}).setdefault(name, {})[panel] = s[m].mean() if m.any() else np.nan
    print(f"CNN tags averaged: {tags}")
    for (kind, split, ds), r in res.items():
        base = r["lgbm"]
        print(f"\n== {kind} {split} {ds} (lgbm {fmean(base):.3f}) ==")
        for name, d in sorted(r.items(), key=lambda kv: -fmean(kv[1]))[:6]:
            diffs = [d[p] - base[p] for p in d if np.isfinite(d[p]) and np.isfinite(base[p])]
            print(f"  {name:<12} {fmean(d):.3f} ({fmean(d) - fmean(base):+.3f}; {sum(x > 1e-9 for x in diffs)}/{sum(x < -1e-9 for x in diffs)})")


if __name__ == "__main__":
    main()
