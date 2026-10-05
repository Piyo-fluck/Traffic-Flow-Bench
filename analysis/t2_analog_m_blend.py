"""Task 2 onset: the state-conditioned analog map (note 47) on top of M's production onset rule, from caches only.

Masked histories (the only ones with cached production probabilities), same blank-aware official-like windows:
    H        production LightGBM onset probabilities (work/cache/t2probs_*, 5 bags, table priors)
    CNN      mean of the 3 production Lovasz CNN seeds (work/cache/t2cnn_t2_cnn_lov*)
    map      the analog map, pca and cand keys (work/cache/t2analog_masked_*)
Variants (expected-IoU ratio rule at T+30), change against M = 0.7 H + 0.3 CNN:
    M then prior   w * M + (1 - w) * map, w in 0.6 .. 0.9
    H + map        w * H + (1 - w) * map (no CNN)
    three-way      a * H + b * CNN + c * map on a small grid
Days 1-14 and all, validation and private, family mean, panels better/worse.

    .venv/Scripts/python.exe analysis/t2_analog_m_blend.py
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

TAGS = ["t2_cnn_lov", "t2_cnn_lov_s1", "t2_cnn_lov_s2"]
CONDS = [("validation", "d1-14", 14), ("validation", "all", 31), ("private", "d1-14", 14), ("private", "all", 31)]


def fmean(d):
    d = {k: v for k, v in d.items() if np.isfinite(v)}
    return family_mean(d) if d else float("nan")


def main() -> None:
    res: dict = {}
    for panel in T2_PANELS:
        for split in ("validation", "private"):
            c = np.load(CACHE / f"t2probs_{split}_{panel}.npz")
            if not len(c["on_d"]):
                continue
            a = np.load(CACHE / f"t2analog_masked_{split}_{panel}.npz")
            assert (a["d"] == c["on_d"]).all(), "window sets differ"
            h = c["on_p"]
            cn = np.mean([np.load(CACHE / f"t2cnn_{t}_{split}_{panel}.npz")["on_p"] for t in TAGS], 0)
            m = 0.7 * h + 0.3 * cn
            v = {"M (0.7 H + 0.3 CNN)": m, "H alone": h}
            for key in ("pca", "cand"):
                mp = a[f"map_{key}"]
                for w in (0.9, 0.8, 0.7, 0.6):
                    v[f"M then {key} w{w}"] = w * m + (1 - w) * mp
                    v[f"H + {key} w{w}"] = w * h + (1 - w) * mp
                for wh, wc, wm in ((0.6, 0.2, 0.2), (0.5, 0.2, 0.3), (0.55, 0.15, 0.3), (0.5, 0.3, 0.2), (0.6, 0.1, 0.3)):
                    v[f"3way {key} {wh}/{wc}/{wm}"] = wh * h + wc * cn + wm * mp
            for name, p in v.items():
                pr = np.zeros(c["on_fut"].shape, bool)
                pr[:, F - 1] = np.stack([select(q, "ratio", 1.0) for q in p])
                s = iou(pr, c["on_fut"], c["on_efut"])
                for sp, ds, dmax in CONDS:
                    if sp == split:
                        mm = c["on_d"] < dmax
                        res.setdefault(name, {}).setdefault((sp, ds), {})[panel] = s[mm].mean() if mm.any() else np.nan
    base = res["M (0.7 H + 0.3 CNN)"]
    rows = []
    for name, r in res.items():
        cells, deltas = [], []
        for sp, ds, _ in CONDS:
            d = fmean(r[(sp, ds)]) - fmean(base[(sp, ds)])
            diffs = [r[(sp, ds)][p] - base[(sp, ds)][p] for p in r[(sp, ds)] if np.isfinite(r[(sp, ds)][p]) and np.isfinite(base[(sp, ds)].get(p, np.nan))]
            deltas.append(d)
            cells.append(f"{fmean(r[(sp, ds)]):.3f} ({d:+.3f} {sum(x > 1e-9 for x in diffs)}/{sum(x < -1e-9 for x in diffs)})")
        rows.append((np.mean(deltas), min(deltas), name, cells))
    print("onset IoU on masked histories, change vs M's onset rule. columns: val d1-14 | val all | priv d1-14 | priv all")
    for mean_d, min_d, name, cells in sorted(rows, key=lambda x: -x[0]):
        print(f"  {name:<26} " + " | ".join(cells) + f"  mean {mean_d:+.4f} min {min_d:+.4f}")


if __name__ == "__main__":
    main()
