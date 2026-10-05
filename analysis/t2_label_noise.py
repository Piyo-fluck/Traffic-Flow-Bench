"""Task 2: how much does label noise matter? The official truth is the noise-free underlying state; ours is not.

SCORING_SPEC: "The ground truth is taken from the underlying state rather than the noisy observation." Our
train labels (and the in-month pseudo labels) threshold the noisy observed speed at 0.6 vf. The underlying
state is not released, so it is approximated by the 3-slot temporal median of the observed speed. Per panel
(train, eligible cells):
    near1 / near2   share of queued cells whose observed speed is within 1 / 2 km/h of the cut
    flip            share of eligible cells whose label differs between noisy and smoothed speed
    IoU_all         IoU(noisy labels, smoothed labels) over all eligible cells
    IoU_og          mean per-window IoU of the two label versions over the horizons of ongoing candidates
IoU_og is how much a perfect predictor of the noisy label would lose against a smoothed truth. It is an upper
bound on what denoised training labels could gain.

    py analysis/t2_label_noise.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
from t2_eda02 import T2_PANELS, candidates  # noqa: E402
from t2_eda03 import F  # noqa: E402
from tfb.data import load  # noqa: E402
from tfb.features import shift  # noqa: E402
from tfb.t1eval import family_mean  # noqa: E402


def main() -> None:
    rows = {}
    for panel in T2_PANELS:
        z = load(panel, "train")
        v, vf, el = z["true_speed"], z["free_speed"], z["elig"]
        ok = el & np.isfinite(v)
        cut = 0.6 * vf
        vv = np.where(ok, v, np.nan)
        with np.errstate(all="ignore"):
            vs = np.nanmedian(np.stack([shift(vv, 1, 1), vv, shift(vv, -1, 1)]), 0)
        qn = ok & (np.nan_to_num(v, nan=1e9) <= cut)
        qs = ok & (np.nan_to_num(vs, nan=1e9) <= cut)
        d_cut = np.abs(np.nan_to_num(v, nan=1e9) - cut)
        c = candidates(z)
        d, t0 = c["d"][c["ongoing"]], c["t0"][c["ongoing"]]
        fi = t0[:, None] + np.arange(1, F + 1)
        a, b, e = qn[d[:, None], fi], qs[d[:, None], fi], ok[d[:, None], fi]
        inter, union = (a & b & e).sum((1, 2)), ((a | b) & e).sum((1, 2))
        iou_og = np.where(union == 0, 1.0, inter / np.maximum(union, 1))
        rows[panel] = dict(
            near1=float((qn & (d_cut < 1)).sum() / max(qn.sum(), 1)),
            near2=float((qn & (d_cut < 2)).sum() / max(qn.sum(), 1)),
            flip=float((qn ^ qs).sum() / max(ok.sum(), 1)),
            IoU_all=float((qn & qs).sum() / max((qn | qs).sum(), 1)),
            IoU_og=float(iou_og.mean()),
        )
        print(f"{panel:<11} " + "  ".join(f"{k} {x:.4f}" for k, x in rows[panel].items()), flush=True)
    print("\n== family mean (train) ==")
    for k in ("near1", "near2", "flip", "IoU_all", "IoU_og"):
        print(f"  {k:<8} {family_mean({p: r[k] for p, r in rows.items()}):.4f}")


if __name__ == "__main__":
    main()
