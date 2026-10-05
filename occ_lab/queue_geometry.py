"""Is the ongoing answer a queue GEOMETRY rather than 900 independent cells, and where is the error?

Every ongoing member we have emits a per-cell probability and the decision thresholds it. But a queue on a
corridor at one instant is an interval of links, and over the horizon its downstream end (the bottleneck) barely
moves while its upstream end travels at the shockwave speed. If that is what the truth looks like, a window's
answer has about four degrees of freedom, not the six hundred cells we predict - and the per-window budget that
`occ_lab/selector_posterior.py` priced at +0.0501 is simply the interval's length, a geometric quantity no
per-cell posterior carries.

Measured here, on the official-like ongoing pool of both scored months:

  shape        how many runs of queued links the truth has per (window, step), and what IoU the truth's own
               convex hull scores against it - the ceiling of any single-interval model
  our shape    the same counts for our thresholded answer
  endpoints    the signed error of the predicted head (downstream end) and tail (upstream end), per step
  oracles      IoU when the posterior's ranking is kept but the answer is clipped to the truth's hull, and when
               only the head or only the tail is corrected - which says which end to spend a model on

    python occ_lab/queue_geometry.py --tag s20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta"))
sys.path.insert(0, str(LAB.parent / "analysis"))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

import common as C  # noqa: E402
from online_adapter import ETA, GATE, global_features, load_split  # noqa: E402
from residual_diag import TAU  # noqa: E402


def runs_of(v):
    """number of runs of True along the last axis"""
    a = v.astype(np.int8)
    return (a[..., :1] + np.maximum(np.diff(a, axis=-1), 0)).sum(-1)


def hull(v, e):
    """[first, last] index of the True cells along the last axis; -1 where empty."""
    idx = np.where(v, np.arange(v.shape[-1]), -1)
    lo = np.where(v, np.arange(v.shape[-1]), v.shape[-1]).min(-1)
    hi = idx.max(-1)
    lo = np.where(hi >= 0, lo, -1)
    _ = e
    return lo, hi


def fill_hull(lo, hi, e):
    s = np.arange(e.shape[-1])
    m = (s[None, None, :] >= lo[..., None]) & (s[None, None, :] <= hi[..., None]) & (hi[..., None] >= 0)
    return m & e


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="s20")
    args = ap.parse_args()
    gbm = lgb.Booster(model_file=str(LAB / "models" / "residual_lgb.txt"))
    agg = {}
    dh, dt = [], []
    for i, panel in enumerate(C.T2_PANELS):
        sp = load_split(i, panel, args.tag)
        for s, D in sp.items():
            base, y, e = D["base"], D["y"], D["e"]
            F = global_features(D["P"], base)
            rh = gbm.predict(F.reshape(-1, F.shape[-1])).reshape(base.shape).astype(np.float32)
            pg = np.clip(base + ETA * np.where(np.abs(rh) >= GATE, rh, 0.0), 0, 1)
            yh = (pg >= TAU) & e
            yt = y & e
            ok = yt.any(-1)                                   # (N, F) steps with a queue in the truth
            agg.setdefault("truth runs", []).append(runs_of(yt)[ok])
            agg.setdefault("our runs", []).append(runs_of(yh)[ok])
            agg.setdefault("truth cells", []).append(yt.sum(-1)[ok])

            tlo, thi = hull(yt, e)
            plo, phi = hull(yh, e)
            both = ok & (phi >= 0)
            dh.append((plo - tlo)[both])                      # head = downstream end (small index)
            dt.append((phi - thi)[both])
            agg.setdefault("hull of truth", []).append(C.iou(fill_hull(tlo, thi, e), y, e))
            agg.setdefault("AS", []).append(C.iou(yh, y, e))
            # keep our ranking, clip to the truth's hull, keep the same number of cells
            for nm, (lo_, hi_) in (("oracle both ends", (tlo, thi)),
                                   ("oracle head only", (tlo, phi)),
                                   ("oracle tail only", (plo, thi))):
                m = fill_hull(lo_, hi_, e)
                agg.setdefault(nm, []).append(C.iou(np.where(m, yh | m, False) & m, y, e))
        print(f"  {panel}", flush=True)

    cat = {k: np.concatenate(v) for k, v in agg.items()}
    dh, dt = np.concatenate(dh), np.concatenate(dt)
    print(f"\nshape of the truth, over {len(cat['truth runs']):,} (window, step) pairs that hold a queue\n")
    for k in ("truth runs", "our runs", "truth cells"):
        v = cat[k]
        print(f"  {k:<14} mean {v.mean():6.2f}   median {np.median(v):5.1f}   "
              f"share == 1 run {np.mean(v == 1):.3f}" if "runs" in k else
              f"  {k:<14} mean {v.mean():6.2f}   median {np.median(v):5.1f}")
    print(f"\nendpoint error (links), predicted minus true, over {len(dh):,} pairs where both are non-empty\n")
    for nm, d in (("head (downstream end)", dh), ("tail (upstream end)", dt)):
        print(f"  {nm:<24} mean {d.mean():+6.2f}  sd {d.std():5.2f}  "
              f"|err| <= 1: {np.mean(np.abs(d) <= 1):.3f}   median |err| {np.median(np.abs(d)):.1f}")
    print(f"\nongoing IoU, posterior ranking kept, geometry replaced\n")
    for k in ("AS", "hull of truth", "oracle head only", "oracle tail only", "oracle both ends"):
        print(f"  {k:<20} {cat[k].mean():.4f}   {cat[k].mean() - cat['AS'].mean():+.4f}")


if __name__ == "__main__":
    main()
