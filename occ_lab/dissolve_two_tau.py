"""A threshold per cell class: queued at T, or not. Note 53 tried tau per panel / queue size / mean probability,
never per class. The calibration table says the queued-at-T population is over-predicted by about +0.09 through
the uncertain middle, so it should want a higher tau than the rest. Fitted on one month, read on the other."""
from __future__ import annotations
import sys
from pathlib import Path
LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB)); sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta"))
sys.path.insert(0, str(LAB.parent / "analysis"))
import numpy as np  # noqa: E402
import common as C  # noqa: E402
CACHE = LAB / "cache"
D = {}
for i, panel in enumerate(C.T2_PANELS):
    if not (CACHE / f"pseudo_s20_{C.SPLITS[0]}_{panel}.npz").exists():
        continue
    ctx, stat, Wd = C.windows(i, panel)
    for s in C.SPLITS:
        og, ref = Wd[s]["og"], Wd[s]["ref"]
        if not len(og["d"]):
            continue
        base = 0.2 * ref["og_gs1"] + 0.6 * ref["og_tfm8"] + 0.2 * ref["og_cnn3"]
        f = CACHE / f"pseudo_s20_{s}_{panel}.npz"
        if f.exists():
            base = 0.4 * base + 0.6 * np.load(f)["og"].astype(np.float32)
        vq = og["ok"] & (np.nan_to_num(og["v"], nan=1e9) <= 0.6 * ctx.vf)
        idx = np.where(og["ok"][:, ::-1], np.arange(og["ok"].shape[1])[::-1][None, :, None], -1).max(1)
        S = vq.shape[2]; last = np.zeros((vq.shape[0], S), bool); ok = idx >= 0
        jj = np.broadcast_to(np.arange(vq.shape[0])[:, None], idx.shape)
        ss = np.broadcast_to(np.arange(S), idx.shape)
        last[ok] = vq[jj[ok], idx[ok], ss[ok]]
        D.setdefault(s, {})[panel] = (base, og["fut"], og["efut"], np.broadcast_to(last[:, None], base.shape))
    ctx.z = None
    print(f"  {panel}", flush=True)

def score(split, tq, tn):
    out = {}
    for panel, (p, y, e, q) in D[split].items():
        pred = np.where(q, p >= tq, p >= tn)
        out[panel] = C.iou(pred, y, e).mean()
    return C.family_mean(out)

grid = np.round(np.arange(0.35, 0.81, 0.025), 3)
print(f"\nproduction tau 0.50 both:   " + "   ".join(f"{s} {score(s, 0.50, 0.50):.4f}" for s in C.SPLITS))
best = {}
for s in C.SPLITS:
    b = max(((score(s, a, b_), a, b_) for a in grid for b_ in grid), key=lambda r: r[0])
    best[s] = b
    print(f"  best on {s:<11} tau_queued {b[1]:.3f} tau_other {b[2]:.3f} -> {b[0]:.4f} "
          f"(+{b[0] - score(s, .5, .5):.4f} in sample)")
print("\nheld out (fit one month, read the other):")
for fit, ev in ((C.SPLITS[0], C.SPLITS[1]), (C.SPLITS[1], C.SPLITS[0])):
    _, a, b_ = best[fit]
    v0, v1 = score(ev, .5, .5), score(ev, a, b_)
    print(f"  fit {fit:<11} -> {ev:<11} tau ({a:.3f}, {b_:.3f})  {v0:.4f} -> {v1:.4f}  ({v1 - v0:+.4f} family, "
          f"{0.15 * (v1 - v0):+.5f} total)")
    v2 = score(ev, a, .5)
    print(f"     queued-only change (tau_other left at 0.50): {v2 - v0:+.4f} family")
