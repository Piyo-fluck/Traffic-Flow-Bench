"""Is the ongoing blend calibrated on the cells queued at T, and how well can it discriminate at all?

`dissolve_anatomy.py` showed the head-first ordering is already tracked to within 0.005 at every horizon. What
remains is per-cell discrimination among cells the model ranks alike. If the probabilities are calibrated, no
re-weighting or feature can help: the +0.007 dissolve oracle then needs information, not a better decision.
"""
from __future__ import annotations
import sys
from pathlib import Path
LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB)); sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta"))
sys.path.insert(0, str(LAB.parent / "analysis"))
import numpy as np  # noqa: E402
import common as C  # noqa: E402
CACHE = LAB / "cache"
BINS = np.array([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0001])
acc = np.zeros((len(BINS) - 1, 3))
auc_n = [0.0, 0.0, 0.0]
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
        S = vq.shape[2]
        last = np.zeros((vq.shape[0], S), bool)
        ok = idx >= 0
        jj = np.broadcast_to(np.arange(vq.shape[0])[:, None], idx.shape)
        ss = np.broadcast_to(np.arange(S), idx.shape)
        last[ok] = vq[jj[ok], idx[ok], ss[ok]]
        m = og["efut"] & last[:, None]
        p, y = base[m], og["fut"][m]
        b = np.clip(np.digitize(p, BINS) - 1, 0, len(BINS) - 2)
        for k in range(len(BINS) - 1):
            sel = b == k
            acc[k] += [sel.sum(), p[sel].sum(), y[sel].sum()]
        pos, neg = p[y], p[~y]
        if len(pos) and len(neg):
            r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1.0
            auc_n[0] += float(r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
            auc_n[1] += len(pos) * len(neg); auc_n[2] += len(pos) + len(neg)
    ctx.z = None
    print(f"  {panel}", flush=True)
print(f"\ncalibration on the {int(acc[:, 0].sum()):,} cells queued at T (label 1 = still queued)")
print(f"  {'bin':<12}{'cells':>10}{'mean p':>9}{'observed':>10}{'gap':>8}")
for k in range(len(BINS) - 1):
    n, sp, sy = acc[k]
    if n < 100:
        continue
    print(f"  {f'{BINS[k]:.1f}-{BINS[k+1]:.1f}':<12}{int(n):>10,}{sp / n:>9.3f}{sy / n:>10.3f}{sp / n - sy / n:>+8.3f}")
print(f"\n  weighted |gap| {float(np.nansum(np.abs(acc[:,1]-acc[:,2]))/acc[:,0].sum()):.4f}")
print(f"  AUC for 'still queued' among the cells queued at T: {auc_n[0]/max(auc_n[1],1):.4f} "
      f"({int(auc_n[2]):,} cells)")
