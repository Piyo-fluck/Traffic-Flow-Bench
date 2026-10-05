"""Does the queue clear from its downstream end, and does the model know?

Note 52's taxonomy of the ongoing error, and note 55's oracle for fixing each class:

    class      meaning                              cells  share of errors  error rate   oracle
    grow       not queued at T, queued later         43k     0.42 / 0.47    0.33 / 0.40  +0.008
    stay       queued at T and still queued         224k     0.27 / 0.24    0.041        +0.005
    dissolve   queued at T, cleared in the horizon  37-40k   0.17 / 0.18    0.154/0.170  +0.007
    far        never queued                        1.63M     0.13 / 0.12    0.003        +0.004

Note 54 attacked `grow` with shockwave features and recovered 6-9% of its oracle. **`dissolve` has never been
attacked** - note 55 said so in 2026-09-16 and sixty notes later it is still true.

Physically a queue clears from its head: the downstream bottleneck releases first and the tail retracts upstream.
If that is how the truth behaves and our probabilities do not track it, the ordering is a free feature. This
measures both sides of that, per horizon, against the distance from the cell to the DOWNSTREAM end of the queue
run it belongs to at T.

    python occ_lab/dissolve_anatomy.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta"))
sys.path.insert(0, str(LAB.parent / "analysis"))
import numpy as np  # noqa: E402

import common as C  # noqa: E402

CACHE = LAB / "cache"
TAU = 0.50


def edge_distance(last):
    """For each queued cell, its index from the DOWNSTREAM end of its run, and the run's length.

    Links are ordered by milepost; `graph_ident.order()` notes they are already in milepost order, so the
    downstream end of a run is the higher index in the direction of travel. Both orientations are reported by
    the caller, since the panel's direction decides which end is the head.
    """
    N, S = last.shape
    dn = np.full((N, S), -1, np.int32)
    up = np.full((N, S), -1, np.int32)
    ln = np.zeros((N, S), np.int32)
    for j in range(N):
        s = 0
        while s < S:
            if not last[j, s]:
                s += 1
                continue
            e = s
            while e + 1 < S and last[j, e + 1]:
                e += 1
            for k in range(s, e + 1):
                up[j, k] = k - s          # from the low-milepost end
                dn[j, k] = e - k          # from the high-milepost end
                ln[j, k] = e - s + 1
            s = e + 1
    return up, dn, ln


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="s20")
    args = ap.parse_args()
    acc = {}
    tot = dict(cells=0, diss=0, err=0, fp=0, fn=0)
    for i, panel in enumerate(C.T2_PANELS):
        if not (CACHE / f"ogmem_{C.SPLITS[0]}_{panel}.npz").exists():
            continue
        ctx, stat, Wd = C.windows(i, panel)
        for s in C.SPLITS:
            og, ref = Wd[s]["og"], Wd[s]["ref"]
            if not len(og["d"]):
                continue
            base = 0.2 * ref["og_gs1"] + 0.6 * ref["og_tfm8"] + 0.2 * ref["og_cnn3"]
            f = CACHE / f"pseudo_{args.tag}_{s}_{panel}.npz"
            if f.exists():
                nw = np.load(f)["og"].astype(np.float32)
                base = 0.4 * base + 0.6 * nw          # candidate AR's ongoing blend
            vis_q = og["ok"] & (np.nan_to_num(og["v"], nan=1e9) <= 0.6 * ctx.vf)
            valid = og["ok"].any(1)
            last = np.zeros(vis_q.shape[0::2], bool)
            idx = np.where(og["ok"][:, ::-1], np.arange(og["ok"].shape[1])[::-1][None, :, None], -1).max(1)
            ok_any = idx >= 0
            j0 = np.arange(vis_q.shape[0])[:, None]
            last[ok_any] = vis_q[j0.repeat(vis_q.shape[2], 1)[ok_any], idx[ok_any],
                                 np.broadcast_to(np.arange(vis_q.shape[2]), idx.shape)[ok_any]]
            up, dn, ln = edge_distance(last)
            y, e = og["fut"], og["efut"]               # (N, F, S)
            for h in range(C.F):
                m = e[:, h] & last & (ln >= 2)
                if not m.any():
                    continue
                cleared = ~y[:, h]
                for nm, dist in (("from the high-milepost end", dn), ("from the low-milepost end", up)):
                    for b in (0, 1, 2, 3):
                        sel = m & (dist == b) if b < 3 else m & (dist >= 3)
                        if sel.sum() < 30:
                            continue
                        a = acc.setdefault((nm, b, h), [0, 0.0, 0.0])
                        a[0] += int(sel.sum())
                        a[1] += float(cleared[sel].sum())              # truth: share that cleared
                        a[2] += float((1 - base[:, h])[sel].sum())     # ours: predicted share cleared
            m = e & last[:, None] & og["ok"].any(1)[:, None, None] if False else e & last[:, None]
            tot["cells"] += int(m.sum())
            tot["diss"] += int((m & ~y).sum())
            pred_q = base >= TAU
            tot["err"] += int((m & (pred_q != y)).sum())
            tot["fp"] += int((m & pred_q & ~y).sum())
            tot["fn"] += int((m & ~pred_q & y).sum())
            _ = valid
        ctx.z = None
        print(f"  {panel}", flush=True)

    print(f"\ncells queued at T: {tot['cells']:,}   of those the truth clears {tot['diss']:,} "
          f"({tot['diss'] / max(tot['cells'], 1):.3f})")
    print(f"  our errors on them {tot['err']:,} ({tot['err'] / max(tot['cells'], 1):.3f}); "
          f"kept queued but cleared {tot['fp']:,}, cleared but stayed {tot['fn']:,}")
    for nm in ("from the high-milepost end", "from the low-milepost end"):
        print(f"\nP(cleared by horizon h) against distance {nm} of the T queue run")
        print(f"  {'dist':<6}" + "".join(f"{'+' + str(5 * (h + 1)):>16}" for h in range(C.F)))
        for b in (0, 1, 2, 3):
            row = f"  {('>=3' if b == 3 else str(b)):<6}"
            for h in range(C.F):
                a = acc.get((nm, b, h))
                row += f"{'-':>16}" if a is None else f"{a[1] / a[0]:>7.3f}/{a[2] / a[0]:<8.3f}"
            print(row)
        print("   (truth / ours, per cell; a gap that grows towards dist 0 is the head-first signal)")


if __name__ == "__main__":
    main()
