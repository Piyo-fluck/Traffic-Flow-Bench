"""Can we tell, at inference, which onset windows we have completely missed?

Note 44: 62-74% of the onset loss is the 12% of windows scored IoU = 0 - the wrong bottleneck picked. Note 78
measured a same-day margin score (today's flow over capacity, [T-3h, T-1h)) and closed it as negative because its
controls matched it overall. But its own table separates the two populations:

    score                                  T-missed (n=29)   other (n=297)
    candidate T itself                          0.14              0.95
    today's flow / capacity, 3h                 0.86              0.68
    control: profile flow / capacity            0.72              0.71
    control: history-only flow / capacity       0.76              0.70

On the windows we miss it is clearly better (0.86 against 0.76 and 0.72); on the windows we already get right it
is not. Averaged over both it disappears, which is how it came to be closed. The loss is not averaged over both.

So the idea is a gate, not a replacement. Everything rests on one question, which this script answers before any
model is built: **is there a feature computable at inference that separates the IoU = 0 windows?** If not, the
whole line is dead, whatever the margin score can do.

Surface: the scored months' onset windows (the right run), the AP blend
(0.35 H + 0.175 ad9b + 0.175 pseudo + 0.30 STID) under the production ratio rule.

    python occ_lab/onset_miss_gate.py
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
from t2_expected_iou import select  # noqa: E402

CACHE = LAB / "cache"
W = (0.35, 0.175, 0.175, 0.30)          # H, ad9b, pseudo, STID - candidate AP's onset blend


def auc(x, y):
    """Rank AUC of x separating y (1 = the window was missed)."""
    m = np.isfinite(x)
    x, y = x[m], y[m]
    if y.sum() < 3 or (~y.astype(bool)).sum() < 3:
        return np.nan
    r = np.argsort(np.argsort(x)) + 1.0
    n1, n0 = y.sum(), len(y) - y.sum()
    return float((r[y.astype(bool)].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="s20")
    args = ap.parse_args()
    rows = []
    for i, panel in enumerate(C.T2_PANELS):
        f = CACHE / f"pseudo_{args.tag}_{C.SPLITS[0]}_{panel}.npz"
        if not f.exists():
            continue
        ctx, stat, Wd = C.windows(i, panel)
        for s in C.SPLITS:
            on, ref = Wd[s]["on"], Wd[s]["ref"]
            if not len(on["d"]):
                continue
            fs = CACHE / f"onset_adapt_stid30_{s}_{panel}.npz"
            if not fs.exists():
                print(f"  {panel} {s}: no onset STID cache, skipped", flush=True)
                continue
            st = np.load(fs)["prob"]
            ps = np.load(CACHE / f"pseudo_{args.tag}_{s}_{panel}.npz")["on"].astype(np.float32)
            p = W[0] * ref["on_H"] + W[1] * ref["on_ad9b"] + W[2] * ps + W[3] * st
            y, e = on["fut"][:, C.F - 1], on["efut"][:, C.F - 1]
            for j in range(len(p)):
                m = e[j]
                if not m.any() or not y[j][m].any():
                    continue
                pv = p[j][m]
                pred = select(pv, "ratio", 1.0)
                inter = int((pred & y[j][m]).sum())
                union = int((pred | y[j][m]).sum())
                iou = inter / max(union, 1)
                o = np.sort(pv)[::-1]
                rows.append(dict(panel=panel, split=s, iou=iou,
                                 # every gate feature below is computable at inference
                                 pmax=float(o[0]), psum=float(pv.sum()), k=int(pred.sum()),
                                 gap12=float(o[0] - o[1]) if len(o) > 1 else float(o[0]),
                                 top3=float(o[:3].sum() / max(pv.sum(), 1e-9)),
                                 ent=float(-(pv / max(pv.sum(), 1e-9)
                                             * np.log(pv / max(pv.sum(), 1e-9) + 1e-12)).sum()),
                                 spread=float(np.std([ref["on_H"][j][m].max(), ref["on_ad9b"][j][m].max(),
                                                      ps[j][m].max(), st[j][m].max()])),
                                 hcov=float(on["ok"][j].mean()), nlink=int(m.sum())))
        ctx.z = None
        print(f"  {panel}", flush=True)

    import pandas as pd
    d = pd.DataFrame(rows)
    miss = (d.iou == 0).astype(int)
    loss = 1 - d.iou
    print(f"\n{len(d)} onset windows, AP blend, production ratio rule")
    print(f"  mean IoU {d.iou.mean():.4f}   exact {float((d.iou == 1).mean()):.3f}   "
          f"IoU=0 {float(miss.mean()):.3f}   partial {float(((d.iou > 0) & (d.iou < 1)).mean()):.3f}")
    print(f"  share of the loss carried by the IoU=0 windows: "
          f"{float(loss[miss == 1].sum() / loss.sum()):.3f}")
    print(f"\nis a miss visible at inference?  AUC of each feature for 'this window is a complete miss'")
    print(f"  {'feature':<10}{'AUC':>8}   {'mean if missed':>15}{'mean if not':>13}")
    for c in ("pmax", "psum", "k", "gap12", "top3", "ent", "spread", "hcov", "nlink"):
        a = auc(d[c].to_numpy(float), miss.to_numpy())
        print(f"  {c:<10}{a:>8.3f}   {d.loc[miss == 1, c].mean():>15.4f}{d.loc[miss == 0, c].mean():>13.4f}")
    print("\n  AUC 0.5 = the feature says nothing. The margin-score idea needs one clearly away from 0.5.")


if __name__ == "__main__":
    main()
