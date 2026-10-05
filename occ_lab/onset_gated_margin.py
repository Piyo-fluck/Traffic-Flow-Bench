"""The decisive test of note 78's re-reading: split by an OBSERVABLE gate, not by the true miss flag.

Note 78 measured the same-day margin score by splitting windows into "T missed this one" and "T did not", and
closed it as negative because its controls matched it on average. But the miss flag is not available at inference,
and the split was where the whole argument lived: on the missed windows today's flow/capacity picked the true
bottleneck 0.86 of the time against 0.76 and 0.72 for its controls, while T itself managed 0.14.

`onset_miss_gate.py` then found that a miss IS visible at inference: the blend's own maximum probability in the
window has AUC 0.218 for "this window is a complete miss" (0.70 when missed against 0.81 when not).

So this asks the only question that matters. Among the windows a **pmax gate** flags - which we can actually
select at inference - does the margin score pick the true bottleneck better than the blend does? The gate is
applied per panel, because misses also concentrate on the smaller corridors (nlink AUC 0.314) and a global cut
would partly just be selecting those.

    python occ_lab/onset_gated_margin.py [--q 0.15]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta"))
sys.path.insert(0, str(LAB.parent / "analysis"))
sys.path.insert(0, str(LAB.parent / "mm_lab" / "mm"))
import numpy as np  # noqa: E402

import common as C  # noqa: E402
from diag_123 import hits, nms_peaks  # noqa: E402
from t2_expected_iou import select  # noqa: E402
from t2_onset_today_demand import SEP, today_scores  # noqa: E402
from tfb.physics import load_ramps  # noqa: E402

CACHE = LAB / "cache"
WB = (0.35, 0.175, 0.175, 0.30)          # candidate AP's onset blend
SCORES = ["qcap", "f3h", "load3h", "prof_qcap", "hist_qcap"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="s20")
    ap.add_argument("--q", type=float, default=0.15, help="gate: the lowest q of pmax, per panel")
    args = ap.parse_args()
    rows = []
    for i, panel in enumerate(C.T2_PANELS):
        if not (CACHE / f"pseudo_{args.tag}_{C.SPLITS[0]}_{panel}.npz").exists():
            continue
        ctx, stat, Wd = C.windows(i, panel)
        for s in C.SPLITS:
            w, z, ref = Wd[s]["on"], Wd[s]["z"], Wd[s]["ref"]
            if not len(w["d"]) or not (CACHE / f"onset_adapt_stid30_{s}_{panel}.npz").exists():
                continue
            st = np.load(CACHE / f"onset_adapt_stid30_{s}_{panel}.npz")["prob"]
            ps = np.load(CACHE / f"pseudo_{args.tag}_{s}_{panel}.npz")["on"].astype(np.float32)
            p = WB[0] * ref["on_H"] + WB[1] * ref["on_ad9b"] + WB[2] * ps + WB[3] * st
            ramps = load_ramps(panel, s, z)
            sc = today_scores(panel, z, ctx, w["d"], w["t0"], ramps)
            with np.errstate(all="ignore"):
                sc["hist_qcap"] = np.nan_to_num(np.nanmean(np.where(w["ok"], w["qf"], np.nan), 1)
                                                / np.maximum(ctx.cap[None, :], 1.0), nan=0.0)
            iou = C.onset_iou(p, w)
            for j in range(len(w["d"])):
                el, tr = w["efut"][j, C.F - 1], w["cells"][j]
                if not el.any() or not tr.any():
                    continue
                cand = [c for c in nms_peaks(p[j], el, 3, SEP)]
                if len(cand) < 2 or not any(hits(c, tr, SEP) for c in cand):
                    continue          # the truth is not among the blend's top 3: no re-ranking can help
                r = dict(panel=panel, split=s, iou=float(iou[j]),
                         pmax=float(p[j][el].max()), blend_hit=int(hits(cand[0], tr, SEP)))
                for nm in SCORES:
                    v = np.nan_to_num(sc[nm][j])
                    val = [v[max(0, c - 1):c + 2].max() for c in cand]
                    r[nm] = int(hits(cand[int(np.argmax(val))], tr, SEP))
                rows.append(r)
        ctx.z = None
        print(f"  {panel}", flush=True)

    import pandas as pd
    d = pd.DataFrame(rows)
    d["cut"] = d.groupby("panel").pmax.transform(lambda x: x.quantile(args.q))
    d["gated"] = d.pmax <= d.cut
    print(f"\n{len(d)} windows whose truth is inside the blend's top 3 "
          f"(re-ranking cannot help the others)\n")
    print(f"{'group':<26}{'n':>5}{'mean IoU':>10}{'blend':>8}"
          + "".join(f"{s:>11}" for s in SCORES))
    for name, g in (("gate fires (low pmax)", d[d.gated]), ("gate quiet", d[~d.gated]), ("all", d)):
        if not len(g):
            continue
        print(f"{name:<26}{len(g):>5}{g.iou.mean():>10.3f}{g.blend_hit.mean():>8.3f}"
              + "".join(f"{g[s].mean():>11.3f}" for s in SCORES))
    print(f"\ngate = the lowest {args.q:.0%} of pmax within each panel")
    print("a usable result: on the gated group a same-day score (qcap / f3h / load3h) beats BOTH the blend and")
    print("the two controls (prof_qcap, hist_qcap), which carry no same-day information.")


if __name__ == "__main__":
    main()
