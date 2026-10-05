"""Plan B': calibrate the blend on the scored month's OWN past days, instead of on the train run.

The blend weights and thresholds of candidate T were chosen on the train month: onset 0.7 LightGBM + 0.3 CNN with
the expected-IoU ratio rule at c = 1.0, ongoing 0.20 gs1 + 0.60 tfm8 + 0.20 cnn3 at tau 0.50. The organizers
confirmed (2026-09-20) that any released data at or before the window origin T may be used, and the labels of the
scored month's earlier days are released observations, so the run's own past days can calibrate those constants.

The official-like windows already cover every day of both months, and their probabilities are cached, so no new
inference is needed. For a window on day d, the calibration set is every window of the same split on days < d,
pooled over panels (the production constants are global too).

Schemes compared, per split:
    fixed        the production constants (candidate T)
    causal       for each window, the constants that maximise the mean IoU on days < d (expanding window)
    oracle       the constants fitted on the whole month, an upper bound that is NOT legal for a submission
Onset searches w_on and the ratio c, ongoing searches the three weights on a coarse simplex and tau.

    .venv/Scripts/python.exe analysis/t2_blend_calibrate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tta_lab" / "tta"))
import common as C  # noqa: E402

W_ON_GRID = (0.5, 0.6, 0.7, 0.8, 0.9)
C_GRID = (0.8, 0.9, 1.0, 1.1, 1.25)
TAU_GRID = (0.40, 0.45, 0.50, 0.55, 0.60)
OG_W_GRID = [(round(a / 10, 2), round(b / 10, 2), round(1 - a / 10 - b / 10, 2))
             for a in range(0, 7) for b in range(2, 11 - a)]        # gs1, tfm8, cnn3; tfm8 >= 0.2
MIN_DAYS = 3


def main() -> None:
    data = {s: {"on": [], "og": []} for s in C.SPLITS}
    for i, panel in enumerate(C.T2_PANELS):
        ctx, stat, W = C.windows(i, panel)
        for s in C.SPLITS:
            w, ref = W[s]["on"], W[s]["ref"]
            if len(w["d"]):
                data[s]["on"].append(dict(panel=panel, d=w["d"], H=ref["on_H"], cnn=ref["on_ad9b"],
                                          fut=w["fut"], efut=w["efut"]))
            og = W[s]["og"]
            if len(og["d"]):
                data[s]["og"].append(dict(panel=panel, d=og["d"], gs1=ref["og_gs1"], tfm8=ref["og_tfm8"],
                                          cnn3=ref["og_cnn3"], fut=og["fut"], efut=og["efut"]))
        print(f"  loaded {panel}", flush=True)

    for s in C.SPLITS:
        # ---------------- onset: per-window IoU for every (w_on, c) so the search is a lookup
        iou_on, day_on, panel_on = {}, [], []
        for p in data[s]["on"]:
            day_on.append(p["d"])
            panel_on.append(np.full(len(p["d"]), p["panel"]))
            for a in W_ON_GRID:
                pr_p = a * p["H"] + (1 - a) * p["cnn"]
                for c in C_GRID:
                    pr = np.zeros(p["fut"].shape, bool)
                    pr[:, C.F - 1] = np.stack([C.select(q, "ratio", c) for q in pr_p])
                    iou_on.setdefault((a, c), []).append(C.iou(pr, p["fut"], p["efut"]))
        iou_on = {k: np.concatenate(v) for k, v in iou_on.items()}
        day_on, panel_on = np.concatenate(day_on), np.concatenate(panel_on)
        # ---------------- ongoing
        iou_og, day_og, panel_og = {}, [], []
        for p in data[s]["og"]:
            day_og.append(p["d"])
            panel_og.append(np.full(len(p["d"]), p["panel"]))
            for wts in OG_W_GRID:
                mix = wts[0] * p["gs1"] + wts[1] * p["tfm8"] + wts[2] * p["cnn3"]
                for tau in TAU_GRID:
                    iou_og.setdefault((wts, tau), []).append(C.iou(mix >= tau, p["fut"], p["efut"]))
        iou_og = {k: np.concatenate(v) for k, v in iou_og.items()}
        day_og, panel_og = np.concatenate(day_og), np.concatenate(panel_og)

        for kind, table, days, panels, fixed in (("onset", iou_on, day_on, panel_on, (0.7, 1.0)),
                                                 ("ongoing", iou_og, day_og, panel_og, ((0.2, 0.6, 0.2), 0.5))):
            keys = list(table)
            M = np.stack([table[k] for k in keys])                 # (K, N)
            fam = lambda v: C.family_mean({p: v[panels == p].mean() for p in np.unique(panels)})  # noqa: E731
            out = {"fixed": table[fixed], "oracle": table[keys[int(np.argmax([fam(table[k]) for k in keys]))]]}
            causal = np.empty(M.shape[1])
            chosen = []
            for d in np.unique(days):
                m = days == d
                past = days < d
                if past.sum() < MIN_DAYS * 8:
                    best = keys.index(fixed)
                else:
                    best = int(np.argmax([fam_past(M[k][past], panels[past]) for k in range(len(keys))]))
                causal[m] = M[best][m]
                chosen.append(keys[best])
            out["causal"] = causal
            base = fam(out["fixed"])
            print(f"\n== {s} {kind}: fixed family {base:.4f}")
            for name in ("causal", "oracle"):
                print(f"   {name:<7} {fam(out[name]) - base:+.4f}  window {C.ci(out[name] - out['fixed'])}")
            uniq, cnt = np.unique([str(c) for c in chosen], return_counts=True)
            print("   causal picks: " + ", ".join(f"{u} x{c}" for u, c in sorted(zip(uniq, cnt), key=lambda t: -t[1])[:4]))


def fam_past(v, panels):
    return C.family_mean({p: v[panels == p].mean() for p in np.unique(panels)})


if __name__ == "__main__":
    main()
