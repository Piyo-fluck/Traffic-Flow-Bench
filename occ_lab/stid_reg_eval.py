"""note 169: onset STID regularization study. Per variant (lab cache onset_adapt_<tag>_<split>_<panel>): the STID alone
and inside BB's blend (0.35 H + 0.175 ad9b + 0.175 ps + 0.30 STID, ratio rule, no set decoder) with the day-adapted
members under the official windows' condition (0-4 days of adaptation, IoU averaged over K = 0..4). Window-weighted
onset IoU per month with the window-level SE of the difference to the retrained baseline (stid_Ss0).

    .venv/Scripts/python.exe occ_lab/stid_reg_eval.py stid_Ss0 stid_idl30s0 ...
"""
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta"))
import numpy as np  # noqa: E402

import common as C  # noqa: E402

CA = LAB / "cache"
TAGS = sys.argv[1:] or ["stid_Ss0"]
KS = range(5)
R = {}
for i, p in enumerate(C.T2_PANELS):
    ctx, stat, W = C.windows(i, p)
    for s in C.SPLITS:
        on, ref = W[s]["on"], W[s]["ref"]
        if not len(on["d"]):
            continue
        H = ref["on_H"]
        others = [0.35 * H + 0.175 * np.load(CA / f"pseudo_h{k}_rec_{s}_{p}.npz")["on"]
                  + 0.175 * np.load(CA / f"pseudo_h{k}_supprec_{s}_{p}.npz")["on"] for k in KS]
        for t in TAGS:
            st = np.load(CA / f"onset_adapt_{t}_{s}_{p}.npz")["prob"]
            R.setdefault((t, s, "alone"), []).append(C.onset_iou(st, on))
            R.setdefault((t, s, "blend"), []).append(np.mean([C.onset_iou(o + 0.30 * st, on) for o in others], 0))
base = TAGS[0]
for kind in ("alone", "blend"):
    print(f"\n### STID {kind} (window-weighted; diff to {base}, SE)")
    for t in TAGS:
        cells = []
        for s in C.SPLITS:
            a, c = np.concatenate(R[(t, s, kind)]), np.concatenate(R[(base, s, kind)])
            d = a - c
            se = d.std(ddof=1) / np.sqrt(len(d)) if d.std() > 0 else 0.0
            cells.append(f"{s} {a.mean():.4f} ({d.mean():+.4f}, SE {se:.4f})")
        print(f"  {t:<16} " + " | ".join(cells), flush=True)
