"""note 171: ceiling of re-localising the set decoder's decision A0 (same site): best IoU over shifts of A0 by
-3..+3 links (shift), over contiguous sets of width 1..4 whose centre lies within +-3 links of A0's centre (relocate)."""
import sys, warnings
from pathlib import Path
LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB)); sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta")); sys.path.insert(0, str(LAB.parent))
import numpy as np
import common as C
import onset_stage2_du as S2
from setdec_early_check import prep
from cbm_override import decode_sets
warnings.simplefilter("ignore")
full = prep(None)
res = {}
for k in range(5):
    early = prep(k)
    for s in C.SPLITS:
        other = [x for x in C.SPLITS if x != s][0]
        sets = decode_sets(full, other, early, s)
        for p_, d in early.items():
            if s not in d:
                continue
            on = d[s]["on"]
            for n in range(len(on["d"])):
                A0 = sets[(p_, n)][0]
                t, e = on["fut"][n, -1], on["efut"][n, -1]
                u0 = S2.iou_of(A0, t, e)
                S = len(A0); aj = np.flatnonzero(A0)
                us, ur = u0, u0
                if len(aj):
                    for sh in range(-3, 4):
                        B = np.zeros(S, bool); j = aj + sh; B[j[(j >= 0) & (j < S)]] = True
                        us = max(us, S2.iou_of(B, t, e))
                    c = int(round(aj.mean()))
                    for cc in range(c - 3, c + 4):
                        for w in range(1, 5):
                            lo = cc - (w - 1) // 2
                            B = np.zeros(S, bool); B[max(0, lo):max(0, min(S, lo + w))] = True
                            ur = max(ur, S2.iou_of(B, t, e))
                r = res.setdefault((s, p_), [0, 0.0, 0.0, 0.0]); r[0] += 1; r[1] += u0; r[2] += us; r[3] += ur
for s in C.SPLITS:
    n = sum(v[0] for (s_, _), v in res.items() if s_ == s)
    tot = [sum(v[i] for (s_, _), v in res.items() if s_ == s) / n for i in (1, 2, 3)]
    print(f"{s}: setdec {tot[0]:.4f}  best shift {tot[1]:.4f} ({tot[1] - tot[0]:+.4f})  best relocate {tot[2]:.4f} ({tot[2] - tot[0]:+.4f})")
    for (s_, p_), v in res.items():
        if s_ == s:
            print(f"   {p_:<10} {v[1] / v[0]:.3f} shift {v[2] / v[0] - v[1] / v[0]:+.3f} relocate {v[3] / v[0] - v[1] / v[0]:+.3f}")
