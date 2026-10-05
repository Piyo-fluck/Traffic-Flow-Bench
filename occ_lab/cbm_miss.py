"""note 171: the 'missed site' windows (no production candidate's ratio-rule region overlaps the truth), K = 0.
Where is the truth relative to production's candidates, and does the CBM point at it?"""
import sys, warnings
from pathlib import Path
LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB)); sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta")); sys.path.insert(0, str(LAB.parent))
import numpy as np
import common as C
from setdec_early_check import prep
from cbm_setdec_check import cbm_prob
warnings.simplefilter("ignore")
early = prep(0)
rows = []
for p_, d in early.items():
    for s, D in d.items():
        on = D["on"]; cb = cbm_prob(s, p_)
        off = 0
        for (n, cand, reg, mass), g in zip(D["meta"], D["g"]):
            U = D["U"][off:off + g]; off += g
            if U.max() > 0:
                continue
            t = on["fut"][n, -1] & on["efut"][n, -1]
            tj = np.flatnonzero(t)
            if not len(tj):
                continue
            p = D["base"][n]
            dist = int(np.min(np.abs(tj[:, None] - cand[None, :])))
            rank_p = int((p > p[tj].max()).sum())                  # links with a higher production prob than the truth
            rank_c = int((cb[n] > cb[n][tj].max()).sum())          # same for the CBM
            rows.append((s, p_, n, len(tj), dist, rank_p, rank_c, float(p[tj].max()), float(cb[n][tj].max()), len(p)))
print("split panel win  ntrue  dist-to-nearest-cand  rank(prod)  rank(CBM)  p_prod  p_cbm  S")
for r in rows:
    print(f"{r[0][:3]} {r[1]:<10} {r[2]:>3} {r[3]:>4} {r[4]:>5} {r[5]:>6} {r[6]:>6}  {r[7]:.3f} {r[8]:.3f} {r[9]}")
