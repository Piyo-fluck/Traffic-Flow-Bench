"""note 171: ceiling of adding CBM-proposed candidates to the set decoder's pool. Per window, oracle = the best
candidate's IoU (U). Production pool = its top-5 peaks' ratio-rule regions. CBM candidates = the CBM's top-3 peaks
(>= 4 links apart), region +-R links, shape = ratio rule on (shape source) inside the region.
    ONSET_SHARE_K=1 .venv/Scripts/python.exe occ_lab/cbm_cand_oracle.py"""
import sys, warnings
from pathlib import Path
LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB)); sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta")); sys.path.insert(0, str(LAB.parent))
import numpy as np
import common as C
import onset_stage2_du as S2
from setdec_early_check import prep
from cbm_setdec_check import cbm_prob
warnings.simplefilter("ignore")


def peaks(p, k=3, sep=4):
    q = p.copy(); out = []
    for _ in range(k):
        j = int(np.argmax(q))
        if q[j] <= 1e-4:
            break
        out.append(j); q[max(0, j - sep + 1):j + sep] = -1
    return out


def cbm_cands(p, cb, R=3):
    out = []
    S = len(p)
    for j in peaks(cb):
        m = np.zeros(S, bool); m[max(0, j - R):j + R + 1] = True
        for nm, src in (("cbm", cb), ("prod", p), ("mix", np.sqrt(np.maximum(p, 1e-9) * cb))):
            out.append((nm, C.select(src * m, "ratio", 1.0) & m))
        one = np.zeros(S, bool); one[j] = True
        out.append(("peak1", one))
    return out


res = {}
for k in range(5):
    early = prep(k)
    for p_, d in early.items():
        for s, D in d.items():
            on = D["on"]; cb = cbm_prob(s, p_)
            off = 0
            for (n, cand, reg, mass), g in zip(D["meta"], D["g"]):
                U = D["U"][off:off + g]; off += g
                t, e = on["fut"][n, -1], on["efut"][n, -1]
                u0 = float(U.max())
                best = {}
                for nm, A in cbm_cands(D["base"][n], cb[n]):
                    best[nm] = max(best.get(nm, 0.0), S2.iou_of(A, t, e))
                r = res.setdefault(s, dict(n=0, u0=0.0, **{f"+{nm}": 0.0 for nm in ("cbm", "prod", "mix", "peak1", "all")},
                                           nwin=0))
                r["n"] += 1; r["u0"] += u0
                for nm, v in best.items():
                    r[f"+{nm}"] += max(u0, v) - u0
                allv = max(best.values()) if best else 0.0
                r["+all"] += max(u0, allv) - u0
                r["nwin"] += int(allv > u0)
for s, r in res.items():
    n = r["n"]
    print(f"{s}: production-pool oracle {r['u0'] / n:.4f}; gain by adding CBM candidates: " +
          ", ".join(f"{k} {v / n:+.4f}" for k, v in r.items() if k.startswith("+")) + f"; windows improved {r['nwin'] / 5:.1f}")
