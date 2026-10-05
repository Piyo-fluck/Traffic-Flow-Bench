"""note 171: localisation inside the chosen region with the CBM. After BB's set decoder stage 1 (production-like,
cross-month rankers), the chosen region's probability mass is kept but redistributed over its links:
    w_j = P1_j^(1-beta) * cbm_j^beta, P1'_j = mass_region * w_j / sum w   (other regions untouched)
The region choice is unchanged; only which links / how many the ratio rule picks inside it can change.

    ONSET_SHARE_K=1 .venv/Scripts/python.exe occ_lab/cbm_loc.py
"""
import sys, warnings
from pathlib import Path
LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB)); sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta")); sys.path.insert(0, str(LAB.parent))
import numpy as np
import common as C
from onset_setdec_fit import SEEDS, fit_rank, apply_rank
from setdec_early_check import prep
from cbm_setdec_check import cbm_prob
warnings.simplefilter("ignore")
BETAS = (0.25, 0.5, 0.75, 1.0)


def localise(P1, meta, cb, beta):
    Q = P1.copy()
    for n, cand, reg, mass in meta:
        p = P1[n]
        m = np.bincount(reg, weights=p, minlength=len(cand))
        r = int(np.argmax(m))
        sel = reg == r
        w = np.maximum(p[sel], 1e-9) ** (1 - beta) * np.maximum(cb[n][sel], 1e-9) ** beta
        Q[n, sel] = m[r] * w / w.sum()
    return np.clip(Q, 0, 1)


def main():
    full = prep(None)
    rk = {s: [fit_rank([d[s] for d in full.values() if s in d], sd) for sd in SEEDS] for s in C.SPLITS}
    res = {}
    for k in range(5):
        early = prep(k)
        for s in C.SPLITS:
            other = [x for x in C.SPLITS if x != s][0]
            for p_, d in early.items():
                if s not in d:
                    continue
                D = d[s]; on = D["on"]
                apply_rank(rk[other], D)
                cb = cbm_prob(s, p_)
                res.setdefault((s, "stage1"), []).append(C.onset_iou(D["P1"], on))
                for b in BETAS:
                    res.setdefault((s, f"loc beta {b}"), []).append(C.onset_iou(localise(D["P1"], D["meta"], cb, b), on))
                    res.setdefault((s, p_, f"loc beta {b}"), []).append(
                        C.onset_iou(localise(D["P1"], D["meta"], cb, b), on) - C.onset_iou(D["P1"], on))
    for s in C.SPLITS:
        base = np.concatenate(res[(s, "stage1")])
        n = len(base) // 5
        print(f"\n{s}: stage 1 IoU {base.mean():.4f}")
        for b in BETAS:
            a = np.concatenate(res[(s, f"loc beta {b}")])
            dd = (a - base).reshape(5, -1).mean(0) if False else a - base
            print(f"  loc beta {b:<5} {a.mean():.4f} ({dd.mean():+.4f}, SE {dd.std(ddof=1) / np.sqrt(n):.4f})  "
                  + " ".join(f"{p_[3:]} {np.concatenate(res[(s, p_, f'loc beta {b}')]).mean():+.3f}"
                             for p_ in C.T2_PANELS if (s, p_, f"loc beta {b}") in res))


if __name__ == "__main__":
    main()
