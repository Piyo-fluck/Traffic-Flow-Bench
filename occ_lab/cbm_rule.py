"""note 171: few-parameter rules for acting on the CBM's proposals (rows from cbm_override.py). Per month: mean IoU
change per window (averaged over K = 0..4) and the number of windows acted on per K."""
import pickle
import sys
from pathlib import Path
import numpy as np
LAB = Path(__file__).resolve().parent
per = pickle.load(open(LAB / "cache" / "cbm_override_rows.pkl", "rb"))
SPL = ("validation", "private")


def evaluate(rule):
    out = {}
    for s in SPL:
        tot, n, act, dus = 0.0, 0, 0, []
        for k in range(5):
            for key, u0, rows in per[(s, k)]:
                best = None
                for f, du, size in rows:
                    if rule(f) and (best is None or f[2] > best[0][2]):
                        best = (f, du)
                d = best[1] if best else 0.0
                dus.append(d); act += best is not None
        dus = np.array(dus).reshape(5, -1).mean(0)
        out[s] = (dus.mean(), dus.std(ddof=1) / np.sqrt(len(dus)), act / 5)
    return out


def show(name, rule):
    r = evaluate(rule)
    print(f"  {name:<44} " + " | ".join(f"{s[:3]} {m:+.4f} (SE {se:.4f}, {a:.1f} win)" for s, (m, se, a) in r.items()))


print("ADD the CBM set (top CBM peak, no overlap with the decision), production unaware there (P1 max in set <= phi)")
for si, sn in ((0, "cbm shape"), (1, "mix shape")):
    for th in (0.3, 0.5, 0.7):
        for phi in (0.05, 0.1, 0.2):
            show(f"{sn} cb>={th} P1max<={phi}", lambda f, si=si, th=th, phi=phi: f[-1] == 1 and f[0] == 0 and f[1] == si
                 and f[2] >= th and f[6] <= phi and f[13] == 0)
print("REPLACE the decision by the CBM set when the CBM is much surer there than at the decision")
for si, sn in ((0, "cbm shape"), (1, "mix shape")):
    for dl in (0.3, 0.5, 0.7):
        show(f"{sn} cb(peak) - cb(decision) >= {dl}", lambda f, si=si, dl=dl: f[-1] == 0 and f[0] == 0 and f[1] == si
             and f[2] - f[5] >= dl)
print("REPLACE, lower margins")
for dl in (0.05, 0.1, 0.15, 0.2, 0.25):
    show(f"cbm shape margin >= {dl}", lambda f, dl=dl: f[-1] == 0 and f[0] == 0 and f[1] == 0 and f[2] - f[5] >= dl)
print("REPLACE, margin as a ratio cb(peak) / cb(decision)")
for rt in (2, 3, 5, 10):
    show(f"cbm shape ratio >= {rt}", lambda f, rt=rt: f[-1] == 0 and f[0] == 0 and f[1] == 0 and f[2] >= rt * max(f[5], 1e-3))
