"""note 169: onset composition under the official windows' condition. The official onset windows sit on days 0-4 of
the scored month, so the day-adapted CNN members have only 0-4 days of adaptation there. Members per condition K
(0..4 days of adaptation, occ_lab/pseudo_adapt.py --hist K): ps = sup+rec (BB's pseudo member), ad = rec (9b recipe);
H (LightGBM) and ST (onset STID lab cache) do not adapt per day. Each window's IoU is averaged over K = 0..4.
Window-weighted onset IoU per month; BB's weights vs a 0.1 simplex grid chosen on one month, measured on the other."""
import sys
from itertools import product
from pathlib import Path
LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB)); sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta"))
import numpy as np
import common as C
CA = LAB / "cache"
KS = [int(k) for k in (sys.argv[1] if len(sys.argv) > 1 else "0,1,2,3,4").split(",")]
D = {}
for i, p in enumerate(C.T2_PANELS):
    ctx, stat, W = C.windows(i, p)
    for s in C.SPLITS:
        on, ref = W[s]["on"], W[s]["ref"]
        if not len(on["d"]):
            continue
        base = {"H": ref["on_H"], "ST": np.load(CA / f"onset_adapt_stid30_{s}_{p}.npz")["prob"]}
        perk = [{**base, "ps": np.load(CA / f"pseudo_h{k}_supprec_{s}_{p}.npz")["on"],
                 "ad": np.load(CA / f"pseudo_h{k}_rec_{s}_{p}.npz")["on"],
                 "G0": np.mean([np.load(CA / f"pseudo_h{k}_ctrl_s{sd}_{s}_{p}.npz")["on"] for sd in (20, 21)], 0)}
                for k in KS]
        full = {**base, "ps": np.load(CA / f"pseudo_s20_{s}_{p}.npz")["on"], "ad": ref["on_ad9b"],
                "G0": np.mean([np.load(CA / f"fam_leap_ctrl_s{sd}{x}_sup+rec_{s}_{p}.npz")["on"] for sd in (20, 21) for x in ("", "_rb")], 0)}
        D.setdefault(s, []).append((on, perk, full))
K4 = ["H", "ad", "ps", "G0", "ST"]


def score(w, s, cond="early"):
    out = []
    for on, perk, full in D[s]:
        if cond == "early":
            out.append(np.mean([C.onset_iou(sum(w[k] * m[k] for k in K4 if w[k]), on) for m in perk], 0))
        else:
            out.append(C.onset_iou(sum(w[k] * full[k] for k in K4 if w[k]), on))
    return np.concatenate(out)


BB = dict(H=0.35, ad=0.175, ps=0.175, G0=0.0, ST=0.30)
for cond in ("full", "early"):
    r = {s: score(BB, s, cond) for s in C.SPLITS}
    print(f"BB weights, {cond:<5} adaptation: " + " ".join(f"{s} {r[s].mean():.4f}" for s in C.SPLITS), flush=True)
grid = [c for c in product(range(11), repeat=4) if sum(c) <= 10]
Ws = [dict(zip(K4, [x / 10 for x in c] + [(10 - sum(c)) / 10])) for c in grid]
print("single members, early: " + " ".join(f"{k} " + "/".join(f"{score({**{x: 0 for x in K4}, k: 1.0}, s, 'early').mean():.4f}" for s in C.SPLITS) for k in K4), flush=True)
for cond in ("early",):
    S = {s: [score(w, s, cond) for w in Ws] for s in C.SPLITS}
    M = {s: np.array([x.mean() for x in S[s]]) for s in C.SPLITS}
    ref = {s: score(BB, s, cond) for s in C.SPLITS}
    print(f"\n### {cond} adaptation: top 5 by the two-month mean")
    both = (M["validation"] * 169 + M["private"] * 162) / 331
    for j in np.argsort(-both)[:5]:
        print(f"  {' '.join(f'{k} {v:.1f}' for k, v in Ws[j].items()):<32} val {M['validation'][j]:.4f} pri {M['private'][j]:.4f}")
    for s_sel, s_ev in (("validation", "private"), ("private", "validation")):
        j = int(np.argmax(M[s_sel])); d = S[s_ev][j] - ref[s_ev]
        print(f"  choose on {s_sel:<10} -> {' '.join(f'{k} {v:.1f}' for k, v in Ws[j].items()):<32} measured on {s_ev} "
              f"{M[s_ev][j]:.4f} (vs BB weights {d.mean():+.4f}, SE {d.std(ddof=1) / np.sqrt(len(d)):.4f})", flush=True)
