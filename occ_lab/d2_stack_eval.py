"""D2STGNN (analysis/t2_d2stgnn.py) as a new member of AU's ongoing blend, frozen and adapted to the scored run.

AU: p0 = 0.40 old + 0.60 p_PS, old = 0.20 gs1 + 0.60 tfm8(adapted) + 0.20 cnn3, then the production residual
corrector (frozen feature columns), tau 0.50. A new kind of member entering the probability itself is what moved
the LB (AN, AP, AU), so D2STGNN enters p0; the corrector is left as it is (AV showed corrector changes do not
transfer).

    .venv/Scripts/python.exe occ_lab/d2_stack_eval.py
"""
from __future__ import annotations

import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta"))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

import common as C  # noqa: E402
from online_adapter import ETA, GATE, global_features, load_split  # noqa: E402
from refit_eval import fam_lab  # noqa: E402

CACHE = LAB / "cache"
SEEDS = range(4)


def d2(mode, s, panel):
    fs = [CACHE / f"fam_d2_s{k}_{mode}_{s}_{panel}.npz" for k in SEEDS]
    fs = [f for f in fs if f.exists()]
    return np.mean([np.load(f)["og"].astype(np.float32) for f in fs], 0) if fs else None


def main() -> None:
    gbm = lgb.Booster(model_file=str(LAB / "models" / "residual_lgb.txt"))

    def corr(base, P):
        Fe = global_features(P, base)
        r = gbm.predict(Fe.reshape(-1, 15)).reshape(base.shape).astype(np.float32)
        return np.clip(base + ETA * np.where(np.abs(r) >= GATE, r, 0.0), 0, 1)

    fam = {}
    for i, panel in enumerate(C.T2_PANELS):
        for s, D in load_split(i, panel, "ssl_w192ssl_avg4").items():
            tad = fam_lab("tfm", s, panel)
            P, ps, y, e, d = D["P"], D["ps"], D["y"], D["e"], D["d"]
            old = 0.2 * P["gs1"] + 0.6 * tad + 0.2 * P["cnn3"]
            V = {"AU": corr(0.4 * old + 0.6 * ps, P)}
            for mode in ("sup", "none"):
                g = d2(mode, s, panel)
                if g is None:
                    continue
                tag = "adapted" if mode == "sup" else "frozen"
                old2 = 0.2 * P["gs1"] + 0.6 * (0.5 * tad + 0.5 * g) + 0.2 * P["cnn3"]
                V[f"d2 {tag} alone"] = g
                V[f"d2 {tag} in the tfm slot (half)"] = corr(0.4 * old2 + 0.6 * ps, P)
                V[f"d2 {tag} 0.10 of p0"] = corr(0.4 * old + 0.5 * ps + 0.1 * g, P)
                V[f"d2 {tag} 0.20 of p0"] = corr(0.4 * old + 0.4 * ps + 0.2 * g, P)
            for nm, q in V.items():
                sc = C.iou(q >= 0.5, y, e)
                fam.setdefault((s, nm, "all"), {})[panel] = sc.mean()
                if (d <= 4).any():
                    fam.setdefault((s, nm, "early"), {})[panel] = sc[d <= 4].mean()
        print(f"  {panel}", flush=True)
    names = sorted({k[1] for k in fam}, key=lambda n: (n != "AU", n))
    for sub in ("all", "early"):
        print(f"\n=== ongoing family IoU, {sub} windows ===")
        ref = np.mean([C.family_mean(fam[(s, "AU", sub)]) for s in C.SPLITS])
        for nm in names:
            if all((s, nm, sub) in fam and len(fam[(s, nm, sub)]) == len(fam[(s, "AU", sub)]) for s in C.SPLITS):
                v = [C.family_mean(fam[(s, nm, sub)]) for s in C.SPLITS]
                print(f"  {nm:<34} val {v[0]:.4f}  priv {v[1]:.4f}  mean {np.mean(v):.4f} ({np.mean(v) - ref:+.4f})")


if __name__ == "__main__":
    main()
