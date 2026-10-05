"""Any new ongoing member in AU's p0 at small weights, with rescue / spoil counts (the 'enter at 0.05' check).

AU: p0 = 0.40 old + 0.60 p_PS, old = 0.20 gs1 + 0.60 tfm8(adapted) + 0.20 cnn3, then the production residual
corrector, tau 0.50. A member X enters as p0' = (1 - w) p0 + w X before the corrector.

    .venv/Scripts/python.exe occ_lab/member_stack_eval.py fam_hydra_r0_run1000 fam_hydra_r0_none ...
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
WS = (0.05, 0.10, 0.20)


def main() -> None:
    names = sys.argv[1:]
    gbm = lgb.Booster(model_file=str(LAB / "models" / "residual_lgb.txt"))

    def corr(base, P):
        Fe = global_features(P, base)
        r = gbm.predict(Fe.reshape(-1, 15)).reshape(base.shape).astype(np.float32)
        return np.clip(base + ETA * np.where(np.abs(r) >= GATE, r, 0.0), 0, 1)

    fam, cnt = {}, {}
    for i, panel in enumerate(C.T2_PANELS):
        for s, D in load_split(i, panel, "ssl_w192ssl_avg4").items():
            tad = fam_lab("tfm", s, panel)
            P, ps, y, e, d = D["P"], D["ps"], D["y"], D["e"], D["d"]
            p0 = 0.4 * (0.2 * P["gs1"] + 0.6 * tad + 0.2 * P["cnn3"]) + 0.6 * ps
            au = corr(p0, P)
            b_sc = C.iou(au >= 0.5, y, e)
            fam.setdefault((s, "AU"), {})[panel] = b_sc.mean()
            for nm in names:
                f = CACHE / f"{nm}_{s}_{panel}.npz"
                if not f.exists():
                    continue
                X = np.load(f)["og"].astype(np.float32)
                fam.setdefault((s, f"{nm} alone"), {})[panel] = C.iou(X >= 0.5, y, e).mean()
                for w in WS:
                    sc = C.iou(corr((1 - w) * p0 + w * X, P) >= 0.5, y, e)
                    k = f"{nm} w{w}"
                    fam.setdefault((s, k), {})[panel] = sc.mean()
                    c = cnt.setdefault(k, [0, 0, 0])
                    c[0] += int((sc > b_sc + 1e-9).sum()); c[1] += int((sc < b_sc - 1e-9).sum()); c[2] += len(sc)
        print(f"  {panel}", flush=True)
    keys = ["AU"] + [f"{n} alone" for n in names] + [f"{n} w{w}" for n in names for w in WS]
    ref = {s: C.family_mean(fam[(s, "AU")]) for s in C.SPLITS}
    print(f"\n{'ongoing family IoU':<40}{'validation':>18}{'private':>18}   windows better / worse")
    for k in keys:
        if not all((s, k) in fam and len(fam[(s, k)]) == len(fam[(s, "AU")]) for s in C.SPLITS):
            continue
        cells = "".join(f"{C.family_mean(fam[(s, k)]):>10.4f} ({C.family_mean(fam[(s, k)]) - ref[s]:+.4f})"
                        for s in C.SPLITS)
        c = cnt.get(k)
        print(f"{k:<40}{cells}" + (f"   {c[0]} / {c[1]} of {c[2]}" if c else ""))


if __name__ == "__main__":
    main()
