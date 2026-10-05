"""General Kaggle techniques on the SSL w192 CNN (occ_lab/leap_t2.py, seed 20): single member and in AU's stack.

Each variant's adapted s20 model replaces p_PS s20 (the other three p_PS seeds unchanged), against the control
(ctrl_s20: the same whole-panel fine-tuning without the technique). All windows and days 1-5.

    .venv/Scripts/python.exe occ_lab/kag_stack_eval.py ctrl_s20 adv_s20 adv2_s20 full_s20 distill_s20 ...
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


def main() -> None:
    tags = sys.argv[1:]
    gbm = lgb.Booster(model_file=str(LAB / "models" / "residual_lgb.txt"))

    def corr(base, P):
        Fe = global_features(P, base)
        r = gbm.predict(Fe.reshape(-1, 15)).reshape(base.shape).astype(np.float32)
        return np.clip(base + ETA * np.where(np.abs(r) >= GATE, r, 0.0), 0, 1)

    fam = {}
    for i, panel in enumerate(C.T2_PANELS):
        for s, D in load_split(i, panel, "ssl_w192ssl_avg4").items():
            P, y, e, d = D["P"], D["y"], D["e"], D["d"]
            tad = fam_lab("tfm", s, panel)
            ps = {k: np.load(CACHE / f"pseudo_ssl_t2_cnn_lov_w192_rec_ssl30k_s{k}_{s}_{panel}.npz")["og"].astype(np.float32)
                  for k in (20, 21, 22, 23)}
            old = 0.2 * P["gs1"] + 0.6 * tad + 0.2 * P["cnn3"]
            V = {"AU": corr(0.4 * old + 0.6 * np.mean(list(ps.values()), 0), P)}
            for t in tags:
                X = np.load(CACHE / f"fam_leap_{t}_sup+rec_{s}_{panel}.npz")["og"].astype(np.float32)
                V[f"{t} alone"] = X
                V[f"{t} in stack"] = corr(0.4 * old + 0.6 * np.mean([X, ps[21], ps[22], ps[23]], 0), P)
            for nm, q in V.items():
                sc = C.iou(q >= 0.5, y, e)
                fam.setdefault((s, nm, "all"), {})[panel] = sc.mean()
                if (d <= 4).any():
                    fam.setdefault((s, nm, "days 1-5"), {})[panel] = sc[d <= 4].mean()
        print(f"  {panel}", flush=True)
    for sub in ("all", "days 1-5"):
        print(f"\n=== ongoing family IoU, {sub} (val / priv / mean; stack rows vs AU) ===")
        ref = np.mean([C.family_mean(fam[(s, "AU", sub)]) for s in C.SPLITS])
        for nm in ["AU"] + [f"{t} alone" for t in tags] + [f"{t} in stack" for t in tags]:
            v = [C.family_mean(fam[(s, nm, sub)]) for s in C.SPLITS]
            extra = f"  ({np.mean(v) - ref:+.4f})" if "stack" in nm or nm == "AU" else ""
            print(f"  {nm:<24} {v[0]:.4f} / {v[1]:.4f} / {np.mean(v):.4f}{extra}")


if __name__ == "__main__":
    main()
