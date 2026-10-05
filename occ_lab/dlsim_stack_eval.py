"""Note 170: the DLSim physics expert (occ_lab/dlsim_expert.py) in AU's stacks, scored-month lab windows.

Ongoing: AU = residual corrector on 0.4 old + 0.6 p_PS (lc_stack_eval); the expert enters (a) after the corrector,
(1 - w) AU + w sim, or (b) inside p0 at weight w taken from p_PS. Onset: the AP ratio-rule blend (bottleneck_cls
base) + w sim. Family IoU per split; single-model IoU of the expert and of persistence for reference.

    .venv/Scripts/python.exe occ_lab/dlsim_stack_eval.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB)); sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta"))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

import bottleneck_cls as BC  # noqa: E402
import common as C  # noqa: E402
from online_adapter import ETA, GATE, global_features, load_split  # noqa: E402
from refit_eval import fam_lab  # noqa: E402

CACHE = LAB / "cache"


def main():
    warnings.simplefilter("ignore")
    gbm = lgb.Booster(model_file=str(LAB / "models" / "residual_lgb.txt"))

    def corr(base, P):
        Fe = global_features(P, base)
        r = gbm.predict(Fe.reshape(-1, 15)).reshape(base.shape).astype(np.float32)
        return np.clip(base + ETA * np.where(np.abs(r) >= GATE, r, 0.0), 0, 1)

    fam, onf = {}, {}
    for i, panel in enumerate(C.T2_PANELS):
        B = BC.build(i, panel)
        for s, D in load_split(i, panel, "ssl_w192ssl_avg4").items():
            sim = np.load(CACHE / f"dlsim_{s}_{panel}.npz")
            so, sn = sim["og"], sim["on"]
            tad = fam_lab("tfm", s, panel)
            P, ps, y, e = D["P"], D["ps"], D["y"], D["e"]
            old = 0.2 * P["gs1"] + 0.6 * tad + 0.2 * P["cnn3"]
            au = corr(0.4 * old + 0.6 * ps, P)
            V = {"AU": au, "sim alone": so, "persistence": np.broadcast_to(D["qT"][:, None], so.shape).astype(np.float32)}
            for w in (0.1, 0.2, 0.3):
                V[f"after corrector w {w}"] = (1 - w) * au + w * so
            for w in (0.1, 0.2):
                V[f"in p0 w {w}"] = corr(0.4 * old + (0.6 - w) * ps + w * so, P)
            dl = so - sim["og_prev"][:, None]                           # the simulated CHANGE from T-5
            dm = np.clip(D["qT"][:, None].astype(np.float32) + dl, 0, 1)
            V["delta member alone"] = dm
            for w in (0.1, 0.2):
                V[f"delta member after corr w {w}"] = (1 - w) * au + w * dm
                V[f"delta member in p0 w {w}"] = corr(0.4 * old + (0.6 - w) * ps + w * dm, P)
            for w in (0.1, 0.2, 0.3, 0.5):
                V[f"AU + {w} x change"] = np.clip(au + w * dl, 0, 1)
            for nm, q in V.items():
                fam.setdefault((s, nm), {})[panel] = C.iou(q >= 0.5, y, e).mean()
            if s in B:
                base, on = B[s]["base"], B[s]["on"]
                for w in (0.0, 0.1, 0.2, 0.3):
                    onf.setdefault((s, w), {})[panel] = C.onset_iou(base + w * sn, on).mean()
                grow = np.clip(sn - sim["on_prev"], 0, 1)
                for w in (0.05, 0.1, 0.2):
                    onf.setdefault((s, f"growth {w}"), {})[panel] = C.onset_iou(base + w * grow, on).mean()
                onf.setdefault((s, "sim alone"), {})[panel] = C.onset_iou(sn, on).mean()
        print(f"  {panel}", flush=True)
    print("\n=== ongoing family IoU (tau 0.5) ===")
    ref = [C.family_mean(fam[(s, "AU")]) for s in C.SPLITS]
    for nm in sorted({k[1] for k in fam}, key=lambda n: (n != "AU", n)):
        v = [C.family_mean(fam[(s, nm)]) for s in C.SPLITS]
        print(f"  {nm:<24} val {v[0]:.4f} ({v[0] - ref[0]:+.4f})  priv {v[1]:.4f} ({v[1] - ref[1]:+.4f})")
    print("\n=== onset family IoU, AP ratio-rule blend + w sim ===")
    for w in (0.0, 0.1, 0.2, 0.3, "sim alone", "growth 0.05", "growth 0.1", "growth 0.2"):
        v = [C.family_mean(onf[(s, w)]) for s in C.SPLITS]
        print(f"  w {w!s:<10} val {v[0]:.4f}  priv {v[1]:.4f}")


if __name__ == "__main__":
    main()
