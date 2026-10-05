"""OneNet-style online ensembling (Zhang et al., NeurIPS 2023) on AU's ongoing stack: run-level member weights by
exponentiated gradient (Hedge) on the losses of the windows released so far, as opposed to note 141's per-window
selection.

For window n at origin T on day d, the released windows are the lab windows of earlier days and of day d with the
horizon ended before T. Their per-member Brier (per horizon step), discounted by gamma^(age in days), gives
    w_{m,h} ~ w_AU,m-prior * exp(-eta * cumulative discounted Brier_{m,h} / n_eff)
p0 = sum_m w p_m, then the production corrector. The same seven members and prior as occ_lab/select_bank.py.

    .venv/Scripts/python.exe occ_lab/onenet_eval.py
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
import select_bank as SB  # noqa: E402
from online_adapter import ETA, GATE, global_features  # noqa: E402

F = C.F


def main() -> None:
    gbm = lgb.Booster(model_file=str(LAB / "models" / "residual_lgb.txt"))

    def corr(base, P):
        Fe = global_features(P, base)
        r = gbm.predict(Fe.reshape(-1, 15)).reshape(base.shape).astype(np.float32)
        return np.clip(base + ETA * np.where(np.abs(r) >= GATE, r, 0.0), 0, 1)

    fam = {}
    for i, panel in enumerate(C.T2_PANELS):
        for s in C.SPLITS:
            B = SB.bank_for(i, panel, s, ps_tag="")                   # shipped p_PS, as AU
            M, d, t0 = B["M"], B["d"], B["t0"]
            yb, eb = B["y"] > 0.5, B["e"] > 0.5
            per = B["brier"].sum(3) / np.maximum(B["e"].sum(2)[None], 1)   # (K, N, F) per-window Brier
            V = {"AU": corr(np.tensordot(SB.WAU, M, 1), B["P"])}
            for eta, gamma in ((20.0, 1.0), (50.0, 1.0), (50.0, 0.7), (100.0, 0.7)):
                p0 = np.tensordot(SB.WAU, M, 1)
                for n in range(len(d)):
                    past = np.flatnonzero((d < d[n]) | ((d == d[n]) & (t0 + F < t0[n])))
                    if not len(past):
                        continue
                    age = (d[n] - d[past]).astype(float)
                    wt = gamma ** age
                    L = (per[:, past] * wt[None, :, None]).sum(1) / wt.sum()   # (K, F)
                    w = SB.PRIOR[:, None] * np.exp(-eta * (L - L.min(0, keepdims=True)))
                    w = w / w.sum(0, keepdims=True)
                    sh = len(past) / (len(past) + 30.0)
                    w = sh * w + (1 - sh) * SB.WAU[:, None]
                    p0[n] = np.einsum("kf,kfs->fs", w, M[:, n])
                V[f"hedge eta{eta:g} gamma{gamma:g}"] = corr(p0, B["P"])
            for nm, q in V.items():
                sc = C.iou(q >= 0.5, yb, eb)
                fam.setdefault((s, nm, "all"), {})[panel] = sc.mean()
                fam.setdefault((s, nm, "days 1-5"), {})[panel] = sc[d <= 4].mean()
        print(f"  {panel}", flush=True)
    names = list(dict.fromkeys(k[1] for k in fam))
    for sub in ("all", "days 1-5"):
        print(f"\n=== ongoing family IoU, {sub} ===")
        ref = {s: C.family_mean(fam[(s, "AU", sub)]) for s in C.SPLITS}
        for nm in names:
            v = {s: C.family_mean(fam[(s, nm, sub)]) for s in C.SPLITS}
            print(f"  {nm:<24}" + "".join(f"  {s[:4]} {v[s]:.4f} ({v[s] - ref[s]:+.4f})" for s in C.SPLITS))


if __name__ == "__main__":
    main()
