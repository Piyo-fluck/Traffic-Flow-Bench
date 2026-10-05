"""Candidate A, decision side: pick each ongoing window's threshold by the expected IoU under the diffusion samples.

The production probability p is rebuilt exactly as AT ships it (occ_lab/ongoing_ablation.py: 0.40 old + 0.60 p_PS,
then the gated residual corrector), with p_PS from the note-123 SSL backbone mean. For each window the candidate
answers are {p >= tau} for tau on a grid; the diffusion model's K joint samples Y^(k) score each candidate by
mean_k IoU(candidate, Y^(k)) on the window's eligible cells, and the best tau is kept. Compared with the flat
tau 0.50 of production and with the oracle tau (chosen on the truth, the ceiling of this candidate family).

    .venv/Scripts/python.exe gen_lab/decide_eval.py --tag a_s20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent
PROJ = LAB.parent
sys.path.insert(0, str(PROJ / "occ_lab"))
sys.path.insert(0, str(PROJ / "tta_lab" / "tta"))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

import common as C  # noqa: E402
from online_adapter import ETA, GATE, global_features, load_split  # noqa: E402

TAUS = np.round(np.arange(0.20, 0.81, 0.05), 2)


def win_iou(pred, y, e):
    """pred (..., F, S) bool vs y (F, S) on eligible e; union 0 -> 1."""
    i = (pred & y & e).sum((-1, -2))
    u = ((pred | y) & e).sum((-1, -2))
    return np.where(u == 0, 1.0, i / np.maximum(u, 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", nargs="+", default=["a_s20"], help="one or more diffusion runs; samples are pooled")
    ap.add_argument("--ps-tag", default="ssl_w192ssl_avg4")
    args = ap.parse_args()
    gbm = lgb.Booster(model_file=str(PROJ / "occ_lab" / "models" / "residual_lgb.txt"))
    fam = {}
    for i, panel in enumerate(C.T2_PANELS):
        sp = load_split(i, panel, args.ps_tag)
        for s, D in sp.items():
            blend = D["base"]
            Fe = global_features(D["P"], blend)
            rh = gbm.predict(Fe.reshape(-1, Fe.shape[-1])).reshape(blend.shape).astype(np.float32)
            p = np.clip(blend + ETA * np.where(np.abs(rh) >= GATE, rh, 0.0), 0, 1)       # = AT's ongoing p
            y, e = D["y"], D["e"]
            S = p.shape[-1]
            smp = []
            for t in args.tag:
                z = np.load(LAB / "cache" / f"diff_{t}_{s}_{panel}.npz")
                smp.append(np.unpackbits(z["og_bits"], axis=-1, count=S).astype(bool))
            smp = np.concatenate(smp, 1)                                 # (N, K, F, S)
            cand = p[:, None] >= TAUS[None, :, None, None]               # (N, n_tau, F, S)
            flat, gen_pick, oracle = [], [], []
            for n in range(len(p)):
                c = cand[n]                                               # (n_tau, F, S)
                exp = np.stack([win_iou(c, smp[n, k], e[n]) for k in range(smp.shape[1])]).mean(0)
                true = win_iou(c, y[n], e[n])
                flat.append(true[list(TAUS).index(0.5)])
                gen_pick.append(true[int(np.argmax(exp))])
                oracle.append(true.max())
            for nm, v in (("flat tau 0.50 (= AT)", flat), ("tau by diffusion E[IoU]", gen_pick),
                          ("oracle tau (ceiling)", oracle)):
                fam.setdefault((s, nm), {})[panel] = float(np.mean(v))
        print(f"  {panel}", flush=True)
    print(f"\nongoing family IoU, official-like pool; diffusion runs {args.tag}\n")
    names = ["flat tau 0.50 (= AT)", "tau by diffusion E[IoU]", "oracle tau (ceiling)"]
    print(f"{'decision':<26}" + "".join(f"{s:>13}" for s in C.SPLITS) + f"{'mean':>10}")
    for nm in names:
        v = [C.family_mean(fam[(s, nm)]) for s in C.SPLITS]
        print(f"{nm:<26}" + "".join(f"{x:>13.4f}" for x in v) + f"{np.mean(v):>10.4f}")


if __name__ == "__main__":
    main()
