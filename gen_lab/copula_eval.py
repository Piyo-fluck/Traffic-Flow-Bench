"""Candidate A, copula decision: production marginals, diffusion correlation.

decide_eval.py let the diffusion samples judge AT's probability p, and lost (0.8606 -> 0.8360): the samples come
from a model that alone scores 0.77-0.79, so they are a worse posterior than p itself. Here the samples keep only
their dependence structure. Per window and cell, the K continuous samples of v/vf are ranked (slowest first) into
u in (0, 1); a copula sample is queued where u < p. Every cell's marginal is then exactly p, and which cells are
slow TOGETHER comes from the diffusion model. Each window's tau is chosen by the expected IoU under these samples.

Controls, same marginals:
    independent   u drawn i.i.d. uniform per cell (no dependence at all)
    shuffled      the diffusion ranks permuted across samples independently per cell (dependence destroyed,
                  rank marginals kept)

    .venv/Scripts/python.exe gen_lab/copula_eval.py --tag c_s20 c_s21
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
import numpy as np  # noqa: E402

import decide_eval as D  # noqa: E402

C = D.C
TAUS = D.TAUS
I5 = list(TAUS).index(0.5)


def ranks(x, rng):
    """x (K, F, S) -> u in (0,1): rank of each sample among the K at that cell, slowest first, random ties."""
    K = x.shape[0]
    key = x.astype(np.float32) + rng.random(x.shape, dtype=np.float32) * 1e-3
    r = np.argsort(np.argsort(key, 0), 0)
    return (r + 0.5) / K


def pick(cand, samples, e):
    ex = np.stack([D.win_iou(cand, samples[k], e) for k in range(len(samples))]).mean(0)
    return int(np.argmax(ex))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", nargs="+", default=["c_s20", "c_s21"])
    ap.add_argument("--ps-tag", default="ssl_w192ssl_avg4")
    args = ap.parse_args()
    gbm = D.lgb.Booster(model_file=str(D.PROJ / "occ_lab" / "models" / "residual_lgb.txt"))
    rng = np.random.default_rng(0)
    names = ["flat tau 0.50 (= AT)", "copula (diffusion dependence)", "control: independent",
             "control: shuffled ranks", "oracle tau (ceiling)"]
    fam = {}
    for i, panel in enumerate(C.T2_PANELS):
        sp = D.load_split(i, panel, args.ps_tag)
        for s, Dd in sp.items():
            blend = Dd["base"]
            Fe = D.global_features(Dd["P"], blend)
            rh = gbm.predict(Fe.reshape(-1, Fe.shape[-1])).reshape(blend.shape).astype(np.float32)
            p = np.clip(blend + D.ETA * np.where(np.abs(rh) >= D.GATE, rh, 0.0), 0, 1)
            y, e = Dd["y"], Dd["e"]
            xs = np.concatenate([np.load(LAB / "cache" / f"diffc_{t}_{s}_{panel}.npy") for t in args.tag], 1)
            cand = p[:, None] >= TAUS[None, :, None, None]
            out = {n: [] for n in names}
            for n in range(len(p)):
                c, en = cand[n], e[n]
                true = D.win_iou(c, y[n], en)
                u = ranks(xs[n], rng)                                   # (K, F, S)
                K = len(u)
                cop = u < p[n][None]
                ind = rng.random(u.shape) < p[n][None]
                perm = np.argsort(rng.random(u.shape), 0)               # independent permutation per cell
                shf = np.take_along_axis(u, perm, 0) < p[n][None]
                out["flat tau 0.50 (= AT)"].append(true[I5])
                out["copula (diffusion dependence)"].append(true[pick(c, cop, en)])
                out["control: independent"].append(true[pick(c, ind, en)])
                out["control: shuffled ranks"].append(true[pick(c, shf, en)])
                out["oracle tau (ceiling)"].append(true.max())
            for nm in names:
                fam.setdefault((s, nm), {})[panel] = float(np.mean(out[nm]))
        print(f"  {panel}", flush=True)
    print(f"\nongoing family IoU, official-like pool; copula of runs {args.tag} on AT's p\n")
    print(f"{'decision':<32}" + "".join(f"{s:>13}" for s in C.SPLITS) + f"{'mean':>10}")
    for nm in names:
        v = [C.family_mean(fam[(s, nm)]) for s in C.SPLITS]
        print(f"{nm:<32}" + "".join(f"{x:>13.4f}" for x in v) + f"{np.mean(v):>10.4f}")


if __name__ == "__main__":
    main()
