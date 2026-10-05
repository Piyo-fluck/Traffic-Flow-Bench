"""Per-window, per-horizon member selection v2 on AU's ongoing stack (causal within the run), optionally with the
SOLID v2 p_PS (occ_lab/adapt_variants.py --method solid2) in p_PS's place.

For window n at origin T on day d: the lab windows of earlier days and those of the same day whose horizon ends
before T (all their labels are released at or before T):
    neighbours   the k earlier windows whose last-15-min v/vf is closest on the target's queue neighbourhood
                 (links queued in its last 15 min +- 8), cosine, minus a time-of-day distance
    competence   each member's Brier score per horizon step on the neighbours' eligible cells
    weights      w_{m,h} ~ prior_m exp(-beta (B_{m,h} / min_m B_{m,h} - 1)), prior = AU's p0 weights plus 0.02 for
                 the members AU does not blend (occ_cnn, occ_tfm, stid); shrunk to the prior by n / (n + kappa)
    answer       p0 = sum_m w_{m,h} p_m, then the production residual corrector, tau 0.50

    .venv/Scripts/python.exe occ_lab/select_v2.py [solid2]
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
F = C.F


def neighbourhood_sig(v, ok, vf, radius=8):
    vr = np.where(ok, v / vf, 1.0)[:, -3:, :].astype(np.float32)          # (N, 3, S)
    q = (np.where(ok, v, 1e9)[:, -3:, :] <= 0.6 * vf).any(1)             # (N, S)
    nb = np.zeros_like(q)
    for r in range(-radius, radius + 1):
        nb |= np.roll(q, r, axis=1)
    return vr, nb


def main() -> None:
    solid_tag = sys.argv[1] if len(sys.argv) > 1 else ""
    use_solid = solid_tag in ("solid2", "solid3")
    gbm = lgb.Booster(model_file=str(LAB / "models" / "residual_lgb.txt"))

    def corr(base, P):
        Fe = global_features(P, base)
        r = gbm.predict(Fe.reshape(-1, 15)).reshape(base.shape).astype(np.float32)
        return np.clip(base + ETA * np.where(np.abs(r) >= GATE, r, 0.0), 0, 1)

    fam = {}
    for i, panel in enumerate(C.T2_PANELS):
        ctx, stat, W = C.windows(i, panel)
        for s, D in load_split(i, panel, "ssl_w192ssl_avg4").items():
            og = W[s]["og"]
            P, ps, y, e, d = D["P"], D["ps"], D["y"], D["e"], D["d"]
            if use_solid:
                ps = np.mean([np.load(CACHE / f"fam_av_s{k}_{solid_tag}_{s}_{panel}.npz")["og"].astype(np.float32)
                              for k in (20, 21, 22, 23)], 0)
            tad = fam_lab("tfm", s, panel)
            mem = {"gs1": P["gs1"], "tfm8": tad, "cnn3": P["cnn3"], "ps": ps,
                   "occ_cnn": P["occ_cnn"], "occ_tfm": P["occ_tfm"], "stid": P["stid"]}
            names = list(mem)
            prior = np.array([0.08, 0.24, 0.08, 0.60, 0.02, 0.02, 0.02]); prior = prior / prior.sum()
            wau = np.array([0.08, 0.24, 0.08, 0.60, 0, 0, 0])
            M = np.stack([mem[m] for m in names])                              # (K, N, F, S)
            p0_au = np.tensordot(wau, M, 1)
            if use_solid:                                               # AU itself, with the shipped p_PS
                p_ship = D["ps"]
                V = {"AU": corr(p0_au - 0.60 * ps + 0.60 * p_ship, P)}
            else:
                V = {"AU": corr(p0_au, P)}
            if use_solid:
                V[f"AU, p_PS -> {solid_tag}"] = corr(p0_au, P)
            vr, nb = neighbourhood_sig(og["v"], og["ok"], ctx.vf)
            ef = e.astype(np.float32); yf = y.astype(np.float32)
            brier = ((M - yf[None]) ** 2) * ef[None]                          # (K, N, F, S)
            for k_nb, beta, kappa in ((40, 10.0, 20.0), (40, 5.0, 20.0), (60, 10.0, 30.0), (80, 10.0, 40.0), (60, 5.0, 30.0), (80, 20.0, 40.0)):
                p0 = p0_au.copy()
                for n in range(len(d)):
                    # earlier days, and the SAME day's windows whose horizon ends before this origin (released by T)
                    past = np.flatnonzero((d < d[n]) | ((d == d[n]) & (og["t0"] + F < og["t0"][n])))
                    if not len(past):
                        continue
                    msk = nb[n] if nb[n].any() else np.ones_like(nb[n])
                    a = vr[n][:, msk].reshape(-1)
                    b = vr[past][:, :, msk].reshape(len(past), -1)
                    sim = (b @ a) / (np.linalg.norm(b, axis=1) * np.linalg.norm(a) + 1e-6)
                    sim = sim - 0.02 * np.abs(og["t0"][past] - og["t0"][n]) / 12.0
                    j = past[np.argsort(-sim)[:k_nb]]
                    den = ef[j].sum((0, 2))                                   # (F,)
                    B = brier[:, j].sum((1, 3)) / np.maximum(den[None], 1)    # (K, F)
                    rel = B / np.maximum(B.min(0, keepdims=True), 1e-6) - 1
                    w = prior[:, None] * np.exp(-beta * rel)
                    w = w / w.sum(0, keepdims=True)
                    sh = len(j) / (len(j) + kappa)
                    w = sh * w + (1 - sh) * wau[:, None]
                    p0[n] = np.einsum("kf,kfs->fs", w, M[:, n])
                V[f"select k{k_nb} b{beta:g} kap{kappa:g}"] = corr(p0, P)
            for nm, q in V.items():
                sc = C.iou(q >= 0.5, y, e)
                fam.setdefault((s, nm, "all"), {})[panel] = sc.mean()
                if (d <= 4).any():
                    fam.setdefault((s, nm, "days 1-5"), {})[panel] = sc[d <= 4].mean()
        print(f"  {panel}", flush=True)
    names = list(dict.fromkeys(k[1] for k in fam))
    for sub in ("all", "days 1-5"):
        print(f"\n=== ongoing family IoU, {sub} ({('p_PS = ' + solid_tag) if use_solid else 'p_PS as shipped'}) ===")
        ref = {s: C.family_mean(fam[(s, "AU", sub)]) for s in C.SPLITS}
        for nm in names:
            v = {s: C.family_mean(fam[(s, nm, sub)]) for s in C.SPLITS}
            print(f"  {nm:<28}" + "".join(f"  {s[:4]} {v[s]:.4f} ({v[s] - ref[s]:+.4f})" for s in C.SPLITS)
                  + f"   mean {np.mean([v[s] - ref[s] for s in C.SPLITS]):+.4f}")


if __name__ == "__main__":
    main()
