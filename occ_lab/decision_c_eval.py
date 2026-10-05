"""C of the 2026-09-27 survey: decision-layer methods on AU's ongoing answer, all causal within the scored run.

Per panel and split, days in order; everything used for day d comes from lab windows of EARLIER days (their labels
are released before day d) or from day d's own predictions (no label).

    prior (sup)     label-shift correction (Saerens 2002 form): the run's queued share on earlier days' windows
                    against AU's mean probability there gives the prior ratio; p' = w1 p / (w1 p + w0 (1 - p))
    prior (EM)      unsupervised EM on day d's own windows and earlier ones, source prior = AU's mean over the
                    pooled lab windows of both months
    platt           online recalibration: logit' = a logit + b_h (per horizon step) fitted on earlier days' eligible
                    cells, ridge-shrunk to (1, 0) with strength lam (so the first days stay at AU)
    select b        per-window member selection (META-DES style): the 20 earlier windows with the closest last-15-min
                    v/vf pattern give each p0 member's IoU there; weights w_m ~ w_AU,m exp(b (IoU_m - mean)),
                    shrunk to AU's weights by n / (n + 10) with n the neighbours found; then the corrector
    lag w           the time-lagged ensemble (occ_lab/adapt_variants.py --method lagged): the four p_PS seeds'
                    forecasts issued at T-5 and T-10 averaged into p0's p_PS slot with weight w on the steps they cover

    .venv/Scripts/python.exe occ_lab/decision_c_eval.py
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


def lg(p):
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def sg(z):
    return 1 / (1 + np.exp(-z))


def signature(v, ok, vf):
    x = np.where(ok, v / vf, 0.0)[:, -3:, :].reshape(len(v), -1).astype(np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-6)


def main() -> None:
    gbm = lgb.Booster(model_file=str(LAB / "models" / "residual_lgb.txt"))

    def corr(base, P):
        Fe = global_features(P, base)
        r = gbm.predict(Fe.reshape(-1, 15)).reshape(base.shape).astype(np.float32)
        return np.clip(base + ETA * np.where(np.abs(r) >= GATE, r, 0.0), 0, 1)

    data = []
    for i, panel in enumerate(C.T2_PANELS):
        ctx, stat, W = C.windows(i, panel)
        for s, D in load_split(i, panel, "ssl_w192ssl_avg4").items():
            og = W[s]["og"]
            P, ps, y, e, d = D["P"], D["ps"], D["y"], D["e"], D["d"]
            tad = fam_lab("tfm", s, panel)
            mem = {"gs1": P["gs1"], "tfm8": tad, "cnn3": P["cnn3"], "ps": ps}
            wau = {"gs1": 0.08, "tfm8": 0.24, "cnn3": 0.08, "ps": 0.60}
            lag = {}
            for L in (1, 2):
                f = [CACHE / f"fam_av_s{k}_lag{L}_{s}_{panel}.npz" for k in (20, 21, 22, 23)]
                if all(x.exists() for x in f):
                    lag[L] = np.mean([np.load(x)["og"].astype(np.float32) for x in f], 0)
            data.append(dict(panel=panel, s=s, P=P, mem=mem, wau=wau, y=y, e=e, d=d, lag=lag,
                             sig=signature(og["v"], og["ok"], ctx.vf)))
        print(f"  {panel}", flush=True)
    src_prior = np.mean([sum(D["wau"][m] * D["mem"][m] for m in D["wau"]).mean() for D in data])

    fam = {}
    for D in data:
        P, mem, wau, y, e, d = D["P"], D["mem"], D["wau"], D["y"], D["e"], D["d"]
        p0 = sum(wau[m] * mem[m] for m in wau)
        au = corr(p0, P)
        V = {"AU": au}
        # ---------------- prior shift and Platt, causal by day
        out = {k: au.copy() for k in ("prior sup", "prior EM", "platt lam 1e3", "platt lam 1e4")}
        for dd in np.unique(d):
            cur, past = d == dd, d < dd
            if past.any():
                m = e[past]
                pi_t = (y[past] & m).sum() / max(m.sum(), 1)
                pi_p = (au[past] * m).sum() / max(m.sum(), 1)
                w1, w0 = pi_t / max(pi_p, 1e-4), (1 - pi_t) / max(1 - pi_p, 1e-4)
                out["prior sup"][cur] = w1 * au[cur] / (w1 * au[cur] + w0 * (1 - au[cur]))
                z, yy, mm = lg(au[past]), y[past].astype(float), m.astype(float)
                for lam in (1e3, 1e4):
                    a, b = 1.0, np.zeros(F)
                    for _ in range(20):                        # Newton on (a, b_h) with a ridge towards (1, 0)
                        pz = sg(a * z + b[None, :, None])
                        g_a = ((pz - yy) * z * mm).sum() + lam * (a - 1)
                        h_a = ((pz * (1 - pz)) * z * z * mm).sum() + lam
                        a -= g_a / h_a
                        pz = sg(a * z + b[None, :, None])
                        g_b = ((pz - yy) * mm).sum((0, 2)) + lam * b
                        h_b = ((pz * (1 - pz)) * mm).sum((0, 2)) + lam
                        b -= g_b / h_b
                    out[f"platt lam {lam:.0e}".replace("+0", "")][cur] = sg(a * lg(au[cur]) + b[None, :, None])
            upto = d <= dd                                     # EM: today's and earlier windows, no label
            pi = src_prior
            for _ in range(30):
                w1, w0 = pi / src_prior, (1 - pi) / (1 - src_prior)
                q = w1 * au[upto] / (w1 * au[upto] + w0 * (1 - au[upto]))
                pi = float(q.mean())
            w1, w0 = pi / src_prior, (1 - pi) / (1 - src_prior)
            out["prior EM"][cur] = w1 * au[cur] / (w1 * au[cur] + w0 * (1 - au[cur]))
        V.update(out)
        # ---------------- per-window member selection
        ious = {m: C.iou(mem[m] >= 0.5, y, e) for m in mem}
        for beta in (5.0, 20.0):
            p0s = p0.copy()
            for n in range(len(d)):
                past = np.flatnonzero(d < d[n])
                if not len(past):
                    continue
                sim = D["sig"][past] @ D["sig"][n]
                nb = past[np.argsort(-sim)[:20]]
                comp = np.array([ious[m][nb].mean() for m in mem])
                w = np.array([wau[m] for m in mem]) * np.exp(beta * (comp - comp.mean()))
                w = w / w.sum()
                sh = len(nb) / (len(nb) + 10)
                w = sh * w + (1 - sh) * np.array([wau[m] for m in mem])
                p0s[n] = sum(wi * mem[m][n] for wi, m in zip(w, mem))
            V[f"select b{beta:g}"] = corr(p0s, P)
        # ---------------- time-lagged ensemble in the p_PS slot
        for wl in (0.2, 0.4):
            if not D["lag"]:
                continue
            psl = mem["ps"].copy()
            for L, arr in D["lag"].items():
                cov = np.isfinite(arr)
                psl = np.where(cov, (1 - wl / len(D["lag"])) * psl + (wl / len(D["lag"])) * np.nan_to_num(arr), psl)
            V[f"lag w{wl}"] = corr(p0 - wau["ps"] * mem["ps"] + wau["ps"] * psl, P)
        for nm, q in V.items():
            sc = C.iou(q >= 0.5, y, e)
            fam.setdefault((D["s"], nm, "all"), {})[D["panel"]] = sc.mean()
            if (d <= 4).any():
                fam.setdefault((D["s"], nm, "days 1-5"), {})[D["panel"]] = sc[d <= 4].mean()
    names = list(dict.fromkeys(k[1] for k in fam))
    for sub in ("all", "days 1-5"):
        print(f"\n=== ongoing family IoU, {sub} ===")
        ref = {s: C.family_mean(fam[(s, "AU", sub)]) for s in C.SPLITS}
        for nm in names:
            if not all((s, nm, sub) in fam for s in C.SPLITS):
                continue
            v = {s: C.family_mean(fam[(s, nm, sub)]) for s in C.SPLITS}
            print(f"  {nm:<16}" + "".join(f"  {s[:4]} {v[s]:.4f} ({v[s] - ref[s]:+.4f})" for s in C.SPLITS)
                  + f"   mean {np.mean([v[s] - ref[s] for s in C.SPLITS]):+.4f}")


if __name__ == "__main__":
    main()
