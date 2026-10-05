"""Decision theory for the per-window budget: a GFM-style expected-IoU decision with a learned size distribution.

Literature
    Dembczynski, Waegeman, Cheng, Huellermeier, "An Exact Algorithm for F-Measure Maximization" (NeurIPS 2011);
    Waegeman et al., "On the Bayes-optimality of F-measure maximizers" (JMLR 2014): the Bayes-optimal set needs,
    instead of independent marginals, p_is = P(y_i = 1, |y| = s).
    Nowozin, "Optimal Decisions from Probabilistic Models: the Intersection-over-Union Case" (CVPR 2014): the
    expected IoU of a set approximated through the intersection's expectation.

Here, per ongoing window (eligible cells only):
    P(s)            a LightGBM multiclass over bins of r = s / sum(p), fitted on the 2031-01/02 OOF windows of the
                    train run (occ_lab/cache/*_oof_*), from window-level statistics of p
    P(y_i=1 | s)    exponential tilting of p: sigmoid(logit p_i + a_s), a_s solved so that the tilted marginals sum
                    to s (the maximum-entropy conditional; it keeps the order of p, so the candidate answers are
                    the top-k sets of p = the threshold family)
    decision        k* = argmax_k sum_s P(s) I_k(s) / (k + s - I_k(s)), I_k(s) = sum of the top-k tilted marginals
Controls: flat tau 0.50 (AT), and the same decision with P(s) a point mass at sum(p) (independence-like).

    .venv/Scripts/python.exe gen_lab/gfm_eval.py
"""
from __future__ import annotations

import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent
PROJ = LAB.parent
sys.path.insert(0, str(LAB))
import numpy as np  # noqa: E402

import decide_eval as D  # noqa: E402

C = D.C
lgb = D.lgb
OCC = PROJ / "occ_lab" / "cache"
R_BINS = np.array([0.0, 0.35, 0.55, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.35, 1.6, 2.0, 2.6, 4.0])   # r = s / sum p
R_MID = np.r_[0.2, (R_BINS[1:-1] + R_BINS[2:]) / 2]                                          # class centres


def lg(p):
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def feats(p, e):
    """window-level statistics of the eligible probabilities: (N, n_feat)"""
    out = []
    for n in range(len(p)):
        q = p[n][e[n]]
        if q.size == 0:
            q = np.zeros(1, np.float32)
        hs = [(p[n][h][e[n][h]]).sum() for h in range(p.shape[1])]
        sp = q.sum()
        out.append([sp, np.log1p(sp), (q > 0.5).sum(), (q * (1 - q)).sum(), q.size, np.quantile(q, 0.9),
                    np.quantile(q, 0.99), q.max(), ((q > 0.2) & (q < 0.8)).sum() / max(sp, 1),
                    hs[0], hs[-1], (hs[-1] - hs[0]) / max(sp, 1), (q > 0.5).sum() / max(sp, 1)])
    return np.asarray(out, np.float32)


def r_class(s, sp):
    r = s / np.maximum(sp, 1e-3)
    return np.clip(np.searchsorted(R_BINS, r, side="right") - 1, 0, len(R_MID) - 1)


def tilt(lgt, target):
    """a with sum sigmoid(lgt + a) = target (bisection)."""
    lo, hi = -20.0, 20.0
    for _ in range(40):
        a = (lo + hi) / 2
        if (1 / (1 + np.exp(-(lgt + a)))).sum() > target:
            hi = a
        else:
            lo = a
    return (lo + hi) / 2


def decide(pw, ew, ps):
    """pw (F,S) probabilities, ew eligibility, ps (n_class,) size distribution -> bool (F,S) answer."""
    q = pw[ew]
    if q.size == 0:
        return np.zeros_like(ew)
    order = np.argsort(-q)
    lgt = lg(q)[order]
    sp = q.sum()
    ks = np.arange(0, len(q) + 1)
    ev = np.zeros(len(ks))
    for c, w in enumerate(ps):
        if w < 1e-4:
            continue
        s = max(R_MID[c] * sp, 0.5)
        pt = 1 / (1 + np.exp(-(lgt + tilt(lgt, s))))
        I = np.r_[0.0, np.cumsum(pt)]
        ev += w * np.where(ks + s - I > 0, I / np.maximum(ks + s - I, 1e-9), 1.0)
    k = int(np.argmax(ev))
    out = np.zeros(q.size, bool)
    out[order[:k]] = True
    ans = np.zeros_like(ew)
    ans[ew] = out
    return ans


def oof_training():
    X, Y = [], []
    for panel in C.T2_PANELS:
        f, fp, fs = OCC / f"og_oof_{panel}.npz", OCC / f"pseudo_oof_{panel}.npz", OCC / f"stid_oof_{panel}.npz"
        if not (f.exists() and fp.exists()):
            continue
        z, zp = np.load(f), np.load(fp)
        old = 0.2 * z["gs1"].astype(np.float32) + 0.6 * z["tfm8"].astype(np.float32) + 0.2 * z["cnn3"].astype(np.float32)
        p = 0.4 * old + 0.6 * zp["pseudo"].astype(np.float32)
        e, y = z["e"], z["y"]
        Xp = feats(p, e)
        s = (y & e).sum((1, 2)).astype(np.float32)
        X.append(Xp)
        Y.append(r_class(s, Xp[:, 0]))
    return np.concatenate(X), np.concatenate(Y)


def main() -> None:
    X, Y = oof_training()
    print(f"size model: {len(Y)} OOF windows, class counts {np.bincount(Y, minlength=len(R_MID))}", flush=True)
    par = {"objective": "multiclass", "num_class": len(R_MID), "learning_rate": 0.05, "num_leaves": 15,
           "min_data_in_leaf": 40, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
           "verbose": -1, "num_threads": 8}
    size_model = lgb.train(par, lgb.Dataset(X, label=Y), num_boost_round=200)
    gbm = lgb.Booster(model_file=str(PROJ / "occ_lab" / "models" / "residual_lgb.txt"))
    names = ["flat tau 0.50 (= AT)", "GFM-IoU, learned P(s)", "GFM-IoU, P(s) = mass at sum p",
             "GFM-IoU, P(s) = OOF marginal"]
    prior = np.bincount(Y, minlength=len(R_MID)) / len(Y)
    fam = {}
    for i, panel in enumerate(C.T2_PANELS):
        sp = D.load_split(i, panel, "ssl_w192ssl_avg4")
        for s, Dd in sp.items():
            blend = Dd["base"]
            Fe = D.global_features(Dd["P"], blend)
            rh = gbm.predict(Fe.reshape(-1, Fe.shape[-1])).reshape(blend.shape).astype(np.float32)
            p = np.clip(blend + D.ETA * np.where(np.abs(rh) >= D.GATE, rh, 0.0), 0, 1)
            y, e = Dd["y"], Dd["e"]
            PS = size_model.predict(feats(p, e))
            point = np.zeros(len(R_MID))
            point[np.searchsorted(R_BINS, 1.0, side="right") - 1] = 1.0
            out = {n: [] for n in names}
            for n in range(len(p)):
                out[names[0]].append(D.win_iou(p[n] >= 0.5, y[n], e[n]))
                out[names[1]].append(D.win_iou(decide(p[n], e[n], PS[n]), y[n], e[n]))
                out[names[2]].append(D.win_iou(decide(p[n], e[n], point), y[n], e[n]))
                out[names[3]].append(D.win_iou(decide(p[n], e[n], prior), y[n], e[n]))
            for nm in names:
                fam.setdefault((s, nm), {})[panel] = float(np.mean(out[nm]))
        print(f"  {panel}", flush=True)
    print(f"\n{'decision':<34}" + "".join(f"{s:>13}" for s in C.SPLITS) + f"{'mean':>10}")
    for nm in names:
        v = [C.family_mean(fam[(s, nm)]) for s in C.SPLITS]
        print(f"{nm:<34}" + "".join(f"{x:>13.4f}" for x in v) + f"{np.mean(v):>10.4f}")


if __name__ == "__main__":
    main()
