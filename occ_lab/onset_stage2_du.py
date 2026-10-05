"""Onset step 2: add a second site by its learned marginal utility (after note 152's IoU-LambdaRank + swap).

Stage 1 (note 152): lambdarank on the candidates' utility U_b, swap -> P1, output S_b = ratio rule on P1 (the actual
output). For every other candidate region c: S_c = the ratio rule on P1 restricted to region c; the teacher is
    dU(c | b) = IoU(Y, S_b u S_c) - IoU(Y, S_b)
(no region-label teacher: a queue split by a region boundary and a true second site are the same question). Low-dof
features, no link identity: score_c, score_top - score_c, distance to the top candidate, both peaks of P1 and the
prominence of c, c's headroom now / q_brk (note 145), cells added and their P1 mass, the ratio estimate of the gain,
number of candidates. A LightGBM regressor predicts dU; the threshold is chosen on the FIT month only (5-fold out
of fold predictions, the largest threshold within 0.001 of the best gain); at most one c per window.
Stage 1 is cross-month; stage 2 is fitted on one month (with stage 1 from the other) and applied to the other.

    .venv/Scripts/python.exe occ_lab/onset_stage2_du.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta"))
sys.path.insert(0, str(LAB.parent))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

import bottleneck_cls as BC  # noqa: E402
import common as C  # noqa: E402
import onset_rerank as RR  # noqa: E402

F = C.F
TAUS = (0.0, 0.02, 0.05, 0.1, 0.15, 0.2)
PAR = {"learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 30, "feature_fraction": 0.8,
       "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "num_threads": 8}
HEAD = 21 + 1        # Xp column: prior headroom now / q_brk (note 129's 21 columns, then note 145's prior rows)


def iou_of(A, t, e):
    a = A & e
    u = (a | t).sum()
    return (a & t).sum() / u if u else 1.0


def stage1(data, sd, key="Xp"):
    """cross-month lambdarank scores and swapped probabilities P1 for every window."""
    for s_fit, s_app in (("validation", "private"), ("private", "validation")):
        Ds = [d[s_fit] for d in data.values() if s_fit in d]
        lab = np.minimum((np.concatenate([D["U"] for D in Ds]) * 10).round().astype(int), 10)
        m = lgb.train({**PAR, "objective": "lambdarank", "seed": sd, "label_gain": list(range(11)),
                       "min_data_in_leaf": 20},
                      lgb.Dataset(np.concatenate([D[key] for D in Ds]), lab,
                                  group=np.concatenate([D["g"] for D in Ds])), 200)
        for d in data.values():
            if s_app not in d:
                continue
            D = d[s_app]
            sc = m.predict(D[key])
            P1 = D["base"].copy()
            D["s1"] = sc
            off = 0
            for (n, cand, reg, mass), g in zip(D["meta"], D["g"]):
                s_ = sc[off:off + g]
                P1[n] = BC.swap(D["base"][n], cand, reg, mass, s_ - s_.min() + 1e-3); off += g
            D["P1"] = P1


def pair_rows(p1, sc, xp, nf, cand, reg):
    """S_b (the ratio rule on p1) and, for each non-top candidate c, (features, S_c) - shared with production."""
    Sb = C.select(p1, "ratio", 1.0)
    top = int(np.argmax(sc))
    g = len(cand)
    out = []
    for c in range(g):
        if c == top:
            continue
        Sc = C.select(p1 * (reg == c), "ratio", 1.0) & (reg == c) & ~Sb
        if not Sc.any():
            continue
        A = Sb | Sc
        est = p1[A].sum() / (A.sum() + p1.sum() - p1[A].sum()) - p1[Sb].sum() / (Sb.sum() + p1.sum() - p1[Sb].sum())
        f = [sc[c], sc[top] - sc[c], abs(int(cand[c]) - int(cand[top])), p1[cand[top]], p1[cand[c]], nf[c, 1],
             nf[c, 2], xp[c, HEAD], Sc.sum(), p1[Sc].sum(), est, g]
        out.append((f, Sc, c))
    return Sb, out


def pairs(D):
    """per window: base IoU of S_b and the (features, dU, S_c) of each other candidate."""
    on = D["on"]
    out = {}
    off = 0
    for (n, cand, reg, mass), g in zip(D["meta"], D["g"]):
        sc = D["s1"][off:off + g]
        xp = D["Xp"][off:off + g]
        nf = D["NF"][off:off + g]
        off += g
        p1 = D["P1"][n]
        t, e = on["fut"][n, -1], on["efut"][n, -1]
        Sb, pr = pair_rows(p1, sc, xp, nf, cand, reg)
        ib = iou_of(Sb, t, e)
        out[n] = (ib, [(f, iou_of(Sb | Sc, t, e) - ib, Sc) for f, Sc, c in pr])
    return out


def main() -> None:
    warnings.simplefilter("ignore")
    data = RR.prepare()
    ref = {}
    res = {}
    for sd in range(6):
        stage1(data, sd)
        T = {p_: {s: pairs(D) for s, D in d.items()} for p_, d in data.items()}
        if sd == 0:
            for s in C.SPLITS:
                du = [dU for p_ in T if s in T[p_] for ib, rows in T[p_][s].values() for _, dU, _ in rows]
                du = np.array(du)
                print(f"{s}: candidate pairs {len(du)}, dU > 0: {int((du > 0).sum())} (sum {du[du > 0].sum():+.1f}), "
                      f"dU < 0: {int((du < 0).sum())} (sum {du[du < 0].sum():+.1f})")
        for s_fit, s_app in (("validation", "private"), ("private", "validation")):
            X = np.array([f for p_ in T if s_fit in T[p_] for _, rows in T[p_][s_fit].values() for f, _, _ in rows],
                         np.float32)
            y = np.array([dU for p_ in T if s_fit in T[p_] for _, rows in T[p_][s_fit].values() for _, dU, _ in rows],
                         np.float32)
            win = np.array([k for k, (p_, n) in enumerate((p_, n) for p_ in T if s_fit in T[p_] for n in T[p_][s_fit])
                            for _ in T[p_][s_fit][n][1]])
            # threshold on the fit month: 5-fold (by window) out-of-fold predictions
            rng = np.random.default_rng(sd)
            fold = rng.integers(0, 5, win.max() + 1)[win]
            oof = np.zeros(len(y))
            for k in range(5):
                mk = lgb.train({**PAR, "objective": "regression", "seed": sd}, lgb.Dataset(X[fold != k], y[fold != k]), 150)
                oof[fold == k] = mk.predict(X[fold == k])
            gain = {}
            for tau in TAUS:
                gsum = 0.0
                for w in np.unique(win):
                    j = np.flatnonzero(win == w)
                    b = j[np.argmax(oof[j])]
                    if oof[b] > tau:
                        gsum += y[b]
                gain[tau] = gsum
            best = max(gain.values())
            tau = max(t_ for t_ in TAUS if gain[t_] >= best - 1e-3) if best > 0 else np.inf
            m = lgb.train({**PAR, "objective": "regression", "seed": sd}, lgb.Dataset(X, y), 150)
            fam, fam0, multi, single, worse, adds = {}, {}, [], [], 0, 0
            for p_, d in data.items():
                if s_app not in d:
                    continue
                D = d[s_app]
                sc_all = []
                for n in range(len(D["on"]["d"])):
                    if n in T[p_][s_app]:
                        ib, rows = T[p_][s_app][n]
                        new = ib
                        if rows:
                            pr = m.predict(np.array([f for f, _, _ in rows], np.float32))
                            j = int(np.argmax(pr))
                            if pr[j] > tau:
                                new = ib + rows[j][1]; adds += 1
                    else:
                        ib = new = iou_of(C.select(D["P1"][n], "ratio", 1.0), D["on"]["fut"][n, -1], D["on"]["efut"][n, -1])
                    sc_all.append((ib, new, D["single"][n]))
                a = np.array(sc_all, float)
                fam0[p_], fam[p_] = a[:, 0].mean(), a[:, 1].mean()
                multi += list(a[a[:, 2] == 0][:, 1] - a[a[:, 2] == 0][:, 0])
                sg = a[a[:, 2] == 1]
                single += list(sg[:, 1] - sg[:, 0]); worse += int((sg[:, 1] < sg[:, 0] - 1e-9).sum())
            res.setdefault(s_app, []).append((C.family_mean(fam) - C.family_mean(fam0), np.mean(multi), np.sum(multi),
                                              np.sum(single), worse, adds, tau))
        print(f"  seed {sd}: " + "  ".join(f"{s[:3]} {res[s][-1][0]:+.4f} (tau {res[s][-1][6]})" for s in C.SPLITS),
              flush=True)
    print(f"\nvs stage 1 (IoU-LambdaRank + swap), mean over 6 seeds:")
    for s in C.SPLITS:
        r = np.array(res[s], float)
        print(f"  applied to {s:<10} all {r[:, 0].mean():+.4f}±{r[:, 0].std():.4f}  multi-site windows mean dIoU "
              f"{r[:, 1].mean():+.4f} (sum {r[:, 2].mean():+.2f})  single-site sum dIoU {r[:, 3].mean():+.2f}, "
              f"worsened {r[:, 4].mean():.1f}  adds {r[:, 5].mean():.1f}")


if __name__ == "__main__":
    main()
