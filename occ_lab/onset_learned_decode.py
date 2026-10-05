"""A learned onset decision rule: predict the IoU of each candidate set, pick the best (vs the fixed ratio rule).

Per window the candidate sets are prefixes of two orderings: all links by p (k = 1..12) and the top region's links
by p (k = 1..8) (regions: note 129's peaks). For each (window, set) the target is the set's actual IoU at T+30;
features describe the set against the window's probabilities (k, sum / mean of p inside, p_k, p_(k+1), the rest's
mass, the ratio rule's own expected-IoU estimate and its k, |k - k_ratio|, blocks of the set, regions touched,
the top region's mass share, peak p, total mass, time of day, prefix family). A LightGBM regressor (cross-month:
fitted on one scored month, applied to the other; LightGBM seeds 0-5) scores the sets; the arg-max is the decision.
Also 'TTT': the ratio rule's c chosen per month-panel on the released earlier days' onset events (last <= 4 days).

    .venv/Scripts/python.exe occ_lab/onset_learned_decode.py
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

import bottleneck_capacity as BK  # noqa: E402
import bottleneck_cls as BC  # noqa: E402
import common as C  # noqa: E402
import onset_block_decode as OB  # noqa: E402

F = C.F
K_ALL, K_REG = 12, 8
CS = (0.6, 0.8, 1.0, 1.25, 1.6)


def ratio_est(ps):
    """the ratio rule's expected-IoU estimate for each prefix of the sorted probabilities."""
    cs = np.cumsum(ps)
    k = np.arange(1, len(ps) + 1)
    return cs / (k + ps.sum() - cs)


def sets_of(p, t0):
    S = len(p)
    order = np.argsort(-p)
    ps = p[order]
    est = ratio_est(ps)
    k_ratio = int(C.select(p, "ratio", 1.0).sum())
    cand = BC.peaks(p)
    reg = np.abs(np.arange(S)[:, None] - cand[None, :]).argmin(1) if len(cand) else np.zeros(S, int)
    mass = np.bincount(reg, weights=p, minlength=max(len(cand), 1))
    top = int(np.argmax(mass))
    tot = p.sum()
    out = []
    fams = [(0, order[:K_ALL])]
    rl = np.flatnonzero(reg == top)
    fams.append((1, rl[np.argsort(-p[rl])][:K_REG]))
    for fam, ordr in fams:
        for k in range(1, len(ordr) + 1):
            A = np.zeros(S, bool)
            A[ordr[:k]] = True
            inside = p[A].sum()
            pk = p[ordr[k - 1]]
            pk1 = p[ordr[k]] if k < len(ordr) else 0.0
            # the ratio estimate of this exact set
            e = inside / (k + tot - inside)
            nb, _ = OB.blocks(A)
            f = [fam, k, inside, inside / k, pk, pk1, tot - inside, e, est.max(), e / max(est.max(), 1e-9),
                 k_ratio, k - k_ratio, nb, len(np.unique(reg[A])), mass[top] / max(tot, 1e-9), ps[0], tot,
                 t0 / 12.0, (ps[:k_ratio] > 0).sum()]
            out.append((A, f))
    return out


def window_table(D, src):
    on = D["on"]
    rows, tgt, grp, sets = [], [], [], []
    for n in range(len(on["d"])):
        ss = sets_of(D[src][n], int(on["t0"][n]))
        t, e = on["fut"][n, -1], on["efut"][n, -1]
        for A, f in ss:
            a = A & e
            u = (a | t).sum()
            rows.append(f); tgt.append((a & t).sum() / u if u else 1.0)
        grp.append(len(ss)); sets.append([A for A, _ in ss])
    return np.asarray(rows, np.float32), np.asarray(tgt, np.float32), np.asarray(grp), sets


def main() -> None:
    warnings.simplefilter("ignore")
    data = {}
    for i, panel in enumerate(C.T2_PANELS):
        B = BC.build(i, panel)
        ctx, stat, W = C.windows(i, panel)
        vf = np.asarray(ctx.vf, np.float32)
        Pr = BK.prior_stats(panel, vf)
        for s, D in B.items():
            A = BK.day_arrays(W[s]["z"], vf)
            D["Xp"] = np.concatenate([D["X"], BK.prior_features(D, Pr), BK.capacity_features_trunc(D, A)], 1)
        data[panel] = B
        print(f"  {panel}", flush=True)
    par = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 30,
           "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "num_threads": 8,
           "seed": 0}
    for s_fit, s_app in (("validation", "private"), ("private", "validation")):
        X = np.concatenate([d[s_fit]["Xp"] for d in data.values() if s_fit in d])
        y = np.concatenate([d[s_fit]["y"] for d in data.values() if s_fit in d])
        m = lgb.train(par, lgb.Dataset(X, y), 200)
        for d in data.values():
            if s_app not in d:
                continue
            D = d[s_app]
            prob = m.predict(D["Xp"])
            P2 = D["base"].copy()
            off = 0
            for (n, cand, reg, mass), g in zip(D["meta"], D["g"]):
                P2[n] = BC.swap(D["base"][n], cand, reg, mass, prob[off:off + g]); off += g
            D["swapped"] = P2
    for src in ("base", "swapped"):
        for d in data.values():
            for D in d.values():
                D["T_" + src] = window_table(D, src)
        ref = {s: C.family_mean({p_: C.onset_iou(d[s][src], d[s]["on"]).mean() for p_, d in data.items() if s in d})
               for s in C.SPLITS}
        orc = {}
        for s in C.SPLITS:
            fam = {}
            for p_, d in data.items():
                if s in d:
                    X, y, g, _ = d[s]["T_" + src]
                    fam[p_] = np.mean([y[o:o + k].max() for o, k in zip(np.r_[0, np.cumsum(g)[:-1]], g)])
            orc[s] = C.family_mean(fam)
        print(f"\n[{src}] ratio rule val {ref['validation']:.4f} priv {ref['private']:.4f}; best candidate set "
              f"(oracle) val {orc['validation']:.4f} priv {orc['private']:.4f}")
        regp = {"objective": "regression", "learning_rate": 0.03, "num_leaves": 15, "min_data_in_leaf": 40,
                "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "num_threads": 8}
        for nm, obj in (("learned IoU (regression)", "regression"), ("learned IoU (lambdarank)", "lambdarank")):
            res = {s: [] for s in C.SPLITS}
            for sd in range(6):
                for s_fit, s_app in (("validation", "private"), ("private", "validation")):
                    tabs = [d[s_fit]["T_" + src] for d in data.values() if s_fit in d]
                    X = np.concatenate([t[0] for t in tabs]); y = np.concatenate([t[1] for t in tabs])
                    g = np.concatenate([t[2] for t in tabs])
                    if obj == "lambdarank":
                        lab = np.minimum((y * 10).round().astype(int), 10)
                        ds = lgb.Dataset(X, lab, group=g)
                        m = lgb.train({**regp, "objective": "lambdarank", "seed": sd,
                                       "label_gain": list(range(11)), "lambdarank_truncation_level": 3}, ds, 300)
                    else:
                        m = lgb.train({**regp, "seed": sd}, lgb.Dataset(X, y), 300)
                    fam = {}
                    for p_, d in data.items():
                        if s_app not in d:
                            continue
                        X2, y2, g2, _ = d[s_app]["T_" + src]
                        sc = m.predict(X2)
                        fam[p_] = np.mean([y2[o:o + k][np.argmax(sc[o:o + k])]
                                           for o, k in zip(np.r_[0, np.cumsum(g2)[:-1]], g2)])
                    res[s_app].append(C.family_mean(fam) - ref[s_app])
            v, pr = np.mean(res["validation"]), np.mean(res["private"])
            print(f"  {nm:<28} val {v:+.4f}±{np.std(res['validation']):.4f}  priv {pr:+.4f}±"
                  f"{np.std(res['private']):.4f}  mean {(v + pr) / 2:+.4f}", flush=True)
        # c: global cross-month and TTT per month-panel on the released earlier days (last <= 4)
        for s in C.SPLITS:
            pass
        glob = {}
        for c in CS:
            for s in C.SPLITS:
                fam = {}
                for p_, d in data.items():
                    if s in d:
                        D = d[s]
                        pr_ = np.stack([C.select(q, "ratio", c) for q in D[src]])
                        fam[p_] = OB.iou_of(pr_, D["on"]).mean()
                glob[(c, s)] = C.family_mean(fam)
        pick = {s: max(CS, key=lambda c: glob[(c, s)]) for s in C.SPLITS}
        v = glob[(pick["private"], "validation")] - ref["validation"]
        pr = glob[(pick["validation"], "private")] - ref["private"]
        print(f"  c cross-month (val <- {pick['private']}, priv <- {pick['validation']})  val {v:+.4f}  priv {pr:+.4f}"
              f"  mean {(v + pr) / 2:+.4f}")
        res = {}
        for s in C.SPLITS:
            fam = {}
            for p_, d in data.items():
                if s not in d:
                    continue
                D = d[s]
                on = D["on"]
                iou_c = {c: OB.iou_of(np.stack([C.select(q, "ratio", c) for q in D[src]]), on) for c in CS}
                sc = []
                for n in range(len(on["d"])):
                    past = (on["d"] < on["d"][n]) & (on["d"] >= on["d"][n] - 4)
                    if past.sum() >= 2:
                        # shrink toward c = 1: a c must beat it by 0.02 on the past events
                        gain = {c: iou_c[c][past].mean() - iou_c[1.0][past].mean() for c in CS}
                        c_ = max(CS, key=lambda c: gain[c] - (0.02 if c != 1.0 else 0))
                    else:
                        c_ = 1.0
                    sc.append(iou_c[c_][n])
                fam[p_] = np.mean(sc)
            res[s] = C.family_mean(fam) - ref[s]
        print(f"  c TTT (month-panel, last <= 4 days)  val {res['validation']:+.4f}  priv {res['private']:+.4f}  "
              f"mean {np.mean(list(res.values())):+.4f}")


if __name__ == "__main__":
    main()
