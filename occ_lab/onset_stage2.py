"""Onset stage 2: after the bottleneck (stage 1, notes 129/145), decide the NUMBER of queue sites.

Loss decomposition of the current candidate (note 150): wrong region 41 %, truth at 2+ sites 36 % (mostly under-
covered: 3-7 true cells vs 2-4 predicted, sites 24-52 links apart), right region but too big 24 %. The ratio rule
sizes the set from p alone, blind to whether this is a multi-site morning. Per window three options:

    keep     the ratio rule's set R
    add      R + the ratio set of the stage-1 runner-up region r2 alone (a second site)
    only     R restricted to the top region r1 (drop the other sites)

Context features: stage-1 classifier probabilities of r1 / r2 (cross-month), peaks and masses of r1 / r2, their
distance, |R| and the regions it touches, the ratio estimate of each option, time of day, weekend, the month's
recent (last <= 4 days) onsets' number of separate sites, the training months' share of multi-site first onsets
around this time, the number of separate slowing clusters (v/vf < 0.8) at T-5 min, r2's capacity headroom (note 145).
Two regressors predict each option's IoU gain over keep; an option is taken when its predicted gain > margin.
Cross-month (fit one scored month, apply the other), LightGBM seeds 0-5.

    .venv/Scripts/python.exe occ_lab/onset_stage2.py
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
import bottleneck_timing as BT  # noqa: E402
import common as C  # noqa: E402

F = C.F
GAP = 4
MARGINS = (0.0, 0.02, 0.05, 0.1)
ADD = sys.argv[1] if len(sys.argv) > 1 else "ratio"      # the second-site set: ratio / peak / hi


def sites(mask):
    """number of separate sites: runs of queued links merged across gaps < GAP."""
    j = np.flatnonzero(mask)
    if not len(j):
        return 0
    return 1 + int((np.diff(j) >= GAP).sum())


def ratio_est(A, p):
    return p[A].sum() / max(A.sum() + p.sum() - p[A].sum(), 1e-9)


def multi_prior(Et, T30):
    """training months: kernel-weighted share of first-onset events with 2+ sites (sigma 30 min)."""
    k = np.exp(-0.5 * ((Et["t"] - T30) / 6.0) ** 2)
    return float(k @ Et["multi"] / max(k.sum(), 1e-6))


def main() -> None:
    warnings.simplefilter("ignore")
    data = {}
    for i, panel in enumerate(C.T2_PANELS):
        B = BC.build(i, panel)
        ctx, stat, W = C.windows(i, panel)
        vf = np.asarray(ctx.vf, np.float32)
        Pr = BK.prior_stats(panel, vf)
        from tfb.data import load
        zt = load(panel, "train")
        ont = C.first_onsets(vf, {k: zt[k] for k in ("obs_speed", "obs_flow", "elig", "weekday")})
        Et = dict(t=ont["t0"] + F, multi=np.array([sites(c) >= 2 for c in ont["cells"]], np.float32))
        for s, D in B.items():
            A = BK.day_arrays(W[s]["z"], vf)
            D["Xc"] = BK.prior_features(D, Pr)
            D["Xp"] = np.concatenate([D["X"], D["Xc"], BK.capacity_features_trunc(D, A)], 1)
            on = D["on"]
            D["msite"] = np.array([sites(on["fut"][n, -1] & on["efut"][n, -1]) for n in range(len(on["d"]))])
            D["mprior"] = np.array([multi_prior(Et, int(t) + F) for t in on["t0"]])
            vv = np.where(on["ok"][:, -1], on["v"][:, -1] / vf, 1.0)
            D["slow"] = np.array([sites(x < 0.8) for x in vv])
            D["we"] = (W[s]["z"]["weekday"][on["d"]] >= 5).astype(np.float32)
        data[panel] = B
        print(f"  {panel}", flush=True)
    # stage 1 (prior4 classifier, cross-month, seed 0): swapped probabilities and per-candidate probabilities
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
            D["pc"] = {}
            off = 0
            for (n, cand, reg, mass), g in zip(D["meta"], D["g"]):
                pc = prob[off:off + g]
                D["pc"][n] = (pc, D["Xc"][off:off + g]); off += g
                P2[n] = BC.swap(D["base"][n], cand, reg, mass, pc)
            D["swapped"] = P2
    # option table
    for d in data.values():
        for D in d.values():
            on = D["on"]
            rows, g_add, g_only, base_iou = [], [], [], []
            for n in range(len(on["d"])):
                p = D["swapped"][n]
                S = len(p)
                t, e = on["fut"][n, -1], on["efut"][n, -1]

                def iou(A):
                    a = A & e
                    u = (a | t).sum()
                    return (a & t).sum() / u if u else 1.0
                R = C.select(p, "ratio", 1.0)
                cand = BC.peaks(p)
                if len(cand) < 2 or n not in D["pc"]:
                    rows.append(None); g_add.append(0.0); g_only.append(0.0); base_iou.append(iou(R))
                    continue
                reg = np.abs(np.arange(S)[:, None] - cand[None, :]).argmin(1)
                r1 = int(reg[np.argmax(p)])
                pc, xc = D["pc"][n]
                # stage-1 candidates are the base-probability peaks; map by nearest peak
                cb = BC.peaks(D["base"][n])
                pcl = np.array([pc[int(np.argmin(np.abs(cb - c)))] for c in cand])
                xcl = np.stack([xc[int(np.argmin(np.abs(cb - c)))] for c in cand])
                order = [r for r in np.argsort(-pcl) if r != r1]
                r2 = int(order[0])
                if ADD == "ratio":
                    S2 = C.select(p * (reg == r2), "ratio", 1.0) & (reg == r2)
                elif ADD == "peak":
                    S2 = np.zeros(S, bool); S2[cand[r2]] = True
                else:
                    S2 = (reg == r2) & (p >= 0.5)
                Aadd = R | S2
                Aonly = R & (reg == r1)
                if not Aonly.any():
                    Aonly = R
                ib = iou(R)
                base_iou.append(ib); g_add.append(iou(Aadd) - ib); g_only.append(iou(Aonly) - ib)
                d_ = int(on["d"][n])
                past = (on["d"] < d_) & (on["d"] >= d_ - 4)
                rows.append([pcl[r1], pcl[r2], p[cand[r1]], p[cand[r2]], p[reg == r1].sum(), p[reg == r2].sum(),
                             abs(int(cand[r1]) - int(cand[r2])), R.sum(), sites(R), len(np.unique(reg[R])),
                             (R & (reg == r2)).sum(), ratio_est(R, p), ratio_est(Aadd, p), ratio_est(Aonly, p),
                             int(on["t0"][n]) / 12.0, D["we"][n],
                             D["msite"][past].mean() if past.any() else np.nan,
                             (D["msite"][past] >= 2).mean() if past.any() else np.nan,
                             D["mprior"][n], D["slow"][n], *xcl[r2][[1, 3, 8]], len(cand)])
            D["opt"] = (rows, np.array(g_add), np.array(g_only), np.array(base_iou))
    # headroom of each option (oracle choice per window)
    for s in C.SPLITS:
        ga = np.concatenate([d[s]["opt"][1] for d in data.values() if s in d])
        go = np.concatenate([d[s]["opt"][2] for d in data.values() if s in d])
        print(f"{s}: windows {len(ga)}; add helps {int((ga > 0).sum())} / hurts {int((ga < 0).sum())} "
              f"(sum {ga.sum():+.1f}); only helps {int((go > 0).sum())} / hurts {int((go < 0).sum())} "
              f"(sum {go.sum():+.1f}); oracle max(0, add, only) sum {np.maximum(0, np.maximum(ga, go)).sum():+.1f}")
    ref = {s: C.family_mean({p_: d[s]["opt"][3].mean() for p_, d in data.items() if s in d}) for s in C.SPLITS}
    rp = {"objective": "regression", "learning_rate": 0.03, "num_leaves": 7, "min_data_in_leaf": 25,
          "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "num_threads": 8}
    for opts in (("add",), ("only",), ("add", "only")):
        for mg in MARGINS:
            res = {s: [] for s in C.SPLITS}
            nch = 0
            for sd in range(6):
                for s_fit, s_app in (("validation", "private"), ("private", "validation")):
                    models = {}
                    for k, o in (("add", 1), ("only", 2)):
                        if k not in opts:
                            continue
                        X = [r for d in data.values() if s_fit in d for r, ok in
                             zip(d[s_fit]["opt"][0], d[s_fit]["opt"][0]) if ok is not None]
                        y = [g for d in data.values() if s_fit in d for r, g in
                             zip(d[s_fit]["opt"][0], d[s_fit]["opt"][o]) if r is not None]
                        models[k] = lgb.train({**rp, "seed": sd}, lgb.Dataset(np.asarray(X, np.float32),
                                                                              np.asarray(y, np.float32)), 150)
                    fam = {}
                    for p_, d in data.items():
                        if s_app not in d:
                            continue
                        rows, ga, go, ib = d[s_app]["opt"]
                        sc = ib.copy()
                        idx = [j for j, r in enumerate(rows) if r is not None]
                        if idx:
                            Xa = np.asarray([rows[j] for j in idx], np.float32)
                            pa = models["add"].predict(Xa) if "add" in models else np.full(len(idx), -9.0)
                            po = models["only"].predict(Xa) if "only" in models else np.full(len(idx), -9.0)
                            for q, j in enumerate(idx):
                                if max(pa[q], po[q]) > mg:
                                    sc[j] += ga[j] if pa[q] >= po[q] else go[j]
                                    nch += 1
                        fam[p_] = sc.mean()
                    res[s_app].append(C.family_mean(fam) - ref[s_app])
            v, pr = np.mean(res["validation"]), np.mean(res["private"])
            print(f"  {'+'.join(opts):<9} margin {mg:<5} val {v:+.4f}±{np.std(res['validation']):.4f}  "
                  f"priv {pr:+.4f}±{np.std(res['private']):.4f}  mean {(v + pr) / 2:+.4f}  changes/run {nch / 12:.1f}",
                  flush=True)


if __name__ == "__main__":
    main()
