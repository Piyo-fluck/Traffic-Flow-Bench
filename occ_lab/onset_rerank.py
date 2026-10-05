"""Onset candidate utility reranker (step 1 of the set-decoder plan, note 152).

Oracle (note 152): choosing the right subset of the top-5 candidate regions (current shapes) lifts IoU 0.870 ->
0.946, ~80 % of it in single-site windows - the candidate ranking, not the extent, is what fails. Here the ranking
is relearned with the teacher changed from 'the region with the most true cells' (notes 129 / 145) to the actual
utility of each candidate:
    U_b = IoU(shape_b, Y),  shape_b = the ratio rule on p restricted to region b
(robust to one true queue being split by a region boundary). Scorers (cross-month, LightGBM seeds 0-5):
    cls        note 145 prior4 classifier (region label)                                  - current
    reg        regression on U_b                  (same features)
    rank       lambdarank on U_b (graded 0..10)   (same features)
    reg+ / rank+   + new candidate evidence: peak width / prominence of p, density k/kcrit last and 12-step slope,
                   up / downstream asymmetry of v/vf, q/cap and p, score features relative to the window's best
Decisions: 'swap' (note 129: exchange the heights of production's and the scorer's top regions, then the ratio rule;
the count stays as now) and 'pick' (output shape_b of the scorer's top candidate only).

    .venv/Scripts/python.exe occ_lab/onset_rerank.py
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

F = C.F


def sites_n(mask, gap=4):
    j = np.flatnonzero(mask)
    return 0 if not len(j) else 1 + int((np.diff(j) >= gap).sum())


def new_feats(p, cand, reg, v, qf, ok, vf, cap, kcrit):
    S = len(p)
    with np.errstate(all="ignore"):
        vv = np.where(ok, v / vf, np.nan)
        qq = np.where(ok, qf / cap, np.nan)
        kk = np.where(ok, qf / np.maximum(v, 1.0) / kcrit, np.nan)
        klast = np.nanmean(kk[-3:], 0)
        kslope = np.nanmean(kk[-3:], 0) - np.nanmean(kk[:3], 0)
        vl = np.nanmean(vv[-3:], 0)
        ql = np.nanmean(qq[-3:], 0)
    rows = []
    for r, j in enumerate(cand):
        mem = np.flatnonzero(reg == r)
        pk = p[j]
        width = int((p[mem] >= 0.5 * pk).sum())
        edge = min(p[mem.min()], p[mem.max()])
        lo, hi = slice(max(0, j - 3), j), slice(j + 1, min(S, j + 4))
        with np.errstate(all="ignore"):
            f = [width, pk - edge, (pk - edge) / max(pk, 1e-6),
                 np.nanmax(klast[max(0, j - 2):j + 3]), np.nanmax(kslope[max(0, j - 2):j + 3]),
                 np.nanmean(vl[lo]) - np.nanmean(vl[hi]), np.nanmean(ql[lo]) - np.nanmean(ql[hi]),
                 np.nansum(p[lo]) - np.nansum(p[hi])]
        rows.append(f)
    return np.asarray(rows, np.float32)


def prepare():
    data = {}
    for i, panel in enumerate(C.T2_PANELS):
        B = BC.build(i, panel)
        ctx, stat, W = C.windows(i, panel)
        vf, cap = np.asarray(ctx.vf, np.float32), np.asarray(ctx.cap, np.float32)
        kcrit = np.asarray(ctx.kcrit, np.float32)
        Pr = BK.prior_stats(panel, vf)
        for s, D in B.items():
            A = BK.day_arrays(W[s]["z"], vf)
            on = D["on"]
            D["Xp"] = np.concatenate([D["X"], BK.prior_features(D, Pr), BK.capacity_features_trunc(D, A)], 1)
            U, NF, shapes = [], [], {}
            for n, cand, reg, mass in D["meta"]:
                p = D["base"][n]
                t, e = on["fut"][n, -1], on["efut"][n, -1]
                sh = []
                for r in range(len(cand)):
                    a = C.select(p * (reg == r), "ratio", 1.0) & (reg == r)
                    sh.append(a)
                    ae = a & e
                    u = (ae | t).sum()
                    U.append((ae & t).sum() / u if u else 1.0)
                shapes[n] = sh
                NF.append(new_feats(p, cand, reg, on["v"][n], on["qf"][n], on["ok"][n], vf, cap, kcrit))
            D["U"] = np.asarray(U, np.float32)
            D["NF"] = np.concatenate(NF)
            D["shapes"] = shapes
            D["single"] = np.array([sites_n(on["fut"][n, -1] & on["efut"][n, -1]) <= 1 for n in range(len(on["d"]))])
        data[panel] = B
        print(f"  {panel}", flush=True)

    def rel(X, g):
        """window-relative copies: value minus the window's max, for the first 5 (production evidence) columns."""
        out = np.empty((len(X), 5), np.float32)
        off = 0
        for k in g:
            out[off:off + k] = X[off:off + k, :5] - X[off:off + k, :5].max(0, keepdims=True); off += k
        return out
    for d in data.values():
        for D in d.values():
            D["Xn"] = np.concatenate([D["Xp"], D["NF"], rel(D["Xp"], D["g"])], 1)
    return data


def main() -> None:
    warnings.simplefilter("ignore")
    data = prepare()
    base_par = {"learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 30, "feature_fraction": 0.8,
                "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "num_threads": 8}
    SC = {"cls": ("Xp", "cls"), "reg": ("Xp", "reg"), "rank": ("Xp", "rank"), "reg+": ("Xn", "reg"),
          "rank+": ("Xn", "rank"), "cls+": ("Xn", "cls")}
    ref = {}
    for s in C.SPLITS:
        fa, fs = {}, []
        for p_, d in data.items():
            if s in d:
                sc = C.onset_iou(d[s]["base"], d[s]["on"])
                fa[p_] = sc.mean(); fs.append(sc[d[s]["single"]])
        ref[s] = (C.family_mean(fa), np.concatenate(fs).mean())
    res = {}
    for sd in range(6):
        for nm, (key, obj) in SC.items():
            for s_fit, s_app in (("validation", "private"), ("private", "validation")):
                Ds = [d[s_fit] for d in data.values() if s_fit in d]
                X = np.concatenate([D[key] for D in Ds])
                if obj == "cls":
                    m = lgb.train({**base_par, "objective": "binary", "seed": sd},
                                  lgb.Dataset(X, np.concatenate([D["y"] for D in Ds])), 200)
                elif obj == "reg":
                    m = lgb.train({**base_par, "objective": "regression", "seed": sd},
                                  lgb.Dataset(X, np.concatenate([D["U"] for D in Ds])), 200)
                else:
                    lab = np.minimum((np.concatenate([D["U"] for D in Ds]) * 10).round().astype(int), 10)
                    m = lgb.train({**base_par, "objective": "lambdarank", "seed": sd, "label_gain": list(range(11)),
                                   "min_data_in_leaf": 20}, lgb.Dataset(X, lab, group=np.concatenate([D["g"] for D in Ds])), 200)
                for dec in ("swap", "pick"):
                    fa, fs = {}, []
                    ch = up = dn = 0
                    for p_, d in data.items():
                        if s_app not in d:
                            continue
                        D = d[s_app]
                        sc_ = m.predict(D[key])
                        P2 = D["base"].copy()
                        pred_pick = {}
                        off = 0
                        b0 = C.onset_iou(D["base"], D["on"])
                        for (n, cand, reg, mass), g in zip(D["meta"], D["g"]):
                            s_ = sc_[off:off + g]
                            if dec == "swap":
                                pc = s_ - s_.min() + 1e-3 if obj != "cls" else s_
                                P2[n] = BC.swap(D["base"][n], cand, reg, mass, pc)
                            else:
                                pred_pick[n] = D["shapes"][n][int(np.argmax(s_))]
                            if int(np.argmax(s_)) != int(np.argmax(mass)):
                                ch += 1
                            off += g
                        if dec == "swap":
                            sc = C.onset_iou(P2, D["on"])
                        else:
                            pr = np.stack([pred_pick[n] if n in pred_pick else C.select(D["base"][n], "ratio", 1.0)
                                           for n in range(len(D["base"]))])
                            full = np.zeros(D["on"]["fut"].shape, bool); full[:, F - 1] = pr
                            sc = C.iou(full, D["on"]["fut"], D["on"]["efut"])
                        up += int((sc > b0 + 1e-9).sum()); dn += int((sc < b0 - 1e-9).sum())
                        fa[p_] = sc.mean(); fs.append(sc[D["single"]])
                    res.setdefault((nm, dec, s_app), []).append(
                        (C.family_mean(fa) - ref[s_app][0], np.concatenate(fs).mean() - ref[s_app][1], ch, up, dn))
        print(f"  seed {sd} done", flush=True)
    print(f"\nproduction: all val {ref['validation'][0]:.4f} priv {ref['private'][0]:.4f}; single-site windows val "
          f"{ref['validation'][1]:.4f} priv {ref['private'][1]:.4f}")
    print(f"{'scorer / decision':<16}{'all val':>9}{'all priv':>9}{'single val':>11}{'single priv':>12}"
          f"   top changed (val/priv)   windows better/worse vs production (val | priv)")
    for nm in SC:
        for dec in ("swap", "pick"):
            r = {s: np.array(res[(nm, dec, s)]) for s in C.SPLITS}
            print(f"{nm + ' / ' + dec:<16}{r['validation'][:, 0].mean():>+9.4f}{r['private'][:, 0].mean():>+9.4f}"
                  f"{r['validation'][:, 1].mean():>+11.4f}{r['private'][:, 1].mean():>+12.4f}"
                  f"   {r['validation'][:, 2].mean():5.1f} / {r['private'][:, 2].mean():5.1f}"
                  f"   {r['validation'][:, 3].mean():4.1f}/{r['validation'][:, 4].mean():4.1f} | "
                  f"{r['private'][:, 3].mean():4.1f}/{r['private'][:, 4].mean():4.1f}")


if __name__ == "__main__":
    main()
