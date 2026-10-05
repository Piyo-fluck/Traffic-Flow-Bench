"""Stage 1 (which bottleneck) learned on the TRAINING months with out-of-fold H (occ_lab/h_oof.py).

The note 129/145 classifier sees ~330 labelled onset windows (two scored months, cross-month). The training months
hold ~1,700 first-onset windows; with H scored out of fold (month folds, priors without the month) they give the
same candidate competition with an honest member probability. The big classifier sees only what exists there:
H's evidence (peak, log, rank, peak / max, region mass share, H at the peak), current state, structure, same-month
evidence (earlier days of the month), time, and note 145's capacity (training-month prior WITHOUT the window's
month + the month's last <= 4 earlier days).

On the scored months the candidates are the production blend's peaks (as notes 129/145) and the features are
computed with the production H (ref on_H) - the big classifier's score pc_big then enters as
    B   the swap decision alone
    A   a stacked feature of the cross-month prior4 classifier (pc_big, its share within the window)
Cross-month, LightGBM seeds 0-5; the big model is fitted once per seed on all training months.

    .venv/Scripts/python.exe occ_lab/stage1_big.py
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
CACHE = LAB / "cache"


def prior_stats_mask(zt, vf, mask):
    """BK.prior_stats on the training days in mask only."""
    z = {k: zt[k][mask] for k in ("obs_speed", "obs_flow", "elig")}
    A = BK.day_arrays(z, vf)
    qq, free, brk, pre = A["qq"], A["free"], A["brk"], A["pre"]
    S = qq.shape[2]
    with np.errstate(all="ignore"):
        q95 = np.nanpercentile(np.where(free, qq, np.nan).reshape(-1, S), 95, axis=0)
        qb = np.nanmedian(np.where(brk, pre, np.nan).reshape(-1, S), axis=0)
        rate = brk.sum((0, 1)) / qq.shape[0]
        a0, a1 = qq[:, :-(F - 1) - 1], qq[:, F:]
        okr = free[:, :-(F - 1) - 1] & free[:, F:] & (a0 > 0)
        R = np.full((qq.shape[1], S), np.nan, np.float32)
        R[1:1 + a0.shape[1]] = np.nanmedian(np.where(okr, a1 / a0, np.nan), 0)
    return dict(q95=q95, qb=qb, rate=rate, R=R)


def rows_h(pH, cand, v, qf, ok, vf, cap, lanes, stat, qd, nd, d, t0):
    """note 129's candidate rows with H as the only member, for GIVEN candidates."""
    S = len(pH)
    reg = np.abs(np.arange(S)[:, None] - cand[None, :]).argmin(1)
    vv = np.where(ok, v / vf, np.nan)
    qq = np.where(ok, qf / cap, np.nan)
    vlast = BC.last_obs(np.nan_to_num(vv, nan=0), ok)
    qlast = BC.last_obs(np.nan_to_num(qq, nan=0), ok)
    with np.errstate(all="ignore"):
        vmin = np.nanmin(vv, 0)
        qslope = np.nanmean(qq[-3:], 0) - np.nanmean(qq[:3], 0)
        share_all = qd[:d].sum(0) / np.maximum(nd[:d].sum(0), 1) if d else np.full(S, np.nan)
        share_last = qd[d - 1] / np.maximum(nd[d - 1], 1) if d else np.full(S, np.nan)
    mass = np.bincount(reg, weights=pH, minlength=len(cand))
    hc = pH[cand]
    rank = np.argsort(np.argsort(-hc))
    rows = []
    for r, j in enumerate(cand):
        nb = slice(max(0, j - 2), j + 3)
        with np.errstate(all="ignore"):
            rows.append([hc[r], np.log(hc[r] + 1e-6), rank[r], hc[r] / max(pH.max(), 1e-9),
                         mass[r] / max(mass.sum(), 1e-9), np.nanmax(pH[nb]),
                         vlast[j], np.nanmin(vmin[nb]) if np.isfinite(vmin[nb]).any() else np.nan,
                         qlast[j], qslope[j], lanes[j], lanes[min(j + 1, S - 1)] - lanes[j], stat[3, j], stat[4, j],
                         np.nanmax(share_all[nb]) if d else np.nan, np.nanmax(share_last[nb]) if d else np.nan,
                         float(d), t0 / 12.0])
    return np.asarray(rows, np.float32), reg


def main() -> None:
    warnings.simplefilter("ignore")
    from tfb.data import load
    train_X, train_y, train_g, train_meta = [], [], [], []
    data = {}
    for i, panel in enumerate(C.T2_PANELS):
        ctx, stat, W = C.windows(i, panel)
        vf, cap = np.asarray(ctx.vf, np.float32), np.asarray(ctx.cap, np.float32)
        lanes = np.asarray(ctx.lanes, np.float32)
        S = len(vf)
        zt = load(panel, "train")
        mon = np.array([str(x)[:7] for x in zt["dates"]])
        H = np.load(CACHE / f"h_oof_{panel}.npz")
        pr_by_m = {m: prior_stats_mask(zt, vf, mon != m) for m in np.unique(mon)}
        # ---- training months
        n_ok = 0
        for m in np.unique(mon):
            days = np.flatnonzero(mon == m)
            zm = {k: zt[k][days] for k in ("obs_speed", "obs_flow", "elig", "weekday")}
            qd, nd = BC.month_share(zm, vf)
            A = BK.day_arrays(zm, vf)
            Pr = pr_by_m[m]
            sel = np.flatnonzero(H["month"] == m)
            for n in sel:
                pH = H["prob"][n]
                if not np.isfinite(pH).all():
                    continue
                cand = BC.peaks(pH)
                if len(cand) < 2:
                    continue
                d = int(np.searchsorted(days, H["d"][n]))
                t0 = int(H["t0"][n])
                X, reg = rows_h(pH, cand, H["v"][n], H["qf"][n], H["ok"][n], vf, cap, lanes, stat, qd, nd, d, t0)
                lo = max(0, d - 4)
                As = {k: w[lo:d + 1] for k, w in A.items()}
                Xc = np.concatenate([BK.prior_rows(Pr, t0, H["qf"][n], H["ok"][n], cand),
                                     BK.cap_rows(As, d - lo, t0, H["qf"][n], H["ok"][n], cand)], 1)
                y30 = H["fut"][n, -1] & H["efut"][n, -1]
                hits = np.bincount(reg[y30], minlength=len(cand))
                if not hits.sum():
                    continue
                train_meta.append((m, int(np.argmax(np.bincount(reg, weights=pH, minlength=len(cand))))))
                train_X.append(np.concatenate([X, Xc], 1))
                train_y.append((np.arange(len(cand)) == np.argmax(hits)).astype(int))
                train_g.append(len(cand))
                n_ok += 1
        # ---- scored months (lab pool): candidates from the production blend, features with the production H
        B = BC.build(i, panel)
        Pr_all = BK.prior_stats(panel, vf)
        for s, D in B.items():
            z = W[s]["z"]
            on = D["on"]
            Hq = W[s]["ref"]["on_H"]
            qd, nd = BC.month_share(z, vf)
            A = BK.day_arrays(z, vf)
            D["Xp"] = np.concatenate([D["X"], BK.prior_features(D, Pr_all), BK.capacity_features_trunc(D, A)], 1)
            xb = []
            for n, cand, reg, mass in D["meta"]:
                d, t0 = int(on["d"][n]), int(on["t0"][n])
                X, _ = rows_h(Hq[n], cand, on["v"][n], on["qf"][n], on["ok"][n], vf, cap, lanes, stat, qd, nd, d, t0)
                lo = max(0, d - 4)
                As = {k: w[lo:d + 1] for k, w in A.items()}
                Xc = np.concatenate([BK.prior_rows(Pr_all, t0, on["qf"][n], on["ok"][n], cand),
                                     BK.cap_rows(As, d - lo, t0, on["qf"][n], on["ok"][n], cand)], 1)
                xb.append(np.concatenate([X, Xc], 1))
            D["Xbig"] = np.concatenate(xb)
            D["Hq"] = Hq
            D["hist"] = [(on["v"][n], on["qf"][n], on["ok"][n]) for n in range(len(on["d"]))]
            D["ctxv"] = (vf, cap, lanes, stat)
            D["msd"] = (qd, nd)
            D["capfn"] = (lambda A_, on_, Pr_: lambda n, cand: np.concatenate(
                [BK.prior_rows(Pr_, int(on_["t0"][n]), on_["qf"][n], on_["ok"][n], cand),
                 BK.cap_rows({k: w[max(0, int(on_["d"][n]) - 4):int(on_["d"][n]) + 1] for k, w in A_.items()},
                             int(on_["d"][n]) - max(0, int(on_["d"][n]) - 4), int(on_["t0"][n]), on_["qf"][n],
                             on_["ok"][n], cand)], 1))(A, on, Pr_all)
        data[panel] = B
        print(f"  {panel}: training-month events {n_ok}", flush=True)
    TX, Ty = np.concatenate(train_X), np.concatenate(train_y)
    print(f"training rows {len(Ty):,}, windows {len(train_g)}")
    # leave-one-month-out on the training months: does the classifier beat H's own mass?
    TG = np.asarray(train_g); offs = np.r_[0, np.cumsum(TG)[:-1]]
    wm = np.array([m for m, _ in train_meta]); hm = np.array([h for _, h in train_meta])
    lab = np.array([int(np.argmax(Ty[o:o + g])) for o, g in zip(offs, TG)])
    hit = []
    for m in np.unique(wm):
        tr = np.flatnonzero(wm != m); te = np.flatnonzero(wm == m)
        rows_tr = np.concatenate([np.arange(offs[k], offs[k] + TG[k]) for k in tr])
        mm = lgb.train({"objective": "binary", "learning_rate": 0.03, "num_leaves": 15, "min_data_in_leaf": 40,
                        "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1,
                        "num_threads": 8, "seed": 0}, lgb.Dataset(TX[rows_tr], Ty[rows_tr]), 400)
        for k in te:
            pc = mm.predict(TX[offs[k]:offs[k] + TG[k]])
            hit.append((int(np.argmax(pc) == lab[k]), int(hm[k] == lab[k])))
    hit = np.array(hit)
    print(f"training months, leave-one-month-out top-1: classifier {hit[:, 0].mean():.3f} vs H mass {hit[:, 1].mean():.3f} "
          f"(n {len(hit)})", flush=True)
    # member-level stage 1: swap H's own peaks with the classifier, then re-blend (ratio rule, no further swap)
    big0 = lgb.train({"objective": "binary", "learning_rate": 0.03, "num_leaves": 15, "min_data_in_leaf": 40,
                      "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1,
                      "num_threads": 8, "seed": 0}, lgb.Dataset(TX, Ty), 400)
    for s_ in C.SPLITS:
        famH, famH2, famB, famB2 = {}, {}, {}, {}
        for p_, d in data.items():
            if s_ not in d:
                continue
            D = d[s_]
            Hq, H2 = D["Hq"], D["Hq"].copy()
            for n in range(len(Hq)):
                cand = BC.peaks(Hq[n])
                if len(cand) < 2:
                    continue
                X, reg = rows_h(Hq[n], cand, *D["hist"][n], *D["ctxv"], *D["msd"], int(D["on"]["d"][n]),
                                int(D["on"]["t0"][n]))
                lo = max(0, int(D["on"]["d"][n]) - 4)
                Xc = D["capfn"](n, cand)
                pc = big0.predict(np.concatenate([X, Xc], 1))
                mass = np.bincount(reg, weights=Hq[n], minlength=len(cand))
                H2[n] = BC.swap(Hq[n], cand, reg, mass, pc)
            famH[p_] = C.onset_iou(Hq, D["on"]).mean(); famH2[p_] = C.onset_iou(H2, D["on"]).mean()
            base2 = D["base"] + 0.35 * (H2 - Hq)
            famB[p_] = C.onset_iou(D["base"], D["on"]).mean(); famB2[p_] = C.onset_iou(base2, D["on"]).mean()
        print(f"  {s_}: H alone {C.family_mean(famH):.4f} -> swapped H {C.family_mean(famH2):.4f};  "
              f"blend {C.family_mean(famB):.4f} -> blend with swapped H {C.family_mean(famB2):.4f}", flush=True)
    # H-only top-1 on the training months (sanity) and on the scored months
    par = {"objective": "binary", "learning_rate": 0.03, "num_leaves": 15, "min_data_in_leaf": 40,
           "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "num_threads": 8}
    ref = {s: C.family_mean({p_: C.onset_iou(d[s]["base"], d[s]["on"]).mean() for p_, d in data.items() if s in d})
           for s in C.SPLITS}
    res = {k: {s: [] for s in C.SPLITS} for k in ("prior4 (145)", "B big alone", "A prior4 + pc_big")}
    top1 = {k: [] for k in ("production mass", "big", "prior4 (145)", "A")}
    for sd in range(int(sys.argv[1]) if len(sys.argv) > 1 else 6):
        big = lgb.train({**par, "seed": sd}, lgb.Dataset(TX, Ty), 400)
        for d in data.values():
            for D in d.values():
                pb = big.predict(D["Xbig"])
                D["pb"] = pb
                share = np.empty_like(pb)
                off = 0
                for g in D["g"]:
                    share[off:off + g] = pb[off:off + g] / max(pb[off:off + g].sum(), 1e-9); off += g
                D["XA"] = np.concatenate([D["Xp"], pb[:, None], share[:, None]], 1)
        for s_fit, s_app in (("validation", "private"), ("private", "validation")):
            ms = {}
            for k, key in (("prior4 (145)", "Xp"), ("A prior4 + pc_big", "XA")):
                X = np.concatenate([d[s_fit][key] for d in data.values() if s_fit in d])
                y = np.concatenate([d[s_fit]["y"] for d in data.values() if s_fit in d])
                ms[k] = (lgb.train({**par, "learning_rate": 0.05, "min_data_in_leaf": 30, "seed": sd},
                                   lgb.Dataset(X, y), 200), key)
            fam = {k: {} for k in res}
            for p_, d in data.items():
                if s_app not in d:
                    continue
                D = d[s_app]
                probs = {k: m.predict(D[key]) for k, (m, key) in ms.items()}
                probs["B big alone"] = D["pb"]
                for k, prob in probs.items():
                    P2 = D["base"].copy()
                    off = 0
                    for (n, cand, reg, mass), g in zip(D["meta"], D["g"]):
                        pc, yy = prob[off:off + g], D["y"][off:off + g]
                        P2[n] = BC.swap(D["base"][n], cand, reg, mass, pc)
                        if sd == 0 and yy.any():
                            nm = {"prior4 (145)": "prior4 (145)", "B big alone": "big", "A prior4 + pc_big": "A"}[k]
                            top1[nm].append(int(np.argmax(pc) == np.argmax(yy)))
                            if k == "B big alone":
                                top1["production mass"].append(int(np.argmax(mass) == np.argmax(yy)))
                        off += g
                    fam[k][p_] = C.onset_iou(P2, D["on"]).mean()
            for k in res:
                res[k][s_app].append(C.family_mean(fam[k]) - ref[s_app])
        print(f"  seed {sd}: " + "  ".join(f"{k[:10]} {np.mean([res[k][s][-1] for s in C.SPLITS]):+.4f}" for k in res),
              flush=True)
    print("\ntop-1 of the true bottleneck (seed 0, both months): " +
          ", ".join(f"{k} {np.mean(v):.3f}" for k, v in top1.items()))
    for k in res:
        v, pr = np.mean(res[k]["validation"]), np.mean(res[k]["private"])
        print(f"  {k:<20} val {v:+.4f}±{np.std(res[k]['validation']):.4f}  priv {pr:+.4f}±"
              f"{np.std(res[k]['private']):.4f}  mean {(v + pr) / 2:+.4f}")


if __name__ == "__main__":
    main()
