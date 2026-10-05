"""Ongoing: the onset machinery (notes 145 / 152) on the queues AWAY from the visible one (note 159: 2/3 of the loss).

Per ongoing window: the T-queue (observed queued cells of the last slot), its zone (<= 2 links); outside the zone,
candidate regions around the top peaks (<= 5, >= 4 links apart, p >= 0.05) of AU's max-over-horizons probability,
each region = the outside-zone links within +-4 of its peak. For each region and each logit shift
delta in (-3, -1.5, +1.5, +3) applied to all of the region's cells, the teacher is the window's actual IoU change
(tau 0.5). A LightGBM regressor on (region features, delta) predicts the change; per region the best delta is taken
when its prediction exceeds a margin (regions decided independently, applied together).

Region features (no link id): AU p at the peak and the region mean per horizon, width, distance to the T-queue, the
7 members' max p in the region, v/vf last / min / trend and q/cap last / trend at the peak (+-2), observed share in the
last 15 min, note 145 capacity headroom (training-month prior + the month's last <= 4 days), time, delta.
Cross-month (fit one scored month, apply the other), LightGBM seeds 0-5; all windows and days 1-5.

    .venv/Scripts/python.exe occ_lab/og_away.py
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
import common as C  # noqa: E402
import select_bank as SB  # noqa: E402
from online_adapter import ETA, GATE, global_features  # noqa: E402

F = C.F
DELTAS = (-3.0, -1.5, 1.5, 3.0)
MARGINS = (0.0, 0.005, 0.01, 0.02)


def peaks(p, zone, k=5, sep=4, pmin=0.05):
    q = np.where(zone, -1.0, p).copy()
    out = []
    for _ in range(k):
        j = int(np.argmax(q))
        if q[j] < pmin:
            break
        out.append(j)
        q[max(0, j - sep + 1):j + sep] = -1
    return out


def iou_w(pred, y, e):
    u = ((pred | y) & e).sum()
    return (pred & y & e).sum() / u if u else 1.0


def main() -> None:
    warnings.simplefilter("ignore")
    gbm = lgb.Booster(model_file=str(LAB / "models" / "residual_lgb.txt"))

    def corr(base, P):
        r = gbm.predict(global_features(P, base).reshape(-1, 15)).reshape(base.shape).astype(np.float32)
        return np.clip(base + ETA * np.where(np.abs(r) >= GATE, r, 0.0), 0, 1)

    data = {}
    for i, panel in enumerate(C.T2_PANELS):
        ctx, stat, W = C.windows(i, panel)
        vf, cap = np.asarray(ctx.vf, np.float32), np.asarray(ctx.cap, np.float32)
        Pr = BK.prior_stats(panel, vf)
        for s in C.SPLITS:
            B = SB.bank_for(i, panel, s, "solid3", "")
            M0 = B["M"].copy(); M0[3] = B["ps_ship"]
            q = corr(np.tensordot(SB.WAU, M0, 1), B["P"])
            og, z = W[s]["og"], W[s]["z"]
            A = BK.day_arrays(z, vf)
            y, e = B["y"] > 0.5, B["e"] > 0.5
            N, _, S = q.shape
            lg = np.log(np.clip(q, 1e-5, 1 - 1e-5)) - np.log1p(-np.clip(q, 1e-5, 1 - 1e-5))
            rows, gains, meta, regions = [], [], [], {}
            rrows, rlab, rmeta_n = [], [], []
            base_iou = np.array([iou_w(q[n] >= 0.5, y[n], e[n]) for n in range(N)])
            for n in range(N):
                ok = og["ok"][n]
                vv = np.where(ok, og["v"][n] / vf, np.nan)
                qq = np.where(ok, og["qf"][n] / cap, np.nan)
                qT = ok[-1] & (np.nan_to_num(og["v"][n, -1], nan=1e9) <= 0.6 * vf)
                zone = np.zeros(S, bool)
                for j in np.flatnonzero(qT):
                    zone[max(0, j - 2):j + 3] = True
                pmax = q[n].max(0)
                cand = peaks(pmax, zone)
                if not cand:
                    continue
                cand = np.array(cand)
                d, t0 = int(og["d"][n]), int(og["t0"][n])
                lo = max(0, d - 4)
                As = {k: w[lo:d + 1] for k, w in A.items()}
                cap_f = np.concatenate([BK.prior_rows(Pr, t0, og["qf"][n], ok, cand),
                                        BK.cap_rows(As, d - lo, t0, og["qf"][n], ok, cand)], 1)
                tq = np.flatnonzero(qT)
                with np.errstate(all="ignore"):
                    vmin = np.nanmin(vv, 0); vtr = np.nanmean(vv[-3:], 0) - np.nanmean(vv[:3], 0)
                    qtr = np.nanmean(qq[-3:], 0) - np.nanmean(qq[:3], 0)
                    vlast = np.nanmean(vv[-2:], 0); qlast = np.nanmean(qq[-2:], 0)
                obs15 = ok[-3:].mean(0)
                for r, j in enumerate(cand):
                    reg = np.zeros(S, bool)
                    reg[max(0, j - 4):j + 5] = True
                    reg &= ~zone
                    regions.setdefault(n, []).append(reg)
                    rlab.append([bool((y[n][:, reg] & e[n][:, reg]).any()), bool((y[n][-3:, reg] & e[n][-3:, reg]).any())] +
                                [bool((y[n][h, reg] & e[n][h, reg]).any()) for h in range(F)])
                    rmeta_n.append((n, len(regions[n]) - 1))
                    nb = slice(max(0, j - 2), j + 3)
                    dist = np.abs(tq - j).min() if len(tq) else 99
                    with np.errstate(all="ignore"):
                        f = [pmax[j], *q[n][:, reg].mean(1), reg.sum(), min(dist, 99),
                             *[B["M"][k, n][:, reg].max() for k in range(7)],
                             np.nanmean(vlast[nb]), np.nanmin(vmin[nb]), np.nanmean(vtr[nb]),
                             np.nanmean(qlast[nb]), np.nanmean(qtr[nb]), obs15[nb].mean(),
                             *cap_f[r], t0 / 12.0, len(cand)]
                    rrows.append(f)
                    for dl in DELTAS:
                        q2 = q[n].copy()
                        q2[:, reg] = 1 / (1 + np.exp(-(lg[n][:, reg] + dl)))
                        rows.append(f + [dl]); gains.append(iou_w(q2 >= 0.5, y[n], e[n]) - base_iou[n])
                        meta.append((n, r, dl))
            data[(panel, s)] = dict(X=np.asarray(rows, np.float32), g=np.asarray(gains, np.float32), meta=meta,
                                    regions=regions, R=np.asarray(rrows, np.float32), RL=np.asarray(rlab), RM=rmeta_n,
                                    q=q, lg=lg, y=y, e=e, d=B["d"], base=base_iou, og=og)
        print(f"  {panel}", flush=True)
    ref = {(s, sub): C.family_mean({p: (data[(p, s)]["base"] if sub == "all" else
                                        data[(p, s)]["base"][data[(p, s)]["d"] <= 4]).mean() for p in C.T2_PANELS})
           for s in C.SPLITS for sub in ("all", "d15")}
    if "--reweight" in sys.argv:
        BETAS = (0.25, 0.5, 0.75, 1.0)
        par2 = {"objective": "binary", "learning_rate": 0.03, "num_leaves": 15, "min_data_in_leaf": 50,
                "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "num_threads": 8}

        def hrows(D):
            """(region, h) rows: region features + h + AU region mean at h; labels at h."""
            R, L = D["R"], D["RL"][:, 2:]
            X = np.concatenate([np.concatenate([R, np.full((len(R), 1), h), R[:, 1 + h:2 + h]], 1) for h in range(F)])
            return X, np.concatenate([L[:, h] for h in range(F)]).astype(int)

        out = {}
        for sd in range(6):
            for s_fit, s_app in (("validation", "private"), ("private", "validation")):
                Xf, yf = zip(*[hrows(data[(p, s_fit)]) for p in C.T2_PANELS if len(data[(p, s_fit)]["R"])])
                Xf, yf = np.concatenate(Xf), np.concatenate(yf)
                full = lgb.train({**par2, "seed": sd}, lgb.Dataset(Xf, yf), 300)
                base = lgb.train({**par2, "seed": sd}, lgb.Dataset(Xf[:, -1:], yf), 100)   # AU's region p alone
                for p in C.T2_PANELS:
                    D = data[(p, s_app)]
                    if not len(D["R"]):
                        for b in BETAS:
                            out.setdefault((s_app, b), {}).setdefault(sd, {})[p] = (D["base"].mean(), D["base"][D["d"] <= 4].mean())
                        continue
                    Xa, _ = hrows(D)
                    lt = lambda z: np.log(np.clip(z, 1e-4, 1 - 1e-4)) - np.log1p(-np.clip(z, 1e-4, 1 - 1e-4))  # noqa: E731
                    dl = (lt(full.predict(Xa)) - lt(base.predict(Xa[:, -1:]))).reshape(F, -1).T   # (regions, F)
                    for b in BETAS:
                        new = D["base"].copy()
                        byw = {}
                        for k, (n, r) in enumerate(D["RM"]):
                            byw.setdefault(n, []).append((r, dl[k]))
                        for n, lst in byw.items():
                            q2 = D["lg"][n].copy()
                            for r, dd in lst:
                                reg = D["regions"][n][r]
                                q2[:, reg] = q2[:, reg] + b * dd[:, None]
                            new[n] = iou_w(q2 >= 0.0, D["y"][n], D["e"][n])
                        out.setdefault((s_app, b), {}).setdefault(sd, {})[p] = (new.mean(), new[D["d"] <= 4].mean())
            print(f"  seed {sd} done", flush=True)
        print()
        print(f"{'beta':<6}{'all val':>9}{'all priv':>10}{'d1-5 val':>10}{'d1-5 priv':>11}")
        for b in BETAS:
            v = [np.mean([C.family_mean({p: out[(s, b)][sd][p][k] for p in C.T2_PANELS}) for sd in range(6)]) - ref[(s, sub)]
                 for k, sub in ((0, "all"), (1, "d15")) for s in C.SPLITS]
            print(f"{b:<6}{v[0]:>+9.4f}{v[1]:>+10.4f}{v[2]:>+10.4f}{v[3]:>+11.4f}")
        return
    if "--auc" in sys.argv:
        from sklearn.metrics import roc_auc_score
        for li, lname in ((0, "any horizon"), (1, "T+20..T+30")):
            for s_fit, s_app in (("validation", "private"), ("private", "validation")):
                Xf = np.concatenate([data[(p, s_fit)]["R"] for p in C.T2_PANELS])
                yf = np.concatenate([data[(p, s_fit)]["RL"][:, li] for p in C.T2_PANELS])
                Xa = np.concatenate([data[(p, s_app)]["R"] for p in C.T2_PANELS])
                ya = np.concatenate([data[(p, s_app)]["RL"][:, li] for p in C.T2_PANELS])
                m = lgb.train({"objective": "binary", "learning_rate": 0.03, "num_leaves": 15, "min_data_in_leaf": 50,
                               "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1,
                               "num_threads": 8}, lgb.Dataset(Xf, yf), 300)
                pm = Xa[:, 1:1 + F].max(1)
                band = (pm >= 0.2) & (pm <= 0.8)
                pc = m.predict(Xa)
                print(f"   band 0.2-0.8: regions {band.sum()}, positive {ya[band].mean():.3f}; AUC AU {roc_auc_score(ya[band], pm[band]):.4f}"
                      f" classifier {roc_auc_score(ya[band], pc[band]):.4f}")
                print(f"[{lname}] applied to {s_app}: regions {len(ya)}, positive {ya.mean():.3f}; AUC AU peak p "
                      f"{roc_auc_score(ya, Xa[:, 0]):.4f}, AU region mean p {roc_auc_score(ya, Xa[:, 1:1 + F].max(1)):.4f}, "
                      f"classifier {roc_auc_score(ya, m.predict(Xa)):.4f}")
        return
    for s in C.SPLITS:
        g = np.concatenate([data[(p, s)]["g"] for p in C.T2_PANELS])
        X = np.concatenate([data[(p, s)]["X"] for p in C.T2_PANELS])
        for dl in DELTAS:
            m = X[:, -1] == dl
            print(f"{s} delta {dl:+}: rows {m.sum()}, gain > 0 {int((g[m] > 1e-9).sum())} (sum {g[m][g[m] > 0].sum():+.1f}),"
                  f" < 0 {int((g[m] < -1e-9).sum())} (sum {g[m][g[m] < 0].sum():+.1f})")
    par = {"objective": "regression", "learning_rate": 0.03, "num_leaves": 15, "min_data_in_leaf": 50,
           "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "num_threads": 8}
    res = {}
    for sd in range(6):
        for s_fit, s_app in (("validation", "private"), ("private", "validation")):
            X = np.concatenate([data[(p, s_fit)]["X"] for p in C.T2_PANELS])
            y = np.concatenate([data[(p, s_fit)]["g"] for p in C.T2_PANELS])
            mdl = lgb.train({**par, "seed": sd}, lgb.Dataset(X, y), 300)
            for mg in MARGINS:
                fa, fd, ch = {}, {}, 0
                for p in C.T2_PANELS:
                    D = data[(p, s_app)]
                    if not len(D["X"]):
                        fa[p] = D["base"].mean(); fd[p] = D["base"][D["d"] <= 4].mean(); continue
                    pr = mdl.predict(D["X"])
                    best = {}
                    for k, (n, r, dl) in enumerate(D["meta"]):
                        if pr[k] > mg and pr[k] > best.get((n, r), (None, -1))[1]:
                            best[(n, r)] = (dl, pr[k])
                    new = D["base"].copy()
                    byw = {}
                    for (n, r), (dl, _) in best.items():
                        byw.setdefault(n, []).append((r, dl))
                    for n, lst in byw.items():
                        q2 = D["q"][n].copy()
                        for r, dl in lst:
                            reg = D["regions"][n][r]
                            q2[:, reg] = 1 / (1 + np.exp(-(D["lg"][n][:, reg] + dl)))
                        new[n] = iou_w(q2 >= 0.5, D["y"][n], D["e"][n]); ch += 1
                    fa[p] = new.mean(); fd[p] = new[D["d"] <= 4].mean()
                res.setdefault((s_app, mg), []).append((C.family_mean(fa) - ref[(s_app, "all")],
                                                         C.family_mean(fd) - ref[(s_app, "d15")], ch))
        print(f"  seed {sd} done", flush=True)
    print(f"\n{'margin':<8}{'all val':>9}{'all priv':>10}{'d1-5 val':>10}{'d1-5 priv':>11}{'windows changed':>17}")
    for mg in MARGINS:
        r = {s: np.array(res[(s, mg)]) for s in C.SPLITS}
        print(f"{mg:<8}{r['validation'][:, 0].mean():>+9.4f}{r['private'][:, 0].mean():>+10.4f}"
              f"{r['validation'][:, 1].mean():>+10.4f}{r['private'][:, 1].mean():>+11.4f}"
              f"{r['validation'][:, 2].mean():>9.0f} /{r['private'][:, 2].mean():>6.0f}")


if __name__ == "__main__":
    main()
