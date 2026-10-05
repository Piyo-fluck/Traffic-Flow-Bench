"""Ongoing decides with a flat threshold; IoU is a per-window set metric. Learn the per-window budget.

`occ_lab/selector_posterior.py` measured the ceiling of this axis on the scored months, with the posterior's own
per-cell ORDER left untouched and only the number of cells taken per window changed:

    AS, tau = 0.50                                        0.8505
    knowing each window's true |Y|, top-k by p            0.9006   +0.0501
    knowing each window's best prefix length              0.9267   +0.0762

So the per-cell ranking is already good and the per-window budget is wrong - which is what a flat threshold must
be, because IoU is not separable over cells. Onset has had a per-window budget since note 31 (the expected-IoU
ratio rule, +0.05..+0.09); ongoing kept the threshold because that rule LOSES there (-0.002..-0.041, note 31),
the cells being strongly correlated in space and time so the independent-Bernoulli calculus is wrong.

The fix is to stop deriving the budget and learn it. Target: the best prefix length k* under the posterior's own
order. Fitted on the OOF tail (days >= FIT_END, `cache_oof.py`), which is earlier than every scored origin and is
the same surface the residual corrector was fitted on, so it is admissible for both months.

Features are posterior-only, so nothing has to be recomputed on the scored side: the sorted probability profile,
the mass by horizon step, counts above several levels, the member spread, and the window's shape.

    python occ_lab/budget_regressor.py --tag s20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta"))
sys.path.insert(0, str(LAB.parent / "analysis"))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

import common as C  # noqa: E402
from online_adapter import ETA, GATE, global_features, load_split, mean_of  # noqa: E402
from residual_diag import TAU  # noqa: E402

CACHE = LAB / "cache"
W_PS = 0.60
QS = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 1.0])


def best_k(p, y, e):
    """k* and |Y| per window, under the posterior's own order."""
    n = len(p)
    ks, ny = np.zeros(n, int), np.zeros(n, int)
    for j in range(n):
        idx = np.flatnonzero(e[j].ravel())
        if not len(idx):
            continue
        pv, yv = p[j].ravel()[idx], (y[j] & e[j]).ravel()[idx]
        o = np.argsort(-pv)
        tot = int(yv.sum())
        ny[j] = tot
        if not tot:
            continue
        hit = np.cumsum(yv[o])
        k = np.arange(1, len(o) + 1)
        ks[j] = int(np.argmax(hit / np.maximum(k + tot - hit, 1))) + 1
    return ks, ny


def feats(p, e, P=None):
    """Posterior-only window descriptors, computable identically on both surfaces."""
    n, Fh, S = p.shape
    rows = []
    for j in range(n):
        idx = np.flatnonzero(e[j].ravel())
        pv = np.sort(p[j].ravel()[idx])[::-1] if len(idx) else np.zeros(1)
        ne = max(len(idx), 1)
        tot = float(pv.sum())
        prof = [float(pv[min(int(q * ne), ne - 1)]) for q in (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5)]
        cnt = [float((pv >= t).sum()) for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)]
        mass = [float((p[j, h] * e[j, h]).sum()) for h in range(Fh)]
        ent = float(-(pv * np.log(np.clip(pv, 1e-6, 1)) +
                      (1 - pv) * np.log(np.clip(1 - pv, 1e-6, 1))).sum())
        sd = float(P[:, j][:, e[j]].std(0).mean()) if P is not None else 0.0
        rows.append([ne, S, tot, tot / ne, float(pv[0]), ent, ent / ne, sd] + prof + cnt + mass
                    + [c / max(tot, 1e-6) for c in cnt[:4]])
    return np.asarray(rows, np.float64)


NAMES = (["n_elig", "n_links", "sum_p", "mean_p", "max_p", "entropy", "entropy_per_cell", "member_sd"]
         + [f"p_q{q}" for q in (1, 2, 5, 10, 20, 30, 50)]
         + [f"n_ge_{t}" for t in (20, 30, 40, 50, 60, 70, 80)]
         + [f"mass_h{h}" for h in range(6)]
         + [f"ratio_ge_{t}" for t in (20, 30, 40, 50)])


def apply_k(p, e, k):
    out = np.zeros_like(e)
    for j in range(len(p)):
        idx = np.flatnonzero(e[j].ravel())
        kk = int(np.clip(k[j], 0, len(idx)))
        if kk <= 0 or not len(idx):
            continue
        o = idx[np.argsort(-p[j].ravel()[idx])][:kk]
        f = out[j].ravel()
        f[o] = True
        out[j] = f.reshape(out[j].shape)
    return out


def hist_feats(ctx, d, t0, Fh):
    """Window observables from the released history, the axis the posterior-only set cannot see."""
    z = ctx.z
    v, okz = z["obs_speed"], z["elig"] & np.isfinite(z["obs_speed"]) & np.isfinite(z["obs_flow"])
    q = okz & (np.nan_to_num(v, nan=1e9) <= 0.6 * ctx.vf)
    hi = t0[:, None] + np.arange(-C.H, 0)
    qh, oh = q[d[:, None], hi], okz[d[:, None], hi]
    n = qh.shape[0]
    cnt = qh.sum(2).astype(np.float64)                       # queued links per history slot
    cov = oh.mean((1, 2)).astype(np.float64)
    lastc, firstc = cnt[:, -1], cnt[:, 0]
    slope = np.array([np.polyfit(np.arange(C.H), c, 1)[0] if c.any() else 0.0 for c in cnt])
    width = qh[:, -1].sum(1).astype(np.float64)
    runs = np.array([int((np.diff(np.flatnonzero(r).astype(int)) > 1).sum()) + (1 if r.any() else 0)
                     for r in qh[:, -1]], float)
    return np.stack([lastc, firstc, lastc - firstc, cnt.mean(1), cnt.max(1), slope, cov, width, runs,
                     (t0 / 12).astype(np.float64), z["weekday"][d].astype(np.float64),
                     lastc / np.maximum(qh.shape[2], 1)], 1)


HNAMES = ["q_last", "q_first", "q_delta", "q_mean", "q_max", "q_slope", "coverage", "width",
          "components", "hour", "weekday", "q_last_share"]


def oof_panel(panel, gbm):
    f, fs, fp = (CACHE / f"og_oof_{panel}.npz", CACHE / f"stid_oof_{panel}.npz",
                 CACHE / f"pseudo_oof_{panel}.npz")
    if not (f.exists() and fs.exists() and fp.exists()):
        return None
    z, zp = np.load(f), np.load(fp)
    P = {"gs1": z["gs1"].astype(np.float32), "tfm8": z["tfm8"].astype(np.float32),
         "cnn3": z["cnn3"].astype(np.float32), "occ_cnn": mean_of(z, "occ_cnn"),
         "occ_tfm": mean_of(z, "occ_tfm"), "stid": mean_of(np.load(fs), "stid")}
    old = 0.2 * P["gs1"] + 0.6 * P["tfm8"] + 0.2 * P["cnn3"]
    base = ((1 - W_PS) * old + W_PS * zp["pseudo"].astype(np.float32)).astype(np.float32)
    F = global_features(P, base)
    rh = gbm.predict(F.reshape(-1, F.shape[-1])).reshape(base.shape).astype(np.float32)
    pg = np.clip(base + ETA * np.where(np.abs(rh) >= GATE, rh, 0.0), 0, 1)
    return pg, z["y"], z["e"], np.stack([P[k] for k in ("gs1", "tfm8", "cnn3", "occ_cnn", "occ_tfm", "stid")])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="s20")
    ap.add_argument("--target", default="kstar", choices=("kstar", "ny"))
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--shrinks", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--hist", action="store_true", help="add window observables from the released history")
    ap.add_argument("--cv", type=int, default=5, help="report the OOF cross-validated R2 of the target")
    args = ap.parse_args()
    gbm = lgb.Booster(model_file=str(LAB / "models" / "residual_lgb.txt"))

    X, T, K0 = [], [], []
    names = list(NAMES)
    for i, panel in enumerate(C.T2_PANELS):
        r = oof_panel(panel, gbm)
        if r is None:
            continue
        pg, y, e, P = r
        ks, ny = best_k(pg, y, e)
        f0 = feats(pg, e, P)
        if args.hist:
            zp = np.load(CACHE / f"pseudo_oof_{panel}.npz")
            ctx = C.Ctx(panel, i)
            f0 = np.concatenate([f0, hist_feats(ctx, zp["d"], zp["t0"], pg.shape[1])], 1)
            ctx.z = ctx.q = ctx.el = None
        X.append(f0)
        T.append((ks if args.target == "kstar" else ny).astype(np.float64))
        K0.append(((pg >= TAU) & e).sum((1, 2)).astype(np.float64))
        print(f"  OOF {panel}: {len(ks)} windows, k* median {np.median(ks):.0f}, "
              f"tau-rule median {np.median(K0[-1]):.0f}", flush=True)
    X, T, K0 = np.concatenate(X), np.concatenate(T), np.concatenate(K0)
    # the target is the RATIO to what the flat threshold already takes, so the model learns a correction
    r = np.log((T + 1) / (K0 + 1))
    print(f"\nOOF: {len(T):,} windows; the flat threshold takes {np.mean(K0):.1f} cells, k* wants "
          f"{np.mean(T):.1f} (log-ratio mean {r.mean():+.3f}, sd {r.std():.3f})", flush=True)
    par = {"objective": "l2", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 40,
           "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1,
           "num_threads": 8}
    if args.hist:
        names = names + HNAMES
    if args.cv:
        rng = np.random.default_rng(0)
        fold = rng.integers(args.cv, size=len(r))
        pred = np.zeros_like(r)
        for k in range(args.cv):
            tr, te = fold != k, fold == k
            m_ = lgb.train(par, lgb.Dataset(X[tr], label=r[tr], feature_name=names),
                           num_boost_round=args.rounds)
            pred[te] = m_.predict(X[te])
        ss = 1 - ((r - pred) ** 2).sum() / ((r - r.mean()) ** 2).sum()
        print(f"  {args.cv}-fold CV on the OOF: R2 {ss:+.4f}, corr {np.corrcoef(r, pred)[0, 1]:+.4f}",
              flush=True)
    mdl = lgb.train(par, lgb.Dataset(X, label=r, feature_name=names), num_boost_round=args.rounds)
    mdl.save_model(str(LAB / "models" / f"budget_lgb_{args.target}.txt"))
    imp = sorted(zip(names, mdl.feature_importance("gain")), key=lambda t: -t[1])[:8]
    print("  top features: " + ", ".join(f"{k} {v:.0f}" for k, v in imp), flush=True)

    if args.hist:
        print("\n--hist fits and cross-validates on the OOF only: the scored-side application would need the "
              "same observables rebuilt there, which is worth doing only if the CV R2 is not zero.", flush=True)
        return
    fam, fam_e = {}, {}
    for i, panel in enumerate(C.T2_PANELS):
        sp = load_split(i, panel, args.tag)
        for s, D in sp.items():
            base, y, e, dd = D["base"], D["y"], D["e"], D["d"]
            F = global_features(D["P"], base)
            rh = gbm.predict(F.reshape(-1, F.shape[-1])).reshape(base.shape).astype(np.float32)
            pg = np.clip(base + ETA * np.where(np.abs(rh) >= GATE, rh, 0.0), 0, 1)
            days = np.unique(dd)
            early = np.isin(dd, days[1:5])
            k0 = ((pg >= TAU) & e).sum((1, 2)).astype(np.float64)
            Pm = np.stack([D["P"][k] for k in ("gs1", "tfm8", "cnn3", "occ_cnn", "occ_tfm", "stid")])
            rr = mdl.predict(feats(pg, e, Pm))
            v0 = C.iou(pg >= TAU, y, e)
            fam.setdefault((s, "AS"), {})[panel] = v0.mean()
            fam_e.setdefault((s, "AS"), {})[panel] = v0[early].mean()
            for sh in args.shrinks:
                k = np.round((k0 + 1) * np.exp(sh * rr) - 1)
                v = C.iou(apply_k(pg, e, k), y, e)
                fam.setdefault((s, f"budget x{sh:g}"), {})[panel] = v.mean()
                fam_e.setdefault((s, f"budget x{sh:g}"), {})[panel] = v[early].mean()
        print(f"  {panel}", flush=True)

    names = ["AS"] + [f"budget x{s:g}" for s in args.shrinks]
    for tag, F_ in (("whole month", fam), ("at most 4 earlier days", fam_e)):
        print(f"\nongoing family IoU, learned per-window budget ({args.target}), {tag}\n")
        print(f"{'setting':<16}" + "".join(f"{s:>13}" for s in C.SPLITS) + f"{'mean':>10}{'vs AS':>9}")
        g0 = None
        for nm in names:
            v = [C.family_mean(F_[(s, nm)]) for s in C.SPLITS]
            m = float(np.mean(v))
            if g0 is None:
                g0 = m
            print(f"{nm:<16}" + "".join(f"{x:>13.4f}" for x in v) + f"{m:>10.4f}{m - g0:>+9.4f}")


if __name__ == "__main__":
    main()
