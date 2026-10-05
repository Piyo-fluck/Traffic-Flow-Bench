"""TabPFN v2 (Prior Labs, 2025; weights Apache 2.0 + attribution) in place of the onset LightGBM H.

H is where onset loses to the run shift: 0.924 on the train tail, 0.79 on the scored months (note 108). TabPFN is
a transformer pretrained on synthetic tabular priors that predicts by in-context learning, so it is a different
function class on the SAME features. Everything is fitted on the 2030 fit days only, then scored twice:

    train tail     the 2031-01/02 earliest-onset windows of the train run (same run)
    scored months  the blank-aware official-like onset windows of validation / private (new run)

Rows are (window, link) at T+30, H's own feature set (scripts/t2_m1_predict.feats_on, no stress). Two feature
variants: `prior` = H's prior-dropout training and train-table priors at scoring, `noprior` = every recurrence
prior zeroed. Controls on exactly the same features: LightGBM on all fit rows (H's recipe, one bag) and LightGBM
on TabPFN's own subsample. TabPFN sees at most 10k context rows, so each of the B bags draws every positive (up
to 4k) plus hard negatives (highest pq30 / nearest the slow region) and random ones, and the bags are averaged.

    .venv/Scripts/python.exe fm_lab/tabpfn_onset.py --variant prior
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parent
PROJ = LAB.parent
for p in (PROJ, PROJ / "analysis", PROJ / "scripts"):
    sys.path.insert(0, str(p))
import lightgbm as lgb  # noqa: E402

from t2_eda02 import T2_PANELS  # noqa: E402
from t2_eda03 import F  # noqa: E402
from t2_expected_iou import select  # noqa: E402
from t2_first_onset_eval import first_onsets, official_blank  # noqa: E402
from t2_m1_models import Ctx, hist_from_truth  # noqa: E402
from t2_m1_predict import feats_on, table_prior, train_prior, zero_prior  # noqa: E402
from t2_prior_robust import dropout_prior  # noqa: E402
from t2_pseudo_eval import iou  # noqa: E402
from tfb.data import CACHE, load  # noqa: E402
from tfb.t1eval import FIT_END, family_mean  # noqa: E402

N_CTX = 10_000


def rows(ctx, d, t0, pri):
    hist = hist_from_truth(ctx, d, t0)
    names, X = feats_on(ctx, False, *hist, t0, ctx.g[d], pri, d, ctx.z["weekday"][d])
    return names, X                                            # (N, S, K)


def onset_iou(p, fut, efut):
    pr = np.zeros(fut.shape, bool)
    if len(p):
        pr[:, F - 1] = np.stack([select(q, "ratio", 1.0) for q in p])
    return iou(pr, fut, efut)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="prior", choices=("prior", "noprior"))
    ap.add_argument("--bags", type=int, default=4)
    ap.add_argument("--n-est", type=int, default=4, help="TabPFN's own ensemble size per fit")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    t_start = time.time()
    tr_X, tr_y, tr_hard = [], [], []
    evals = []            # (panel, kind, X (N,S,K), fut, efut, day)
    names = None
    for i, panel in enumerate(T2_PANELS):
        ctx = Ctx(panel, i)                                     # priors from fit days only (default fitd)
        d, t0 = ctx.eo
        fit = np.asarray(ctx.z["dates"])[d].astype(str) < FIT_END
        pri_tr = zero_prior(ctx) if args.variant == "noprior" else dropout_prior(ctx)
        names, X = rows(ctx, d[fit], t0[fit], pri_tr)
        y = ctx.q[d[fit], t0[fit] + F]
        e = ctx.el[d[fit], t0[fit] + F]
        k_pq = names.index("pq30")
        tr_X.append(X[e]); tr_y.append(y[e]); tr_hard.append(X[e][:, k_pq])
        # train tail, same run: leave-one-day-out train priors (tail days are not fit days, so no leak)
        pri_te = zero_prior(ctx) if args.variant == "noprior" else train_prior(ctx)
        _, Xt = rows(ctx, d[~fit], t0[~fit], pri_te)
        fi = t0[~fit, None] + np.arange(1, F + 1)
        evals.append((panel, "train tail", Xt, ctx.q[d[~fit, None], fi], ctx.el[d[~fit, None], fi], d[~fit]))
        # scored months, new run: the cached official-like windows' construction (analysis/t2_decision_variants)
        for split in ("validation", "private"):
            z = load(panel, split)
            on = first_onsets(ctx.vf, z, blank=official_blank(panel, split, z))
            if not len(on["d"]):
                continue
            grp = (z["weekday"][on["d"]] >= 5).astype(int)
            pri_s = zero_prior(ctx) if args.variant == "noprior" else table_prior(ctx)
            _, Xs = feats_on(ctx, False, on["v"], on["qf"], on["ok"], on["t0"], grp, pri_s,
                             np.stack([on["d"], grp], 1), z["weekday"][on["d"]].astype(int))
            evals.append((panel, split, Xs, on["fut"], on["efut"], on["d"]))
        print(f"[{panel}] fit windows {fit.sum()}, tail {(~fit).sum()} ({time.time() - t_start:.0f}s)", flush=True)
    X, y, hard = np.concatenate(tr_X), np.concatenate(tr_y).astype(int), np.concatenate(tr_hard)
    print(f"fit rows {len(y):,}, positives {y.sum():,}, {X.shape[1]} features", flush=True)

    par = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=50, feature_fraction=0.8,
               bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=8)
    models = {"lgbm all rows (H recipe)": lgb.train(par, lgb.Dataset(X, y), 600)}
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    hard_neg = neg[np.argsort(-hard[neg])[:len(neg) // 5]]
    subs = []
    for b in range(args.bags):
        p_ = rng.choice(pos, min(len(pos), 4000), replace=False)
        n_h = rng.choice(hard_neg, min(len(hard_neg), (N_CTX - len(p_)) // 2), replace=False)
        n_r = rng.choice(np.setdiff1d(neg, n_h), N_CTX - len(p_) - len(n_h), replace=False)
        subs.append(np.concatenate([p_, n_h, n_r]))
    models["lgbm same subsample"] = [lgb.train(dict(par, min_data_in_leaf=20), lgb.Dataset(X[s], y[s]), 300)
                                     for s in subs]
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    tab = []
    for s in subs:
        clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2, device="cuda", n_estimators=args.n_est,
                                                          ignore_pretraining_limits=True)
        clf.fit(X[s], y[s])
        tab.append(clf)
    models["TabPFN v2"] = tab
    print(f"fitted ({time.time() - t_start:.0f}s)", flush=True)

    def predict(m, Xf):
        if isinstance(m, list):
            if hasattr(m[0], "predict_proba"):
                return np.mean([c.predict_proba(Xf)[:, 1] for c in m], 0)
            return np.mean([c.predict(Xf) for c in m], 0)
        return m.predict(Xf)

    res = {}
    for panel, kind, Xe, fut, efut, dd in evals:
        flat = Xe.reshape(-1, Xe.shape[-1])
        for nm, m in models.items():
            p = np.concatenate([predict(m, flat[j:j + 20000]) for j in range(0, len(flat), 20000)])
            res.setdefault((kind, nm), {})[panel] = onset_iou(p.reshape(Xe.shape[:2]), fut, efut).mean()
        print(f"  scored {panel} {kind} ({time.time() - t_start:.0f}s)", flush=True)
    print(f"\nonset family IoU (ratio rule, T+30), variant '{args.variant}', fitted on 2030 days only\n")
    kinds = ["train tail", "validation", "private"]
    print(f"{'model':<28}" + "".join(f"{k:>13}" for k in kinds) + f"{'drop':>9}")
    for nm in models:
        v = [family_mean(res[(k, nm)]) for k in kinds]
        print(f"{nm:<28}" + "".join(f"{x:>13.4f}" for x in v) + f"{np.mean(v[1:]) - v[0]:>+9.4f}")


if __name__ == "__main__":
    main()
