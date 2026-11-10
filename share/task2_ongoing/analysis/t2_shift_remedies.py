"""Task 2: run-shift remedies, judged inside a scored month on pseudo windows (past days only).

Protocol for one split (default validation):
  - Pseudo windows come from the split's masked layer (see analysis/t2_pseudo_eval.py).
  - The first ADAPT_DAYS days are adaptation days: their windows may be used for training.
  - The remaining days are evaluation days.
  - The priors for a window on day d use only split days < d (month-past prior) or none.
Variants, all scored on the same evaluation-day windows:
  E          production models (train only) + train priors          (what was submitted)
  E+month    production models + month-past priors (kappa 4)
  noprior    models retrained on train windows with every prior feature set to 0 (removes the train-run
             memorisation, as with location embeddings in STAEformer)
  adapt      models retrained on train windows + adaptation-day pseudo windows (weight W), month-past priors
             (test-time adaptation on the new run)
  adapt-np   as adapt, without priors

    py analysis/t2_shift_remedies.py validation
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "scripts"))
from t2_eda02 import T2_PANELS, candidates  # noqa: E402
from t2_eda03 import F  # noqa: E402
from t2_m1_models import Ctx, hist_from_truth, onset_decide  # noqa: E402
from t2_m1_predict import month_prior, ongoing_X, onset_X, table_prior, train_prior  # noqa: E402
from t2_pseudo_eval import iou, pseudo_windows  # noqa: E402
from tfb.data import load  # noqa: E402
from tfb.t1eval import family_mean  # noqa: E402

ADAPT_DAYS, W_ADAPT, MAX_OG = 15, 5.0, 350_000
ROUNDS_ON, ROUNDS_OG = 600, 2000
rng = np.random.default_rng(0)
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=50, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=0, num_threads=8)


def zero_prior(ctx: Ctx):
    return lambda name, key, t: np.zeros((len(t), ctx.S), np.float32)


def sample_og(Xh, y, e, base):
    near = np.minimum(base["dist_slow_up"], base["dist_slow_dn"]) <= 10
    keep = np.flatnonzero(e & (np.broadcast_to(near[:, None, :], y.shape) | y | (rng.random(y.shape) < 0.02)))
    return Xh.reshape(-1, Xh.shape[-1])[keep], y.reshape(-1)[keep]


def train_rows(ctx: Ctx, pri):
    d, t0 = ctx.eo
    names_on, X = onset_X(ctx, *hist_from_truth(ctx, d, t0), t0, ctx.g[d], pri, d)
    on = (X.reshape(-1, X.shape[-1]), ctx.q[d, t0 + F].reshape(-1))
    c = candidates(ctx.z)
    d, t0 = c["d"][c["ongoing"]], c["t0"][c["ongoing"]]
    xs, ys = [], []
    for i in range(0, len(d), 400):
        dd, tt = d[i:i + 400], t0[i:i + 400]
        names_og, Xh, base = ongoing_X(ctx, *hist_from_truth(ctx, dd, tt), tt, ctx.g[dd], pri, dd)
        y = ctx.q[dd[:, None], tt[:, None] + np.arange(1, F + 1)]
        e = ctx.el[dd[:, None], tt[:, None] + np.arange(1, F + 1)]
        a, b = sample_og(Xh, y, e, base)
        xs.append(a); ys.append(b)
    X, y = np.concatenate(xs), np.concatenate(ys)
    if len(y) > MAX_OG:
        idx = rng.choice(len(y), MAX_OG, replace=False)
        X, y = X[idx], y[idx]
    return names_on, on, names_og, (X, y)


def pseudo_rows(ctx: Ctx, w: dict, sel: np.ndarray, grp: np.ndarray, pri, key):
    on = sel & w["onset"]
    _, X = onset_X(ctx, w["v"][on], w["qf"][on], w["ok"][on], w["t0"][on], grp[on], pri, key[on])
    e = w["efut"][on][:, F - 1]
    on_rows = (X[e], w["fut"][on][:, F - 1][e])
    idx = np.flatnonzero(sel & w["ongoing"])
    xs, ys = [], []
    for j in range(0, len(idx), 200):
        b = idx[j:j + 200]
        _, Xh, base = ongoing_X(ctx, w["v"][b], w["qf"][b], w["ok"][b], w["t0"][b], grp[b], pri, key[b])
        a, c = sample_og(Xh, w["fut"][b], w["efut"][b], base)
        xs.append(a); ys.append(c)
    og_rows = (np.concatenate(xs), np.concatenate(ys)) if xs else (None, None)
    return on_rows, og_rows


def fit(names, parts, rounds, leaves):
    X = np.concatenate([p[0] for p in parts]); y = np.concatenate([p[1] for p in parts])
    wt = np.concatenate([np.full(len(p[1]), p[2]) for p in parts])
    return lgb.train(dict(PARAMS, num_leaves=leaves), lgb.Dataset(X, y, weight=wt, feature_name=names), rounds)


def score(ctx, w, sel, grp, pri, key, m_on, m_og, alpha=1.2, tau=0.65):
    on = sel & w["onset"]
    res = {}
    if on.any():
        _, X = onset_X(ctx, w["v"][on], w["qf"][on], w["ok"][on], w["t0"][on], grp[on], pri, key[on])
        p = m_on.predict(X.reshape(-1, X.shape[-1])).reshape(X.shape[:2])
        pr = np.zeros((on.sum(), F, ctx.S), bool)
        pr[:, F - 1] = onset_decide(p, ("sum", alpha))
        res["onset"] = iou(pr, w["fut"][on], w["efut"][on]).mean()
    idx = np.flatnonzero(sel & w["ongoing"])
    ious = []
    for j in range(0, len(idx), 200):
        b = idx[j:j + 200]
        _, Xh, _ = ongoing_X(ctx, w["v"][b], w["qf"][b], w["ok"][b], w["t0"][b], grp[b], pri, key[b])
        p = m_og.predict(Xh.reshape(-1, Xh.shape[-1])).reshape(Xh.shape[:3])
        ious.append(iou(p >= tau, w["fut"][b], w["efut"][b]))
    res["ongoing"] = np.concatenate(ious).mean() if ious else np.nan
    return res


def main() -> None:
    split = sys.argv[1] if len(sys.argv) > 1 else "validation"
    t_start = time.time()
    ctxs = []
    for i, p in enumerate(T2_PANELS):
        ctx = Ctx(p, i)
        ctx.fitd = np.ones_like(ctx.fitd)
        ctxs.append(ctx)
    m_on_E = lgb.Booster(model_file=str(ROOT / "work" / "sub_t2m1" / "t2_onset.txt"))
    m_og_E = lgb.Booster(model_file=str(ROOT / "work" / "sub_t2m1" / "t2_ongoing.txt"))

    # ---- training rows: train windows (with and without priors), adaptation-day pseudo windows
    tr_p, tr_0, ad_p, ad_0, split_data = [], [], [], [], {}
    for ctx in ctxs:
        names_on, on_p, names_og, og_p = train_rows(ctx, train_prior(ctx))
        _, on_0, _, og_0 = train_rows(ctx, zero_prior(ctx))
        tr_p.append((on_p, og_p)); tr_0.append((on_0, og_0))
        z = load(ctx.panel, split)
        w = pseudo_windows(ctx, z)
        grp = (z["weekday"][w["d"]] >= 5).astype(int)
        key = np.stack([w["d"], grp], 1)
        adapt = w["d"] < ADAPT_DAYS
        pri_m = month_prior(ctx, z, 4.0)
        ad_p.append(pseudo_rows(ctx, w, adapt, grp, pri_m, key))
        ad_0.append(pseudo_rows(ctx, w, adapt, grp, zero_prior(ctx), key))
        split_data[ctx.panel] = (z, w, grp, key, pri_m, table_prior(ctx))
        # Scoring needs only the static link attributes and the two prior callables built above,
        # so release the large train arrays now (8 panels would not fit in RAM together).
        ctx.z = {"link_ids": ctx.z["link_ids"]}
        ctx.q = ctx.el = ctx.xs_q = ctx.xs_on = None
        print(f"[{ctx.panel}] rows ready: adapt onset {int((adapt & w['onset']).sum())} ongoing {int((adapt & w['ongoing']).sum())}; "
              f"eval onset {int((~adapt & w['onset']).sum())} ongoing {int((~adapt & w['ongoing']).sum())} ({time.time() - t_start:.0f}s)", flush=True)

    def parts(tr, ad, k, with_adapt):
        ps = [(x[k][0], x[k][1], 1.0) for x in tr]
        if with_adapt:
            ps += [(x[k][0], x[k][1], W_ADAPT) for x in ad if x[k][0] is not None and len(x[k][1])]
        return ps

    models = {"E": (m_on_E, m_og_E)}
    for name, tr, ad, with_adapt in (("noprior", tr_0, ad_0, False), ("adapt", tr_p, ad_p, True), ("adapt-np", tr_0, ad_0, True)):
        models[name] = (fit(names_on, parts(tr, ad, 0, with_adapt), ROUNDS_ON, 63),
                        fit(names_og, parts(tr, ad, 1, with_adapt), ROUNDS_OG, 127))
        print(f"trained {name} ({time.time() - t_start:.0f}s)", flush=True)

    variants = {"E": ("E", "table"), "E+month": ("E", "month"), "noprior": ("noprior", "zero"),
                "adapt": ("adapt", "month"), "adapt-np": ("adapt-np", "zero")}
    res = {v: {"onset": {}, "ongoing": {}} for v in variants}
    for ctx in ctxs:
        z, w, grp, key, pri_m, pri_t = split_data[ctx.panel]
        ev = w["d"] >= ADAPT_DAYS
        pris = {"table": pri_t, "month": pri_m, "zero": zero_prior(ctx)}
        for v, (mname, pname) in variants.items():
            r = score(ctx, w, ev, grp, pris[pname], key, *models[mname])
            for k2, val in r.items():
                if not np.isnan(val):
                    res[v][k2][ctx.panel] = val
        print(f"[{ctx.panel}] scored", flush=True)
    print(f"\n== {split}: evaluation days >= {ADAPT_DAYS} (pseudo windows), family mean ==")
    for v, r in res.items():
        a, b = family_mean(r["onset"]), family_mean(r["ongoing"])
        print(f"  {v:<9} onset {a:.3f}  ongoing {b:.3f}  S_queue {(a + b) / 2:.3f}")
    print(f"total {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
