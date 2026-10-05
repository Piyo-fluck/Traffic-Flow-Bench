"""Task 2 production run for the learned models (note 15): train on every train day, predict validation/private.

Training windows come from train truth (earliest-onset origins for onset, every-30-min ongoing origins).
Their priors are leave-one-day-out over all train days. Validation/private features come only from
window_history.parquet (T-60..T-5, speed/flow/eligibility), plus priors:
  default          : priors fitted on every train day
  --month-kappa K  : the queue prior P_q also uses the scored month's own visible cells on days strictly
                     before the window's date (same weekday type), shrunk to the train prior by K pseudo-days
                     (note 17; user decision: past days only). The onset-start prior P_on stays train-only.
Decisions: onset flags the top round(alpha * sum p) links at T+30 (clipped to 1..6); ongoing flags cells
with p >= tau. --load-models DIR reuses saved models instead of retraining.

    py scripts/t2_m1_predict.py --rounds-on 600 --rounds-og 3300 --alpha 1.2 --tau 0.65 --out-dir work/sub_t2m1
    py scripts/t2_m1_predict.py --load-models work/sub_t2m1 --month-kappa 4 --out-dir work/sub_t2m1_month
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "scripts"))
from t2_eda02 import T2_PANELS, candidates  # noqa: E402
from t2_eda03 import F, H  # noqa: E402
from t2_m1_models import Ctx, hist_from_truth, link_features, onset_decide  # noqa: E402
from t2_predict_rules import slot  # noqa: E402
from tfb.data import RELEASE, load  # noqa: E402
from tfb.features import shift  # noqa: E402
from tfb.month import MonthPastPrior  # noqa: E402

SEED = 0
MAX_OG_ROWS = 400_000
rng = np.random.default_rng(SEED)


# ------------------------------------------------------------------ priors
# A prior callable is pri(name, key, t) -> (N, S), with name "q" (queued) or "on" (onset start).
def train_prior(ctx: Ctx):
    """Train windows: leave-one-day-out over every train day. key = train day index (N,)."""
    def pri(name, key, t):
        return ctx.prior(ctx.xs_q if name == "q" else ctx.xs_on, key, t)
    return pri


def train_tables(ctx: Ctx) -> dict[str, np.ndarray]:
    return {name: np.stack([xs[ctx.g == g].mean(0) for g in range(int(ctx.g.max()) + 1)])
            for name, xs in (("q", ctx.xs_q), ("on", ctx.xs_on))}


def table_prior(ctx: Ctx):
    """Scored windows, train priors only. key = (N, 2) array of (split day index, weekday type)."""
    tabs = train_tables(ctx)

    def pri(name, key, t):
        return tabs[name][key[:, 1], np.clip(t, 0, tabs[name].shape[1] - 1)]
    return pri


def zero_prior(ctx: Ctx):
    """No recurrence priors: every prior feature is 0 (removes the train-run memorisation; see t2_shift_remedies)."""
    return lambda name, key, t: np.zeros((len(t), ctx.S), np.float32)


def month_prior(ctx: Ctx, z_split: dict, kappa: float):
    """Scored windows: P_q from the split's past days blended with train; P_on from train. key as table_prior."""
    tabs = train_tables(ctx)
    mp = MonthPastPrior(z_split, ctx.vf, tabs["q"], kappa, smooth=1)

    def pri(name, key, t):
        if name == "q":
            return mp(key[:, 0], key[:, 1], t)
        return tabs[name][key[:, 1], np.clip(t, 0, tabs[name].shape[1] - 1)]
    return pri


# ------------------------------------------------------------------ features (identical for train and scoring)
def common(ctx, v, qf, ok, t0, grp):
    f = link_features(ctx, v, qf, ok)
    N = len(t0)
    bc = lambda x: np.broadcast_to(np.asarray(x, np.float32)[:, None], (N, ctx.S))  # noqa: E731
    f.update({"tod": bc(t0 / 12), "group": bc(grp), "panel": bc(np.full(N, ctx.pid))})
    return f


def onset_X(ctx, v, qf, ok, t0, grp, pri, key):
    f = common(ctx, v, qf, ok, t0, grp)
    pq30 = pri("q", key, t0 + F)
    f.update({"pq30": pq30, "pq25": pri("q", key, t0 + F - 1), "pq35": pri("q", key, t0 + F + 1),
              "pon30": pri("on", key, t0 + F), "pq_hist": pri("q", key, t0 - 1)})
    for k in (1, -1, 2, -2):
        f[f"pq30_s{k:+d}"] = shift(pq30, k, 1)
    f["pq30_rank"] = (np.argsort(np.argsort(-pq30, 1), 1) / ctx.S).astype(np.float32)
    f["pq30_rel"] = pq30 / np.maximum(pq30.max(1, keepdims=True), 1e-6)
    names = sorted(f)
    return names, np.stack([np.asarray(f[k], np.float32) for k in names], -1)  # (N, S, K)


def ongoing_X(ctx, v, qf, ok, t0, grp, pri, key):
    base = common(ctx, v, qf, ok, t0, grp)
    pq_hist = pri("q", key, t0 - 1)
    names = sorted(base) + ["h", "pq_h", "pq_diff"]
    B = np.stack([np.asarray(base[k], np.float32) for k in sorted(base)], -1)
    per = []
    for h in range(1, F + 1):
        ph = pri("q", key, t0 + h)
        per.append(np.concatenate([B, np.stack([np.full_like(ph, h), ph, ph - pq_hist], -1)], -1))
    return names, np.stack(per, 1), base  # (N, F, S, K)


def _tq(ctx):
    if not hasattr(ctx, "tq"):
        from t2_demand_stress import load_tq
        ctx.tq = load_tq(ctx.panel)
    return ctx.tq


def feats_on(ctx, stress: bool, v, qf, ok, t0, grp, pri, key, wd):
    """onset_X, plus the demand-stress features of analysis/t2_demand_stress.py when stress is set (note 22)."""
    if stress:
        from t2_demand_stress import onset_Xs
        return onset_Xs(ctx, v, qf, ok, t0, grp, pri, key, wd, _tq(ctx), True)
    return onset_X(ctx, v, qf, ok, t0, grp, pri, key)


def feats_og(ctx, stress: bool, v, qf, ok, t0, grp, pri, key, wd):
    if stress:
        from t2_demand_stress import ongoing_Xs
        return ongoing_Xs(ctx, v, qf, ok, t0, grp, pri, key, wd, _tq(ctx), True)
    return ongoing_X(ctx, v, qf, ok, t0, grp, pri, key)


# ------------------------------------------------------------------ window_history -> arrays
def hist_from_release(ctx, H_df: pd.DataFrame, W: pd.DataFrame):
    s_of = {l: i for i, l in enumerate(ctx.z["link_ids"])}
    N = len(W)
    v = np.full((N, H, ctx.S), np.nan, np.float32)
    qf = np.full((N, H, ctx.S), np.nan, np.float32)
    ok = np.zeros((N, H, ctx.S), bool)
    t0 = slot(W.forecast_origin).astype(int)
    w_of = {w: i for i, w in enumerate(W.window_id)}
    wi = H_df.window_id.map(w_of).to_numpy()
    hi = slot(H_df.timestamp) - t0[wi] + H
    si = H_df.link_id.astype(str).map(s_of).to_numpy()
    good = (hi >= 0) & (hi < H)
    wi, hi, si = wi[good], hi[good], si[good]
    sp, fl = H_df.speed_kmh.to_numpy(float)[good], H_df.flow_vph.to_numpy(float)[good]
    el = H_df.is_score_eligible.astype(bool).to_numpy()[good]
    v[wi, hi, si], qf[wi, hi, si] = sp, fl
    ok[wi, hi, si] = el & np.isfinite(sp) & np.isfinite(fl)
    grp = np.array([int(pd.Timestamp(str(d)).weekday() >= 5) for d in W.date])
    return v, qf, ok, t0, grp


def train_models(ctxs, args, t_start):
    n_bags = max(1, args.onset_bags)
    Xon, yon, Xog, yog = [[] for _ in range(n_bags)], [], [], []
    names_on = names_og = None
    def train_pri(ctx, mode):
        if mode == "zero":
            return zero_prior(ctx)
        if mode == "dropout":
            from t2_prior_robust import dropout_prior  # domain randomisation (note 20); a fresh draw per call
            return dropout_prior(ctx)
        return train_prior(ctx)

    for ctx in ctxs:
        pri_og = train_pri(ctx, args.ongoing_prior)
        d, t0 = ctx.eo
        hist = hist_from_truth(ctx, d, t0)
        for b in range(n_bags):  # onset bagging: one prior-dropout draw per bag (the onset score varies by draw)
            names_on, X = feats_on(ctx, args.stress, *hist, t0, ctx.g[d], train_pri(ctx, args.onset_prior), d,
                                   ctx.z["weekday"][d])
            Xon[b].append(X.reshape(-1, X.shape[-1]))
        yon.append(ctx.q[d, t0 + F].reshape(-1))
        c = candidates(ctx.z)
        d, t0 = c["d"][c["ongoing"]], c["t0"][c["ongoing"]]
        xs, ys = [], []
        for i in range(0, len(d), 400):
            dd, tt = d[i:i + 400], t0[i:i + 400]
            names_og, Xh, base = feats_og(ctx, args.stress, *hist_from_truth(ctx, dd, tt), tt, ctx.g[dd], pri_og, dd,
                                          ctx.z["weekday"][dd])
            y = ctx.q[dd[:, None], tt[:, None] + np.arange(1, F + 1)]
            e = ctx.el[dd[:, None], tt[:, None] + np.arange(1, F + 1)]
            near = np.minimum(base["dist_slow_up"], base["dist_slow_dn"]) <= 10
            keep = np.flatnonzero(e & (np.broadcast_to(near[:, None, :], y.shape) | y | (rng.random(y.shape) < 0.02)))
            xs.append(Xh.reshape(-1, Xh.shape[-1])[keep]); ys.append(y.reshape(-1)[keep])
        X, y = np.concatenate(xs), np.concatenate(ys)
        if len(y) > MAX_OG_ROWS:
            idx = rng.choice(len(y), MAX_OG_ROWS, replace=False)
            X, y = X[idx], y[idx]
        Xog.append(X); yog.append(y)
        print(f"[train {ctx.panel}] onset windows {len(ctx.eo[0])}, ongoing rows {len(y):,} ({time.time() - t_start:.0f}s)", flush=True)
    params = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=50, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=SEED, num_threads=8)
    m_on = []
    for b in range(n_bags):
        m = lgb.train(dict(params, seed=SEED + b), lgb.Dataset(np.concatenate(Xon[b]), np.concatenate(yon), feature_name=names_on),
                      args.rounds_on)
        m.save_model(str(args.out_dir / ("t2_onset.txt" if b == 0 else f"t2_onset_{b}.txt")))
        m_on.append(m)
    m_og = lgb.train(dict(params, num_leaves=127), lgb.Dataset(np.concatenate(Xog), np.concatenate(yog), feature_name=names_og), args.rounds_og)
    m_og.save_model(str(args.out_dir / "t2_ongoing.txt"))
    return m_on, m_og


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds-on", type=int, default=600)
    ap.add_argument("--rounds-og", type=int, default=3300)
    ap.add_argument("--alpha", type=float, default=1.2)
    ap.add_argument("--tau", type=float, default=0.65)
    ap.add_argument("--month-kappa", type=float, default=None)
    ap.add_argument("--load-models", type=Path, default=None)
    ap.add_argument("--no-priors", action="store_true", help="train and score without any recurrence prior feature")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--onset-prior", choices=["train", "zero", "dropout"], default="train",
                    help="prior features when training the onset model (zero = none at scoring too)")
    ap.add_argument("--ongoing-prior", choices=["train", "zero", "dropout"], default="train",
                    help="prior features when training the ongoing model (zero = none at scoring too)")
    ap.add_argument("--onset-bags", type=int, default=1,
                    help="onset models trained on independent prior-dropout draws; probabilities are averaged")
    ap.add_argument("--stress", action="store_true",
                    help="add the demand-stress features (projected flow / capacity; note 22). Needed again with --load-models")
    args = ap.parse_args()
    if args.no_priors:
        args.onset_prior = args.ongoing_prior = "zero"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    ctxs = []
    for i, p in enumerate(T2_PANELS):
        ctx = Ctx(p, i)
        ctx.fitd = np.ones_like(ctx.fitd)  # priors from every train day (leave-one-day-out for train windows)
        ctxs.append(ctx)
    if args.load_models:
        m_on = [lgb.Booster(model_file=str(f)) for f in sorted(args.load_models.glob("t2_onset*.txt"))]
        m_og = lgb.Booster(model_file=str(args.load_models / "t2_ongoing.txt"))
        print(f"loaded {len(m_on)} onset model(s) and the ongoing model from {args.load_models}", flush=True)
    else:
        m_on, m_og = train_models(ctxs, args, t_start)
        print(f"models trained ({time.time() - t_start:.0f}s)", flush=True)
    print(f"priors: onset model {args.onset_prior}, ongoing model {args.ongoing_prior}; scoring priors "
          f"{'month-past (kappa ' + str(args.month_kappa) + ')' if args.month_kappa is not None else 'train table'}; "
          f"alpha {args.alpha}, tau {args.tau}", flush=True)

    for split in ("validation", "private"):
        frames = []
        for ctx in ctxs:
            base_dir = RELEASE / "task2" / ctx.panel / split
            W = pd.read_csv(base_dir / "window_index.csv")
            Hh = pd.read_parquet(base_dir / "window_history.parquet")
            Q = pd.read_csv(base_dir / "sample_submission_queue.csv", usecols=["window_id", "timestamp", "link_id"])
            s_of = {l: i for i, l in enumerate(ctx.z["link_ids"])}
            v, qf, ok, t0, grp = hist_from_release(ctx, Hh, W)
            z_split = load(ctx.panel, split)
            d_of = {d: i for i, d in enumerate(z_split["dates"])}
            key = np.stack([W.date.astype(str).map(d_of).to_numpy(), grp], 1).astype(int)
            pri_full = month_prior(ctx, z_split, args.month_kappa) if args.month_kappa is not None else table_prior(ctx)
            pri_on = zero_prior(ctx) if args.onset_prior == "zero" else pri_full
            pri_og = zero_prior(ctx) if args.ongoing_prior == "zero" else pri_full
            pred = np.zeros((len(W), F, ctx.S), bool)
            prob = np.zeros((len(W), F, ctx.S), np.float32)
            is_on = (W.condition == "queue_onset").to_numpy()
            wd = np.array([pd.Timestamp(str(x)).weekday() for x in W.date])
            if is_on.any():
                _, X = feats_on(ctx, args.stress, v[is_on], qf[is_on], ok[is_on], t0[is_on], grp[is_on], pri_on,
                                key[is_on], wd[is_on])
                p = np.mean([m.predict(X.reshape(-1, X.shape[-1])) for m in m_on], 0).reshape(X.shape[:2])
                prob[is_on, F - 1] = p
                pred[is_on, F - 1] = onset_decide(p, ("sum", args.alpha))
            if (~is_on).any():
                _, Xh, _ = feats_og(ctx, args.stress, v[~is_on], qf[~is_on], ok[~is_on], t0[~is_on], grp[~is_on], pri_og,
                                    key[~is_on], wd[~is_on])
                p = m_og.predict(Xh.reshape(-1, Xh.shape[-1])).reshape(Xh.shape[:3])
                prob[~is_on] = p
                pred[~is_on] = p >= args.tau
            # Probabilities are kept so the decision rules can be re-applied without retraining.
            np.savez(args.out_dir / f"prob_{split}_{ctx.panel}.npz", prob=prob, is_onset=is_on,
                     window_id=W.window_id.to_numpy().astype(str), t0=t0)
            w_of = {w: i for i, w in enumerate(W.window_id)}
            wi = Q.window_id.map(w_of).to_numpy()
            hi = slot(Q.timestamp) - t0[wi] - 1
            si = Q.link_id.astype(str).map(s_of).to_numpy()
            if (hi < 0).any() or (hi >= F).any():
                raise ValueError(f"{ctx.panel}/{split}: template timestamps outside T+5..T+30")
            Q["queue_pred"] = pred[wi, hi, si].astype(int)
            frames.append(Q)
            print(f"[{split} {ctx.panel}] onset flagged cells {int(pred[is_on].sum())} "
                  f"(per window {pred[is_on].sum((1, 2)).mean() if is_on.any() else 0:.1f}), "
                  f"ongoing flagged cells {int(pred[~is_on].sum())}", flush=True)
        out = pd.concat(frames, ignore_index=True)
        out.to_csv(args.out_dir / f"queue_{split}.csv", index=False)
        print(f"wrote {args.out_dir / f'queue_{split}.csv'}: {len(out):,} rows, share 1 = {out.queue_pred.mean():.4f}")


if __name__ == "__main__":
    main()
