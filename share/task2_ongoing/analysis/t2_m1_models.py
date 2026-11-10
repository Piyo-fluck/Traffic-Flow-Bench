"""Task 2 method M1: learned models, pooled over the 8 Task 2 panels.

  onset   : per (window, link) at T+30, P(link starts a queue); pick the top-k per window
  ongoing : per (window, step, link), P(queued at T+5..T+30); threshold for IoU

Inputs are only what a window offers (history T-60..T-5: speed, flow, eligibility) plus
static link attributes and recurrence priors. Days: Jun-Nov 2030 train the models,
Dec 2030 early-stops them and tunes the decision rules, Jan-Feb 2031 test them. Priors
come from Jun-Dec 2030: leave-one-day-out for Jun-Dec windows, plain for Jan-Feb windows.
The rules v2 are scored on the same windows.

    py analysis/t2_m1_models.py
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
from t2_eda02 import T2_PANELS, candidates  # noqa: E402
from t2_eda03 import F, H, earliest_onset, labels  # noqa: E402
from t2_eda04 import gids, official, prior_at, smooth  # noqa: E402
from tfb.features import neighbors_along, shift  # noqa: E402
from tfb.physics import fd_params  # noqa: E402
from tfb.t1eval import FIT_END, family_mean  # noqa: E402

ES_START = "2030-12-01"
SEED = 0
MAX_OG_ROWS = 400_000
rng = np.random.default_rng(SEED)


# ----------------------------------------------------------------------------- context
class Ctx:
    def __init__(self, panel: str, pid: int):
        z = load_train(panel)
        self.panel, self.pid, self.z = panel, pid, z
        self.q, self.el = labels(z)
        prm = fd_params(panel, z["link_ids"])
        self.vf, self.cap, self.kcrit, self.lanes = z["free_speed"], prm["cap"], prm["kcrit"], prm["lanes"]
        self.S = len(self.vf)
        self.g = gids(z, "wt2")
        dates = z["dates"]
        self.fitd = dates < FIT_END
        self.part = np.where(dates < ES_START, 0, np.where(dates < FIT_END, 1, 2))  # 0 train, 1 es, 2 test
        eo = earliest_onset(self.q)
        onset_cell = np.zeros_like(self.q)
        dd, tt = np.nonzero(eo)
        onset_cell[dd, tt + F] = self.q[dd, tt + F]
        self.eo = (dd, tt)
        self.xs_q = smooth(self.q.astype(np.float32), 1)
        self.xs_on = smooth(onset_cell.astype(np.float32), 1)

    def prior(self, xs: np.ndarray, d: np.ndarray, t: np.ndarray) -> np.ndarray:
        t = np.clip(t, 0, xs.shape[1] - 1)
        infit = self.fitd[d]
        out = np.empty((len(d), self.S), np.float32)
        if infit.any():
            out[infit] = prior_at(xs, self.g, self.fitd, d[infit], t[infit], loo=True)
        if (~infit).any():
            out[~infit] = prior_at(xs, self.g, self.fitd, d[~infit], t[~infit], loo=False)
        return out


def load_train(panel: str) -> dict:
    from tfb.data import load
    return load(panel, "train")


# ----------------------------------------------------------------------------- features
def nanlast(x: np.ndarray, ok: np.ndarray) -> np.ndarray:
    idx = np.where(ok, np.arange(x.shape[1])[None, :, None], -1).max(1)
    v = np.take_along_axis(x, np.clip(idx, 0, None)[:, None, :], 1)[:, 0]
    return np.where(idx >= 0, v, np.nan)


def link_features(ctx: Ctx, v: np.ndarray, qf: np.ndarray, ok: np.ndarray) -> dict[str, np.ndarray]:
    """(N, S) features from the history arrays v, qf, ok of shape (N, H, S)."""
    with np.errstate(invalid="ignore", divide="ignore", all="ignore"):
        vr = np.where(ok, v / ctx.vf, np.nan)
        qc = np.where(ok, qf / ctx.cap, np.nan)
        kc = np.where(ok, qf / np.maximum(v, 1.0) / ctx.kcrit, np.nan)
        qflag = ok & (np.nan_to_num(v, nan=1e9) <= 0.6 * ctx.vf)
        f = {
            "vr_last": nanlast(vr, ok), "vr_min3": np.nanmin(vr[:, -3:], 1), "vr_min12": np.nanmin(vr, 1),
            "vr_mean": np.nanmean(vr, 1), "vr_trend": np.nanmean(vr[:, -3:], 1) - np.nanmean(vr[:, :3], 1),
            "qc_last": nanlast(qc, ok), "qc_max3": np.nanmax(qc[:, -3:], 1), "qc_mean": np.nanmean(qc, 1),
            "kc_last": nanlast(kc, ok), "kc_max3": np.nanmax(kc[:, -3:], 1),
            "kc_trend": np.nanmean(kc[:, -3:], 1) - np.nanmean(kc[:, :3], 1),
            "q_cnt": qflag.sum(1).astype(np.float32), "q_last3": qflag[:, -3:].any(1).astype(np.float32),
            "n_ok3": ok[:, -3:].sum(1).astype(np.float32),
        }
        last_q = np.where(qflag, np.arange(H)[None, :, None], -1).max(1)
        f["since_q"] = np.where(last_q >= 0, H - 1 - last_q, H + 1).astype(np.float32)
    for key in ("vr_min3", "kc_max3", "q_last3", "qc_max3"):
        for k in (1, -1, 2, -2, 4, -4):
            f[f"{key}_s{k:+d}"] = shift(f[key], k, 1)
    slow = np.nan_to_num(f["vr_min3"], nan=9) < 0.8
    pos = np.broadcast_to(np.arange(ctx.S, dtype=np.float32), slow.shape)
    _, _, dp, dn = neighbors_along(np.where(slow, pos, np.nan), slow, axis=1)
    f["dist_slow_up"], f["dist_slow_dn"] = np.nan_to_num(dp, nan=99), np.nan_to_num(dn, nan=99)
    f["n_slow"] = np.broadcast_to(slow.sum(1, keepdims=True), slow.shape).astype(np.float32)
    N = slow.shape[0]
    for name, vec in (("pos", np.arange(ctx.S) / max(ctx.S - 1, 1)), ("lanes", ctx.lanes), ("vf", ctx.vf),
                      ("cap_lane", ctx.cap / ctx.lanes), ("kcrit_lane", ctx.kcrit / ctx.lanes)):
        f[name] = np.broadcast_to(np.asarray(vec, np.float32)[None, :], (N, ctx.S))
    return {k: np.asarray(x, np.float32) for k, x in f.items()}


def hist_from_truth(ctx: Ctx, d: np.ndarray, t0: np.ndarray):
    hi = t0[:, None] + np.arange(-H, 0)
    ok = ctx.el[d[:, None], hi]
    return ctx.z["true_speed"][d[:, None], hi], ctx.z["true_flow"][d[:, None], hi], ok


def window_cols(ctx: Ctx, d: np.ndarray, t0: np.ndarray, N: int) -> dict[str, np.ndarray]:
    bc = lambda x: np.broadcast_to(np.asarray(x, np.float32)[:, None], (N, ctx.S))  # noqa: E731
    return {"tod": bc(t0 / 12), "group": bc(ctx.g[d]), "panel": bc(np.full(N, ctx.pid))}


def onset_features(ctx: Ctx, d, t0):
    f = link_features(ctx, *hist_from_truth(ctx, d, t0))
    f.update(window_cols(ctx, d, t0, len(d)))
    pq30 = ctx.prior(ctx.xs_q, d, t0 + F)
    f["pq30"], f["pq25"], f["pq35"] = pq30, ctx.prior(ctx.xs_q, d, t0 + F - 1), ctx.prior(ctx.xs_q, d, t0 + F + 1)
    f["pon30"] = ctx.prior(ctx.xs_on, d, t0 + F)
    f["pq_hist"] = ctx.prior(ctx.xs_q, d, t0 - 1)
    for k in (1, -1, 2, -2):
        f[f"pq30_s{k:+d}"] = shift(pq30, k, 1)
    f["pq30_rank"] = (np.argsort(np.argsort(-pq30, 1), 1) / ctx.S).astype(np.float32)
    f["pq30_rel"] = pq30 / np.maximum(pq30.max(1, keepdims=True), 1e-6)
    return f


def ongoing_features(ctx: Ctx, d, t0):
    base = link_features(ctx, *hist_from_truth(ctx, d, t0))
    base.update(window_cols(ctx, d, t0, len(d)))
    pq_hist = ctx.prior(ctx.xs_q, d, t0 - 1)
    per_h = []
    for h in range(1, F + 1):
        ph = ctx.prior(ctx.xs_q, d, t0 + h)
        per_h.append({"h": np.full_like(ph, h), "pq_h": ph, "pq_diff": ph - pq_hist})
    return base, per_h


# ----------------------------------------------------------------------------- data assembly
def onset_rows(ctx: Ctx, d, t0):
    f = onset_features(ctx, d, t0)
    names = sorted(f)
    X = np.stack([f[k] for k in names], -1)  # (N, S, K)
    return names, X, ctx.q[d, t0 + F], ctx.el[d, t0 + F]


def ongoing_chunk(ctx: Ctx, d, t0):
    """(N, F, S, K) features, labels and eligibility for a chunk of ongoing windows."""
    base, per_h = ongoing_features(ctx, d, t0)
    names = sorted(base) + ["h", "pq_h", "pq_diff"]
    B = np.stack([base[k] for k in sorted(base)], -1)  # (N, S, Kb)
    Xh = np.stack([np.concatenate([B, np.stack([ph["h"], ph["pq_h"], ph["pq_diff"]], -1)], -1) for ph in per_h], 1)
    y = ctx.q[d[:, None], t0[:, None] + np.arange(1, F + 1)]
    e = ctx.el[d[:, None], t0[:, None] + np.arange(1, F + 1)]
    return names, Xh, y, e, base


def ongoing_train_rows(ctx: Ctx, d, t0, chunk: int = 400):
    """Sampled training rows: eligible cells near the history's slow links (or queued), plus 2% of the rest."""
    Xs, ys, names = [], [], None
    for i in range(0, len(d), chunk):
        names, Xh, y, e, base = ongoing_chunk(ctx, d[i:i + chunk], t0[i:i + chunk])
        near = np.minimum(base["dist_slow_up"], base["dist_slow_dn"]) <= 10
        keep = e & (np.broadcast_to(near[:, None, :], y.shape) | y | (rng.random(y.shape) < 0.02))
        idx = np.flatnonzero(keep)
        Xs.append(Xh.reshape(-1, Xh.shape[-1])[idx]); ys.append(y.reshape(-1)[idx])
    X, y = np.concatenate(Xs), np.concatenate(ys)
    if len(y) > MAX_OG_ROWS:
        idx = rng.choice(len(y), MAX_OG_ROWS, replace=False)
        X, y = X[idx], y[idx]
    return names, X, y


def split_windows(ctx: Ctx, d, t0):
    return {p: (d[ctx.part[d] == p], t0[ctx.part[d] == p]) for p in (0, 1, 2)}


# ----------------------------------------------------------------------------- decisions and scoring
def iou_windows(pred, truth, elig):
    axes = tuple(range(1, pred.ndim))
    p = pred & elig
    inter, union = (p & truth).sum(axes), (p | truth).sum(axes)
    return np.where(union == 0, 1.0, inter / np.maximum(union, 1))


def onset_decide(p: np.ndarray, rule) -> np.ndarray:
    kind, val = rule
    if kind == "k":
        k = np.full(len(p), val)
    else:
        k = np.clip(np.round(val * p.sum(1)), 1, 6).astype(int)
    order = np.argsort(-p, 1)
    ranks = np.argsort(order, 1)
    return ranks < k[:, None]


def main() -> None:
    t_start = time.time()
    ctxs = [Ctx(p, i) for i, p in enumerate(T2_PANELS)]
    print(f"contexts ready ({time.time() - t_start:.0f}s)", flush=True)
    params = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=50, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=SEED, num_threads=8)

    # ---------------- onset ----------------
    on = {}
    for ctx in ctxs:
        sp = split_windows(ctx, *ctx.eo)
        for part, (d, t0) in sp.items():
            names, X, y, e = onset_rows(ctx, d, t0)
            on.setdefault(part, []).append((ctx, d, t0, X, y, e))
    Xtr = np.concatenate([x[3].reshape(-1, x[3].shape[-1]) for x in on[0]])
    ytr = np.concatenate([x[4].reshape(-1) for x in on[0]])
    Xes = np.concatenate([x[3].reshape(-1, x[3].shape[-1]) for x in on[1]])
    yes = np.concatenate([x[4].reshape(-1) for x in on[1]])
    m_on = lgb.train(params, lgb.Dataset(Xtr, ytr, feature_name=names), 3000,
                     valid_sets=[lgb.Dataset(Xes, yes, reference=None)], callbacks=[lgb.early_stopping(100, verbose=False)])
    imp = m_on.feature_importance("gain")
    print(f"onset model: {len(ytr):,} rows, best_iter {m_on.best_iteration}; top: " +
          ", ".join(f"{names[i]} {imp[i] / imp.sum():.2f}" for i in np.argsort(-imp)[:8]), flush=True)
    rules_grid = [("k", k) for k in range(1, 7)] + [("sum", a) for a in (0.8, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6)]
    es_score = {r: [] for r in rules_grid}
    for ctx, d, t0, X, y, e in on[1]:
        p = m_on.predict(X.reshape(-1, X.shape[-1]), num_iteration=m_on.best_iteration).reshape(y.shape)
        for r in rules_grid:
            es_score[r].append((ctx.panel, iou_windows(onset_decide(p, r), y, e).mean()))
    es_fam = {r: family_mean(dict(v)) for r, v in es_score.items()}
    best_on = max(es_fam, key=es_fam.get)
    print("onset ES by rule: " + ", ".join(f"{r[0]}{r[1]}={v:.3f}" for r, v in es_fam.items()) + f" -> {best_on}", flush=True)
    res_on_model, res_on_rules = {}, {}
    for ctx, d, t0, X, y, e in on[2]:
        p = m_on.predict(X.reshape(-1, X.shape[-1]), num_iteration=m_on.best_iteration).reshape(y.shape)
        res_on_model[ctx.panel] = iou_windows(onset_decide(p, best_on), y, e).mean()
        sc = ctx.prior(ctx.xs_on, d, t0 + F) + 0.5 * ctx.prior(ctx.xs_q, d, t0 + F)
        res_on_rules[ctx.panel] = iou_windows(onset_decide(sc, ("k", 3)), y, e).mean()

    # ---------------- ongoing ----------------
    og = {0: [], 1: [], 2: []}
    for ctx in ctxs:
        c = candidates(ctx.z)
        d, t0 = c["d"][c["ongoing"]], c["t0"][c["ongoing"]]
        sp = split_windows(ctx, d, t0)
        for part in (0, 1, 2):
            og[part].append((ctx, *sp[part]))
    tr = [ongoing_train_rows(ctx, d, t0) for ctx, d, t0 in og[0]]
    names_og = tr[0][0]
    Xtr = np.concatenate([x[1] for x in tr]); ytr = np.concatenate([x[2] for x in tr])
    es_rows = [ongoing_train_rows(ctx, d, t0) for ctx, d, t0 in og[1]]
    Xes = np.concatenate([x[1] for x in es_rows]); yes = np.concatenate([x[2] for x in es_rows])
    del tr, es_rows
    params_og = dict(params, num_leaves=127)
    m_og = lgb.train(params_og, lgb.Dataset(Xtr, ytr, feature_name=names_og), 5000,
                     valid_sets=[lgb.Dataset(Xes, yes)], callbacks=[lgb.early_stopping(100, verbose=False)])
    imp = m_og.feature_importance("gain")
    print(f"ongoing model: {len(ytr):,} rows, best_iter {m_og.best_iteration}; top: " +
          ", ".join(f"{names_og[i]} {imp[i] / imp.sum():.2f}" for i in np.argsort(-imp)[:8]), flush=True)

    def og_predict(ctx, d, t0, chunk: int = 200):
        ps, ys, es = [], [], []
        for i in range(0, len(d), chunk):
            _, X, y, e, _ = ongoing_chunk(ctx, d[i:i + chunk], t0[i:i + chunk])
            ps.append(m_og.predict(X.reshape(-1, X.shape[-1]), num_iteration=m_og.best_iteration).reshape(y.shape))
            ys.append(y); es.append(e)
        return np.concatenate(ps), np.concatenate(ys), np.concatenate(es)

    taus = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8)
    es_t = {t: {} for t in taus}
    for ctx, d, t0 in og[1]:
        p, y, e = og_predict(ctx, d, t0)
        for t in taus:
            es_t[t][ctx.panel] = iou_windows(p >= t, y, e).mean()
    best_tau = max(taus, key=lambda t: family_mean(es_t[t]))
    print("ongoing ES by tau: " + ", ".join(f"{t}={family_mean(es_t[t]):.3f}" for t in taus) + f" -> {best_tau}", flush=True)
    res_og_model, res_og_rules = {}, {}
    for ctx, d, t0 in og[2]:
        p, y, e = og_predict(ctx, d, t0)
        res_og_model[ctx.panel] = iou_windows(p >= best_tau, y, e).mean()
        v, _, ok = hist_from_truth(ctx, d, t0)
        with np.errstate(invalid="ignore", all="ignore"):
            link = np.nan_to_num(np.nanmin(np.where(ok[:, -3:], v[:, -3:] / ctx.vf, np.nan), 1), nan=9) < 0.8
        res_og_rules[ctx.panel] = iou_windows(np.repeat(link[:, None], F, 1), y, e).mean()

    print("\n== test (Jan-Feb 2031) windows, family mean over 8 panels ==")
    on_m, on_r = family_mean(res_on_model), family_mean(res_on_rules)
    og_m, og_r = family_mean(res_og_model), family_mean(res_og_rules)
    print(f"  onset   : rules {on_r:.4f} -> model {on_m:.4f}")
    print(f"  ongoing : rules {og_r:.4f} -> model {og_m:.4f}")
    print(f"  S_queue : rules {(on_r + og_r) / 2:.4f} -> model {(on_m + og_m) / 2:.4f}")
    for p in T2_PANELS:
        print(f"    {p:<11} onset {res_on_rules[p]:.3f}->{res_on_model[p]:.3f}  ongoing {res_og_rules[p]:.3f}->{res_og_model[p]:.3f}")
    print(f"total {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
