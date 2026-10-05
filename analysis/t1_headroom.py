"""How well could ANY local imputer do? The full-information ceiling for a Task 1 cell.

Five diagnostics have now come back empty (notes 96-99): the error does not depend on mask geometry, corridor
conservation cannot reach it, there is no day-level bias, no low-rank (day x TOD x link) component survives even
when half the answers are handed over, and a different function class converges worse and does not blend. That
leaves two possibilities, and they lead to opposite decisions:

    (a) the residual is intrinsically unpredictable -> Task 1 is finished, stop investing
    (b) it is predictable from information the current inputs do not carry -> the input set is the problem

This separates them. A gradient-boosted model is given the TRUE values of a cell's neighbourhood - the same link
at t-2, t-1, t+1, t+2, the neighbouring links at the same slot, the cell's own occupancy, and the weekday x
time-of-day profile - with no masking at all, fitted on the fit days and scored on the same target cells the CNN
is scored on. It is an upper bound on what any imputer using local information could achieve, because a real
imputer only ever sees a masked version of these inputs.

    py analysis/t1_headroom.py [panels...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tfb.data import PANELS, load  # noqa: E402
from tfb.features import profile, truth_valid  # noqa: E402
from tfb.t1eval import split_days  # noqa: E402

PRED = ROOT / "work" / "preds"
PANEL_ARGS = sys.argv[1:] or ["D7_I10_E", "D7_I210_E", "D7_I405_N", "D12_I5_N"]
ROUNDS = 400
PARAMS = {"objective": "l2", "learning_rate": 0.05, "num_leaves": 127, "min_data_in_leaf": 200,
          "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "num_threads": 8}
MAX_FIT = 3_000_000


def shift_t(a, k):
    out = np.full_like(a, np.nan)
    if k > 0:
        out[:, k:] = a[:, :-k]
    elif k < 0:
        out[:, :k] = a[:, -k:]
    else:
        out = a.copy()
    return out


def shift_s(a, k):
    out = np.full_like(a, np.nan)
    if k > 0:
        out[:, :, k:] = a[:, :, :-k]
    elif k < 0:
        out[:, :, :k] = a[:, :, -k:]
    else:
        out = a.copy()
    return out


def build(z, base_v, base_q):
    ts, tq, occ, lanes = z["true_speed"], z["true_flow"], z["true_occ"], z["lanes"]
    ql = tq / lanes
    bl = base_q / lanes
    feats = {"base_v": np.broadcast_to(base_v, ts.shape), "base_q": np.broadcast_to(bl, ts.shape)}
    for k in (-2, -1, 1, 2):                       # the same link, earlier and later in the day
        feats[f"q_t{k:+d}"] = shift_t(ql, k)
        feats[f"v_t{k:+d}"] = shift_t(ts, k)
    for k in (-2, -1, 1, 2):                       # neighbouring links, same slot
        feats[f"q_s{k:+d}"] = shift_s(ql, k)
        feats[f"v_s{k:+d}"] = shift_s(ts, k)
    feats["occ"] = occ
    feats["occ_t-1"] = shift_t(occ, 1)
    feats["occ_t+1"] = shift_t(occ, -1)
    feats["lanes"] = np.broadcast_to(lanes[None, None, :], ts.shape)
    feats["vf"] = np.broadcast_to(z["free_speed"][None, None, :], ts.shape)
    names = sorted(feats)
    X = np.stack([feats[n] for n in names], -1).astype(np.float32)
    return X, names, ql, ts


def main() -> None:
    rng = np.random.default_rng(0)
    acc = {}
    for panel in PANEL_ARGS:
        z = load(panel, "train")
        fit_d, test_d = split_days(z)
        valid = truth_valid(z)
        base_v = profile(z["true_speed"], valid, z["weekday"], fit_d)
        base_q = profile(z["true_flow"], valid, z["weekday"], fit_d)
        X, names, ql, ts = build(z, base_v, base_q)
        ok = np.isfinite(ql) & np.isfinite(ts) & z["elig"]
        fitm = ok & fit_d[:, None, None]
        tgt = z["target"] & ok & test_d[:, None, None]
        idx = np.flatnonzero(fitm.ravel())
        if len(idx) > MAX_FIT:
            idx = rng.choice(idx, MAX_FIT, replace=False)
        Xf = X.reshape(-1, X.shape[-1])[idx]
        models = {}
        for ch, y in (("q", ql), ("v", ts)):
            models[ch] = lgb.train(PARAMS, lgb.Dataset(Xf, label=y.ravel()[idx], feature_name=names,
                                                       free_raw_data=False), num_boost_round=ROUNDS)
        ti = np.flatnonzero(tgt.ravel())
        Xt = X.reshape(-1, X.shape[-1])[ti]
        pq = models["q"].predict(Xt)
        pv = models["v"].predict(Xt)
        yq, yv = ql.ravel()[ti], ts.ravel()[ti]
        free = yv > 0.85 * np.broadcast_to(z["free_speed"][None, None, :], ts.shape).ravel()[ti]
        f = np.load(PRED / f"t1_m3all_{panel}_test.npz")
        days = f["days"]
        cnn_q = ((f["pq"] / z["lanes"])[np.isin(np.flatnonzero(test_d), days).nonzero()[0]]
                 if len(days) != test_d.sum() else f["pq"] / z["lanes"])
        cnn_v = f["pv"]
        m_t = z["target"][days] & np.isfinite(ql[days]) & np.isfinite(ts[days])
        for nm, sel in (("all", slice(None)), ("free", free)):
            a = acc.setdefault(("oracle", nm), [0.0, 0.0, 0])
            a[0] += float(((pv[sel] - yv[sel]) ** 2).sum())
            a[1] += float(((pq[sel] - yq[sel]) ** 2).sum())
            a[2] += int(np.sum(sel) if sel is not slice(None) else len(yq))
        free_c = ts[days] > 0.85 * z["free_speed"][None, None, :]
        for nm, m in (("all", m_t), ("free", m_t & free_c)):
            a = acc.setdefault(("cnn", nm), [0.0, 0.0, 0])
            a[0] += float(((cnn_v - ts[days])[m] ** 2).sum())
            a[1] += float(((cnn_q - ql[days])[m] ** 2).sum())
            a[2] += int(m.sum())
        imp = sorted(zip(models["q"].feature_importance("gain"), names), reverse=True)[:6]
        tot = sum(g for g, _ in imp) or 1
        print(f"  {panel}: flow gain " + ", ".join(f"{n} {g / tot * 100:.0f}%" for g, n in imp), flush=True)

    print(f"\n{'model':<10} {'group':<6} {'cells':>10} {'speed RMSE':>11} {'flow RMSE':>10}")
    for (mdl, nm), a in acc.items():
        print(f"{mdl:<10} {nm:<6} {a[2]:10,} {np.sqrt(a[0] / a[2]):11.3f} {np.sqrt(a[1] / a[2]):10.2f}")
    print("\n(the oracle sees UNMASKED true neighbours; a real imputer never does, so this is an upper bound)")


if __name__ == "__main__":
    main()
