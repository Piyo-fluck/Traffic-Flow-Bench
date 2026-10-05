"""Does a run's DYNAMICS, measured on its earlier days, predict how the model will be wrong today?

The user's second Task 2 line: instead of feeding past-day labels to the model, estimate a low-dimensional
latent run state - extension tendency, dissipation tendency, queue width change, horizon acceleration, breakdown
propensity - and condition the prediction on it. Explicitly NOT the past days' queue LOCATIONS, which failed
before (note 79's onset map); only the run's dynamics parameters.

Conditioning is only worth building if the thing conditioned on explains the error. This is the gate, and it
needs no training: every component of z_run is a model-free statistic of the released grid, so it is computable
from the one to four earlier days an official window actually has (note 120).

    z_run, measured on strictly earlier days of the same run
        ext   P(queued at t+1 | not queued at t)                a breakdown / spreading rate
        dis   P(not queued at t+1 | queued at t)                a recovery rate
        wid   mean queued links per slot, and its day-over-day change
        spd   mean upstream displacement of the queue head per slot
        brk   queue onsets per day
        occ   mean queued share over the day

    the error, on the current day's windows, in the same coordinates
        d_ext  true minus predicted number of cells that turn ON relative to the persistence answer
        d_dis  true minus predicted number that turn OFF

If a run that extends more than usual is a run where we under-predict extension, `ext` correlates with `d_ext`
and the latent is worth building. The control re-measures z_run on a RANDOM other day of the month, which keeps
every marginal and destroys only the run-day pairing.

    python occ_lab/run_dynamics_gate.py --tag s20
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
ZN = ["ext", "dis", "wid", "d_wid", "spd", "brk", "occ"]


def day_dynamics(q, ok):
    """z for one day's released grid: (D, T, S) slices in, seven numbers out."""
    a, b = q[:-1], q[1:]
    m = ok[:-1] & ok[1:]
    on = int((b & ~a & m).sum())
    off = int((~b & a & m).sum())
    n_off = max(int((~a & m).sum()), 1)
    n_on = max(int((a & m).sum()), 1)
    wid = q.sum(1).astype(float)                              # queued links per slot
    head = np.array([np.flatnonzero(r).min() if r.any() else np.nan for r in q], float)
    spd = float(np.nanmean(np.diff(head))) if np.isfinite(head).sum() > 1 else 0.0
    any_t = q.any(1)
    brk = float((any_t[1:] & ~any_t[:-1]).sum())
    return np.array([on / n_off, off / n_on, wid.mean(), 0.0, spd, brk, float(q.mean())])


def run_state(ctx, z, days):
    """z_run averaged over the given days, with the width trend filled in."""
    if not len(days):
        return np.zeros(len(ZN))
    ok = z["elig"] & np.isfinite(z["obs_speed"]) & np.isfinite(z["obs_flow"])
    q = ok & (np.nan_to_num(z["obs_speed"], nan=1e9) <= 0.6 * ctx.vf)
    rows = np.stack([day_dynamics(q[d], ok[d]) for d in days])
    out = rows.mean(0)
    out[3] = float(rows[-1, 2] - rows[0, 2]) if len(rows) > 1 else 0.0
    return out


def err_coords(pg, y, e, per):
    """The error in extension / dissipation coordinates, per window."""
    yh = (pg >= TAU) & e
    yt = y & e
    ax = (1, 2)
    t_ext = (yt & ~per & e).sum(ax).astype(float)
    p_ext = (yh & ~per & e).sum(ax).astype(float)
    t_dis = (~yt & per & e).sum(ax).astype(float)
    p_dis = (~yh & per & e).sum(ax).astype(float)
    n = np.maximum(e.sum(ax), 1)
    return np.stack([(t_ext - p_ext) / n, (t_dis - p_dis) / n], 1)


def persist_of(v, ok, vf, shape):
    vq = ok & (np.nan_to_num(v, nan=1e9) <= 0.6 * vf)
    idx = np.where(ok[:, ::-1], np.arange(ok.shape[1])[::-1][None, :, None], -1).max(1)
    N, S = idx.shape
    last = np.zeros((N, S), bool)
    m = idx >= 0
    jj = np.broadcast_to(np.arange(N)[:, None], idx.shape)
    ss = np.broadcast_to(np.arange(S), idx.shape)
    last[m] = vq[jj[m], idx[m], ss[m]]
    return np.broadcast_to(last[:, None, :], shape)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="s20")
    ap.add_argument("--max-past", type=int, default=4, help="the day budget an official window actually has")
    args = ap.parse_args()
    gbm = lgb.Booster(model_file=str(LAB / "models" / "residual_lgb.txt"))
    Z, Zc, E, P = [], [], [], []
    for i, panel in enumerate(C.T2_PANELS):
        ctx, _, Wd = C.windows(i, panel, need_train=True)
        sp = load_split(i, panel, args.tag)
        for s, D in sp.items():
            og, z = Wd[s]["og"], Wd[s]["z"]
            base, y, e, dd = D["base"], D["y"], D["e"], D["d"]
            F = global_features(D["P"], base)
            rh = gbm.predict(F.reshape(-1, F.shape[-1])).reshape(base.shape).astype(np.float32)
            pg = np.clip(base + ETA * np.where(np.abs(rh) >= GATE, rh, 0.0), 0, 1)
            per = persist_of(og["v"], og["ok"], ctx.vf, y.shape)
            ec = err_coords(pg, y, e, per)
            days = np.unique(dd)
            rng = np.random.default_rng(3 + i)
            zs, zc = {}, {}
            for k, day in enumerate(days):
                past = days[max(0, k - args.max_past):k]
                zs[day] = run_state(ctx, z, past)
                alt = rng.choice(days[days != day], min(len(past), max(len(days) - 1, 1)), replace=False) \
                    if len(past) and len(days) > 1 else past
                zc[day] = run_state(ctx, z, alt)
            keep = np.array([len(days[max(0, list(days).index(d) - args.max_past):list(days).index(d)]) > 0
                             for d in dd])
            if not keep.any():
                continue
            Z.append(np.stack([zs[d] for d in dd[keep]]))
            Zc.append(np.stack([zc[d] for d in dd[keep]]))
            E.append(ec[keep])
            P.append(np.full(int(keep.sum()), i))
        ctx.z = ctx.q = ctx.el = None
        print(f"  {panel}", flush=True)

    Z, Zc, E, P = (np.concatenate(a) for a in (Z, Zc, E, P))
    print(f"\n{len(Z):,} windows with at least one earlier day\n")
    print(f"{'z component':<10}" + f"{'corr d_ext':>12}{'control':>10}{'corr d_dis':>12}{'control':>10}")
    for j, nm in enumerate(ZN):
        r = [np.corrcoef(Z[:, j], E[:, c])[0, 1] for c in (0, 1)]
        rc = [np.corrcoef(Zc[:, j], E[:, c])[0, 1] for c in (0, 1)]
        print(f"{nm:<10}{r[0]:>12.4f}{rc[0]:>10.4f}{r[1]:>12.4f}{rc[1]:>10.4f}")

    par = {"objective": "l2", "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 80,
           "feature_fraction": 0.9, "verbose": -1, "num_threads": 8}
    print(f"\n5-fold CV R2 of the error coordinates on z_run (panel id included, so only the WITHIN-panel "
          f"variation is new)\n")
    print(f"{'target':<10}{'z_run':>10}{'control':>10}{'panel only':>12}")
    rng = np.random.default_rng(0)
    fold = rng.integers(5, size=len(Z))
    for c, nm in enumerate(("d_ext", "d_dis")):
        out = []
        for X in (np.column_stack([Z, P]), np.column_stack([Zc, P]), P[:, None]):
            pred = np.zeros(len(Z))
            for k in range(5):
                tr, te = fold != k, fold == k
                m_ = lgb.train(par, lgb.Dataset(X[tr], label=E[tr, c]), num_boost_round=200)
                pred[te] = m_.predict(X[te])
            out.append(1 - ((E[:, c] - pred) ** 2).sum() / ((E[:, c] - E[:, c].mean()) ** 2).sum())
        print(f"{nm:<10}{out[0]:>10.4f}{out[1]:>10.4f}{out[2]:>12.4f}")


if __name__ == "__main__":
    main()
