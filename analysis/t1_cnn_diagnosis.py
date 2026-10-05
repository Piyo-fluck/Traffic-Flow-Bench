"""What does the Task 1 CNN actually succeed at, and where does it fail?

Four questions, all on the saved production predictions over the local test block (train days 2031-01/02, the
real official mask, 10 panels):

    1. SKILL      against two references it must beat - the weekday x time-of-day profile, and interpolation in
                  time from the nearest observed slots on the same link - by group. Where does the network add
                  value, and where is it no better than arithmetic?
    2. SHRINKAGE  regress the predicted deviation from the profile on the true deviation. A slope below 1 means
                  the network pulls its answers back towards the profile - a systematic, correctable failure,
                  unlike noise.
    3. STRUCTURE  autocorrelation of the residual between neighbouring target cells, in time and along the
                  corridor. Correlated residuals are left-over signal; independent ones are noise.
    4. SCALE      how the error grows with the flow level and with the link's local volatility that day.

    py analysis/t1_cnn_diagnosis.py [tag]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tfb.data import PANELS, load  # noqa: E402
from tfb.features import profile, truth_valid  # noqa: E402
from tfb.t1eval import split_days  # noqa: E402

PRED = ROOT / "work" / "preds"
TAG = sys.argv[1] if len(sys.argv) > 1 else "t1_m3all"


def nearest_interp(q, obs):
    """Linear interpolation in time from the nearest observed slots on the same link."""
    D, T, S = q.shape
    vb = np.full_like(q, np.nan); db = np.full(q.shape, 999, np.int32)
    cur = np.full((D, S), np.nan, np.float32); curd = np.full((D, S), 999, np.int32)
    for t in range(T):
        vb[:, t], db[:, t] = cur, curd
        cur = np.where(obs[:, t], q[:, t], cur); curd = np.where(obs[:, t], 0, np.minimum(curd + 1, 999))
    vf_ = np.full_like(q, np.nan); df = np.full(q.shape, 999, np.int32)
    cur = np.full((D, S), np.nan, np.float32); curd = np.full((D, S), 999, np.int32)
    for t in range(T - 1, -1, -1):
        vf_[:, t], df[:, t] = cur, curd
        cur = np.where(obs[:, t], q[:, t], cur); curd = np.where(obs[:, t], 0, np.minimum(curd + 1, 999))
    w = db / np.maximum(db + df, 1)
    out = np.where(np.isnan(vb), vf_, np.where(np.isnan(vf_), vb, (1 - w) * vb + w * vf_))
    return out


def rmse(e, m):
    return float(np.sqrt((e[m] ** 2).mean())) if m.any() else float("nan")


def main() -> None:
    skill, shrink, struct, scale = {}, {}, {}, {}
    for panel in PANELS:
        f = PRED / f"{TAG}_{panel}_test.npz"
        if not f.exists():
            continue
        p = np.load(f)
        z = load(panel, "train")
        fit_d, _ = split_days(z)
        days = p["days"]
        lanes, vf = z["lanes"], z["free_speed"]
        ts, tq = z["true_speed"][days], z["true_flow"][days]
        ql = tq / lanes
        valid = truth_valid(z)
        bq = profile(z["true_flow"], valid, z["weekday"], fit_d)[days] / lanes
        bv = profile(z["true_speed"], valid, z["weekday"], fit_d)[days]
        tgt = z["target"][days] & np.isfinite(ts) & np.isfinite(tq)
        obs = z["elig"][days] & np.isfinite(ql) & ~z["target"][days]
        pq, pv = p["pq"] / lanes, p["pv"]
        iq = nearest_interp(np.where(obs, ql, np.nan).astype(np.float32), obs)
        free = tgt & (np.nan_to_num(ts, nan=0.0) > 0.85 * vf)
        queued = tgt & (ts <= 0.6 * vf)
        hour = np.broadcast_to((np.arange(ts.shape[1]) * 24 // ts.shape[1])[None, :, None], ts.shape)
        peak = tgt & ((hour >= 7) & (hour < 10) | (hour >= 16) & (hour < 19))
        night = tgt & ((hour < 5) | (hour >= 22))
        groups = {"all": tgt, "free flow": free, "queued": queued, "peak hours": peak, "night": night}

        # 1 skill
        for g, m in groups.items():
            for nm, est in (("CNN", pq), ("profile", bq), ("time interp", iq)):
                a = skill.setdefault((g, nm), [0.0, 0])
                e = np.where(np.isfinite(est), est - ql, 0.0)
                a[0] += float((e[m] ** 2).sum()); a[1] += int(m.sum())
        # 2 shrinkage: predicted deviation vs true deviation
        for g, m in groups.items():
            for ch, (pp, bb, tt) in (("flow", (pq, bq, ql)), ("speed", (pv, bv, ts))):
                dt_, dp = (tt - bb)[m], (pp - bb)[m]
                a = shrink.setdefault((g, ch), [0.0, 0.0, 0.0, 0])
                a[0] += float((dt_ * dp).sum()); a[1] += float((dt_ ** 2).sum())
                a[2] += float((dp ** 2).sum()); a[3] += int(m.sum())
        # 3 residual structure
        r = np.where(tgt, ql - pq, np.nan)
        for nm, a_, b_ in (("time +1", r[:, :-1], r[:, 1:]), ("time +2", r[:, :-2], r[:, 2:]),
                           ("link +1", r[:, :, :-1], r[:, :, 1:])):
            m = np.isfinite(a_) & np.isfinite(b_)
            s = struct.setdefault(nm, [0.0, 0.0, 0.0, 0])
            s[0] += float((a_[m] * b_[m]).sum()); s[1] += float((a_[m] ** 2).sum())
            s[2] += float((b_[m] ** 2).sum()); s[3] += int(m.sum())
        # 4 scale
        vol = np.abs(np.diff(ql, axis=1, prepend=ql[:, :1]))
        for nm, edges, v in (("flow level", (0, 400, 800, 1200, 1800, 1e9), ql),
                             ("volatility", (0, 20, 40, 80, 160, 1e9), vol)):
            for i in range(len(edges) - 1):
                m = tgt & (v >= edges[i]) & (v < edges[i + 1])
                a = scale.setdefault((nm, f"{edges[i]:.0f}-{edges[i + 1]:.0f}"), [0.0, 0])
                a[0] += float(((pq - ql)[m] ** 2).sum()); a[1] += int(m.sum())
        print(f"  {panel}", flush=True)

    print("\n=== 1. skill: flow RMSE against the references ===")
    print(f"{'group':<12} {'cells':>10} {'CNN':>8} {'profile':>9} {'time interp':>12} {'CNN gain vs best':>17}")
    for g in ("all", "free flow", "queued", "peak hours", "night"):
        row = {nm: np.sqrt(skill[(g, nm)][0] / max(skill[(g, nm)][1], 1)) for nm in ("CNN", "profile",
                                                                                     "time interp")}
        best_ref = min(row["profile"], row["time interp"])
        print(f"{g:<12} {skill[(g, 'CNN')][1]:10,} {row['CNN']:8.2f} {row['profile']:9.2f} "
              f"{row['time interp']:12.2f} {best_ref - row['CNN']:17.2f}")

    print("\n=== 2. shrinkage: slope of predicted deviation on true deviation (1.0 = no shrinkage) ===")
    print(f"{'group':<12} {'flow slope':>11} {'flow corr':>10} {'speed slope':>12} {'speed corr':>11}")
    for g in ("all", "free flow", "queued", "peak hours", "night"):
        out = []
        for ch in ("flow", "speed"):
            c, vt, vp, _ = shrink[(g, ch)]
            out += [c / max(vt, 1e-9), c / max(np.sqrt(vt * vp), 1e-9)]
        print(f"{g:<12} {out[0]:11.3f} {out[1]:10.3f} {out[2]:12.3f} {out[3]:11.3f}")

    print("\n=== 3. residual correlation between neighbouring target cells ===")
    for nm, s in struct.items():
        print(f"   {nm:<8} r = {s[0] / max(np.sqrt(s[1] * s[2]), 1e-9):+.3f}   ({s[3]:,} pairs)")

    print("\n=== 4. error against level and volatility ===")
    for nm in ("flow level", "volatility"):
        print(f"   {nm}:")
        for k, a in scale.items():
            if k[0] == nm and a[1]:
                print(f"      {k[1]:>12} veh/h/lane: {a[1]:9,} cells, flow RMSE {np.sqrt(a[0] / a[1]):7.2f}")


if __name__ == "__main__":
    main()
