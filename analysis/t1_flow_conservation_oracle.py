"""How much of a masked cell's FLOW is already determined by its observed neighbours and the ramps?

The error anatomy (notes/96) puts 90% of the Task 1 flow error on free-flow cells, where flow is demand rather
than a queue phenomenon, and the production CNN sits at 28.9 veh/h/lane against a noise floor near 18. Masking is
i.i.d. per (day, timestamp, link), so at a masked cell the neighbouring LINKS at the same timestamp are observed
with probability 0.5-0.8, and along a corridor conservation ties them together:

    q(i) = q(i-1) + on(i) - off(i)        (steady state, per hour, summed over the links in between)

This measures, with no training at all, what several such estimators achieve on the real targets of 2031-01/02,
next to the production CNN:

    time      linear interpolation in time on the same link, from the nearest observed slots
    up/down   the nearest observed link upstream / downstream at the same timestamp, corrected by the ramp flows
              attached to the links in between
    cons      the mean of up and down where both exist
    cnn       the production prediction
    blends    0.5 cnn + 0.5 cons, and the per-cell oracle min(|cnn|, |cons|) as an upper bound

    py analysis/t1_flow_conservation_oracle.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tfb.data import PANELS, load  # noqa: E402
from tfb.physics import load_ramps  # noqa: E402

PRED = ROOT / "work" / "preds"
TAG = "t1_m3all"
MAXSTEP = 3          # how far along the corridor an estimator may reach


def nearest_obs(q, obs, axis, back):
    """Nearest observed value along `axis` (and how far away), searching backwards or forwards."""
    n = q.shape[axis]
    val = np.full_like(q, np.nan)
    dist = np.full(q.shape, 99, np.int16)
    cur = np.full(np.delete(q.shape, axis), np.nan, np.float32)
    curd = np.full(np.delete(q.shape, axis), 99, np.int16)
    idx = range(n) if back else range(n - 1, -1, -1)
    sl = [slice(None)] * q.ndim
    for k in idx:
        sl[axis] = k
        t = tuple(sl)
        val[t], dist[t] = cur, curd
        o = obs[t]
        cur = np.where(o, q[t], cur)
        curd = np.where(o, 0, np.minimum(curd + 1, 99)).astype(np.int16)
    return val, dist


def main() -> None:
    acc = {}
    for panel in PANELS:
        f = PRED / f"{TAG}_{panel}_test.npz"
        if not f.exists():
            continue
        p = np.load(f)
        z = load(panel, "train")
        days = p["days"]
        ts, tq = z["true_speed"][days], z["true_flow"][days]
        lanes, vf = z["lanes"], z["free_speed"]
        elig = z["elig"][days]
        tgt = z["target"][days] & np.isfinite(ts) & np.isfinite(tq)
        obs = elig & np.isfinite(tq) & ~z["target"][days]          # what the release actually shows
        r = load_ramps(panel, "train", z)
        on_q, off_q = r["on_q"][days], r["off_q"][days]
        q = np.where(obs, tq, np.nan).astype(np.float32)

        est = {}
        # --- time: interpolate on the same link between the nearest observed slots
        vb, db = nearest_obs(q, obs, 1, True)
        vfw, dfw = nearest_obs(q, obs, 1, False)
        w = db / np.maximum(db + dfw, 1)
        est["time"] = np.where(np.isnan(vb), vfw, np.where(np.isnan(vfw), vb, (1 - w) * vb + w * vfw))
        # --- along the corridor, corrected by the ramps in between
        net_in = on_q - off_q
        for name, back in (("up", True), ("down", False)):
            vv, dd = nearest_obs(q, obs, 2, back)
            corr = np.zeros_like(q)
            cum = np.cumsum(net_in, axis=2)
            for k in range(1, MAXSTEP + 1):                        # ramps strictly between source and target
                m = dd == k
                if not m.any():
                    continue
                s_idx = np.broadcast_to(np.arange(q.shape[2]), q.shape)
                src = s_idx - k if back else s_idx + k
                src = np.clip(src, 0, q.shape[2] - 1)
                d_ax, t_ax = np.indices(q.shape)[:2]
                delta = cum[d_ax, t_ax, s_idx] - cum[d_ax, t_ax, src]
                corr = np.where(m, delta if back else -delta, corr)
            est[name] = np.where(dd <= MAXSTEP, vv + corr, np.nan)
        both = np.isfinite(est["up"]) & np.isfinite(est["down"])
        est["cons"] = np.where(both, 0.5 * (np.nan_to_num(est["up"]) + np.nan_to_num(est["down"])),
                               np.where(np.isfinite(est["up"]), est["up"], est["down"]))
        est["cnn"] = p["pq"]
        est["cnn+cons"] = np.where(np.isfinite(est["cons"]), 0.5 * est["cnn"] + 0.5 * est["cons"], est["cnn"])

        free = tgt & (np.nan_to_num(ts, nan=0.0) > 0.85 * vf)
        for grp, m in (("all targets", tgt), ("free flow", free), ("queued", tgt & (ts <= 0.6 * vf))):
            for k, v in est.items():
                ok = m & np.isfinite(v)
                a = acc.setdefault((grp, k), [0.0, 0, 0])
                a[0] += float((((v - tq) / lanes)[ok] ** 2).sum()); a[1] += int(ok.sum()); a[2] += int(m.sum())
            ok = m & np.isfinite(est["cons"])
            d1 = np.abs(est["cnn"] - tq); d2 = np.abs(np.nan_to_num(est["cons"], nan=1e9) - tq)
            a = acc.setdefault((grp, "oracle cnn/cons"), [0.0, 0, 0])
            best = np.where(d2 < d1, est["cons"], est["cnn"])
            a[0] += float((((best - tq) / lanes)[m] ** 2).sum()); a[1] += int(m.sum()); a[2] += int(m.sum())
        print(f"  {panel}", flush=True)

    for grp in ("all targets", "free flow", "queued"):
        rows = [(k[1], v) for k, v in acc.items() if k[0] == grp]
        n_grp = max(v[2] for _, v in rows)
        print(f"\n=== {grp} ({n_grp:,} cells) ===")
        print(f"   {'estimator':<16} {'coverage':>9} {'flow RMSE':>10}")
        for k, v in rows:
            if v[1]:
                print(f"   {k:<16} {v[1] / v[2]:9.1%} {np.sqrt(v[0] / v[1]):10.2f}")


if __name__ == "__main__":
    main()
