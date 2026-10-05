"""Task 1 EDA 01: structure, where the baseline loses, what surrounds a target, noise floor.

    py analysis/t1_eda01.py D12_I5_N
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tfb.data import REGIMES, T, load  # noqa: E402
from tfb.metrics import fmt, state_score  # noqa: E402


def interp_along(x: np.ndarray, avail: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    """Linear interpolation along one axis from the available cells; nearest value at the ends.

    Returns the filled array and the gap (distance between the bracketing
    available cells, 0 when the cell itself is available, -1 when one-sided).
    """
    x = np.moveaxis(x, axis, -1)
    a = np.moveaxis(avail, axis, -1)
    L = x.shape[-1]
    idx = np.arange(L, dtype=np.int32)
    prev = np.maximum.accumulate(np.where(a, idx, -1), axis=-1)
    nxt = np.minimum.accumulate(np.where(a, idx, L)[..., ::-1], axis=-1)[..., ::-1]
    hp, hn = prev >= 0, nxt < L
    xp = np.take_along_axis(x, np.clip(prev, 0, L - 1), -1)
    xn = np.take_along_axis(x, np.clip(nxt, 0, L - 1), -1)
    w = (idx - prev) / np.maximum(nxt - prev, 1)
    out = np.where(hp & hn, xp + w * (xn - xp), np.where(hp, xp, np.where(hn, xn, np.nan)))
    gap = np.where(hp & hn, nxt - prev, -1)
    return np.moveaxis(out.astype(np.float32), -1, axis), np.moveaxis(gap, -1, axis)


def robust_sigma(e: np.ndarray) -> float:
    e = e[np.isfinite(e)]
    return float(1.4826 * np.median(np.abs(e - np.median(e)))) if e.size else float("nan")


def main() -> None:
    panel = sys.argv[1] if len(sys.argv) > 1 else "D12_I5_N"
    t0 = time.time()
    z = load(panel, "train")
    print(f"loaded {panel} train in {time.time() - t0:.0f}s")
    D, _, S = z["obs_speed"].shape
    tgt, elig, pct = z["target"], z["elig"], z["pct"]
    ts, tq = z["true_speed"], z["true_flow"]
    os_, oq = z["obs_speed"], z["obs_flow"]
    lanes, reg = z["lanes"], z["regime"]

    print("\n== 1. structure ==")
    print(f"days={D} stations={S} unique links={len(set(z['link_ids']))} "
          f"regime days={ {r: int((reg == r).sum()) for r in REGIMES} }")
    print(f"has_sensor=1 stations: {int(z['has_sensor'].sum())}/{S}")
    dm = np.diff(z["milepost"])
    print(f"milepost along topology order: increasing {np.mean(dm > 0):.2f}, decreasing {np.mean(dm < 0):.2f}")
    u, c = np.unique(lanes, return_counts=True)
    print("lanes:", dict(zip(u.tolist(), c.tolist())))
    valued = np.isfinite(ts) & np.isfinite(tq)
    print(f"truth cells valued {valued.mean():.4f} | eligible {elig.mean():.4f} | "
          f"valued but ineligible {(valued & ~elig).mean():.4f} | pct==0 {(pct == 0).mean():.4f} | "
          f"pct==100 {(pct == 100).mean():.4f} | 0<pct<75 {((pct > 0) & (pct < 75)).mean():.4f} | "
          f"75<=pct<100 {((pct >= 75) & (pct < 100)).mean():.4f}")
    vis = np.isfinite(os_)
    print(f"masked layer: visible {vis.mean():.4f} | visible & ineligible {(vis & ~elig).mean():.4f} | "
          f"visible == truth: {bool(np.array_equal(os_[vis], ts[vis]))}")
    for r in REGIMES:
        dmask = (reg == r)[:, None, None]
        e = elig & dmask
        print(f"  {r}: targets {int((tgt & dmask).sum()):>9,} = {(tgt & dmask).sum() / e.sum():.3f} of eligible | "
              f"blank eligible non-target {((~vis) & e & ~tgt).sum() / e.sum():.4f} | "
              f"targets ineligible {(tgt & dmask & ~elig).sum()}")

    print("\n== 2. baseline (weekday x time-of-day mean) and where it loses ==")
    wk = z["weekday"]
    valid = (pct >= 75) & valued
    base_v = np.zeros_like(ts)
    base_q = np.zeros_like(tq)
    link_v = np.nansum(np.where(valid, ts, 0), axis=(0, 1)) / np.maximum(valid.sum((0, 1)), 1)
    link_q = np.nansum(np.where(valid, tq, 0), axis=(0, 1)) / np.maximum(valid.sum((0, 1)), 1)
    for w in range(7):
        dd = wk == w
        n = valid[dd].sum(0)
        pv = np.where(valid[dd], ts[dd], 0).sum(0) / np.maximum(n, 1)
        pq = np.where(valid[dd], tq[dd], 0).sum(0) / np.maximum(n, 1)
        base_v[dd] = np.where(n > 0, pv, link_v)
        base_q[dd] = np.where(n > 0, pq, link_q)
    print("baseline:", fmt(state_score(z, base_v, base_q)))
    m = tgt & valued
    ev = (base_v - ts)[m]
    eq = ((base_q - tq) / lanes)[m]
    tv = ts[m]
    print("by true speed bin (targets):  share_cells  rmse_v  share_sq_err_v  rmse_q/lane  share_sq_err_q")
    for lo, hi in [(0, 40), (40, 60), (60, 80), (80, 100), (100, 999)]:
        b = (tv >= lo) & (tv < hi)
        print(f"  [{lo:>3},{hi:>3}) {b.mean():>8.3f} {np.sqrt(np.mean(ev[b] ** 2)):>8.2f} "
              f"{np.sum(ev[b] ** 2) / np.sum(ev ** 2):>12.3f} {np.sqrt(np.mean(eq[b] ** 2)):>10.1f} "
              f"{np.sum(eq[b] ** 2) / np.sum(eq ** 2):>12.3f}")
    tod = np.broadcast_to(np.arange(T)[None, :, None], tgt.shape)[m] // 36
    print("by 3-hour block: rmse_v / rmse_q  ", "  ".join(
        f"{3 * h:02d}h {np.sqrt(np.mean(ev[tod == h] ** 2)):.1f}/{np.sqrt(np.mean(eq[tod == h] ** 2)):.0f}"
        for h in range(8)))
    day_sq = np.bincount(np.nonzero(m)[0], weights=ev ** 2, minlength=D)
    top = np.sort(day_sq)[::-1]
    print(f"speed sq-error share of worst 10% days: {top[: max(1, D // 10)].sum() / top.sum():.3f}; "
          f"worst 5 days: {[str(z['dates'][i]) for i in np.argsort(day_sq)[::-1][:5]]}")

    print("\n== 3. what surrounds a target ==")
    avail = vis
    tt = np.nonzero(tgt)
    d_, t_, s_ = tt

    def p(mask_fn):
        return float(np.mean(mask_fn))
    tprev = np.where(t_ > 0, avail[d_, np.maximum(t_ - 1, 0), s_], False)
    tnext = np.where(t_ < T - 1, avail[d_, np.minimum(t_ + 1, T - 1), s_], False)
    sup = np.where(s_ > 0, avail[d_, t_, np.maximum(s_ - 1, 0)], False)
    sdn = np.where(s_ < S - 1, avail[d_, t_, np.minimum(s_ + 1, S - 1)], False)
    print(f"P(visible) same station t-5min {p(tprev):.3f}, t+5min {p(tnext):.3f}, both {p(tprev & tnext):.3f} | "
          f"upstream station same t {p(sup):.3f}, downstream {p(sdn):.3f}, both {p(sup & sdn):.3f} | "
          f"none of the four {p(~(tprev | tnext | sup | sdn)):.3f}")
    for r in REGIMES:
        rr = reg[d_] == r
        print(f"  {r}: both time-neighbours visible {np.mean((tprev & tnext)[rr]):.3f}, "
              f"both space-neighbours visible {np.mean((sup & sdn)[rr]):.3f}")
    _, gap_t = interp_along(os_, avail, axis=1)
    g = gap_t[tgt]
    print("time gap between bracketing visible cells (5-min steps): "
          + ", ".join(f"q{q}={np.percentile(g[g > 0], q):.0f}" for q in (50, 90, 99))
          + f", one-sided {np.mean(g < 0):.4f}")

    print("\n== 4. noise floor (truth, second difference in time, all three cells eligible) ==")
    ok = elig[:, 1:-1] & elig[:, :-2] & elig[:, 2:]
    e2v = ts[:, 1:-1] - 0.5 * (ts[:, :-2] + ts[:, 2:])
    e2q = (tq[:, 1:-1] - 0.5 * (tq[:, :-2] + tq[:, 2:])) / lanes
    free = ts[:, 1:-1] > 90
    cong = ts[:, 1:-1] < 60
    k = 1 / np.sqrt(1.5)
    print(f"speed sigma: free-flow {k * robust_sigma(e2v[ok & free]):.2f} km/h, congested {k * robust_sigma(e2v[ok & cong]):.2f}, "
          f"all {k * robust_sigma(e2v[ok]):.2f} | flow/lane sigma: free {k * robust_sigma(e2q[ok & free]):.1f}, "
          f"congested {k * robust_sigma(e2q[ok & cong]):.1f}, all {k * robust_sigma(e2q[ok]):.1f}")
    nb = elig[:, :-2] & elig[:, 2:]
    low = (pct[:, 1:-1] > 0) & (pct[:, 1:-1] < 75) & np.isfinite(ts[:, 1:-1])
    mid = (pct[:, 1:-1] >= 75) & (pct[:, 1:-1] < 100)
    full = pct[:, 1:-1] == 100
    for name, cm in [("pct 1-74 (ineligible)", low), ("pct 75-99", mid), ("pct 100", full)]:
        cc = nb & cm & free
        print(f"  centre {name:<22} free-flow speed sigma {k * robust_sigma(e2v[cc]):.2f}, flow/lane {k * robust_sigma(e2q[cc]):.1f}  (n={int(cc.sum()):,})")

    print("\n== 5. probes on the targets ==")
    t_all, _ = interp_along(os_, avail, 1)
    q_all, _ = interp_along(oq, np.isfinite(oq), 1)
    print("time-interp, any visible     :", fmt(state_score(z, np.where(np.isnan(t_all), base_v, t_all), np.where(np.isnan(q_all), base_q, q_all))))
    ae = avail & elig
    t_el, _ = interp_along(np.where(ae, os_, np.nan), ae, 1)
    q_el, _ = interp_along(np.where(ae, oq, np.nan), ae, 1)
    print("time-interp, eligible only   :", fmt(state_score(z, np.where(np.isnan(t_el), base_v, t_el), np.where(np.isnan(q_el), base_q, q_el))))
    s_all, _ = interp_along(os_, avail, 2)
    sq_all, _ = interp_along(oq / lanes, np.isfinite(oq), 2)
    print("space-interp, any visible    :", fmt(state_score(z, np.where(np.isnan(s_all), base_v, s_all), np.where(np.isnan(sq_all), base_q, sq_all * lanes))))
    an_v, _ = interp_along(os_ - base_v, avail, 1)
    an_q, _ = interp_along(oq - base_q, np.isfinite(oq), 1)
    pv = base_v + np.nan_to_num(an_v)
    pq = base_q + np.nan_to_num(an_q)
    print("baseline + time-interp anomaly:", fmt(state_score(z, pv, pq)))
    mv = np.where(np.isnan(s_all), t_all, 0.5 * (t_all + s_all))
    mq = np.where(np.isnan(sq_all), q_all, 0.5 * (q_all + sq_all * lanes))
    print("mean(time, space)            :", fmt(state_score(z, np.where(np.isnan(mv), base_v, mv), np.where(np.isnan(mq), base_q, mq))))


if __name__ == "__main__":
    main()
