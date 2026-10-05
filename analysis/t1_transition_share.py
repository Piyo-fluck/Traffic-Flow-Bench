"""How much of S_LWR's residual sits at congestion transitions?

Reading the worst cells one by one (`t1_look_at_cases.py`) showed a single repeated failure: the truth holds a
queue, then flushes it in one slot - flow spikes above the free-flow level while speed jumps - and we put that
release one slot early, so we miss the discharge spike entirely. Every one of the largest flow errors had that
shape, on different corridors and different days.

S_LWR scores |e[t+1] - e[t]| L, so a one-slot timing error is its worst case: one large residual, then another
of the opposite sign. And transitions are only about 0.2% of cells (note 96), which is exactly the regime an
aggregate correlation cannot see - every structure test so far pooled all cells.

This splits the S_LWR numerator and denominator by distance to the nearest congestion crossing, where congested
means v < 0.6 vf, the same threshold the queue task uses.

    .venv/Scripts/python.exe analysis/t1_transition_share.py [tag]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tfb.data import PANELS, load  # noqa: E402
from tfb.physics import fd_params, ramp_ok  # noqa: E402

PRED = ROOT / "work" / "preds"
TAG = sys.argv[1] if len(sys.argv) > 1 else "t1_m3all"
BANDS = [(0, "at the crossing"), (1, "within 1 slot"), (2, "within 2"), (3, "within 3")]


def dist_to_cross(cong):
    """Slots from each transition (t -> t+1) to the nearest congestion crossing, along time."""
    cross = cong[:, 1:] != cong[:, :-1]                      # (D, T-1, S) at transition index t
    T = cross.shape[1]
    d = np.full(cross.shape, 10_000, np.int32)
    d[cross] = 0
    for _ in range(6):                                       # cheap distance transform, enough for 3 bands
        left = np.full_like(d, 10_000); left[:, 1:] = d[:, :-1] + 1
        right = np.full_like(d, 10_000); right[:, :-1] = d[:, 1:] + 1
        d = np.minimum(d, np.minimum(left, right))
    _ = T
    return d, cross


def main() -> None:
    num = {k: 0.0 for k, _ in BANDS}
    den = {k: 0.0 for k, _ in BANDS}
    cnt = {k: 0 for k, _ in BANDS}
    sq = {k: 0.0 for k, _ in BANDS}
    tot = [0.0, 0.0, 0, 0.0]
    disc = [0.0, 0.0]                                        # numerator at discharge / at onset crossings
    for panel in PANELS:
        f = PRED / f"{TAG}_{panel}_test.npz"
        if not f.exists():
            continue
        p = np.load(f)
        z = load(panel, "train")
        days = p["days"]
        prm = fd_params(panel, z["link_ids"])
        L = prm["length"].astype(np.float64)
        vf = np.asarray(z["free_speed"], np.float64)
        ok = ramp_ok(panel, "train", z)[days]
        ts, tq = z["true_speed"][days].astype(np.float64), z["true_flow"][days].astype(np.float64)
        lanes = np.broadcast_to(z["lanes"], ts.shape).astype(np.float64)
        tgt = z["target"][days] & np.isfinite(ts) & np.isfinite(tq)
        use = z["elig"][days] & np.isfinite(ts) & np.isfinite(tq)
        k_t = tq / np.maximum(ts, 1.0)
        k_p = np.where(tgt, p["pq"].astype(np.float64) / np.maximum(p["pv"].astype(np.float64), 1.0), k_t)
        e = np.where(tgt, k_p - k_t, 0.0)
        trans = use[:, :-1] & use[:, 1:] & ok[:, :-1]
        res = np.abs(e[:, 1:] - e[:, :-1]) * L
        rhs = np.abs(k_t[:, 1:] - k_t[:, :-1]) * L
        qerr = np.where(tgt, (p["pq"].astype(np.float64) - tq) / lanes, 0.0) ** 2

        cong = np.nan_to_num(ts, nan=1e9) < 0.6 * vf[None, None, :]
        d, cross = dist_to_cross(cong)
        release = cross & cong[:, :-1] & ~cong[:, 1:]        # congested -> free, the discharge
        onset = cross & ~cong[:, :-1] & cong[:, 1:]

        tot[0] += float(res[trans].sum()); tot[1] += float(rhs[trans].sum())
        tot[2] += int(trans.sum()); tot[3] += float(qerr[tgt].sum())
        disc[0] += float(res[trans & release].sum()); disc[1] += float(res[trans & onset].sum())
        for k, _ in BANDS:
            m = trans & (d <= k)
            num[k] += float(res[m].sum()); den[k] += float(rhs[m].sum()); cnt[k] += int(m.sum())
            mc = tgt.copy(); mc[:, :-1] &= (d <= k); mc[:, -1] = False
            sq[k] += float(qerr[mc].sum())
        print(f"  {panel}", flush=True)
        del z

    print(f"\ncongested = v < 0.6 vf; distance measured in slots from the nearest crossing")
    print(f"total scored transitions {tot[2]:,}, S_LWR numerator {tot[0]:,.0f}, denominator {tot[1]:,.0f}\n")
    print(f"{'band':<18} {'transitions':>12} {'share':>7} | {'S_LWR numerator':>16} {'share':>7} | "
          f"{'denominator':>12} {'share':>7} | {'flow SSE share':>15}")
    for k, name in BANDS:
        print(f"{name:<18} {cnt[k]:>12,} {cnt[k] / tot[2] * 100:>6.2f}% | {num[k]:>16,.0f} "
              f"{num[k] / tot[0] * 100:>6.2f}% | {den[k]:>12,.0f} {den[k] / tot[1] * 100:>6.2f}% | "
              f"{sq[k] / tot[3] * 100:>14.2f}%")
    print(f"\nof the numerator at a crossing: discharge {disc[0] / tot[0] * 100:.2f}% of the total, "
          f"onset {disc[1] / tot[0] * 100:.2f}%")


if __name__ == "__main__":
    main()
