"""Do bottleneck discharge capacities transfer across simulation runs, when the released FD parameters do not fit?

The released fd_parameters disagree with the generator: links with 1-5 lanes and nominal capacity 5,756-13,162
all top out near the same flow, and 3-lane links carry more than their nominal capacity without queueing
(t2_ctm_sim diagnosis). The effective capacity must come from data. At a queue head (link s free-flowing,
link s-1 queued in the same slot) the flow at s is the bottleneck discharge rate. Per link, per split:
    dis[s]    median discharge flow over head events (>= 20 events)
    nhead[s]  number of head events (where the active bottlenecks are)
    q99.5[s]  near-maximum flow;  qshare[s]  share of queued cells
Train uses the truth; validation/private use the visible observations (a different run, note 17).

    py analysis/t2_capacity_transfer.py D7_I10_E D12_I5_N D7_I405_N
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tfb.data import load  # noqa: E402
from tfb.physics import fd_params  # noqa: E402


def stats(z: dict, truth: bool):
    v = z["true_speed"] if truth else z["obs_speed"]
    q = z["true_flow"] if truth else z["obs_flow"]
    ok = z["elig"] & np.isfinite(v) & np.isfinite(q)
    vf = z["free_speed"]
    qd = ok & (np.nan_to_num(v, nan=1e9) <= 0.6 * vf)
    fr = ok & (np.nan_to_num(v, nan=0) > 0.85 * vf)
    S = v.shape[2]
    head = np.zeros_like(ok)
    head[:, :, 1:] = fr[:, :, 1:] & qd[:, :, :-1]
    nhead = head.sum((0, 1))
    dis = np.array([np.median(q[:, :, s][head[:, :, s]]) if nhead[s] >= 20 else np.nan for s in range(S)])
    qmax = np.array([np.percentile(q[:, :, s][ok[:, :, s]], 99.5) for s in range(S)])
    qshare = qd.sum((0, 1)) / np.maximum(ok.sum((0, 1)), 1)
    return dis, nhead, qmax, qshare


def main() -> None:
    for p in sys.argv[1:] or ["D7_I10_E", "D12_I5_N", "D7_I405_N"]:
        out = {}
        for sp in ("train", "validation", "private"):
            z = load(p, sp)
            out[sp] = stats(z, sp == "train")
        cap = fd_params(p, z["link_ids"])["cap"]
        d0, n0, m0, s0 = out["train"]
        print(f"== {p}: S={len(cap)}; links with >= 20 train head events: {int(np.isfinite(d0).sum())}")
        for sp in ("validation", "private"):
            d1, n1, m1, s1 = out[sp]
            both = np.isfinite(d0) & np.isfinite(d1)
            top0, top1 = set(np.argsort(-n0)[:8]), set(np.argsort(-n1)[:8])
            print(f"  {sp}: discharge links in both {int(both.sum())}; median |dis/dis_train - 1| "
                  f"{np.median(np.abs(d1[both] / d0[both] - 1)):.3f}; corr {np.corrcoef(d0[both], d1[both])[0, 1]:.3f}; "
                  f"top-8 head-link overlap {len(top0 & top1)}/8; q99.5 corr {np.corrcoef(m0, m1)[0, 1]:.3f}, "
                  f"median ratio {np.median(m1 / m0):.3f}; queue-share corr {np.corrcoef(s0, s1)[0, 1]:.3f}")
        dv, dp = out["validation"][0], out["private"][0]
        rows = [f"s{s} n{n0[s]} dis {d0[s]:.0f} (val {dv[s]:.0f}, priv {dp[s]:.0f}) cap_fd {cap[s]:.0f}"
                for s in np.argsort(-n0)[:8]]
        print("  train top head links: " + "; ".join(rows), flush=True)


if __name__ == "__main__":
    main()
