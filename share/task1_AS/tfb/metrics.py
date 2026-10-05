"""Task 1 score on dense arrays, matching score_task1.py (per-lane flow, macro mean over regimes)."""
from __future__ import annotations

import numpy as np

from tfb.data import REGIMES

SPEED_NORM, FLOW_NORM, SPEED_W, FLOW_W = 25.0, 600.0, 0.54, 0.46


def state_score(data: dict, pred_speed: np.ndarray, pred_flow: np.ndarray,
                cells: np.ndarray | None = None) -> dict:
    """S_state and its parts over the target cells (optionally a subset of them)."""
    target = data["target"] if cells is None else data["target"] & cells
    lanes = np.broadcast_to(data["lanes"], target.shape)
    ts, tq = data["true_speed"], data["true_flow"]
    rows = {}
    for r in REGIMES:
        m = target & (data["regime"] == r)[:, None, None] & np.isfinite(ts) & np.isfinite(tq)
        n = int(m.sum())
        if n == 0:
            rows[r] = dict(n=0, rmse_v=np.nan, rmse_q=np.nan, S_v=0.0, S_q=0.0, S=0.0)
            continue
        ps = np.nan_to_num(pred_speed[m], nan=0.0)
        pq = np.nan_to_num(pred_flow[m], nan=0.0)
        rv = float(np.sqrt(np.mean((ps - ts[m]) ** 2)))
        rq = float(np.sqrt(np.mean(((pq - tq[m]) / lanes[m]) ** 2)))
        sv, sq = max(0.0, 1 - rv / SPEED_NORM), max(0.0, 1 - rq / FLOW_NORM)
        rows[r] = dict(n=n, rmse_v=rv, rmse_q=rq, S_v=sv, S_q=sq, S=SPEED_W * sv + FLOW_W * sq)
    rows["mean"] = dict(n=sum(rows[r]["n"] for r in REGIMES),
                        rmse_v=np.nan, rmse_q=np.nan,
                        S_v=float(np.mean([rows[r]["S_v"] for r in REGIMES])),
                        S_q=float(np.mean([rows[r]["S_q"] for r in REGIMES])),
                        S=float(np.mean([rows[r]["S"] for r in REGIMES])))
    return rows


def fmt(rows: dict) -> str:
    parts = [f"{r}: v={rows[r]['rmse_v']:.2f} q={rows[r]['rmse_q']:.1f} S={rows[r]['S']:.4f}" for r in REGIMES]
    return " | ".join(parts) + f" || S_state={rows['mean']['S']:.4f} (S_v={rows['mean']['S_v']:.3f}, S_q={rows['mean']['S_q']:.3f})"
