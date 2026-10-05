"""Month-level adaptation for a split that comes from a different simulation run (note 17).

month_deviation: Task 1 (offline). The weekday-type x slot x station mean deviation of a month's visible
    observations from a base profile, shrunk toward 0 by kappa pseudo-days. Add it to the train profile to
    get a month-specific base.
month_past_prior: Task 2 (online). A queue-frequency prior that, for a window on day d, uses only the
    split's visible cells on days before d of the same weekday type, blended with a train prior table by
    kappa pseudo-days.
"""
from __future__ import annotations

import numpy as np


def _smooth_t(x: np.ndarray, w: int, axis: int) -> np.ndarray:
    return x if w == 0 else sum(np.roll(x, k, axis=axis) for k in range(-w, w + 1))


def month_deviation(z: dict, base_v: np.ndarray, base_q: np.ndarray, days: np.ndarray,
                    kappa_days: float | None, smooth: int = 1,
                    kappa_q: float | None = "same") -> tuple[np.ndarray, np.ndarray]:
    """(dev_v, dev_q), each (D, T, S): filled on `days`, zero elsewhere.

    kappa_days shrinks the speed deviation, and kappa_q the flow deviation. kappa_q defaults to
    kappa_days. None means no deviation for that channel.
    """
    if kappa_q == "same":
        kappa_q = kappa_days
    ov, oq = z["obs_speed"], z["obs_flow"]
    vis = z["elig"] & np.isfinite(ov) & np.isfinite(oq)
    rv = np.where(vis, ov - base_v, 0.0).astype(np.float32)
    rq = np.where(vis, oq - base_q, 0.0).astype(np.float32)
    wk = z["weekday"] >= 5
    dev_v, dev_q = np.zeros_like(base_v), np.zeros_like(base_q)
    for grp in (False, True):
        m = days & (wk == grp)
        if not m.any():
            continue
        n = _smooth_t(vis[m].sum(0).astype(np.float32), smooth, 0)
        if kappa_days is not None:
            dev_v[m] = (_smooth_t(rv[m].sum(0), smooth, 0) / (n + kappa_days * (2 * smooth + 1)))[None]
        if kappa_q is not None:
            dev_q[m] = (_smooth_t(rq[m].sum(0), smooth, 0) / (n + kappa_q * (2 * smooth + 1)))[None]
    return dev_v, dev_q


class MonthPastPrior:
    """P(queued | slot, link) for a window on split day d, from days < d of the same weekday type, shrunk to a train table."""

    def __init__(self, z: dict, vf: np.ndarray, train_table: np.ndarray, kappa_days: float, smooth: int = 1):
        ov = z["obs_speed"]
        vis = z["elig"] & np.isfinite(ov)
        vq = vis & (np.nan_to_num(ov, nan=1e9) <= 0.6 * vf)
        num = _smooth_t(vq.astype(np.float32), smooth, 1)
        den = _smooth_t(vis.astype(np.float32), smooth, 1)
        g = (z["weekday"] >= 5).astype(int)
        D, T, S = num.shape
        self.cum_num = np.zeros((2, D + 1, T, S), np.float32)
        self.cum_den = np.zeros((2, D + 1, T, S), np.float32)
        for k in (0, 1):
            m = (g == k)[:, None, None]
            self.cum_num[k, 1:] = np.cumsum(num * m, 0)
            self.cum_den[k, 1:] = np.cumsum(den * m, 0)
        self.table, self.k = train_table, kappa_days * (2 * smooth + 1)

    def __call__(self, d: np.ndarray, g: np.ndarray, t: np.ndarray) -> np.ndarray:
        """(N, S) prior for windows on split days d (index into the split), weekday types g, slots t."""
        t = np.clip(t, 0, self.table.shape[1] - 1)
        base = self.table[g, t]
        n, c = self.cum_num[g, d, t], self.cum_den[g, d, t]  # sums over days strictly before d
        return (n + self.k * base) / (c + self.k)


class MonthLOOPrior:
    """P(queued | slot, link) for a window on split day d, from every OTHER day of the same weekday type.

    The window's own day is always left out. Future days of the month are included. This is the
    measurement-only option B of 2026-09-15: it is not for submissions unless the user decides so.
    """

    def __init__(self, z: dict, vf: np.ndarray, train_table: np.ndarray, kappa_days: float, smooth: int = 1):
        ov = z["obs_speed"]
        vis = z["elig"] & np.isfinite(ov)
        vq = vis & (np.nan_to_num(ov, nan=1e9) <= 0.6 * vf)
        self.num = _smooth_t(vq.astype(np.float32), smooth, 1)
        self.den = _smooth_t(vis.astype(np.float32), smooth, 1)
        self.g = (z["weekday"] >= 5).astype(int)
        self.tot_num = np.stack([self.num[self.g == k].sum(0) for k in (0, 1)])
        self.tot_den = np.stack([self.den[self.g == k].sum(0) for k in (0, 1)])
        self.table, self.k = train_table, kappa_days * (2 * smooth + 1)

    def __call__(self, d: np.ndarray, g: np.ndarray, t: np.ndarray) -> np.ndarray:
        t = np.clip(t, 0, self.table.shape[1] - 1)
        own = (self.g[d] == g)[:, None]
        n = self.tot_num[g, t] - own * self.num[d, t]
        c = self.tot_den[g, t] - own * self.den[d, t]
        return (n + self.k * self.table[g, t]) / (c + self.k)
