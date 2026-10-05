"""Task 1 building blocks on the dense arrays from tfb.data."""
from __future__ import annotations

import numpy as np


def interp_along(x: np.ndarray, avail: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    """Linear interpolation along one axis from the available cells; nearest value at the ends.

    Returns the filled array (NaN where the whole line is unavailable) and the
    gap between the bracketing available cells: 0 at an available cell, -1
    where only one side exists or none.
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
    return np.moveaxis(out.astype(np.float32), -1, axis), np.moveaxis(gap.astype(np.int16), -1, axis)


def neighbors_along(x: np.ndarray, avail: np.ndarray, axis: int) -> tuple[np.ndarray, ...]:
    """Nearest available value before and after each cell along one axis, and their distances.

    Returns (x_prev, x_next, d_prev, d_next) as float32, NaN where that side
    has no available cell. At an available cell both sides are the cell itself.
    """
    x = np.moveaxis(x, axis, -1)
    a = np.moveaxis(avail, axis, -1)
    L = x.shape[-1]
    idx = np.arange(L, dtype=np.int32)
    prev = np.maximum.accumulate(np.where(a, idx, -1), axis=-1)
    nxt = np.minimum.accumulate(np.where(a, idx, L)[..., ::-1], axis=-1)[..., ::-1]
    hp, hn = prev >= 0, nxt < L
    xp = np.where(hp, np.take_along_axis(x, np.clip(prev, 0, L - 1), -1), np.nan)
    xn = np.where(hn, np.take_along_axis(x, np.clip(nxt, 0, L - 1), -1), np.nan)
    dp = np.where(hp, idx - prev, np.nan)
    dn = np.where(hn, nxt - idx, np.nan)
    return tuple(np.moveaxis(v.astype(np.float32), -1, axis) for v in (xp, xn, dp, dn))


def shift(x: np.ndarray, k: int, axis: int) -> np.ndarray:
    """x shifted by k along axis (out[i] = x[i - k]), NaN-padded."""
    out = np.full_like(x, np.nan)
    src = [slice(None)] * x.ndim
    dst = [slice(None)] * x.ndim
    if k > 0:
        src[axis], dst[axis] = slice(None, -k), slice(k, None)
    elif k < 0:
        src[axis], dst[axis] = slice(-k, None), slice(None, k)
    out[tuple(dst)] = x[tuple(src)]
    return out


def profile(values: np.ndarray, valid: np.ndarray, weekday: np.ndarray, use_days: np.ndarray) -> np.ndarray:
    """Weekday x time-of-day x station mean over the valid cells of use_days, laid out for every day.

    This is the official historical-mean baseline when use_days is every train
    day. A station/slot with no valid cell falls back to the station mean.
    """
    v = valid & use_days[:, None, None]
    filled = np.where(v, values, 0)
    fallback = filled.sum((0, 1)) / np.maximum(v.sum((0, 1)), 1)
    out = np.empty_like(values)
    for w in range(7):
        dd = weekday == w
        uw = dd & use_days
        n = v[uw].sum(0)
        out[dd] = np.where(n > 0, filled[uw].sum(0) / np.maximum(n, 1), fallback)
    return out


def profile_table(values: np.ndarray, valid: np.ndarray, weekday: np.ndarray,
                  use_days: np.ndarray) -> np.ndarray:
    """The (7, T, S) weekday x time-of-day x station table behind profile(), to apply to another split."""
    v = valid & use_days[:, None, None]
    filled = np.where(v, values, 0)
    fallback = filled.sum((0, 1)) / np.maximum(v.sum((0, 1)), 1)
    table = np.empty((7,) + values.shape[1:], dtype=values.dtype)
    for w in range(7):
        uw = (weekday == w) & use_days
        n = v[uw].sum(0)
        table[w] = np.where(n > 0, filled[uw].sum(0) / np.maximum(n, 1), fallback)
    return table


def truth_valid(z: dict) -> np.ndarray:
    """Cells the baseline profile learns from: pct_observed >= 75 and both channels present."""
    return (z["pct"] >= 75) & np.isfinite(z["true_speed"]) & np.isfinite(z["true_flow"])
