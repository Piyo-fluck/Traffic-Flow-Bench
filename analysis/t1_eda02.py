"""Task 1 EDA 02: why do spatial neighbours help so little? Level vs anomaly, lags, sensors.

    py analysis/t1_eda02.py D7_I210_W
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tfb.data import T, load  # noqa: E402
from tfb.metrics import fmt, state_score  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t1_eda01 import interp_along  # noqa: E402


def profile(z: dict, x: np.ndarray) -> np.ndarray:
    """Weekday x time-of-day x station mean over eligible truth (the official baseline)."""
    valid = (z["pct"] >= 75) & np.isfinite(z["true_speed"]) & np.isfinite(z["true_flow"])
    out = np.zeros_like(x)
    fallback = np.where(valid, x, 0).sum((0, 1)) / np.maximum(valid.sum((0, 1)), 1)
    for w in range(7):
        dd = z["weekday"] == w
        n = valid[dd].sum(0)
        out[dd] = np.where(n > 0, np.where(valid[dd], x[dd], 0).sum(0) / np.maximum(n, 1), fallback)
    return out


def corr(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 10 else float("nan")


def main() -> None:
    panel = sys.argv[1] if len(sys.argv) > 1 else "D7_I210_W"
    z = load(panel, "train")
    D, _, S = z["obs_speed"].shape
    ts, tq, el = z["true_speed"], z["true_flow"], z["elig"]
    lanes = z["lanes"]
    tql = tq / lanes
    base_v, base_q = profile(z, ts), profile(z, tq)
    an_v = np.where(el, ts - base_v, np.nan)
    an_q = np.where(el, (tq - base_q) / lanes, np.nan)

    print(f"== {panel}: {S} stations, has_sensor {int(z['has_sensor'].sum())}/{S} ==")
    print("\n-- 1. station levels (eligible truth) --")
    night = np.zeros(T, bool)
    night[24:48] = True  # 02:00-04:00
    vnight = np.nanmedian(np.where(el & night[None, :, None], ts, np.nan), axis=(0, 1))
    qday = np.nanmean(np.where(el, tql, np.nan), axis=(0, 1))
    print(f"night median speed by station: min {vnight.min():.1f} max {vnight.max():.1f} "
          f"std {vnight.std():.2f}; adjacent |diff| median {np.median(np.abs(np.diff(vnight))):.2f}")
    print(f"links.csv free_speed: min {z['free_speed'].min():.1f} max {z['free_speed'].max():.1f}; "
          f"corr(night speed, free_speed) {corr(vnight, z['free_speed']):.3f}")
    print(f"mean flow/lane by station: min {qday.min():.0f} max {qday.max():.0f}; "
          f"adjacent |diff| median {np.median(np.abs(np.diff(qday))):.0f}, max {np.abs(np.diff(qday)).max():.0f}")

    print("\n-- 2. adjacent stations, same time --")
    pair = el[:, :, :-1] & el[:, :, 1:]
    dv = ts[:, :, 1:] - ts[:, :, :-1]
    dq = tql[:, :, 1:] - tql[:, :, :-1]
    cong = pair & ((ts[:, :, 1:] < 60) | (ts[:, :, :-1] < 60))
    free = pair & ~cong
    print(f"raw speed diff RMS: free {np.sqrt(np.nanmean(dv[free] ** 2)):.2f}, congested {np.sqrt(np.nanmean(dv[cong] ** 2)):.2f} "
          f"| raw flow/lane diff RMS: free {np.sqrt(np.nanmean(dq[free] ** 2)):.1f}, congested {np.sqrt(np.nanmean(dq[cong] ** 2)):.1f}")
    print(f"corr of anomaly s vs s+1 (same t): speed {corr(an_v[:, :, :-1], an_v[:, :, 1:]):.3f}, flow {corr(an_q[:, :, :-1], an_q[:, :, 1:]):.3f}")
    print(f"corr of anomaly t vs t+1 (same s): speed {corr(an_v[:, :-1], an_v[:, 1:]):.3f}, flow {corr(an_q[:, :-1], an_q[:, 1:]):.3f}")
    print("corr of speed anomaly by station distance k (same t):",
          ", ".join(f"k={k}: {corr(an_v[:, :, :-k], an_v[:, :, k:]):.3f}" for k in (1, 2, 3, 5, 10)))

    print("\n-- 3. propagation: corr(anomaly at s, anomaly at s+1 shifted by lag), speed, congested periods --")
    anyc = np.nanmin(np.where(el, ts, np.nan), axis=2) < 60  # (D,T): some station congested
    for name, a in (("speed", an_v), ("flow", an_q)):
        cells = []
        for lag in range(-4, 5):
            if lag >= 0:
                x, y, m = a[:, : T - lag, :-1], a[:, lag:, 1:], anyc[:, : T - lag]
            else:
                x, y, m = a[:, -lag:, :-1], a[:, : T + lag, 1:], anyc[:, -lag:]
            mm = np.broadcast_to(m[:, :, None], x.shape)
            cells.append(f"{lag:+d}:{corr(np.where(mm, x, np.nan), np.where(mm, y, np.nan)):.3f}")
        print(f"  {name} (downstream at t+lag vs upstream at t): " + " ".join(cells))

    print("\n-- 4. probes on targets --")
    os_, oq = z["obs_speed"], z["obs_flow"]
    vis_v, vis_q = np.isfinite(os_), np.isfinite(oq)
    at_v, gt = interp_along(os_ - base_v, vis_v, 1)
    at_q, _ = interp_along((oq - base_q) / lanes, vis_q, 1)
    as_v, gs = interp_along(os_ - base_v, vis_v, 2)
    as_q, _ = interp_along((oq - base_q) / lanes, vis_q, 2)
    at_v, at_q, as_v, as_q = (np.nan_to_num(a) for a in (at_v, at_q, as_v, as_q))
    print("base + time anomaly      :", fmt(state_score(z, base_v + at_v, base_q + at_q * lanes)))
    print("base + space anomaly     :", fmt(state_score(z, base_v + as_v, base_q + as_q * lanes)))
    for w in (0.2, 0.35, 0.5):
        print(f"base + {1 - w:.2f}*time + {w:.2f}*space:",
              fmt(state_score(z, base_v + (1 - w) * at_v + w * as_v, base_q + ((1 - w) * at_q + w * as_q) * lanes)))
    tgt = z["target"]
    for lo, hi in ((1, 2), (3, 4), (5, 8), (9, 999)):
        cells = tgt & (gt >= lo) & (gt <= hi)
        if cells.sum():
            r = state_score(z, base_v + at_v, base_q + at_q * lanes, cells=cells)["mean"]
            print(f"  time-anomaly by time gap {lo}-{hi}: share {cells.sum() / tgt.sum():.3f}, S_state {r['S']:.4f}")


if __name__ == "__main__":
    main()
