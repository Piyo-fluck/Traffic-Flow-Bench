"""Does the Task 1 flow residual carry an upstream advection signature?

Note 01's 'noise floor' is not noise (`analysis/t1_noise_floor_doubt.py`): obs == true exactly, and the second
difference fails the i.i.d. autocorrelation test. So the residual is deterministic signal the CNN does not
capture. The physical candidate is advection: a platoon passing station u at time t reaches station s at
t + travel_time, and the lag DEPENDS ON SPEED, which a fixed convolution kernel cannot represent.

Adjacent links are 0.184 km apart (5.7 s at free speed), so the lag only reaches one 5-minute slot about 10 km
and ~50 links upstream. This scans the (link offset, time lag) plane for the high-pass flow anomaly:

    r(d, t, s)          = (predicted flow - true flow) / lanes, at target cells
    a(d, t, u)          = observed flow at u minus its own centred 7-slot mean, per lane
    corr(r, a(t - lag)) for u = s + offset

The advection signature is a ridge whose peak lag grows linearly with |offset|. A flat surface at zero means the
residual has no upstream origin and the branch is dead.

    .venv/Scripts/python.exe analysis/t1_advection_probe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tfb.data import load  # noqa: E402

PANELS = ["D12_I5_N", "D7_I405_N"]
OFFSETS = [-160, -80, -40, -20, -10, -3, -1, 1, 3, 10, 20, 40, 80, 160]
LAGS = list(range(0, 9))
W = 7                                                    # high-pass window, slots


def highpass(q):
    """q (days, T, links) with NaN where unobserved -> deviation from its own centred nan-mean of width W."""
    ok = np.isfinite(q).astype(np.float32)
    x = np.nan_to_num(q, nan=0.0).astype(np.float32)
    k = W // 2
    cs = np.cumsum(np.pad(x, ((0, 0), (1, 0), (0, 0))), 1, dtype=np.float64)
    cn = np.cumsum(np.pad(ok, ((0, 0), (1, 0), (0, 0))), 1, dtype=np.float64)
    T = q.shape[1]
    lo = np.clip(np.arange(T) - k, 0, T)
    hi = np.clip(np.arange(T) + k + 1, 0, T)
    s = cs[:, hi] - cs[:, lo]
    n = cn[:, hi] - cn[:, lo]
    mean = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    return (q - mean).astype(np.float32)


def main() -> None:
    print(f"offsets in links, lags in 5-min slots; correlation of the flow residual with the upstream anomaly\n")
    for panel in PANELS:
        f = ROOT / "work" / "preds" / f"t1_m3all_{panel}_test.npz"
        if not f.exists():
            print(f"{panel}: no saved predictions")
            continue
        p = np.load(f)
        z = load(panel, "train")
        days = p["days"]
        lanes = z["lanes"].astype(np.float32)
        tq = z["true_flow"][days]
        tgt = z["target"][days] & np.isfinite(tq)
        r = ((p["pq"] - tq) / lanes).astype(np.float32)
        r = np.where(tgt, r, np.nan)
        a = highpass(np.where(z["target"][days], np.nan, z["obs_flow"][days]) / lanes)
        mp = z["milepost"]
        print(f"=== {panel} ({tgt.sum():,} targets, {len(mp)} links, corridor {mp.max():.1f} km) ===")
        print(f"{'offset':>7} {'km':>7} " + " ".join(f"lag{l:<2}" for l in LAGS))
        best = (0.0, None)
        for off in OFFSETS:
            S = len(mp)
            i = np.arange(S)
            j = i + off
            m = (j >= 0) & (j < S)
            if m.sum() < 5:
                continue
            km = float(np.median(np.abs(mp[j[m]] - mp[i[m]])))
            row = []
            for lag in LAGS:
                rr = r[:, lag:, i[m]]
                aa = a[:, :a.shape[1] - lag if lag else None, j[m]]
                n = min(rr.shape[1], aa.shape[1])
                x, y = rr[:, :n].ravel(), aa[:, :n].ravel()
                k = np.isfinite(x) & np.isfinite(y)
                if k.sum() < 1000:
                    row.append(float("nan")); continue
                x, y = x[k], y[k]
                x = x - x.mean(); y = y - y.mean()
                c = float(x @ y / np.sqrt((x @ x) * (y @ y)))
                row.append(c)
                if abs(c) > abs(best[0]):
                    best = (c, (off, lag, km))
            print(f"{off:>7} {km:>7.2f} " + " ".join(f"{v:>+5.3f}" for v in row))
        print(f"  strongest |corr| {best[0]:+.4f} at offset {best[1][0]}, lag {best[1][1]} ({best[1][2]:.2f} km)\n")
        del z


if __name__ == "__main__":
    main()
