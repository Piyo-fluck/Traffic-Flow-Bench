"""Does the residual's spatial correlation travel with the traffic?

The user's reading of notes 116-117 is that most of the unexplained error variance is microscopic realisation -
the particular platoon that happened to pass - rather than a deterministic function of the released macro state.
That reading makes a falsifiable prediction. A platoon that is on link i at time t is on link i+k at
t + travel_time, so the residual's correlation with a link k downstream should peak at a LAG THAT GROWS WITH k,
tracking the travel time. A static local common cause would instead keep the peak at lag zero at every offset.

Adjacent links are about 0.19 km apart, 6 seconds at free speed, so the first slot of lag needs roughly 9.5 km -
50 to 90 links, depending on the corridor. `t1_residual_axes.py` only looked at lag zero and found the spatial
correlation gone by 5 links; that is consistent with either story, because it never reached the offsets where the
two differ.

Free flow only (96% of cells, and congestion behaves differently). Control: the partner taken from a day seven
days away.

    .venv/Scripts/python.exe analysis/t1_cohort_ridge.py [tag]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tfb.data import PANELS, load  # noqa: E402

PRED = ROOT / "work" / "preds"
TAG = sys.argv[1] if len(sys.argv) > 1 else "t1_m3all"
OFFSETS = [-100, -50, -25, -10, -3, -1, 1, 3, 10, 25, 50, 100]
LAGS = [0, 1, 2, 3]


class Acc:
    def __init__(self):
        self.n = self.sx = self.sy = self.sxx = self.syy = self.sxy = 0.0

    def add(self, x, y):
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        self.n += x.size
        self.sx += x.sum(); self.sy += y.sum()
        self.sxx += (x * x).sum(); self.syy += (y * y).sum(); self.sxy += (x * y).sum()

    def rho(self):
        if self.n < 2000:
            return np.nan, 0
        n = self.n
        cxy = self.sxy - self.sx * self.sy / n
        cxx = self.sxx - self.sx ** 2 / n
        cyy = self.syy - self.sy ** 2 / n
        return (float(cxy / np.sqrt(cxx * cyy)) if cxx > 0 and cyy > 0 else np.nan), int(n)


def main() -> None:
    A = {(k, l, w): Acc() for k in OFFSETS for l in LAGS for w in ("real", "ctrl")}
    km = {k: [] for k in OFFSETS}
    spd = []
    for panel in PANELS:
        f = PRED / f"{TAG}_{panel}_test.npz"
        if not f.exists():
            continue
        p = np.load(f)
        z = load(panel, "train")
        d = p["days"]
        mp = np.asarray(z["milepost"], np.float64)
        vf = np.asarray(z["free_speed"], np.float64)
        ts, tq = z["true_speed"][d].astype(np.float64), z["true_flow"][d].astype(np.float64)
        tgt = z["target"][d] & np.isfinite(ts) & np.isfinite(tq)
        k_t = tq / np.maximum(ts, 1.0)
        k_p = p["pq"].astype(np.float64) / np.maximum(p["pv"].astype(np.float64), 1.0)
        free = np.nan_to_num(ts, nan=1e9) >= 0.6 * vf[None, None, :]
        e = np.where(tgt & free, k_p - k_t, np.nan)
        ec = np.roll(e, 7, axis=0)
        spd.append(float(np.nanmedian(np.where(tgt & free, ts, np.nan))))
        S = e.shape[2]
        for k in OFFSETS:
            a0, a1 = (0, S - k) if k > 0 else (-k, S)
            b0, b1 = (k, S) if k > 0 else (0, S + k)
            if a1 - a0 < 5:
                continue
            km[k].append(float(np.median(np.abs(mp[b0:b1] - mp[a0:a1]))))
            for l in LAGS:
                x = e[:, :e.shape[1] - l if l else None, a0:a1]
                xc = ec[:, :e.shape[1] - l if l else None, a0:a1]
                y = e[:, l:, b0:b1]
                A[(k, l, "real")].add(x, y)
                A[(k, l, "ctrl")].add(xc, y)
        print(f"  {panel}", flush=True)
        del z
    v = float(np.mean(spd))
    print(f"\nfree-flow median speed {v:.1f} km/h; one slot of lag is {v * 5 / 60:.2f} km of travel\n")
    print(f"{'offset':>7}{'km':>7}{'lag implied':>12} |" + "".join(f"{'lag ' + str(l):>10}" for l in LAGS)
          + f" |{'ctrl lag0':>10}")
    for k in OFFSETS:
        if not km[k]:
            continue
        dkm = float(np.mean(km[k]))
        implied = dkm / (v * 5 / 60)
        row = f"{k:>7}{dkm:>7.2f}{implied:>12.2f} |"
        for l in LAGS:
            r, n = A[(k, l, "real")].rho()
            row += f"{r:>+10.4f}" if n else f"{'-':>10}"
        c, _ = A[(k, 0, "ctrl")].rho()
        print(row + f" |{c:>+10.4f}")
    print("\npositive offset = higher milepost. A travelling cohort puts the peak on the lag column nearest")
    print("the implied value on one side; a static local cause keeps every peak at lag 0.")


if __name__ == "__main__":
    main()
