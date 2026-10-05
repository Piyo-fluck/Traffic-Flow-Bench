"""Task 1 method M1: time/space anomaly interpolation with weights learned per gap bin.

For each channel, the anomaly (value minus the weekday x tod profile) at a
target is predicted as a*time_interp + b*space_interp. The pair (a, b) is
fitted by ridge least squares separately for every (time gap, space gap) bin,
on fit days only, and scored on the held-out test days.

    py analysis/t1_m1_blend.py D12_I5_N D7_I210_W
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tfb.data import PANELS, load  # noqa: E402
from tfb.features import interp_along, profile, truth_valid  # noqa: E402
from tfb.metrics import fmt, state_score  # noqa: E402
from tfb.t1eval import family_mean, split_days  # noqa: E402

NB = 10  # gap bins: 0 = one-sided or none, 1..8 = exact gap, 9 = 9 or more


def gap_bin(g: np.ndarray) -> np.ndarray:
    return np.where(g < 0, 0, np.minimum(g, NB - 1)).astype(np.int16)


class BinnedBlend:
    """y ≈ a[b]*x1 + b[b]*x2 with one (a, b) per bin, ridge-regularised toward 0."""

    def fit(self, y, x1, x2, bins, ridge=1e-3):
        k = NB * NB
        n = np.bincount(bins, minlength=k).astype(float)
        s11 = np.bincount(bins, x1 * x1, k); s12 = np.bincount(bins, x1 * x2, k)
        s22 = np.bincount(bins, x2 * x2, k); s1y = np.bincount(bins, x1 * y, k)
        s2y = np.bincount(bins, x2 * y, k)
        r = ridge * np.maximum(n, 1) * np.maximum(s11 / np.maximum(n, 1), 1e-6)
        A = np.stack([np.stack([s11 + r, s12], -1), np.stack([s12, s22 + r], -1)], -2)
        rhs = np.stack([s1y, s2y], -1)
        self.coef = np.linalg.solve(A, rhs[..., None])[..., 0]
        self.n = n
        return self

    def predict(self, x1, x2, bins):
        return self.coef[bins, 0] * x1 + self.coef[bins, 1] * x2


def run_panel(panel: str) -> dict[str, float]:
    t0 = time.time()
    z = load(panel, "train")
    fit, test = split_days(z)
    ts, tq, lanes = z["true_speed"], z["true_flow"], z["lanes"]
    wk = z["weekday"]
    valid = truth_valid(z)
    base_v = profile(ts, valid, wk, fit)
    base_q = profile(tq, valid, wk, fit)
    ov, oq = z["obs_speed"], z["obs_flow"]
    vis = np.isfinite(ov) & np.isfinite(oq)

    feats = {}
    for ch, obs, base, div in (("v", ov, base_v, 1.0), ("q", oq, base_q, lanes)):
        an = (obs - base) / div
        at, gt = interp_along(an, vis, 1)
        as_, gs = interp_along(an, vis, 2)
        feats[ch] = (np.nan_to_num(at), np.nan_to_num(as_))
    bins = gap_bin(gt) * NB + gap_bin(gs)

    tgt = z["target"] & np.isfinite(ts) & np.isfinite(tq)
    mf = tgt & fit[:, None, None]
    preds = {}
    coefs = {}
    for ch, base, truth, div in (("v", base_v, ts, 1.0), ("q", base_q, tq, lanes)):
        at, as_ = feats[ch]
        y = (truth - base) / div
        model = BinnedBlend().fit(y[mf], at[mf], as_[mf], bins[mf])
        coefs[ch] = model.coef
        preds[ch] = base + model.predict(at, as_, bins) * div
    cells = test[:, None, None]
    out = {}
    rows = {
        "baseline (fit-day profile)": (base_v, base_q),
        "base + time anomaly": (base_v + feats["v"][0], base_q + feats["q"][0] * lanes),
        "base + 0.8 time + 0.2 space": (base_v + 0.8 * feats["v"][0] + 0.2 * feats["v"][1],
                                         base_q + (0.8 * feats["q"][0] + 0.2 * feats["q"][1]) * lanes),
        "M1 binned blend": (preds["v"], preds["q"]),
    }
    print(f"\n== {panel} (test days {int(test.sum())}, {time.time() - t0:.0f}s) ==")
    for name, (pv, pq) in rows.items():
        r = state_score(z, pv, pq, cells=cells)
        out[name] = r["mean"]["S"]
        print(f"  {name:<30} {fmt(r)}")
    for ch in ("v", "q"):
        c = coefs[ch].reshape(NB, NB, 2)
        print(f"  M1 weights {ch} (time, space) for space gap 2, by time gap 1..9: "
              + " ".join(f"{g}:({c[g, 2, 0]:.2f},{c[g, 2, 1]:.2f})" for g in range(1, NB)))
    return out


def main() -> None:
    panels = sys.argv[1:] or PANELS
    results = {p: run_panel(p) for p in panels}
    if len(panels) > 1:
        print("\n== family-weighted mean over panels ==")
        for name in next(iter(results.values())):
            print(f"  {name:<30} {family_mean({p: r[name] for p, r in results.items()}):.4f}")


if __name__ == "__main__":
    main()
