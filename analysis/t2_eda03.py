"""Task 2 EDA 03: onset windows = origins 30 min before onset -> predict only T+30, from recurrence priors.

"Earliest-onset" candidates mimic the official selection: no ongoing queue in
the history (release rule), nothing queued at T+5..T+25, at least one queued
cell at T+30. Priors are computed leave-one-day-out, so a window never sees
its own day:
  P_q  : P(link queued at slot t | weekday type)
  P_on : P(link is an onset cell at slot t | weekday type), an onset cell
         being a T+30 queued cell of an earliest-onset origin
Both are smoothed over +-SMOOTH slots. Predictors choose the top-k links (or
all links above a threshold) at T+30 only.

    py analysis/t2_eda03.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
from t2_eda02 import T2_PANELS  # noqa: E402
from tfb.data import RELEASE, T, load  # noqa: E402
from tfb.t1eval import family_mean  # noqa: E402

H, F, SMOOTH = 12, 6, 2
KS = (1, 2, 3, 4, 6)


def labels(z: dict) -> tuple[np.ndarray, np.ndarray]:
    ts = z["true_speed"]
    el = z["elig"] & np.isfinite(ts)
    return (np.nan_to_num(ts, nan=1e9) <= 0.6 * z["free_speed"]) & el, el


def earliest_onset(q: np.ndarray) -> np.ndarray:
    """(D, T) bool: origin t0 is an earliest-onset origin."""
    D, Tn, S = q.shape
    ok = np.zeros((D, Tn), bool)
    for t0 in range(H, Tn - F):
        hist = q[:, t0 - H:t0]
        ongoing = (hist.sum((1, 2)) >= 2) & (hist.sum(1).max(1) >= 2)
        ok[:, t0] = ~ongoing & ~q[:, t0 + 1:t0 + F].any((1, 2)) & q[:, t0 + F].any(1)
    return ok


def loo_prior(x: np.ndarray, wtype: np.ndarray) -> np.ndarray:
    """Leave-one-day-out mean over days of the same weekday type, smoothed in time. x: (D,T,S) float."""
    xs = np.zeros_like(x, dtype=np.float32)
    for k in range(-SMOOTH, SMOOTH + 1):
        xs += np.roll(x, k, axis=1)
    xs /= 2 * SMOOTH + 1
    out = np.empty_like(xs)
    for w in (False, True):
        m = wtype == w
        tot, n = xs[m].sum(0), m.sum()
        out[m] = (tot[None] - xs[m]) / max(n - 1, 1)
    return out


def score(prior_t30: np.ndarray, truth_t30: np.ndarray, elig_t30: np.ndarray, rule) -> np.ndarray:
    """IoU per window when predicting only at T+30. prior/truth/elig: (N, S)."""
    pred = rule(prior_t30) & elig_t30
    inter = (pred & truth_t30).sum(1)
    union = (pred | truth_t30).sum(1)
    return np.where(union == 0, 1.0, inter / np.maximum(union, 1))


def topk(k: int):
    def rule(p):
        idx = np.argsort(-p, axis=1)[:, :k]
        out = np.zeros_like(p, bool)
        np.put_along_axis(out, idx, True, axis=1)
        return out
    return rule


def main() -> None:
    res: dict[str, dict[str, dict[str, float]]] = {}
    for panel in T2_PANELS:
        z = load(panel, "train")
        q, el = labels(z)
        wt = z["weekday"] >= 5
        eo = earliest_onset(q)
        onset_cell = np.zeros_like(q)
        d_i, t_i = np.nonzero(eo)
        onset_cell[d_i, t_i + F] = q[d_i, t_i + F]
        priors = {"P_q": loo_prior(q.astype(np.float32), wt), "P_on": loo_prior(onset_cell.astype(np.float32), wt)}
        W = pd.read_csv(RELEASE / "task2" / panel / "train" / "window_index.csv")
        W = W[W.condition == "queue_onset"]
        d_of = {d: i for i, d in enumerate(z["dates"])}
        s = W.forecast_origin.astype(str)
        od = W.date.astype(str).map(d_of).to_numpy()
        ot = (s.str.slice(11, 13).astype(int) * 12 + s.str.slice(14, 16).astype(int) // 5).to_numpy()
        sets = {"cand": (d_i, t_i), "official": (od, ot)}
        print(f"{panel}: earliest-onset candidates {len(d_i)} on {len(np.unique(d_i))} days; official onset windows {len(od)}")
        for sname, (dd, tt) in sets.items():
            truth, elig = q[dd, tt + F], el[dd, tt + F]
            for pname, P in priors.items():
                pt = P[dd, tt + F]
                for k in KS:
                    res.setdefault(f"{pname} top{k}", {}).setdefault(sname, {})[panel] = score(pt, truth, elig, topk(k)).mean()
                for thr in (0.1, 0.2, 0.3):
                    res.setdefault(f"{pname} >{thr}", {}).setdefault(sname, {})[panel] = score(pt, truth, elig, lambda p, t=thr: p > t).mean()
    print(f"\n{'rule (T+30 only)':<16} {'candidates':>10} {'official':>9}")
    for name, r in res.items():
        print(f"{name:<16} {family_mean(r['cand']):>10.4f} {family_mean(r['official']):>9.4f}")


if __name__ == "__main__":
    main()
