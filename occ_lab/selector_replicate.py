"""Replicate the organizers' Task 2 window selector on the train split (labels known) and score the match.

config/synthetic_release_v1.json: 5 queue_onset + 5 queue_ongoing windows per panel and split, minimum spacing 360
min, minimum history coverage 0.7, maximum persistence IoU 0.9, a queue in the horizon required. The organizers
confirmed (2026-09-23) that onset keeps the EARLIEST origin whose horizon holds a queue. If ongoing is scanned the
same way, the origin before an ongoing window failed a condition, which is evidence about the horizon.

Variants of the reconstruction are scored by how many of the 80 official train windows (8 panels x 10) they
reproduce exactly (same day and origin).

    .venv/Scripts/python.exe occ_lab/selector_replicate.py
"""
from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
for p in (PROJ, PROJ / "analysis", PROJ / "scripts"):
    sys.path.insert(0, str(p))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from t2_eda02 import T2_PANELS  # noqa: E402
from tfb.data import load  # noqa: E402

H, F, T = 12, 6, 288


def arrays(panel, split, layer):
    z = load(panel, split)
    vf = z["free_speed"]
    v = z["true_speed"] if layer == "true" else z["obs_speed"]
    qf = z["true_flow"] if layer == "true" else z["obs_flow"]
    ok = z["elig"] & np.isfinite(v) & np.isfinite(qf)
    fin = np.isfinite(v)
    q = fin & (np.nan_to_num(v, nan=1e9) <= 0.6 * vf)
    return z, q, ok, fin


def persistence_iou(q, m_hist, m_fut, d, t, last_mode):
    """IoU of the last visible history state held over the horizon, on horizon cells in m_fut."""
    hq, hm = q[d, t - H:t], m_hist[d, t - H:t]
    S = hq.shape[1]
    if last_mode == "last_obs":
        idx = np.where(hm[::-1], np.arange(H)[::-1][:, None], -1).max(0)
        last = np.zeros(S, bool)
        k = idx >= 0
        last[k] = hq[idx[k], np.arange(S)[k]]
    elif last_mode == "at_T":                                # the old baseline: the corridor state AT the origin
        last = q[d, t] & m_hist[d, t]
    else:                                                    # the final history slot only
        last = hq[-1] & hm[-1]
    fut, fm = q[d, t + 1:t + 1 + F], m_fut[d, t + 1:t + 1 + F]
    p = np.broadcast_to(last[None], fut.shape) & fm
    y = fut & fm
    u = (p | y).sum()
    return 1.0 if u == 0 else (p & y).sum() / u


def candidates(z, q, ok, fin, var):
    """Per (d, t): onset / ongoing eligibility under the variant."""
    D = min(q.shape[0], 45)                                  # the official windows sit in the first weeks
    mh = ok if var["hist_mask"] == "elig" else fin
    mf = ok if var["fut_mask"] == "elig" else fin
    on, og = np.zeros((D, T), bool), np.zeros((D, T), bool)
    for d in range(D):
        for t in range(H + 1, T - F):
            hour = t / 12
            cov = z["elig"][d, t - H:t].mean()
            if cov < 0.7:
                continue
            hq = (q[d, t - H:t] & mh[d, t - H:t]).any()
            fq = (q[d, t + 1:t + 1 + F] & mf[d, t + 1:t + 1 + F]).any()
            if not fq:
                continue
            if not hq:
                on[d, t] = var["on_hours"][0] <= hour < var["on_hours"][1]
            elif var["og_hours"][0] <= hour < var["og_hours"][1]:
                og[d, t] = persistence_iou(q, mh, mf, d, t, var["last"]) <= 0.9
    return on, og


def select(on, og, order):
    """Chronological scan, earliest first, spacing 360 min between all chosen windows of the panel-split."""
    chosen = []

    def free(d, t):
        return all(abs((d * T + t) - (dd * T + tt)) >= 72 for dd, tt, _ in chosen)

    def scan(mask, kind):
        n = 0
        for d in range(mask.shape[0]):
            for t in range(mask.shape[1]):
                if n >= 5:
                    return
                if mask[d, t] and free(d, t):
                    chosen.append((d, t, kind)); n += 1

    if order == "onset_first":
        scan(on, "queue_onset"); scan(og, "queue_ongoing")
    else:
        scan(og, "queue_ongoing"); scan(on, "queue_onset")
    return chosen


def main() -> None:
    variants = []
    for layer, hm, fm, last, order in product(("true",), ("elig", "fin"), ("elig", "fin"), ("at_T", "final", "last_obs"),
                                              ("onset_first",)):
        variants.append(dict(layer=layer, hist_mask=hm, fut_mask=fm, last=last, order=order,
                             on_hours=(0, 24), og_hours=(7, 21)))
    official = {}
    for panel in T2_PANELS:
        W = pd.read_csv(PROJ / "data" / "kaggle_public" / "task2" / panel / "train" / "window_index.csv")
        ts = pd.to_datetime(W.forecast_origin)
        official[panel] = {(str(r.date), t.hour * 12 + t.minute // 5, r.condition) for r, t in zip(W.itertuples(), ts)}
    cache = {}
    for var in variants:
        hit = {"queue_onset": 0, "queue_ongoing": 0}
        for panel in T2_PANELS:
            key = (panel, var["layer"], var["hist_mask"], var["fut_mask"], var["last"])
            if key not in cache:
                z, q, ok, fin = arrays(panel, "train", var["layer"])
                cache[key] = (np.asarray(z["dates"]).astype(str), candidates(z, q, ok, fin, var))
            dates, (on, og) = cache[key]
            for d, t, kind in select(on, og, var["order"]):
                if (dates[d], t, kind) in official[panel]:
                    hit[kind] += 1
        print(f"{var['layer']:<5}{var['hist_mask']:<5}{var['fut_mask']:<5}{var['last']:<9}{var['order']:<14}"
              f"onset {hit['queue_onset']:>2}/40  ongoing {hit['queue_ongoing']:>2}/40", flush=True)


if __name__ == "__main__":
    main()
