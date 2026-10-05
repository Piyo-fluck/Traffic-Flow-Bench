"""Task 2 EDA 02: candidate windows from train truth, onset precursors, and heuristic predictors.

A window uses only its 12 history steps (T-60..T-5), the raw observation with
eligibility. Local labels: truth speed <= v_cut (0.6 x free speed) on eligible
cells. Candidate origins every 30 min on every train day; the condition
follows the release rule: ongoing when the history holds >= 2 queued cells
with >= 2 on one link, onset otherwise; every window needs a queued cell in
the horizon.

    py analysis/t2_eda02.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tfb.data import PANELS, RELEASE, T, load  # noqa: E402
from tfb.t1eval import family_mean  # noqa: E402

T2_PANELS = [p for p in PANELS if not p.startswith("D12_I405")]
H, F, STEP = 12, 6, 6


def build(z: dict, d: np.ndarray, t0: np.ndarray) -> dict:
    vf = z["free_speed"]
    ts = z["true_speed"]
    el = z["elig"] & np.isfinite(ts)
    q = (np.nan_to_num(ts, nan=1e9) <= 0.6 * vf) & el
    hi = t0[:, None] + np.arange(-H, 0)
    fi = t0[:, None] + np.arange(1, F + 1)
    hist = q[d[:, None], hi]
    c = dict(d=d, t0=t0, hist=hist, fut=q[d[:, None], fi], efut=el[d[:, None], fi],
             vhist=ts[d[:, None], hi] / vf, ehist=el[d[:, None], hi])
    c["ongoing"] = (hist.sum((1, 2)) >= 2) & (hist.sum(1).max(1) >= 2)
    return c


def candidates(z: dict) -> dict:
    D = len(z["dates"])
    t0s = np.arange(H, T - F, STEP)
    d, t0 = np.repeat(np.arange(D), len(t0s)), np.tile(t0s, D)
    c = build(z, d, t0)
    keep = c["fut"].sum((1, 2)) >= 1
    return {k: v[keep] for k, v in c.items()}


def iou(pred: np.ndarray, c: dict) -> np.ndarray:
    p = pred & c["efut"]
    inter = (p & c["fut"]).sum((1, 2))
    union = (p | c["fut"]).sum((1, 2))
    return np.where(union == 0, 1.0, inter / np.maximum(union, 1))


def dilate(link_pred: np.ndarray, up: int, down: int = 0) -> np.ndarray:
    """Spread a (N, S) link set to u stations upstream (lower index) and d downstream."""
    out = link_pred.copy()
    for k in range(1, up + 1):
        out[:, :-k] |= link_pred[:, k:]
    for k in range(1, down + 1):
        out[:, k:] |= link_pred[:, :-k]
    return out


def over_horizon(link_pred: np.ndarray) -> np.ndarray:
    return np.repeat(link_pred[:, None, :], F, axis=1)


def growing(link_pred: np.ndarray, rate: float) -> np.ndarray:
    """Horizon step h spreads upstream by ceil(h*rate) stations."""
    return np.stack([dilate(link_pred, int(np.ceil((h + 1) * rate))) for h in range(F)], axis=1)


def heuristics(c: dict) -> dict[str, np.ndarray]:
    last = c["hist"][:, -1]
    any2, any3 = c["hist"][:, -2:].any(1), c["hist"][:, -3:].any(1)
    vmin3 = np.nanmin(np.where(c["ehist"][:, -3:], c["vhist"][:, -3:], np.nan), axis=1)
    vlast = np.where(c["ehist"][:, -1], c["vhist"][:, -1], np.nan)
    out = {"persist": over_horizon(last), "persist_any2": over_horizon(any2), "persist_any3": over_horizon(any3)}
    for u in (1, 2, 3):
        out[f"persist_up{u}"] = over_horizon(dilate(last, u))
    for r in (0.25, 0.5, 1.0):
        out[f"persist_grow{r}"] = growing(last, r)
    for a in (0.65, 0.7, 0.75, 0.8, 0.85):
        with np.errstate(invalid="ignore"):
            out[f"onset_vmin3<{a}"] = over_horizon(np.nan_to_num(vmin3, nan=9) < a)
            out[f"onset_vlast<{a}"] = over_horizon(np.nan_to_num(vlast, nan=9) < a)
    return out


def main() -> None:
    res_c: dict[str, dict] = {}
    res_o: dict[str, dict] = {}
    for panel in T2_PANELS:
        z = load(panel, "train")
        c = candidates(z)
        on, og = ~c["ongoing"], c["ongoing"]
        nt = c["fut"].sum((1, 2))
        tod = c["t0"] / 12
        print(f"\n== {panel}: candidates onset {int(on.sum())}, ongoing {int(og.sum())} ==")
        print(f"  horizon queued cells, onset: median {np.median(nt[on]):.0f}, q90 {np.percentile(nt[on], 90):.0f} | "
              f"ongoing: median {np.median(nt[og]):.0f} | onset origin hour: {np.percentile(tod[on], [10, 50, 90]).round(1)}")
        # Onset precursor: the last visible speed ratio of links that queue in the horizon vs the rest.
        qlinks = c["fut"].any(1)
        vl = np.where(c["ehist"][:, -1], c["vhist"][:, -1], np.nan)
        a, b = vl[on][qlinks[on]], vl[on][~qlinks[on]]
        first = np.argmax(c["fut"][on].any(2), axis=1) + 1
        print(f"  onset: v(T-5)/vf of links that queue: q10/50/90 {np.nanpercentile(a, [10, 50, 90]).round(2)} | "
              f"other links: q1/5/50 {np.nanpercentile(b, [1, 5, 50]).round(2)} | first queued step (x5 min): "
              f"{np.bincount(first, minlength=F + 1)[1:].tolist()} | links queued per onset window: {qlinks[on].sum(1).mean():.1f}")
        W = pd.read_csv(RELEASE / "task2" / panel / "train" / "window_index.csv")
        d_of = {d: i for i, d in enumerate(z["dates"])}
        s = W.forecast_origin.astype(str)
        co = build(z, W.date.astype(str).map(d_of).to_numpy(),
                   (s.str.slice(11, 13).astype(int) * 12 + s.str.slice(14, 16).astype(int) // 5).to_numpy())
        co["ongoing"] = (W.condition == "queue_ongoing").to_numpy()
        for name, pred in heuristics(c).items():
            i = iou(pred, c)
            res_c.setdefault(name, {})[panel] = (i[on].mean(), i[og].mean())
        for name, pred in heuristics(co).items():
            i = iou(pred, co)
            res_o.setdefault(name, {})[panel] = (i[~co["ongoing"]].mean(), i[co["ongoing"]].mean())

    print("\n== heuristic IoU (family mean over the 8 panels) ==")
    print(f"{'heuristic':<18} {'cand onset':>10} {'cand ongoing':>12} {'official onset':>14} {'official ongoing':>16}")
    for name in res_c:
        fo = lambda r, k: family_mean({p: v[k] for p, v in r[name].items()})  # noqa: E731
        print(f"{name:<18} {fo(res_c, 0):>10.4f} {fo(res_c, 1):>12.4f} {fo(res_o, 0):>14.4f} {fo(res_o, 1):>16.4f}")


if __name__ == "__main__":
    main()
