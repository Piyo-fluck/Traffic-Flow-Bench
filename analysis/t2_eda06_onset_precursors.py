"""Task 2 EDA 06: does FLOW (not speed) announce where a queue starts at T+30?

On a triangular FD the speed stays at v_f until breakdown, so speed has no
precursor (note 09). A bottleneck breaks down when the demand reaching it
exceeds its capacity, so the history's flow/capacity should point at the onset
links. For earliest-onset windows (all train days; history = truth T-60..T-5
on eligible cells), compare link features between onset links (queued at T+30)
and the rest, and score top-3 picks at T+30 by each feature and by
combinations with the recurrence prior (leave-one-day-out).

    py analysis/t2_eda06_onset_precursors.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
from t2_eda02 import T2_PANELS  # noqa: E402
from t2_eda03 import F, H, earliest_onset, labels  # noqa: E402
from t2_eda04 import gids, prior_at, smooth  # noqa: E402
from tfb.data import load  # noqa: E402
from tfb.physics import fd_params  # noqa: E402
from tfb.t1eval import family_mean  # noqa: E402


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    x = np.concatenate([pos, neg])
    r = x.argsort().argsort().astype(float) + 1
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def nanlast(x: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """Last valid value along axis 1 of (N, H, S)."""
    idx = np.where(ok, np.arange(x.shape[1])[None, :, None], -1).max(1)
    v = np.take_along_axis(x, np.clip(idx, 0, None)[:, None, :], 1)[:, 0]
    return np.where(idx >= 0, v, np.nan)


def top3_iou(score: np.ndarray, truth: np.ndarray, elig: np.ndarray, k: int = 3) -> float:
    s = np.nan_to_num(score, nan=-1e9)
    pred = np.zeros_like(truth)
    np.put_along_axis(pred, np.argsort(-s, axis=1)[:, :k], True, axis=1)
    pred &= elig
    inter, union = (pred & truth).sum(1), (pred | truth).sum(1)
    return float(np.where(union == 0, 1.0, inter / np.maximum(union, 1)).mean())


def main() -> None:
    feats_auc: dict[str, dict[str, float]] = {}
    picks: dict[str, dict[str, float]] = {}
    for panel in T2_PANELS:
        z = load(panel, "train")
        q, el = labels(z)
        prm = fd_params(panel, z["link_ids"])
        d, t0 = np.nonzero(earliest_onset(q))
        hi = t0[:, None] + np.arange(-H, 0)
        ok = el[d[:, None], hi]
        qc = np.where(ok, z["true_flow"][d[:, None], hi] / prm["cap"], np.nan)          # flow / capacity
        occ = np.where(ok, z["true_occ"][d[:, None], hi], np.nan)
        vr = np.where(ok, z["true_speed"][d[:, None], hi] / z["free_speed"], np.nan)
        kc = np.where(ok, (z["true_flow"] / np.maximum(z["true_speed"], 1))[d[:, None], hi] / prm["kcrit"], np.nan)
        with np.errstate(invalid="ignore", all="ignore"):
            f = {
                "q/cap last": nanlast(qc, ok),
                "q/cap max3": np.nanmax(qc[:, -3:], 1),
                "q/cap trend": np.nanmean(qc[:, -3:], 1) - np.nanmean(qc[:, :3], 1),
                "k/kcrit last": nanlast(kc, ok),
                "k/kcrit max3": np.nanmax(kc[:, -3:], 1),
                "occ last": nanlast(occ, ok),
                "v/vf min3 (low=pos)": -np.nanmin(vr[:, -3:], 1),
                "downstream k/kcrit max3": np.concatenate([np.nanmax(kc[:, -3:, 1:], 1), np.full((len(d), 1), np.nan)], 1),
            }
        truth, elig = q[d, t0 + F], el[d, t0 + F]
        prior = prior_at(smooth(q.astype(np.float32), 1), gids(z, "wt2"), np.ones(len(z["dates"]), bool), d, t0 + F, loo=True)
        for name, v in f.items():
            feats_auc.setdefault(name, {})[panel] = auc(v[truth & elig], v[~truth & elig])
        feats_auc.setdefault("prior P_q", {})[panel] = auc(prior[truth & elig], prior[~truth & elig])
        kcl = np.nan_to_num(f["k/kcrit max3"], nan=0)
        cands = {
            "prior": prior,
            "k/kcrit max3": f["k/kcrit max3"],
            "q/cap max3": f["q/cap max3"],
            "prior x k/kcrit": prior * kcl,
            "prior x k/kcrit^2": prior * kcl ** 2,
            "prior x q/cap^4": prior * np.nan_to_num(f["q/cap max3"], nan=0) ** 4,
            "prior + 0.2 k/kcrit": prior + 0.2 * kcl,
        }
        for name, s in cands.items():
            picks.setdefault(name, {})[panel] = top3_iou(s, truth, elig)
        print(f"{panel}: {len(d)} earliest-onset windows; onset links per window {truth.sum(1).mean():.1f}; "
              f"k/kcrit of onset links (median) {np.nanmedian(f['k/kcrit max3'][truth]):.2f} vs others {np.nanmedian(f['k/kcrit max3'][~truth & elig]):.2f}", flush=True)
    print("\n== AUC (onset link vs other links, pooled within panel; family mean) ==")
    for name, r in feats_auc.items():
        print(f"  {name:<26} {family_mean(r):.3f}")
    print("\n== top-3 at T+30 (IoU, family mean) ==")
    for name, r in picks.items():
        print(f"  {name:<20} {family_mean(r):.3f}")


if __name__ == "__main__":
    main()
