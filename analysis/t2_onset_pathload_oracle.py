"""Task 2 onset: can DEMAND (path load) rank the rare bottleneck we miss? Oracle ceiling first.

Notes 51/59: on the ~12% of official-like onset windows M misses, the true bottleneck is a median 26 links away,
rare in train, and its past-day activity in the same run does NOT rank it above the habitual one. The remaining
hypothesis (the user's): what distinguishes those days is where TODAY's demand goes, which a Task-4-style path/OD
solve on the window's own history could expose.

Before building the causal estimator, measure the ceiling. Four scores per candidate bottleneck (the Voronoi cells
of the train first-onset peaks, as in notes 51/59):

    now            mean q/cap in the cell over the last 3 history slots          (baseline: local demand now)
    path           path flows fitted to the HISTORY link flows, extrapolated to T+30 by the train profile growth,
                   mapped back to links as A x_hat, over capacity                (the proposal, estimated)
    oracle_flow    the link flow at T+30 in the cell, over capacity              (ceiling: future flows known)
    oracle_demand  path flows fitted to the T+30 link flows, as A x / cap        (ceiling: future demand known)

The scored splits ship no hidden truth arrays (only `obs_*`), so "the flow at T+30" means the released observed
layer at that slot — which is exactly the layer the queue labels of these windows are built from, so the oracle is
self-consistent with what we are scored on.

If the oracles do not rank the true bottleneck first, no amount of estimation quality can make this line work.

Solve: min_x ||A x - y||^2 + lambda ||x - b||^2 has the closed form x = b + A^T (A A^T + lambda I)^-1 (y - A b),
which needs one factorisation of a (links x links) matrix per panel instead of an NNLS over 4278 unknowns per
window (mathematically the same problem as the official `[A; sqrt(lambda) I]` stack; negatives are clipped after).
The released weak prior is a PM-period (15-19h) object while onset windows are in the morning, so the prior is
tried both as published and as zero.

    .venv/Scripts/python.exe analysis/t2_onset_pathload_oracle.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "scripts"))
from t2_demand_stress import load_tq  # noqa: E402
from t2_eda02 import T2_PANELS  # noqa: E402
from t2_eda03 import F  # noqa: E402
from t2_first_onset_eval import first_onsets, official_blank  # noqa: E402
from t2_m1_models import Ctx  # noqa: E402
from t2_onset_gate_diag import N_CAND, SEP, iou1, peaks, sel, voronoi  # noqa: E402
from tfb.data import CACHE, RELEASE, load  # noqa: E402

TAGS = ["t2_cnn_lov", "t2_cnn_lov_s1", "t2_cnn_lov_s2"]
W_M, LAM = 0.7, 0.05
SCORES = ["now", "path", "path0", "oracle_flow", "oracle_demand"]


def operator(panel: str, link_ids: np.ndarray):
    """(A, keep) with A (n_keep, P) the path-link incidence restricted to this panel's station links."""
    net = RELEASE / "corridors" / panel / "network"
    inc = pd.read_csv(net / "path_link_incidence.csv", dtype={"link_id": str, "path_id": str})
    paths = pd.read_csv(net / "path_set.csv", dtype={"path_id": str})
    pid = {p: i for i, p in enumerate(paths.path_id)}
    lid = {str(l): i for i, l in enumerate(link_ids)}
    m = inc.link_id.map(lid)
    keep_rows = m.notna()
    A = np.zeros((len(link_ids), len(pid)), np.float64)
    A[m[keep_rows].to_numpy(int), inc.path_id[keep_rows].map(pid).to_numpy(int)] = 1.0
    keep = A.any(1)                                   # station links that appear on at least one path
    return A, keep, list(paths.path_id)


def prior_vec(panel: str, split: str, path_ids) -> np.ndarray:
    f = RELEASE / "task4" / panel / split / "synthetic_weak_prior.csv"
    if not f.exists():
        return np.zeros(len(path_ids))
    pr = pd.read_csv(f, dtype={"path_id": str}).set_index("path_id").reindex(path_ids)
    return pd.to_numeric(pr.path_flow, errors="coerce").fillna(0.0).to_numpy(float)


def make_solver(A: np.ndarray, lam: float):
    G = A @ A.T + lam * np.eye(A.shape[0])
    c = cho_factor(G)

    def solve(Y: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Y (n_links, n_rhs) -> x (P, n_rhs), x = b + A^T (A A^T + lam I)^-1 (y - A b)."""
        r = Y - (A @ b)[:, None]
        return np.maximum(b[:, None] + A.T @ cho_solve(c, r), 0.0)
    return solve


def main() -> None:
    t0 = time.time()
    rows: list = []
    for i, panel in enumerate(T2_PANELS):
        ctx = Ctx(panel, i)
        ctx.fitd = np.ones_like(ctx.fitd)
        tq = load_tq(panel)
        A_full, keep, path_ids = operator(panel, ctx.z["link_ids"])
        A = A_full[keep]
        solve = make_solver(A, LAM)
        tr = first_onsets(ctx.vf, ctx.z, truth=True)
        cands = peaks(tr["cells"].sum(0), N_CAND, SEP)
        NB = voronoi(cands, ctx.S)
        cap_cell = np.array([max(ctx.cap[m].sum(), 1.0) for m in NB])
        for split in ("validation", "private"):
            z = load(panel, split)
            on = first_onsets(ctx.vf, z, blank=official_blank(panel, split, z))
            n = len(on["d"])
            if not n:
                continue
            b = prior_vec(panel, split, path_ids)
            c = np.load(CACHE / f"t2probs_{split}_{panel}.npz")
            assert (c["on_d"] == on["d"]).all()
            cn = np.mean([np.load(CACHE / f"t2cnn_{t}_{split}_{panel}.npz")["on_p"] for t in TAGS], 0)
            pM = W_M * c["on_p"] + (1 - W_M) * cn
            truth, elig = on["fut"][:, F - 1], on["efut"][:, F - 1]
            cov = truth.astype(int) @ NB.T
            b_true = np.where(cov.max(1) > 0, cov.argmax(1), -1)
            d, t0w = on["d"], on["t0"]
            wd = z["weekday"][d].astype(int)
            # history flows (observed layer, as a real estimator would see) and the true T+30 flows (oracle)
            hi = t0w[:, None] + np.arange(-3, 0)
            q_hist = np.nanmean(np.where(z["elig"][d[:, None], hi], z["obs_flow"][d[:, None], hi], np.nan), 1)
            q_hist = np.nan_to_num(q_hist, nan=0.0)                       # (n, S)
            ti = t0w + F                                                   # the scored splits ship no hidden truth
            q_true = np.where(z["elig"][d, ti] & np.isfinite(z["obs_flow"][d, ti]),
                              np.nan_to_num(z["obs_flow"][d, ti], nan=0.0), 0.0)   # (n, S)
            growth = (tq[wd, np.clip(t0w + F, 0, tq.shape[1] - 1)].sum(1)
                      / np.maximum(tq[wd, t0w - 1].sum(1), 1.0))           # (n,) corridor-level profile growth
            x_hist = solve(q_hist[:, keep].T, b)                           # (P, n)
            x_hist0 = solve(q_hist[:, keep].T, np.zeros_like(b))
            x_true = solve(q_true[:, keep].T, b)
            load_hist = (A_full @ x_hist).T * growth[:, None]               # (n, S) projected link load
            load_hist0 = (A_full @ x_hist0).T * growth[:, None]
            load_true = (A_full @ x_true).T
            for j in range(n):
                sc = {
                    "now": np.array([q_hist[j][m].sum() for m in NB]) / cap_cell,
                    "path": np.array([load_hist[j][m].sum() for m in NB]) / cap_cell,
                    "path0": np.array([load_hist0[j][m].sum() for m in NB]) / cap_cell,
                    "oracle_flow": np.array([q_true[j][m].sum() for m in NB]) / cap_cell,
                    "oracle_demand": np.array([load_true[j][m].sum() for m in NB]) / cap_cell,
                }
                bt = int(b_true[j])
                r = dict(panel=panel, split=split, iou=iou1(sel(pM[j]), truth[j], elig[j]), b_true=bt,
                         m_top1=int(np.argmax(pM[j] @ NB.T)), n_cand=len(cands))
                for k, v in sc.items():
                    r[f"rank_{k}"] = int((v > v[bt]).sum()) + 1 if bt >= 0 else -1
                rows.append(r)
        print(f"[{panel}] paths {A.shape[1]}, observed links {int(keep.sum())}/{ctx.S} ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "work" / "t2_onset_pathload_oracle.csv", index=False)
    miss = df.iou == 0
    print(f"\n=== {len(df)} official-like onset windows, {int(miss.sum())} missed (IoU = 0) ===")
    print(f"  M's own bottleneck top-1: all {np.mean(df.m_top1 == df.b_true):.2f}, "
          f"missed {np.mean(df.loc[miss, 'm_top1'] == df.loc[miss, 'b_true']):.2f}")
    print("\n  rank of the TRUE bottleneck among the candidates (1 = best):")
    print(f"  {'score':<14} {'all: top1':>10} {'top3':>6} {'med':>5} | {'missed: top1':>13} {'top3':>6} {'med':>5}")
    for k in SCORES:
        r = df[f"rank_{k}"]
        rm = df.loc[miss, f"rank_{k}"]
        print(f"  {k:<14} {np.mean(r == 1):>10.2f} {np.mean(r <= 3):>6.2f} {r.median():>5.0f} | "
              f"{np.mean(rm == 1):>13.2f} {np.mean(rm <= 3):>6.2f} {rm.median():>5.0f}")
    print(f"\n  saved work/t2_onset_pathload_oracle.csv; total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
