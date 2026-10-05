"""Is the Task 1 CNN residual low-rank over (day x time-of-day x link)? The ceiling, measured first.

The user's residual-LATC proposal: keep the CNN and model its residual tensor R = q_true - q_CNN with a low-rank
(day, TOD, link) structure, the idea being that the free-flow bulk carries a shared "demand level x day shape x
link share" pattern. Note 98 already found no day-level MEAN bias (0.69 veh/h/lane against RMSE 29.1), but a mean
over links cannot see a pattern that is positive on some links and negative on others, which is exactly what a
low-rank factor would capture.

This measures the CEILING before anything is built. The tensor is fitted by alternating least squares on HALF of
the target cells and scored on the other half - information we would never have in a scored month, where only
probes (5% of observed cells, note 98) are available. If even this cannot take the free-flow flow RMSE down by
the agreed 0.3 veh/h/lane, no probe-estimated version can.

    py analysis/t1_residual_lowrank.py [tag] [ranks...]
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
RANKS = [int(a) for a in sys.argv[2:]] or [1, 2, 4, 8]
ITERS = 25
RIDGE = float(__import__("os").environ.get("LR_RIDGE", 1.0))


def als(R, M, K, iters=ITERS, seed=0):
    """CP factors for a (D, T, S) tensor observed on M, by alternating ridge least squares."""
    rng = np.random.default_rng(seed)
    D, T, S = R.shape
    A = rng.normal(0, 0.1, (D, K))
    B = rng.normal(0, 0.1, (T, K))
    C = rng.normal(0, 0.1, (S, K))
    Rm = np.where(M, R, 0.0)
    for _ in range(iters):
        for mode in (0, 1, 2):
            if mode == 0:
                Z = np.einsum("tk,sk->tsk", B, C)                      # (T, S, K)
                G = np.einsum("tsk,tsl->kl", Z, Z * 0 + Z)             # placeholder, recomputed per slice below
                for d in range(D):
                    m = M[d]
                    if not m.any():
                        continue
                    Zi = Z[m]
                    G = Zi.T @ Zi + RIDGE * np.eye(K)
                    A[d] = np.linalg.solve(G, Zi.T @ R[d][m])
            elif mode == 1:
                Z = np.einsum("dk,sk->dsk", A, C)
                for t in range(T):
                    m = M[:, t]
                    if not m.any():
                        continue
                    Zi = Z[m]
                    G = Zi.T @ Zi + RIDGE * np.eye(K)
                    B[t] = np.linalg.solve(G, Zi.T @ R[:, t][m])
            else:
                Z = np.einsum("dk,tk->dtk", A, B)
                for s in range(S):
                    m = M[:, :, s]
                    if not m.any():
                        continue
                    Zi = Z[m]
                    G = Zi.T @ Zi + RIDGE * np.eye(K)
                    C[s] = np.linalg.solve(G, Zi.T @ R[:, :, s][m])
    _ = Rm
    return A, B, C


def main() -> None:
    rng = np.random.default_rng(0)
    acc = {}
    for panel in PANELS:
        f = PRED / f"{TAG}_{panel}_test.npz"
        if not f.exists():
            continue
        p = np.load(f)
        z = load(panel, "train")
        days = p["days"]
        ts, tq, lanes, vf = z["true_speed"][days], z["true_flow"][days], z["lanes"], z["free_speed"]
        tgt = z["target"][days] & np.isfinite(ts) & np.isfinite(tq)
        R = ((tq - p["pq"]) / lanes).astype(np.float64)                # per lane, the metric's units
        fit = tgt & (rng.random(tgt.shape) < 0.5)
        hold = tgt & ~fit
        free = hold & (np.nan_to_num(ts, nan=0.0) > 0.85 * vf)
        a = acc.setdefault("base", [0.0, 0.0, 0, 0])
        a[0] += float((R[hold] ** 2).sum()); a[2] += int(hold.sum())
        a[1] += float((R[free] ** 2).sum()); a[3] += int(free.sum())
        # rank 0 references: per-day, per-tod and per-link means fitted on the same half
        for name, axes in (("mean/day", (1, 2)), ("mean/tod", (0, 2)), ("mean/link", (0, 1))):
            s = np.where(fit, R, 0.0).sum(axis=axes)
            n = fit.sum(axis=axes)
            m = (s / np.maximum(n, 1))[(None,) * 0]
            shape = [1, 1, 1]
            keep = [i for i in range(3) if i not in axes][0]
            shape[keep] = -1
            corr = m.reshape(shape)
            e = acc.setdefault(name, [0.0, 0.0, 0, 0])
            e[0] += float(((R - corr)[hold] ** 2).sum()); e[2] += int(hold.sum())
            e[1] += float(((R - corr)[free] ** 2).sum()); e[3] += int(free.sum())
        for K in RANKS:
            A, B, C = als(R, fit, K)
            corr = np.einsum("dk,tk,sk->dts", A, B, C)
            e = acc.setdefault(f"rank {K}", [0.0, 0.0, 0, 0])
            e[0] += float(((R - corr)[hold] ** 2).sum()); e[2] += int(hold.sum())
            e[1] += float(((R - corr)[free] ** 2).sum()); e[3] += int(free.sum())
            f_in = acc.setdefault(f"rank {K} IN-SAMPLE", [0.0, 0.0, 0, 0])
            f_in[0] += float(((R - corr)[fit] ** 2).sum()); f_in[2] += int(fit.sum())
            f_in[1] += float((R[fit] ** 2).sum()); f_in[3] += int(fit.sum())
            print(f"    rank {K}: in-sample {np.sqrt(((R - corr)[fit] ** 2).mean()):.3f} vs "
                  f"{np.sqrt((R[fit] ** 2).mean()):.3f}; |corr| mean {np.abs(corr[hold]).mean():.3f}",
                  flush=True)
        print(f"  {panel}", flush=True)

    b = acc["base"]
    print(f"\nheld-out target cells {b[2]:,} (free flow {b[3]:,})")
    print(f"{'model':<12} {'q RMSE':>9} {'q RMSE free':>12} {'delta free':>11}")
    base_free = np.sqrt(b[1] / b[3])
    for k, e in acc.items():
        if "IN-SAMPLE" in k:
            print(f"{k:<20} fitted {np.sqrt(e[0] / e[2]):.3f} against {np.sqrt(e[1] / e[3]):.3f} unfitted")
            continue
        qf = np.sqrt(e[1] / e[3])
        print(f"{k:<12} {np.sqrt(e[0] / e[2]):9.2f} {qf:12.2f} {qf - base_free:+11.3f}")
    print("\n(fitted on half the TARGET cells - information a scored month never provides, so this is a ceiling)")


if __name__ == "__main__":
    main()
