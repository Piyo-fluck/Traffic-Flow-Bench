"""Experiment F: ensemble of adaptation trajectories, gated by the self-supervised loss of the CURRENT window.

Input: several finished run_adapt.py runs (different adaptation settings = different trajectories). For each window
    w_k = exp(-beta * L_k) / sum_j exp(-beta * L_j),   L_k = fixed-mask reconstruction loss of run k on that window
    p   = sum_k w_k * p_k
L_k is computed from the window history only (saved as on_ssl / og_ssl), so the gate is legal at prediction time.
beta is chosen on one month and applied to the other (cross-split), and the uniform mean is always shown.
L_k is standardised per window across runs (divided by the window's median over runs) so beta is scale-free.

    python tta/run_F_gate.py --runs rep9b C_norm D_nd A2T --name F_gate
"""
from __future__ import annotations

import argparse

import numpy as np

import common as C

BETAS = (0.0, 1.0, 3.0, 10.0, 30.0, 100.0)


def gate(ps, ls, beta):
    L = np.stack(ls, 0)                                  # (K, N)
    L = L / np.maximum(np.median(L, 0, keepdims=True), 1e-9)
    w = np.exp(-beta * (L - L.min(0, keepdims=True)))
    w /= w.sum(0, keepdims=True)
    P = np.stack(ps, 0)
    return (w.reshape(w.shape + (1,) * (P.ndim - 2)) * P).sum(0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    # 1) score every beta on each split
    fam = {}
    data = {}
    for i, panel in enumerate(C.T2_PANELS):
        ctx, stat, W = C.windows(i, panel)
        for split in C.SPLITS:
            rs = [C.load_run(r, split, panel) for r in args.runs]
            assert all(r is not None for r in rs), f"missing run output for {split} {panel}"
            w = W[split]
            data[(split, panel)] = (rs, w)
            for kind in ("on", "og"):
                if f"{kind}_p" not in rs[0] or not len(w[kind]["d"]):
                    continue
                for b in BETAS:
                    p = gate([r[f"{kind}_p"] for r in rs], [r[f"{kind}_ssl"] for r in rs], b)
                    name = "onset T-blend" if kind == "on" else "ongoing T-blend (x replaces cnn3)"
                    bl = C.blends(w["ref"], p if kind == "on" else None, p if kind == "og" else None)[name]
                    s = C.score(name, bl, w[kind])
                    fam.setdefault((split, kind, b), {})[panel] = s.mean()
    best = {}
    for kind in ("on", "og"):
        for split in C.SPLITS:
            row = {b: C.family_mean(fam[(split, kind, b)]) for b in BETAS if (split, kind, b) in fam}
            if row:
                print(f"{kind} {split}: " + "  ".join(f"beta {b:g} {v:.4f}" for b, v in row.items()))
                best[(kind, split)] = max(row, key=row.get)
    # 2) save the cross-split choice: beta fitted on the OTHER month
    for (split, panel), (rs, w) in data.items():
        other = "private" if split == "validation" else "validation"
        out = {}
        for kind in ("on", "og"):
            if (kind, other) not in best:
                continue
            b = best[(kind, other)]
            out[f"{kind}_p"] = gate([r[f"{kind}_p"] for r in rs], [r[f"{kind}_ssl"] for r in rs], b)
            out[f"{kind}_ssl"] = np.min(np.stack([r[f"{kind}_ssl"] for r in rs]), 0)
        C.save_run(args.name, split, panel, **out)
        um = {}
        for kind in ("on", "og"):
            if f"{kind}_p" in rs[0]:
                um[f"{kind}_p"] = np.mean([r[f"{kind}_p"] for r in rs], 0)
                um[f"{kind}_ssl"] = np.min(np.stack([r[f"{kind}_ssl"] for r in rs]), 0)
        C.save_run(args.name + "_uniform", split, panel, **um)
    print("cross-split betas:", best)
    C.report(list(args.runs) + [args.name + "_uniform", args.name], file=C.RUNS / args.name / "report.txt")


if __name__ == "__main__":
    main()
