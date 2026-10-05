"""Task 4 path flows by an L1 solve, to match the norms S_ODME actually uses.

    S_ODME = 0.45 S_od + 0.25 S_link + 0.15 S_dev + 0.15 S_attr
    S_od   = 1 - sum|fhat - f*| / sum f*        an L1 distance to the withheld truth
    S_dev  = exp(-|Dhat/Dstar - 1|),  Dhat = sum|fhat - b|
    S_link = 1 - sum|A fhat - c| / sum c

Three of the four terms are L1. The organizers' reference solve, which our submissions have used since the
beginning, is `nnls([A; sqrt(0.05) I], [c; sqrt(0.05) b])` - an L2 ridge toward the prior. With about 4,000 paths
against 100-350 counted links the counts leave roughly 4,000 free dimensions, so the regulariser decides where
the answer lands, and an L2 regulariser spreads the adjustment thinly over every path while an L1 one
concentrates it on few. The released priors say which is right: between two periods the true demand change puts
0.65 of its L1 mass in the top 10% of paths, while our current answer spreads the same kind of adjustment to
0.30. The estimator is using the wrong norm.

So: minimise the L1 deviation from the prior, with the count fit weighted heavily enough to keep S_link where it
is (measured at 0.99994 today).

    minimise  sum|f - b| + gamma * sum|A_c f - c|   subject to  f >= 0

as a linear program. S_link is checkable locally with `src/task4/score_task4.py`; S_od, S_dev and S_attr are only
scored on the leaderboard, so this is verified by submitting it.

    .venv/Scripts/python.exe scripts/t4_l1_solve.py --splits validation private --out-dir work/t4_l1
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import bmat, eye, csr_matrix

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "trafficflowbench-public" / "src"))
sys.path.insert(0, str(ROOT))
from task4.build_task4_odme_artifacts import load_operator, released_counts, released_prior  # noqa: E402
from tfb.data import PANELS, RELEASE  # noqa: E402


def solve_panel(panel: str, split: str, gamma: float) -> pd.DataFrame:
    path_ids, link_ids, A, paths = load_operator(RELEASE / "corridors" / panel / "network")
    cf = released_counts(RELEASE, panel, split)
    cf["link_id"] = cf.link_id.astype(str)
    li = {l: i for i, l in enumerate(map(str, link_ids))}
    keep = [l for l in cf.link_id if l in li]
    Ac = csr_matrix(A[[li[l] for l in keep], :])
    c = cf.set_index("link_id").reindex(keep)["count"].fillna(0.0).to_numpy(float)
    pr = released_prior(RELEASE, panel, split)
    b = np.nan_to_num(pr.set_index("path_id").reindex(path_ids)["path_flow"].to_numpy(float))
    n, m = len(path_ids), len(keep)

    # variables [f (n), u (n), v (n), sp (m), sn (m)]
    #   f - u + v = b            u, v are the positive and negative parts of the deviation
    #   Ac f - sp + sn = c       sp, sn absorb any count misfit
    I = eye(n, format="csr")
    Z = csr_matrix((n, m))
    top = bmat([[I, -I, I, Z, Z]], format="csr")
    bot = bmat([[Ac, csr_matrix((m, n)), csr_matrix((m, n)), -eye(m, format="csr"),
                 eye(m, format="csr")]], format="csr")
    Aeq = bmat([[top], [bot]], format="csr")
    beq = np.concatenate([b, c])
    cost = np.concatenate([np.zeros(n), np.ones(2 * n), gamma * np.ones(2 * m)])
    bounds = [(0, None)] * (3 * n + 2 * m)
    r = linprog(cost, A_eq=Aeq, b_eq=beq, bounds=bounds, method="highs")
    if not r.success:
        raise RuntimeError(f"{panel} {split}: {r.message}")
    f = r.x[:n]
    fit = float(np.abs(Ac @ f - c).sum() / max(c.sum(), 1e-9))
    dev = float(np.abs(f - b).sum())
    base = np.abs(b - b).sum()
    print(f"  {panel:<12} {split:<11} paths {n:<5} counts {m:<4} "
          f"S_link {1 - fit:.6f}  |f-b| {dev:,.0f} ({dev / max(b.sum(), 1e-9):.3f} of sum b)", flush=True)
    _ = base
    out = pr[["panel", "departure_time", "path_id", "origin_zone", "destination_zone"]].copy()
    out["path_id"] = out.path_id.astype(str)
    out = out.set_index("path_id").reindex(path_ids).reset_index()
    out["panel"] = panel
    out["path_flow"] = np.maximum(f, 0.0)
    return out[["panel", "departure_time", "path_id", "origin_zone", "destination_zone", "path_flow"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["validation", "private"])
    ap.add_argument("--gamma", type=float, default=200.0, help="weight on the count misfit")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--panels", nargs="+", default=PANELS)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for split in args.splits:
        frames = [solve_panel(p, split, args.gamma) for p in args.panels]
        d = pd.concat(frames, ignore_index=True)
        f = args.out_dir / f"t4_{split}.csv"
        d.to_csv(f, index=False)
        print(f"wrote {f}: {len(d):,} rows, total flow {d.path_flow.sum():,.0f} "
              f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
