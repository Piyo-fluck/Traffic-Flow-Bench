"""The degree of freedom S_state cannot see: the correlation between the speed and flow errors.

S_state is two separate RMSEs, one on v and one on q, so it depends only on their MARGINAL accuracy. S_LWR is
scored on k = q/v, and

    dk/k ~= dq/q - dv/v,   Var(dk/k) = s_q^2 + s_v^2 - 2 rho s_q s_v

so the correlation rho between the two errors is decisive for S_LWR and completely invisible to S_state. Unlike
the length weighting of note 112, exploiting it asks the network for no extra accuracy at all - only that the two
errors it is already making point the same way.

This measures rho as it stands, and what Var(dk/k) would be at higher rho with the marginals held fixed.

    .venv/Scripts/python.exe analysis/t3_qv_error_correlation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tfb.data import PANELS, load  # noqa: E402

PRED = ROOT / "work" / "preds"
TAGS = tuple(sys.argv[1:]) or ("t1_m3all", "t1_m3A")   # prediction tags to compare


def main() -> None:
    for tag in TAGS:
        rows = []
        for panel in PANELS:
            f = PRED / f"{tag}_{panel}_test.npz"
            if not f.exists():
                continue
            p = np.load(f)
            z = load(panel, "train")
            days = p["days"]
            ts, tq = z["true_speed"][days].astype(np.float64), z["true_flow"][days].astype(np.float64)
            m = z["target"][days] & np.isfinite(ts) & np.isfinite(tq)
            v, q = p["pv"].astype(np.float64)[m], p["pq"].astype(np.float64)[m]
            tv, tqq = ts[m], tq[m]
            rv = (v - tv) / np.maximum(tv, 1.0)                 # relative errors, which is what dk/k sees
            rq = (q - tqq) / np.maximum(tqq, 1.0)
            k_ours, k_true = q / np.maximum(v, 1.0), tqq / np.maximum(tv, 1.0)
            rk = (k_ours - k_true) / np.maximum(k_true, 1e-9)
            keep = np.isfinite(rv) & np.isfinite(rq) & np.isfinite(rk) & (np.abs(rk) < 2)
            rv, rq, rk = rv[keep], rq[keep], rk[keep]
            rho = float(np.corrcoef(rv, rq)[0, 1])
            rows.append((panel, rho, float(rv.std()), float(rq.std()), float(rk.std()), len(rv)))
            del z
        if not rows:
            print(f"{tag}: no predictions")
            continue
        print(f"\n=== {tag} ===")
        print(f"{'panel':<12} {'rho(dv,dq)':>11} {'sd dv/v':>9} {'sd dq/q':>9} {'sd dk/k':>9} "
              f"{'implied sd':>11}")
        for panel, rho, sv, sq, sk, n in rows:
            imp = np.sqrt(max(sv ** 2 + sq ** 2 - 2 * rho * sv * sq, 0))
            print(f"{panel:<12} {rho:>+11.3f} {sv:>9.4f} {sq:>9.4f} {sk:>9.4f} {imp:>11.4f}")
        rho = float(np.mean([r[1] for r in rows]))
        sv = float(np.mean([r[2] for r in rows]))
        sq = float(np.mean([r[3] for r in rows]))
        sk = float(np.mean([r[4] for r in rows]))
        print(f"{'mean':<12} {rho:>+11.3f} {sv:>9.4f} {sq:>9.4f} {sk:>9.4f}")
        print(f"\n  what sd(dk/k) would be at a higher correlation, marginals unchanged:")
        base = np.sqrt(sv ** 2 + sq ** 2 - 2 * rho * sv * sq)
        for r in (rho, 0.2, 0.4, 0.6, 0.8):
            s = np.sqrt(max(sv ** 2 + sq ** 2 - 2 * r * sv * sq, 0))
            print(f"    rho {r:+.2f} -> sd {s:.4f}  ({(s / base - 1) * 100:+.1f}% on the density error)")


if __name__ == "__main__":
    main()
