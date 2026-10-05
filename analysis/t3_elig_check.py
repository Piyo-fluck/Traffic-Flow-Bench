"""Task 3: does the real S_LWR (~0.51) differ from the proxy (0.65) because the official state includes cells
the proxy skips?

The local proxy (tfb/physics.py) scores only score-eligible cells with a finite truth. The official evaluator
builds the state from every organizer truth row plus the Task 1 targets, whether or not a cell is eligible.
The CNN loss never covers ineligible cells, so its predictions there may be poor. On the train holdout (saved
`m3all` predictions), this script reports per panel:
    share of target cells that are ineligible (with a finite truth)
    S_LWR~ with the proxy rule (eligible only) vs with every finite-truth cell
    density MAE (per lane) at eligible vs ineligible target cells

    py analysis/t3_elig_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tfb.data import PANELS, REGIMES, load  # noqa: E402
from tfb.features import profile, truth_valid  # noqa: E402
from tfb.physics import fd_params, ramp_ok  # noqa: E402
from tfb.t1eval import family_mean, split_days  # noqa: E402

TAG = "m3all"


def s_lwr(z, pv, pq, L, ok, use, days):
    tgt = z["target"]
    ts, tq = z["true_speed"], z["true_flow"]
    k_st = np.where(tgt, pq, tq) / np.maximum(np.where(tgt, pv, ts), 1.0)
    k_tr = tq / np.maximum(ts, 1.0)
    trans = use[:, :-1] & use[:, 1:] & ok[:, :-1]
    res = np.where(trans, np.abs((k_st[:, 1:] - k_st[:, :-1]) - (k_tr[:, 1:] - k_tr[:, :-1])) * L, 0.0)
    rhs = np.where(trans, np.abs(k_tr[:, 1:] - k_tr[:, :-1]) * L, 0.0)
    vals = []
    for r in REGIMES:
        dm = (z["regime"] == r) & days
        if dm.any():
            vals.append(max(0.0, 1 - min(1.0, res[dm].sum() / (rhs[dm].sum() + 1e-9))))
    return float(np.mean(vals))


def main() -> None:
    rows = {}
    for panel in PANELS:
        f = ROOT / "work" / "preds" / f"t1_{TAG}_{panel}_test.npz"
        if not f.exists():
            continue
        z = load(panel, "train")
        fit, _ = split_days(z)
        p = np.load(f)
        valid = truth_valid(z)
        pv = profile(z["true_speed"], valid, z["weekday"], fit)
        pq = profile(z["true_flow"], valid, z["weekday"], fit)
        pv[p["days"]], pq[p["days"]] = p["pv"], p["pq"]
        days = np.zeros(len(z["dates"]), bool)
        days[p["days"]] = True
        L = fd_params(panel, z["link_ids"])["length"]
        ok = ramp_ok(panel, "train", z)
        ts, tq, tgt, el = z["true_speed"], z["true_flow"], z["target"], z["elig"]
        fin = np.isfinite(ts) & np.isfinite(tq)
        dsel = days[:, None, None]
        t_el, t_in = tgt & el & fin & dsel, tgt & ~el & fin & dsel
        k_err = np.abs(pq / np.maximum(pv, 1.0) - tq / np.maximum(ts, 1.0)) / z["lanes"]
        rows[panel] = dict(
            share_inelig=t_in.sum() / max((t_el | t_in).sum(), 1),
            lwr_elig=s_lwr(z, pv, pq, L, ok, el & fin, days),
            lwr_all=s_lwr(z, pv, pq, L, ok, fin, days),
            kmae_el=float(k_err[t_el].mean()) if t_el.any() else np.nan,
            kmae_in=float(k_err[t_in].mean()) if t_in.any() else np.nan,
        )
        r = rows[panel]
        print(f"{panel:<11} ineligible share of targets {r['share_inelig']:.3f} | S_LWR~ eligible-only {r['lwr_elig']:.4f} "
              f"vs all finite {r['lwr_all']:.4f} | density MAE per lane: eligible {r['kmae_el']:.2f}, ineligible {r['kmae_in']:.2f}",
              flush=True)
    print("\n== family mean ==")
    for k in ("share_inelig", "lwr_elig", "lwr_all", "kmae_el", "kmae_in"):
        print(f"  {k:<13} {family_mean({p: r[k] for p, r in rows.items() if np.isfinite(r[k])}):.4f}")


if __name__ == "__main__":
    main()
