"""Task 1 check: does adding a month-level deviation to the CNN's base profile hurt within one simulation run?

Uses the all-panel M3 checkpoint (trained on Jun-Nov, profile from Jun-Dec). On the Jan-Feb test days,
compare the base = fit-day profile (as trained) with base = fit-day profile + that calendar month's
deviation (visible cells of the month, shrunk by kappa pseudo-days). Train is a single run, so this cannot
show the cross-run gain. It only shows whether the adaptation is harmless before it is used on
validation/private.

    py analysis/t1_month_base_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
import t1_m3_cnn as m3  # noqa: E402
from tfb.data import PANELS, load  # noqa: E402
from tfb.features import profile, truth_valid  # noqa: E402
from tfb.metrics import state_score  # noqa: E402
from tfb.month import month_deviation  # noqa: E402
from tfb.physics import fd_params, physics_scores, ramp_ok  # noqa: E402
from tfb.t1eval import family_mean, split_days  # noqa: E402

KAPPAS = (None, 10.0, 4.0, 2.0)


def main() -> None:
    panels = sys.argv[1:] or PANELS
    net = m3.Net(10, 64).to(m3.DEV)
    net.load_state_dict(torch.load(ROOT / "work" / "models" / "t1_m3_all.pt", map_location=m3.DEV))
    net.eval()
    res = {k: ({}, {}) for k in KAPPAS}
    for panel in panels:
        z = load(panel, "train")
        fit, test = split_days(z)
        valid = truth_valid(z)
        bv0 = profile(z["true_speed"], valid, z["weekday"], fit)
        bq0 = profile(z["true_flow"], valid, z["weekday"], fit)
        months = np.array([d[:7] for d in z["dates"]])
        prm, ok = fd_params(panel, z["link_ids"]), ramp_ok(panel, "train", z)
        days = np.flatnonzero(test)
        for kappa in KAPPAS:
            bv, bq = bv0.copy(), bq0.copy()
            if kappa is not None:
                for mo in np.unique(months[test]):
                    dv, dq = month_deviation(z, bv0, bq0, test & (months == mo), kappa)
                    bv += dv
                    bq += dq
            p = dict(X=m3.make_inputs(z, bv, bq), base_v=bv, base_q=bq, z=z)
            pv, pq = m3.predict_days(net, p, days)
            s = state_score(z, pv, pq, cells=test[:, None, None])["mean"]["S"]
            f = physics_scores(z, pv, pq, prm, ok, days=test)["mean"]["S_phys"]
            res[kappa][0][panel], res[kappa][1][panel] = s, f
            print(f"{panel:<11} kappa={kappa}  S_state {s:.4f}  S_phys~ {f:.4f}", flush=True)
    print("\nfamily mean")
    for kappa, (s, f) in res.items():
        a, b = family_mean(s), family_mean(f)
        print(f"  kappa={str(kappa):<5} S_state {a:.4f}  S_phys~ {b:.4f}  0.35*S_state+0.15*S_phys~ {0.35 * a + 0.15 * b:.4f}")


if __name__ == "__main__":
    main()
