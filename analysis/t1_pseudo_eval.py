"""Task 1 pseudo-evaluation INSIDE the validation/private months (the other simulation run, note 17).

Hide a random 20% of the visible eligible cells of each day (seeded), reconstruct them with a saved CNN, and
score the reconstruction against the hidden observations (per-lane flow, as in S_state). Because the months
come from a different run than train, this measures whether a month-specific base profile (train table +
the month's own deviation, shrunk by kappa pseudo-days) helps where it matters. The deviation is computed
after hiding, so the pseudo targets never inform the base.

    py analysis/t1_pseudo_eval.py validation
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
from tfb.features import profile_table, truth_valid  # noqa: E402
from tfb.metrics import FLOW_NORM, FLOW_W, SPEED_NORM, SPEED_W  # noqa: E402
from tfb.month import month_deviation  # noqa: E402
from tfb.t1eval import family_mean  # noqa: E402

KAPPAS = (None, 10.0, 4.0, 2.0, 1.0)  # same kappa for speed and flow
if len(sys.argv) > 2 and sys.argv[2] == "pairs":  # (kappa_speed, kappa_flow)
    KAPPAS = ((None, None), (2.0, None), (2.0, 10.0), (2.0, 20.0), (4.0, 10.0), (1.0, 10.0), (2.0, 4.0))
HIDE = 0.2


def main() -> None:
    split = sys.argv[1] if len(sys.argv) > 1 else "validation"
    net = m3.Net(10, 64).to(m3.DEV)
    net.load_state_dict(torch.load(ROOT / "work" / "sub_m3v1" / "t1_m3_seed0.pt", map_location=m3.DEV))
    net.eval()
    rng = np.random.default_rng(0)
    res = {k: {} for k in KAPPAS}
    for panel in PANELS:
        zt = load(panel, "train")
        alld = np.ones(len(zt["dates"]), bool)
        valid = truth_valid(zt)
        tv = profile_table(zt["true_speed"], valid, zt["weekday"], alld)
        tq = profile_table(zt["true_flow"], valid, zt["weekday"], alld)
        del zt
        z = load(panel, split)
        vis = z["elig"] & np.isfinite(z["obs_speed"]) & np.isfinite(z["obs_flow"])
        hide = vis & (rng.random(vis.shape) < HIDE)
        truth_v, truth_q = z["obs_speed"].copy(), z["obs_flow"].copy()
        zm = dict(z)
        for k in ("obs_speed", "obs_flow", "obs_occ"):
            zm[k] = np.where(hide, np.nan, z[k]).astype(np.float32)
        bv0, bq0 = tv[z["weekday"]], tq[z["weekday"]]
        lanes = z["lanes"]
        for kappa in KAPPAS:
            bv, bq = bv0, bq0
            kv, kq = kappa if isinstance(kappa, tuple) else (kappa, kappa)
            if kv is not None or kq is not None:
                dv, dq = month_deviation(zm, bv0, bq0, np.ones(len(z["dates"]), bool), kv, kappa_q=kq)
                bv, bq = bv0 + dv, bq0 + dq
            p = dict(X=m3.make_inputs(zm, bv, bq), base_v=bv, base_q=bq, z=zm)
            pv, pq = m3.predict_days(net, p, range(len(z["dates"])))
            ev = (pv - truth_v)[hide]
            eq = ((pq - truth_q) / lanes)[hide]
            rv, rq = float(np.sqrt(np.mean(ev ** 2))), float(np.sqrt(np.mean(eq ** 2)))
            s = SPEED_W * max(0, 1 - rv / SPEED_NORM) + FLOW_W * max(0, 1 - rq / FLOW_NORM)
            res[kappa][panel] = s
            print(f"{split} {panel:<11} kappa={str(kappa):<5} rmse_v {rv:.3f} rmse_q/lane {rq:.2f}  S {s:.4f}", flush=True)
    print(f"\n== Task 1 pseudo-evaluation inside {split} (family mean, 20% hidden visible cells) ==")
    for kappa, r in res.items():
        print(f"  kappa={str(kappa):<5} S {family_mean(r):.4f}")


if __name__ == "__main__":
    main()
