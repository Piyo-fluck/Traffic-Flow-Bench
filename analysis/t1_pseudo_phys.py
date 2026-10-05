"""Task 1/3 pseudo-evaluation inside a scored month, with a conservation (S_LWR-like) score.

Hide a random 20% of the visible eligible cells (seeded), reconstruct them with CNN checkpoints, and score:
  S_state-like : on the hidden cells (per-lane flow), as in analysis/t1_pseudo_eval.py
  S_LWR-like   : over 5-min transitions (t -> t+1, same link) between cells whose observation is known (visible
                 eligible) and that pass the Mode A ramp rule. The state is the observation at unhidden cells
                 and the prediction at hidden ones, and rhs := the observation's own dN. Transitions touching a
                 real Task 1 target are excluded (their truth is not released).
  E = sum|dN_state - dN_obs| / sum|dN_obs|,  S_LWR~ = 1 - E
This is the in-run counterpart of tfb/physics.py, for judging Task 3 methods under the run shift.
Variants are given as name=ckpt:width[,ckpt:width...][@kv,kq] (month-base kappas optional).

    py analysis/t1_pseudo_phys.py validation base=work/sub_m3v1/t1_m3_seed0.pt:64 month=work/sub_m3v1/t1_m3_seed0.pt:64@2,10
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
import t1_m3_cnn as m3  # noqa: E402
from tfb.data import CACHE, PANELS, load  # noqa: E402
from tfb.metrics import FLOW_NORM, FLOW_W, SPEED_NORM, SPEED_W  # noqa: E402
from tfb.month import month_deviation  # noqa: E402
from tfb.physics import fd_params, ramp_ok  # noqa: E402
from tfb.t1eval import family_mean  # noqa: E402

HIDE = 0.2


def parse(spec: str):
    name, rest = spec.split("=", 1)
    kap = None
    if "@" in rest:
        rest, k = rest.split("@")
        kv, kq = k.split(",")
        kap = (float(kv) if kv != "none" else None, float(kq) if kq != "none" else None)
    nets = []
    for c in rest.split(","):
        path, width = c.rsplit(":", 1)
        net = m3.Net(10, int(width)).to(m3.DEV)
        net.load_state_dict(torch.load(path, map_location=m3.DEV))
        net.eval()
        nets.append(net)
    return name, nets, kap


def main() -> None:
    split, *specs = sys.argv[1:]
    variants = [parse(s) for s in specs]
    rng = np.random.default_rng(0)
    res = {v[0]: ({}, {}) for v in variants}
    for panel in PANELS:
        with np.load(CACHE / f"{panel}_t1tables.npz") as f:
            tv, tq = f["tv"], f["tq"]
        z = load(panel, split)
        prm = fd_params(panel, z["link_ids"])
        ok = ramp_ok(panel, split, z)
        vis = z["elig"] & np.isfinite(z["obs_speed"]) & np.isfinite(z["obs_flow"])
        hide = vis & (rng.random(vis.shape) < HIDE)
        ov, oq = z["obs_speed"], z["obs_flow"]
        zm = dict(z)
        for k in ("obs_speed", "obs_flow", "obs_occ"):
            zm[k] = np.where(hide, np.nan, z[k]).astype(np.float32)
        bv0, bq0 = tv[z["weekday"]], tq[z["weekday"]]
        lanes, L = z["lanes"], prm["length"]
        k_obs = oq / np.maximum(ov, 1.0)
        trans = vis[:, :-1] & vis[:, 1:] & ok[:, :-1] & (hide[:, :-1] | hide[:, 1:])
        known = vis[:, :-1] & vis[:, 1:] & ok[:, :-1]
        dn_obs = (k_obs[:, 1:] - k_obs[:, :-1]) * L
        for name, nets, kap in variants:
            bv, bq = bv0, bq0
            if kap is not None:
                dv, dq = month_deviation(zm, bv0, bq0, np.ones(len(z["dates"]), bool), kap[0], kappa_q=kap[1])
                bv, bq = bv0 + dv, bq0 + dq
            p = dict(X=m3.make_inputs(zm, bv, bq), base_v=bv, base_q=bq, z=zm)
            preds = [m3.predict_days(net, p, range(len(z["dates"]))) for net in nets]
            pv = np.mean([x[0] for x in preds], 0)
            pq = np.mean([x[1] for x in preds], 0)
            ev, eq = (pv - ov)[hide], ((pq - oq) / lanes)[hide]
            rv, rq = float(np.sqrt(np.mean(ev ** 2))), float(np.sqrt(np.mean(eq ** 2)))
            s_state = SPEED_W * max(0, 1 - rv / SPEED_NORM) + FLOW_W * max(0, 1 - rq / FLOW_NORM)
            k_st = np.where(hide, pq / np.maximum(pv, 1.0), k_obs)
            dn_st = (k_st[:, 1:] - k_st[:, :-1]) * L
            e = np.abs(dn_st - dn_obs)[trans].sum() / (np.abs(dn_obs)[known].sum() + 1e-9)
            res[name][0][panel], res[name][1][panel] = s_state, 1 - e
            print(f"{split} {panel:<11} {name:<10} S_state~ {s_state:.4f}  S_LWR~ {1 - e:.4f}  (rmse_v {rv:.3f}, rmse_q {rq:.2f})", flush=True)
    print(f"\n== {split}: in-month pseudo T1/T3 (20% hidden), family mean ==")
    for name, (s, l) in res.items():
        a, b = family_mean(s), family_mean(l)
        print(f"  {name:<10} S_state~ {a:.4f}  S_LWR~ {b:.4f}  0.35*S_state~ + 0.10*S_LWR~ = {0.35 * a + 0.10 * b:.4f}")


if __name__ == "__main__":
    main()
