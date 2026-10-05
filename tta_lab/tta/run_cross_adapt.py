"""Idea 9b with a CROSS-PANEL adaptation pool.

The organizers clarified (2026-09-20) that a window with origin T may use any released data timestamped at or
before T, in any file - so the other corridors' released observations of earlier dates are allowed too. Idea 9b
adapts on 16 history windows drawn from the panel's own earlier days; that is a few tens of days of one corridor.
Here the same 30 steps per day draw from EVERY panel's earlier dates (the Task 2 CNN is convolutional over links
and is trained on all panels, so a window of another corridor is a valid input).

Each adaptation step picks a panel - the window's own with probability --own-share, else a random other one - and
draws a batch of history windows from that panel's days strictly before the current date. Prediction, the order
(predict-then-adapt), the losses, the steps and the learning rate are those of 9b.

    python tta/run_cross_adapt.py --name cross50 --own-share 0.5
    python tta/run_cross_adapt.py --name cross_all --own-share 0.125     # panels drawn uniformly
Outputs runs/<name>/probs_<split>_<panel>.npz, readable by common.report like any run_adapt output.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

import common as C


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--ckpt", default=str(C.MODELS / "t2_cnn_lov_rec_s20.pt"))
    ap.add_argument("--own-share", type=float, default=0.5, help="probability that a step uses the own panel")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--splits", nargs="*", default=list(C.SPLITS))
    args = ap.parse_args()
    (C.RUNS / args.name).mkdir(parents=True, exist_ok=True)
    (C.RUNS / args.name / "args.json").write_text(json.dumps(vars(args), indent=1))
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    t_all = time.time()

    # every panel's context, static channels and both scored months, so any of them can feed the adaptation
    P = []
    for i, panel in enumerate(C.T2_PANELS):
        ctx, stat, W = C.windows(i, panel)
        P.append(dict(panel=panel, ctx=ctx, stat=stat, W=W))
        print(f"loaded {panel} ({time.time() - t_all:.0f}s)", flush=True)

    for pi, p in enumerate(P):
        ctx, stat = p["ctx"], p["stat"]
        for split in args.splits:
            w = p["W"][split]
            z = w["z"]
            net, params = C.build_net(args.ckpt, "full")
            res = {k: np.zeros((len(w[k]["d"]), C.F, ctx.S), np.float32) for k in ("on", "og")}
            days = sorted(set(w["on"]["d"].tolist()) | set(w["og"]["d"].tolist()))
            for day in days:
                date = str(z["dates"][day])
                for k in ("on", "og"):
                    m = np.flatnonzero(w[k]["d"] == day)
                    if len(m):
                        res[k][m] = C.cnn.predict(net, C.make_x(ctx, stat, w[k], m, z))

                def batch(date=date, pi=pi):
                    q = P[pi] if rng.random() < args.own_share else P[int(rng.integers(len(P)))]
                    zz = q["W"][split]["z"]
                    pool = np.flatnonzero(np.array([str(x) for x in zz["dates"]]) < date)
                    if not len(pool):
                        return None
                    return C.past_batch(q["ctx"], q["stat"], zz, pool, args.bs, rng)

                if batch() is not None:
                    C.adapt(net, params, batch, args.steps, args.lr)
            C.save_run(args.name, split, p["panel"], on_p=res["on"][:, C.F - 1], og_p=res["og"],
                       on_d=w["on"]["d"], og_d=w["og"]["d"],
                       on_ssl=np.zeros(len(w["on"]["d"]), np.float32), og_ssl=np.zeros(len(w["og"]["d"]), np.float32))
            print(f"[{p['panel']} {split}] {len(days)} days ({time.time() - t_all:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
