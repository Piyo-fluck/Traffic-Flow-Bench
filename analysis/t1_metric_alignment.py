"""Is the Task 1 training objective aligned with the official score? A 10-minute diagnostic.

The official Task 1 score is a macro average: S_state is the mean over 5 families, each the mean over its 2
panels, each the mean over 3 regimes, and each regime's score is

    S_g = 0.54 max(0, 1 - RMSE_v,g / 25) + 0.46 max(0, 1 - RMSE_q,g / 600)

So the loss to minimise is  sum_g w_g [0.0216 RMSE_v,g + 0.000767 RMSE_q,g]  with **w_g = 1/30 for every
(panel, regime) group**, because every family has exactly two panels and three regimes. Training, by contrast,
minimises a plain mean over cells, which weights a group by its cell count n_g.

Two mismatches are measured here:

  1. group weighting   the official per-cell weight is proportional to w_g / (n_g RMSE_g); training's is 1/N.
                       Groups with few cells, or with a low RMSE, are worth more per cell than training thinks.
  2. channel weighting the score is LINEAR in RMSE, so the value of one unit of squared error is
                       w_g a / (2 RMSE_g). As the model improved on speed (3 km/h when LOSS_W was set, 1.3
                       today) the metric-optimal weight on speed rose. LOSS_W has not moved.

    py analysis/t1_metric_alignment.py [tag]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tfb.data import PANELS, load  # noqa: E402
from tfb.metrics import FLOW_NORM, FLOW_W, REGIMES, SPEED_NORM, SPEED_W  # noqa: E402

PRED = ROOT / "work" / "preds"
TAG = sys.argv[1] if len(sys.argv) > 1 else "t1_m3A"
SCALE_V, SCALE_Q = 10.0, 100.0          # the training units of t1_m3_cnn
LOSS_W = (0.75, 0.25)


def main() -> None:
    rows = []
    for panel in PANELS:
        f = PRED / f"{TAG}_{panel}_test.npz"
        if not f.exists():
            continue
        p = np.load(f)
        z = load(panel, "train")
        days = p["days"]
        lanes = z["lanes"]
        ts, tq = z["true_speed"][days], z["true_flow"][days]
        tgt = z["target"][days] & np.isfinite(ts) & np.isfinite(tq)
        reg = np.asarray(z["regime"], str)[days]
        dv = p["pv"] - ts
        dq = (p["pq"] - tq) / lanes
        for r in REGIMES:
            m = tgt & (reg == r)[:, None, None]
            if not m.any():
                continue
            rows.append(dict(panel=panel, family=panel.rsplit("_", 1)[0], regime=r, n=int(m.sum()),
                             rv=float(np.sqrt((dv[m] ** 2).mean())), rq=float(np.sqrt((dq[m] ** 2).mean()))))
        print(f"  {panel}", flush=True)

    fams = sorted({r["family"] for r in rows})
    for r in rows:
        n_pan = len({x["panel"] for x in rows if x["family"] == r["family"]})
        n_reg = len([x for x in rows if x["panel"] == r["panel"]])
        r["w"] = 1.0 / (len(fams) * n_pan * n_reg)
    a, b = SPEED_W / SPEED_NORM, FLOW_W / FLOW_NORM
    N = sum(r["n"] for r in rows)
    L = sum(r["w"] * (a * r["rv"] + b * r["rq"]) for r in rows)
    print(f"\n{len(rows)} (panel, regime) groups, {N:,} target cells; official loss {L:.5f} "
          f"-> S_state {1 - L:.4f}")
    print(f"speed carries {sum(r['w'] * a * r['rv'] for r in rows) / L:.1%} of the official loss, "
          f"flow {sum(r['w'] * b * r['rq'] for r in rows) / L:.1%}")

    # ---- 1. group weighting
    print(f"\n{'panel':<12} {'reg':<4} {'cells':>9} {'RMSE v':>7} {'RMSE q':>8} {'score share':>12} "
          f"{'MSE share':>10} {'per-cell w':>11}")
    tot_mse = sum(r["n"] * (LOSS_W[0] * (r["rv"] / SCALE_V) ** 2 + LOSS_W[1] * (r["rq"] / SCALE_Q) ** 2)
                  for r in rows)
    mult = []
    for r in rows:
        sc = r["w"] * (a * r["rv"] + b * r["rq"]) / L
        ms = r["n"] * (LOSS_W[0] * (r["rv"] / SCALE_V) ** 2 + LOSS_W[1] * (r["rq"] / SCALE_Q) ** 2) / tot_mse
        r["mult"] = (r["w"] / (r["n"] * max(r["rv"], 1e-9))) * N
        mult.append(r["mult"])
    mean_mult = float(np.mean(mult))
    for r in sorted(rows, key=lambda x: -x["mult"] / mean_mult)[:6] + \
            sorted(rows, key=lambda x: x["mult"] / mean_mult)[:4]:
        sc = r["w"] * (a * r["rv"] + b * r["rq"]) / L
        ms = r["n"] * (LOSS_W[0] * (r["rv"] / SCALE_V) ** 2 + LOSS_W[1] * (r["rq"] / SCALE_Q) ** 2) / tot_mse
        print(f"{r['panel']:<12} {r['regime']:<4} {r['n']:9,} {r['rv']:7.3f} {r['rq']:8.2f} {sc:12.2%} "
              f"{ms:10.2%} {r['mult'] / mean_mult:11.2f}")
    print("   (top six and bottom four by the per-cell weight the official score implies, 1.00 = average)")

    # ---- 2. channel weighting
    rv = float(np.sqrt(sum(r["n"] * r["rv"] ** 2 for r in rows) / N))
    rq = float(np.sqrt(sum(r["n"] * r["rq"] ** 2 for r in rows) / N))
    gv = sum(r["w"] * a / (2 * max(r["rv"], 1e-9)) for r in rows) * SCALE_V ** 2
    gq = sum(r["w"] * b / (2 * max(r["rq"], 1e-9)) for r in rows) * SCALE_Q ** 2
    print(f"\npooled RMSE: speed {rv:.3f} km/h, flow {rq:.2f} veh/h/lane")
    print(f"value of one unit of TRAINING-scaled squared error: speed {gv:.4f}, flow {gq:.4f}")
    print(f"  -> metric-optimal loss weights  speed {gv / (gv + gq):.3f} / flow {gq / (gv + gq):.3f}")
    print(f"  -> t1_m3_cnn currently uses     speed {LOSS_W[0]:.3f} / flow {LOSS_W[1]:.3f}")
    print(f"  (LOSS_W was set when speed RMSE was about 3 km/h; at 3.0 the optimal speed weight would be "
          f"{gv * rv / 3.0 / (gv * rv / 3.0 + gq):.3f})")


if __name__ == "__main__":
    main()
