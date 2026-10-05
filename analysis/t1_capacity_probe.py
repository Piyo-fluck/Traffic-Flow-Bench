"""Can the architecture fit the Task 1 mapping at all, or is the residual the Bayes error of the inputs?

`t1_train_test_gap.py` found train and test almost equal (sd(dk/k) 0.0298 against 0.0319 over 183 training days).
That is consistent with two very different worlds and cannot tell them apart:

    underfitting          capacity / optimisation / data are the binding constraint -> generating better
                          training data is worth doing
    at the Bayes error    the inputs do not determine the target -> no data-side intervention can help

A capacity probe separates them. Train the same network on a handful of days only, with no extra hiding and no
early stopping, and watch the error ON THOSE DAYS. A network that can represent the mapping will drive it far
below 0.030; one that plateaus there cannot represent it whatever data it is given.

    .venv/Scripts/python.exe analysis/t1_capacity_probe.py [n_days] [width] [epochs]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT))
import t1_m3_cnn as M  # noqa: E402

N_DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
WIDTH = int(sys.argv[2]) if len(sys.argv) > 2 else 64
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 300
PANEL = "D12_I5_N"
STEPS = 30


def density_sd(p, days, pv, pq):
    z = p["z"]
    ts, tq = z["true_speed"][days].astype(np.float64), z["true_flow"][days].astype(np.float64)
    m = z["target"][days] & np.isfinite(ts) & np.isfinite(tq)
    v, q = pv[days].astype(np.float64)[m], pq[days].astype(np.float64)[m]
    k_o, k_t = q / np.maximum(v, 1.0), tq[m] / np.maximum(ts[m], 1.0)
    rk = (k_o - k_t) / np.maximum(k_t, 1e-9)
    rk = rk[np.isfinite(rk) & (np.abs(rk) < 2)]
    return (float(np.sqrt(np.mean((v - ts[m]) ** 2))), float(np.sqrt(np.mean((q - tq[m]) ** 2))),
            float(rk.std()))


def main() -> None:
    print(f"panel {PANEL}, {N_DAYS} days, width {WIDTH}, {EPOCHS} epochs, no extra hiding, no early stopping",
          flush=True)
    p = M.panel_tensors(PANEL)
    days = p["tr_days"][:N_DAYS]
    p = dict(p)
    p["tr_days"] = days
    net = M.Net(p["X"].shape[1], WIDTH).to(M.DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=M.LR, weight_decay=0.0)
    rng = np.random.default_rng(0)
    t0 = time.time()
    print(f"{'epoch':>6} {'loss':>10} {'RMSE v':>9} {'RMSE q':>9} {'sd dk/k':>9}")
    for ep in range(EPOCHS):
        net.train()
        tot = 0.0
        for _ in range(STEPS):
            x, y, m, e, w = M.sample_batch([p], rng, None)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = net(x)
            loss = M.loss_fn(out.float(), y, m, x, w)          # the real cells only, no hide_extra
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += loss.item()
        if (ep + 1) % 20 == 0 or ep == 0:
            net.eval()
            pv, pq = M.predict_days(net, p, days)
            rv, rq, sk = density_sd(p, days, pv, pq)
            print(f"{ep + 1:>6} {tot / STEPS:>10.5f} {rv:>9.4f} {rq:>9.3f} {sk:>9.5f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    print("\nreference, production on its own training days: RMSE v 1.09, RMSE q 120.4, sd dk/k 0.0298")


if __name__ == "__main__":
    main()
