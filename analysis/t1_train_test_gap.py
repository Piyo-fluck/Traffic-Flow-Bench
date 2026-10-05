"""Is the Task 1 residual irreducible given the inputs, or is the model simply not fitting?

Every note from 96 to 113 reports test-side numbers, and all of them conclude the residual cannot be reduced.
None of them reports how well the model fits the days it was TRAINED on. The distinction decides whether any
data-side intervention - augmentation, a privileged teacher, a learned generator - can help at all:

    train error ~= test error   the model is underfitting; capacity, optimisation or data have room, and
                                generating better training data is a sensible thing to try
    train error << test error   the model is memorising; the inputs do not carry the information, and no
                                generator can put it there

Same checkpoint, same cells, `tr_days` against `te_days`.

    .venv/Scripts/python.exe analysis/t1_train_test_gap.py [checkpoint]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT))
import t1_m3_cnn as M  # noqa: E402
from tfb.metrics import state_score  # noqa: E402

CKPT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "work" / "models" / "t1_lenw0_all.pt"


def stats(p, days, pv, pq):
    z = p["z"]
    ts, tq = z["true_speed"][days].astype(np.float64), z["true_flow"][days].astype(np.float64)
    m = z["target"][days] & np.isfinite(ts) & np.isfinite(tq)
    v, q = pv[days].astype(np.float64)[m], pq[days].astype(np.float64)[m]
    tv, tqq = ts[m], tq[m]
    k_o, k_t = q / np.maximum(v, 1.0), tqq / np.maximum(tv, 1.0)
    rk = (k_o - k_t) / np.maximum(k_t, 1e-9)
    rk = rk[np.isfinite(rk) & (np.abs(rk) < 2)]
    cells = np.zeros(len(z["dates"]), bool)
    cells[days] = True
    r = state_score(z, pv, pq, cells=cells[:, None, None])["mean"]
    return r["S"], float(np.sqrt(np.mean((v - tv) ** 2))), float(np.sqrt(np.mean((q - tqq) ** 2))), float(rk.std())


def main() -> None:
    print(f"checkpoint {CKPT.name}", flush=True)
    parts = [M.panel_tensors(p) for p in M.PANELS]
    net = M.Net(parts[0]["X"].shape[1], M.WIDTH).to(M.DEV)
    net.load_state_dict(torch.load(CKPT, map_location=M.DEV))
    net.eval()
    acc = {}
    for p in parts:
        allday = np.arange(len(p["z"]["dates"]))
        pv = np.zeros(p["z"]["true_speed"].shape, np.float32)
        pq = np.zeros_like(pv)
        for key in ("tr_days", "te_days"):
            a, b = M.predict_days(net, p, p[key])
            pv[p[key]], pq[p[key]] = a[p[key]], b[p[key]]
        for key in ("tr_days", "te_days"):
            acc.setdefault(key, []).append(stats(p, p[key], pv, pq))
        print(f"  {p['panel']}  train {len(p['tr_days'])}d  test {len(p['te_days'])}d", flush=True)
        _ = allday

    print(f"\n{'days':<14} {'S_state':>9} {'RMSE v':>9} {'RMSE q':>9} {'sd dk/k':>9}")
    out = {}
    for key, label in (("tr_days", "TRAINED on"), ("te_days", "held out")):
        v = acc[key]
        row = tuple(float(np.mean([x[i] for x in v])) for i in range(4))
        out[key] = row
        print(f"{label:<14} {row[0]:>9.5f} {row[1]:>9.4f} {row[2]:>9.3f} {row[3]:>9.5f}")
    a, b = out["tr_days"], out["te_days"]
    print(f"\n  gap: S_state {a[0] - b[0]:+.5f}, RMSE v {a[1] - b[1]:+.4f}, RMSE q {a[2] - b[2]:+.3f}, "
          f"sd dk/k {a[3] - b[3]:+.5f}")
    print("  train ~= test -> underfitting, data-side work has room."
          "\n  train << test -> memorising, the inputs do not carry it.")


if __name__ == "__main__":
    main()
