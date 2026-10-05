"""Production Task 1 test-time training: fine-tune the 3 production CNNs on one scored month (note 32).

Same recipe as analysis/t1_ttt.py (validated there: +0.0019 S_state~, +0.0033 S_LWR~ on validation): 400 steps,
lr 2e-4, 10-50% of the month's visible eligible cells hidden per sample, fixed seeds. Two differences: no cells
are held out for evaluation (every visible cell is used), and the inputs are built exactly like
scripts/t1_predict_ckpt.py with --month-kappa 2 --month-kappa-q 10 (train table + month deviation on the full
masked layer). T1 is offline, so the whole scored month may be used.

Writes <out-dir>/ttt_<split>_<i>.pt for the checkpoints in CKPTS order; use them with t1_predict_ckpt.py --splits <split>.

    py scripts/t1_ttt_train.py --split validation --out-dir work/sub_m3ttt
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
import t1_m3_cnn as m3  # noqa: E402
from t1_ttt import CKPTS, TTT_LR, TTT_STEPS, ttt_hide  # noqa: E402
from tfb.data import CACHE, PANELS, load  # noqa: E402
from tfb.month import month_deviation  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["validation", "private"], required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    parts = []
    for panel in PANELS:
        with np.load(CACHE / f"{panel}_t1tables.npz") as f:
            tv, tq = f["tv"], f["tq"]
        z = load(panel, args.split)
        bv, bq = tv[z["weekday"]], tq[z["weekday"]]
        dv, dq = month_deviation(z, bv, bq, np.ones(len(z["dates"]), bool), 2.0, kappa_q=10.0)
        bv, bq = bv + dv, bq + dq
        vis = z["elig"] & np.isfinite(z["obs_speed"]) & np.isfinite(z["obs_flow"])
        Y = np.stack([np.where(vis, (z["obs_speed"] - bv) / m3.SCALE_V, 0),
                      np.where(vis, (z["obs_flow"] - bq) / z["lanes"] / m3.SCALE_Q, 0)], axis=1).astype(np.float16)
        parts.append(dict(panel=panel, X=m3.make_inputs(z, bv, bq), Y=Y, M=np.zeros(vis.shape, bool), E=vis,
                          tr_days=np.arange(len(z["dates"]))))
        print(f"[{args.split} {panel}] ready ({time.time() - t_start:.0f}s)", flush=True)
    gen = torch.Generator(device=m3.DEV).manual_seed(0)
    brng = np.random.default_rng(1)
    for i, (path, width) in enumerate(CKPTS):
        net = m3.Net(10, width).to(m3.DEV)
        net.load_state_dict(torch.load(ROOT / path, map_location=m3.DEV))
        net.train()
        opt = torch.optim.AdamW(net.parameters(), lr=TTT_LR, weight_decay=1e-4)
        tot = 0.0
        for _ in range(TTT_STEPS):
            x, y, _, e, _w = m3.sample_batch(parts, brng)   # m3.sample_batch gained a weight slot (note 101)
            x, m = ttt_hide(x, e, gen)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = net(x)
            loss = m3.loss_fn(out.float(), y, m)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += loss.item()
        dst = args.out_dir / f"ttt_{args.split}_{i}.pt"
        torch.save(net.state_dict(), dst)
        print(f"[{args.split}] {path} -> {dst.name}: {TTT_STEPS} steps, mean loss {tot / TTT_STEPS:.4f} "
              f"({time.time() - t_start:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
