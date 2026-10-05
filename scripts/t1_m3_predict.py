"""Task 1 production run for M3 (CNN inpainting): train on every train day, predict validation and private.

Same network, inputs, loss and augmentation as analysis/t1_m3_cnn.py, including
its M3_V2 / M3_DNLOSS options (read from the environment). The
profile is the weekday x tod table over all of train. No days are held out,
so training runs for a fixed number of epochs (chosen on the holdout) and
keeps the final weights.

    py scripts/t1_m3_predict.py --epochs 40 --out-dir work/sub_m3
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
import t1_m3_cnn as m3  # noqa: E402
from tfb.data import PANELS, load  # noqa: E402
from tfb.features import profile_table, truth_valid  # noqa: E402
from tfb.submit import append_csv, check_against_template, state_frame  # noqa: E402


def train_part(panel: str) -> tuple[dict, tuple[np.ndarray, np.ndarray]]:
    z = load(panel, "train")
    alldays = np.ones(len(z["dates"]), bool)
    valid = truth_valid(z)
    tv = profile_table(z["true_speed"], valid, z["weekday"], alldays)
    tq = profile_table(z["true_flow"], valid, z["weekday"], alldays)
    base_v, base_q = tv[z["weekday"]], tq[z["weekday"]]
    ts, tf, lanes = z["true_speed"], z["true_flow"], z["lanes"]
    part = dict(
        panel=panel,
        X=m3.make_inputs(z, base_v, base_q, m3.v2_channels(panel, "train", z) if m3.V2 else None),
        Y=np.stack([np.nan_to_num((ts - base_v) / m3.SCALE_V),
                    np.nan_to_num((tf - base_q) / lanes / m3.SCALE_Q)], axis=1).astype(np.float16),
        M=z["target"] & np.isfinite(ts) & np.isfinite(tf),
        E=z["elig"],
        tr_days=np.arange(len(z["dates"])),
    )
    return part, (tv, tq)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", nargs="+", default=PANELS)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    gen = torch.Generator(device=m3.DEV).manual_seed(args.seed)

    t0 = time.time()
    parts, tables = [], {}
    for panel in args.panels:
        part, tables[panel] = train_part(panel)
        parts.append(part)
    print(f"loaded {len(parts)} panels in {time.time() - t0:.0f}s", flush=True)

    net = m3.Net(parts[0]["X"].shape[1], m3.WIDTH).to(m3.DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=m3.LR, weight_decay=1e-4)
    total = args.epochs * m3.STEPS_PER_EPOCH
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=m3.LR, total_steps=total, pct_start=0.1)
    net.train()
    for ep in range(args.epochs):
        t1, tot = time.time(), 0.0
        for _ in range(m3.STEPS_PER_EPOCH):
            x, y, m, e = m3.sample_batch(parts, rng)
            x, m = m3.hide_extra(x, m, e, gen)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = net(x)
            outf = out.float()
            loss = m3.loss_fn(outf, y, m)
            if m3.DN_LAMBDA > 0:
                loss = loss + m3.DN_LAMBDA * m3.dn_loss(outf, x, y, m, e)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item()
        print(f"epoch {ep + 1:>2}: loss {tot / m3.STEPS_PER_EPOCH:.4f} ({time.time() - t1:.0f}s)", flush=True)
    torch.save(net.state_dict(), args.out_dir / f"t1_m3_seed{args.seed}.pt")
    del parts

    net.eval()
    for split in ("validation", "private"):
        out = args.out_dir / f"state_{split}.csv"
        for i, panel in enumerate(args.panels):
            z = load(panel, split)
            tv, tq = tables[panel]
            base_v, base_q = tv[z["weekday"]], tq[z["weekday"]]
            extra = m3.v2_channels(panel, split, z) if m3.V2 else None
            p = dict(X=m3.make_inputs(z, base_v, base_q, extra), base_v=base_v, base_q=base_q, z=z)
            pv, pq = m3.predict_days(net, p, range(len(z["dates"])))
            np.savez(args.out_dir / f"pred_{split}_{panel}.npz", pv=pv, pq=pq)
            frame = state_frame(panel, z, pv, pq)
            check_against_template(panel, split, frame)
            append_csv(frame, out, first=(i == 0))
            print(f"[{split} {panel}] {len(frame):,} rows, speed mean {frame.speed_kmh.mean():.1f}", flush=True)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
