"""Onset as what it actually is: a single-horizon spatial problem at T+30.

The organizers' announcement (notes/onset_discussion_2026-09-23.txt) confirms that every queue_onset window is
built by taking the earliest origin whose 30-minute horizon contains a queue, so the queue lands on T+30 by
construction and T+5..T+25 are empty in all 120 shipped windows. Our pipeline already places predictions only at
T+30, and our onset training windows already come from the same selector (`earliest_onset`) - but the network is
still trained with the loss spread over all six horizons, five of which are empty by definition and never read.

This trains the onset model on the T+30 map alone:

    windows   onset windows only (T2CNN_ONSET_ONLY), i.e. the day's first onset, the official condition
    loss      BCE + 0.5 Lovasz on horizon T+30 only
    start     fine-tuned from a production checkpoint (note 46: fine-tuning on onset windows lifted the single
              model in 8/8 cells; its listwise "which link breaks first" term did not help, so it stays off)
    ES        BCE at T+30 on the 2031-01/02 windows

    CUDA_VISIBLE_DEVICES=0 python occ_lab/train_onset.py --seed 0 --init work/models/t2_cnn_lov.pt
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("T2CNN_LOVASZ", "0.5")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ["T2CNN_ONSET_ONLY"] = "1"
os.environ["T2CNN_ARCH"] = os.environ.get("ONSET_ARCH", "cnn")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

LAB = Path(__file__).resolve().parent
PROJ = LAB.parent
for p in (PROJ, PROJ / "analysis", PROJ / "scripts", LAB):
    sys.path.insert(0, str(p))
import t2_cnn as cnn  # noqa: E402
from t2_eda02 import T2_PANELS  # noqa: E402
from t2_eda03 import F  # noqa: E402
from t2_m1_models import Ctx  # noqa: E402
from tfb.physics import load_ramps  # noqa: E402

MODELS = LAB / "models"
REMOTE = PROJ / "work" / ("models_remote" if (PROJ / "work" / "models_remote").exists() else "models")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--init", type=Path, default=None, help="production checkpoint to fine-tune from")
    ap.add_argument("--arch", default=os.environ.get("ONSET_ARCH", "cnn"), choices=("cnn", "tfm"))
    args = ap.parse_args()
    MODELS.mkdir(parents=True, exist_ok=True)
    cnn.rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    t0 = time.time()
    parts = []
    for i, panel in enumerate(T2_PANELS):
        ctx = Ctx(panel, i)
        ctx.loff = 0
        stat = cnn.static_channels(ctx, load_ramps(panel, "train", ctx.z))
        parts.append(cnn.train_windows(ctx, stat))
        print(f"[{panel}] onset windows {len(parts[-1]['x']):,} ({time.time() - t0:.0f}s)", flush=True)
        ctx.z = ctx.q = ctx.el = None
    pool = {id(p): np.flatnonzero(~p["es"]) for p in parts}
    net = (cnn.T2Tfm if args.arch == "tfm" else cnn.T2Net)(parts[0]["x"].shape[1], cnn.WIDTH).to(cnn.DEV)
    if args.init is not None:
        net.load_state_dict(torch.load(args.init, map_location=cnn.DEV))
        print(f"fine-tuning from {args.init.name}", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    best, best_state = 1e9, None
    for step in range(1, args.steps + 1):
        x, y, m, _ = cnn.batch(parts, pool, cnn.BATCH)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = net(x)
        logit = out[:, F - 1:F].float()                     # T+30 only: the other five are empty by construction
        yl, ml = y[:, F - 1:F], m[:, F - 1:F]
        loss = cnn.loss_fn(logit, yl, ml) + cnn.LOVASZ * cnn.lovasz_hinge(logit, yl, ml)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 250 == 0:
            with torch.no_grad():
                tot, n = 0.0, 0.0
                for p in parts:
                    es = np.flatnonzero(p["es"])[:400]
                    if not len(es):
                        continue
                    pr = cnn.predict(net, p["x"][es])[:, F - 1]
                    yv, mv = p["y"][es][:, F - 1].astype(np.float32), p["m"][es][:, F - 1].astype(np.float32)
                    bce = -(yv * np.log(pr + 1e-6) + (1 - yv) * np.log(1 - pr + 1e-6))
                    tot += float((bce * mv).sum()); n += float(mv.sum())
            es_loss = tot / max(n, 1)
            flag = ""
            if es_loss < best:
                best, best_state, flag = es_loss, {k: v.detach().clone() for k, v in net.state_dict().items()}, " *"
            print(f"step {step}: train {loss.item():.4f}  ES bce(T+30) {es_loss:.5f}{flag} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    torch.save(best_state, MODELS / f"onset30_{args.arch}_s{args.seed}.pt")
    print(f"saved onset30_{args.arch}_s{args.seed}.pt (ES {best:.5f}, {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
