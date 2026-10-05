"""One ongoing model to replace the twelve.

`occ_lab/ongoing_ablation.py` priced the stack: the frozen half of the blend is 0.20 gs1 + 0.60 tfm8 + 0.20 cnn3
- one LightGBM and eleven neural seeds - and it scores 0.8295, while a single model adapted to the scored run
scores 0.8291. Seeds are not what the blend is made of, so the replacement has to carry a KIND of information the
twelve do not, not more of the same.

What is missing from the frozen half is occupancy: `make_x` gives the twelve v/v_free, q/capacity, the observed
flag and the queued flag, and never the loading that separates a full lane from an empty one at the same speed
(note 84 measured the AUC on the disputed cells at 0.772 -> 0.785 with it). `occ_lab/train_occ.py` already builds
those three channels. This trains one wider model on them, on the production windows, losses and schedule, so
the only differences from a production seed are the input and the width.

The family to replace is the transformer one. Alone, `occ_lab/member_alone.py` puts tfm8 at 0.8191 against
cnn3's 0.7901 and gs1's 0.7846, and blended with the adapted member tfm8 reaches **0.8437, above the full
twelve-model 0.8424** - so the CNN seeds and the LightGBM are not merely redundant, they cost. The target for one
model is therefore 0.8191 alone and 0.8437 blended.

    T2CNN_WIDTH=128 python occ_lab/train_one.py --steps 20000
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("T2CNN_LOVASZ", "0.5")
os.environ.setdefault("T2CNN_ARCH", "tfm")
os.environ.setdefault("OMP_NUM_THREADS", "4")
LAB = Path(__file__).resolve().parent
PROJ = LAB.parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(PROJ / "analysis"))
sys.path.insert(0, str(PROJ / "scripts"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import t2_cnn as cnn  # noqa: E402
from t2_eda02 import T2_PANELS  # noqa: E402
from t2_eda03 import F  # noqa: E402
from t2_m1_models import Ctx  # noqa: E402
from tfb.physics import load_ramps  # noqa: E402
from train_occ import windows  # noqa: E402

MODELS = LAB / "models"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--ema", type=float, default=0.0,
                    help="decay of an exponential moving average of the weights. Eight seeds buy variance "
                         "reduction by averaging models; this buys it by averaging one model's own trajectory, "
                         "at no extra cost. The EMA weights are what gets scored and saved")
    ap.add_argument("--occ", action="store_true",
                    help="keep the three occupancy channels. They cost on the scored months: occupancy is only "
                         "~76%% observed there while training sees it everywhere inside eligibility, and it "
                         "carries no observed flag of its own, so a masked cell reads as an empty road")
    args = ap.parse_args()
    MODELS.mkdir(parents=True, exist_ok=True)
    cnn.rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    t0 = time.time()
    parts, profs = [], {}
    for i, panel in enumerate(T2_PANELS):
        ctx = Ctx(panel, i)
        ctx.loff = 0
        stat = cnn.static_channels(ctx, load_ramps(panel, "train", ctx.z))
        p, profs[panel] = windows(ctx, stat, cnn.rng)
        if not args.occ:
            p["x"] = np.ascontiguousarray(p["x"][:, :-3])
        parts.append(p)
        print(f"[{panel}] windows {len(p['x']):,}, channels {p['x'].shape[1]} "
              f"({time.time() - t0:.0f}s)", flush=True)
        ctx.z = ctx.q = ctx.el = None
    np.savez(MODELS / "occ_profiles_one.npz", **profs)
    pool = {id(p): np.flatnonzero(~p["es"]) for p in parts}
    cin = parts[0]["x"].shape[1]
    net = (cnn.T2Tfm if cnn.ARCH == "tfm" else cnn.T2Net)(cin, cnn.WIDTH).to(cnn.DEV)
    n_par = sum(q.numel() for q in net.parameters())
    print(f"{cnn.ARCH} width {cnn.WIDTH}, {cin} input channels, {n_par:,} parameters, "
          f"batch {cnn.BATCH}, lovasz {cnn.LOVASZ}", flush=True)
    ema = {k: v.detach().clone().float() for k, v in net.state_dict().items()} if args.ema else None
    enet = (cnn.T2Tfm if cnn.ARCH == "tfm" else cnn.T2Net)(cin, cnn.WIDTH).to(cnn.DEV) if args.ema else None
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    best, best_state, best_step = 1e9, None, 0
    for step in range(1, args.steps + 1):
        x, y, m, _ = cnn.batch(parts, pool, cnn.BATCH)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = net(x)
        logit = out[:, :F].float()
        loss = cnn.loss_fn(logit, y, m) + cnn.LOVASZ * cnn.lovasz_hinge(logit, y, m)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if ema is not None:
            with torch.no_grad():
                for k, v in net.state_dict().items():
                    ema[k].mul_(args.ema).add_(v.detach().float(), alpha=1.0 - args.ema)
        if step % 500 == 0:
            scored = net
            if ema is not None:
                enet.load_state_dict({k: v.to(dtype=w.dtype) for (k, v), w
                                      in zip(ema.items(), net.state_dict().values())})
                enet.eval()
                scored = enet
            with torch.no_grad():
                tot, n = 0.0, 0.0
                for p in parts:
                    es = np.flatnonzero(p["es"])[:400]
                    if not len(es):
                        continue
                    pr = cnn.predict(scored, p["x"][es])
                    yv, mv = p["y"][es].astype(np.float32), p["m"][es].astype(np.float32)
                    bce = -(yv * np.log(pr + 1e-6) + (1 - yv) * np.log(1 - pr + 1e-6))
                    tot += float((bce * mv).sum())
                    n += float(mv.sum())
            es_loss = tot / max(n, 1)
            flag = ""
            if es_loss < best:
                best, best_step, flag = es_loss, step, " *"
                best_state = {k: v.detach().clone() for k, v in scored.state_dict().items()}
            print(f"step {step}: train {loss.item():.4f}  ES bce {es_loss:.4f}{flag} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    name = (f"one_{cnn.ARCH}_w{cnn.WIDTH}{'_occ' if args.occ else ''}"
            f"{'_ema' if args.ema else ''}{args.tag}_s{args.seed}.pt")
    torch.save(best_state, MODELS / name)
    print(f"saved {name} (ES bce {best:.4f} at step {best_step}, {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
