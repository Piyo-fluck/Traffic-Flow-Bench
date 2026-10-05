"""Task 2 CNN trained on MORE ongoing windows: the cap and the origin spacing are the only changes.

The production recipe (analysis/t2_cnn.py, the rec_s20 model candidate T uses) builds its training set from
  every earliest-onset window                             (about 2,350 over the 8 panels)
  ongoing candidates at 30-minute origins, capped at 3,000 per panel   (21,663 in total)
There are 98,243 ongoing candidates at 10-minute origins, so the model currently sees under a quarter of them.
This script keeps the network, the inputs, the labels, the losses, the steps and the seed identical and only
changes `--og-step` (origin spacing in 5-minute slots) and `--max-og` (cap per panel).

    python data_lab/train_more_og.py --name og_step2_nocap --og-step 2 --max-og 100000
    python data_lab/train_more_og.py --name ctrl_prod --og-step 6 --max-og 3000      # control, same code path

Evaluate the checkpoint like any other rec-head model (idea 9b is applied by run_adapt.py):
    cd tta_lab && ../.venv/bin/python tta/run_adapt.py --name <name> --ckpt ../data_lab/out/<name>.pt
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("T2CNN_REC", "0.3")
os.environ.setdefault("T2CNN_LOVASZ", "0.5")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

LAB = Path(__file__).resolve().parent
PROJ = LAB.parent
for p in (PROJ, PROJ / "analysis", PROJ / "scripts"):
    sys.path.insert(0, str(p))
import t2_cnn as cnn  # noqa: E402
import t2_eda02 as e2  # noqa: E402
from t2_eda03 import F, H, earliest_onset  # noqa: E402
from t2_m1_models import Ctx  # noqa: E402
from tfb.physics import load_ramps  # noqa: E402
from tfb.t1eval import FIT_END  # noqa: E402


def windows(ctx, stat, rng, og_step, max_og):
    z = ctx.z
    d_on, t_on = np.nonzero(earliest_onset(ctx.q))
    e2.STEP = og_step
    c = e2.candidates(z)
    e2.STEP = 6
    d_og, t_og = c["d"][c["ongoing"]], c["t0"][c["ongoing"]]
    if len(d_og) > max_og:
        s = rng.choice(len(d_og), max_og, replace=False)
        d_og, t_og = d_og[s], t_og[s]
    d, t0 = np.concatenate([d_on, d_og]), np.concatenate([t_on, t_og])
    hi = t0[:, None] + np.arange(-H, 0)
    fi = t0[:, None] + np.arange(1, F + 1)
    x = cnn.make_x(ctx, stat, z["true_speed"][d[:, None], hi], z["true_flow"][d[:, None], hi],
                   ctx.el[d[:, None], hi], t0, z["weekday"][d] >= 5)
    return dict(x=x, y=ctx.q[d[:, None], fi].astype(np.float16), m=ctx.el[d[:, None], fi],
                es=z["dates"][d] >= FIT_END, lid=np.arange(ctx.S),
                is_on=np.r_[np.ones(len(d_on), bool), np.zeros(len(d_og), bool)]), len(d_on), len(d_og)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--og-step", type=int, default=2, help="ongoing origin spacing in 5-min slots (production: 6)")
    ap.add_argument("--max-og", type=int, default=100000, help="cap per panel (production: 3000)")
    ap.add_argument("--onset-share", type=float, default=0.0,
                    help="repeat the onset windows in the sampling pool until they are this share of it "
                         "(production is 0.098); 0 = no repetition")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=20)
    args = ap.parse_args()
    (LAB / "out").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    cnn.rng = rng
    torch.manual_seed(args.seed)
    t0 = time.time()
    parts = []
    for i, panel in enumerate(cnn.T2_PANELS):
        ctx = Ctx(panel, i)
        ctx.loff = 0
        stat = cnn.static_channels(ctx, load_ramps(panel, "train", ctx.z))
        p, n_on, n_og = windows(ctx, stat, rng, args.og_step, args.max_og)
        parts.append(p)
        ctx.z = ctx.q = ctx.el = None
        print(f"[{panel}] onset {n_on}, ongoing {n_og} ({time.time() - t0:.0f}s)", flush=True)
    pool = {}
    for p in parts:
        idx = np.flatnonzero(~p["es"])
        if args.onset_share > 0:
            on, og = idx[p["is_on"][idx]], idx[~p["is_on"][idx]]
            k = max(1, int(round(args.onset_share * len(og) / max(len(on), 1) / (1 - args.onset_share))))
            idx = np.concatenate([np.repeat(on, k), og])
            print(f"  onset windows repeated {k}x -> share {len(on) * k / len(idx):.3f}", flush=True)
        pool[id(p)] = idx
    print(f"total windows {sum(len(p['x']) for p in parts):,}", flush=True)
    net = cnn.T2Net(parts[0]["x"].shape[1], cnn.WIDTH).to(cnn.DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    best, best_state = 1e9, None
    for step in range(1, args.steps + 1):
        x, y, m, _ = cnn.batch(parts, pool, cnn.BATCH)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = net(x)
        logit = out[:, :F].float()
        loss = cnn.loss_fn(logit, y, m) + cnn.LOVASZ * cnn.lovasz_hinge(logit, y, m)
        hid = torch.rand_like(x[:, 2]) < 0.2
        xin = x.clone()
        xin[:, 0:4] = xin[:, 0:4] * (~hid).unsqueeze(1).float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            rec = net.rec(net.blocks(net.inp(xin))).float()
        w = (x[:, 2] > 0.5).float().unsqueeze(1) * hid.unsqueeze(1).float()
        loss = loss + cnn.REC * ((rec - x[:, 0:2].float()) ** 2 * w).sum() / w.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0:
            with torch.no_grad():
                tot, n = 0.0, 0.0
                for p in parts:
                    es = np.flatnonzero(p["es"])[:400]
                    if len(es):
                        pr = cnn.predict(net, p["x"][es])
                        yv, mv = p["y"][es].astype(np.float32), p["m"][es].astype(np.float32)
                        bce = -(yv * np.log(pr + 1e-6) + (1 - yv) * np.log(1 - pr + 1e-6))
                        tot += float((bce * mv).sum()); n += float(mv.sum())
            es_loss = tot / max(n, 1)
            flag = ""
            if es_loss < best:
                best, best_state, flag = es_loss, {k: v.detach().clone() for k, v in net.state_dict().items()}, " *"
            print(f"step {step}: train loss {loss.item():.4f}  ES bce {es_loss:.4f}{flag} ({time.time() - t0:.0f}s)",
                  flush=True)
    torch.save(best_state, LAB / "out" / f"{args.name}.pt")
    print(f"saved {args.name}.pt (ES bce {best:.4f}, {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
