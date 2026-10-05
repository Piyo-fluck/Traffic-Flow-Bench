"""Train the Task 2 CNN (with the reconstruction head of idea 9b) on synthetic runs, then on the real train run.

Modes (--mode):
  real_only          control: the rec_s20 recipe re-run by this script (same code path, no synthetic data)
  synth_only         the zero-shot gate: synthetic runs only; early stopping still on the real 2031-01/02 windows
  pretrain_finetune  --steps-pre on synthetic runs, then --steps-ft on the real train run (fresh LR schedule)
  mix                --steps-pre + --steps-ft steps on a mixture: each batch window is real with prob --real-share
The network, inputs, labels and losses are those of analysis/t2_cnn.py (Ada version) with
T2CNN_REC=0.3 T2CNN_LOVASZ=0.5: BCE + 0.5 Lovasz + 0.3 masked reconstruction, 64-link crops, batch 32.
Synthetic windows are built exactly like the real ones: every earliest-onset window plus up to 3000 ongoing
candidates per (panel, run). The static link channels come from the real panel (same network).

The checkpoint (best real-ES BCE, like t2_cnn.py) has the rec head, so idea 9b applies at evaluation:
    cd tta_lab && ../.venv/bin/python tta/run_adapt.py --name sim_X --ckpt ../sim_lab/out/X/model.pt
    cd tta_lab && ../.venv/bin/python tta/run_adapt.py --name sim_X_noad --ckpt ../sim_lab/out/X/model.pt --no-adapt

    python sim/train_sim.py --name pf_20k10k --mode pretrain_finetune --data sim_lab/data
"""
from __future__ import annotations

import argparse
import json
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

LAB = Path(__file__).resolve().parents[1]
PROJ = LAB.parent
for p in (PROJ, PROJ / "analysis", PROJ / "scripts"):
    sys.path.insert(0, str(p))
import t2_cnn as cnn  # noqa: E402
from t2_eda02 import T2_PANELS, candidates  # noqa: E402
from t2_eda03 import F, H, earliest_onset, labels  # noqa: E402
from t2_m1_models import Ctx  # noqa: E402
from tfb.physics import load_ramps  # noqa: E402

DEV = cnn.DEV


def windows(ctx, z, q, el, stat, rng, es, max_og=3000):
    """(x, y, m, es, lid, is_on) exactly as cnn.train_windows, for any split-like dict z."""
    d_on, t_on = np.nonzero(earliest_onset(q))
    c = candidates(z)
    d_og, t_og = c["d"][c["ongoing"]], c["t0"][c["ongoing"]]
    if len(d_og) > max_og:
        s = rng.choice(len(d_og), max_og, replace=False)
        d_og, t_og = d_og[s], t_og[s]
    d, t0 = np.concatenate([d_on, d_og]), np.concatenate([t_on, t_og])
    hi = t0[:, None] + np.arange(-H, 0)
    fi = t0[:, None] + np.arange(1, F + 1)
    x = cnn.make_x(ctx, stat, z["true_speed"][d[:, None], hi], z["true_flow"][d[:, None], hi], el[d[:, None], hi], t0,
                   z["weekday"][d] >= 5)
    return dict(x=x, y=q[d[:, None], fi].astype(np.float16), m=el[d[:, None], fi], es=es(d),
                lid=np.arange(ctx.S), is_on=np.r_[np.ones(len(d_on), bool), np.zeros(len(d_og), bool)])


def load_parts(args, rng):
    real, synth = [], []
    for i, panel in enumerate(T2_PANELS):
        ctx = Ctx(panel, i)
        ctx.loff = 0
        stat = cnn.static_channels(ctx, load_ramps(panel, "train", ctx.z))
        zr = ctx.z                                      # real parts are always built: they carry the ES windows
        if True:
            from tfb.t1eval import FIT_END
            real.append(windows(ctx, zr, ctx.q, ctx.el, stat, rng, lambda d: zr["dates"][d] >= FIT_END))
        if args.mode != "real_only":
            files = sorted((args.data / panel).glob("run_*.npz"))[:args.max_runs]
            for f in files:
                zz = dict(np.load(f))
                q, el = labels(zz)
                synth.append(windows(ctx, zz, q, el, stat, rng, lambda d: np.zeros(len(d), bool), args.syn_max_og))
        ctx.z = {k: ctx.z[k] for k in ("link_ids", "weekday")}
        ctx.q = ctx.el = None
        print(f"[{panel}] real windows {len(real[-1]['x']):,}, synthetic parts {len(synth)} "
              f"({sum(len(p['x']) for p in synth):,} windows so far)", flush=True)
    return real, synth


def train_phase(net, parts_fn, steps, lr, real, t0, tag, best):
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(steps, 1), pct_start=0.05)
    for step in range(1, steps + 1):
        parts, pool = parts_fn()
        x, y, m, _ = cnn.batch(parts, pool, cnn.BATCH, 0)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = net(x)
        logit = out[:, :F].float()
        loss = cnn.loss_fn(logit, y, m) + cnn.LOVASZ * cnn.lovasz_hinge(logit, y, m)
        if cnn.REC > 0:
            hid = torch.rand_like(x[:, 2]) < 0.2
            xin = x.clone()
            xin[:, 0:4] = xin[:, 0:4] * (~hid).unsqueeze(1).float()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                rec = net.rec(net.blocks(net.inp(xin))).float()
            wgt = (x[:, 2] > 0.5).float().unsqueeze(1) * hid.unsqueeze(1).float()
            loss = loss + cnn.REC * ((rec - x[:, 0:2].float()) ** 2 * wgt).sum() / wgt.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0:
            es = es_bce(net, real)
            flag = ""
            if es < best["bce"]:
                best.update(bce=es, state={k: v.detach().clone() for k, v in net.state_dict().items()})
                flag = " *"
            print(f"[{tag}] step {step}: train loss {loss.item():.4f}  real ES bce {es:.4f}{flag} "
                  f"({time.time() - t0:.0f}s)", flush=True)


@torch.no_grad()
def es_bce(net, real):
    tot, n = 0.0, 0.0
    for p in real:
        idx = np.flatnonzero(p["es"])[:400]
        if not len(idx):
            continue
        pr = cnn.predict(net, p["x"][idx])
        yv, mv = p["y"][idx].astype(np.float32), p["m"][idx].astype(np.float32)
        bce = -(yv * np.log(pr + 1e-6) + (1 - yv) * np.log(1 - pr + 1e-6))
        tot += float((bce * mv).sum())
        n += float(mv.sum())
    return tot / max(n, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--mode", required=True, choices=("real_only", "synth_only", "pretrain_finetune", "mix"))
    ap.add_argument("--data", type=Path, default=LAB / "data")
    ap.add_argument("--syn-max-og", type=int, default=500,
                    help="ongoing windows kept per synthetic run (memory: about 13 MB per run at 500)")
    ap.add_argument("--max-runs", type=int, default=1000, help="synthetic runs per panel to use")
    ap.add_argument("--steps-pre", type=int, default=20000)
    ap.add_argument("--steps-ft", type=int, default=10000)
    ap.add_argument("--lr-pre", type=float, default=1e-3)
    ap.add_argument("--lr-ft", type=float, default=3e-4)
    ap.add_argument("--real-share", type=float, default=0.5, help="mix mode: probability a batch is real")
    ap.add_argument("--seed", type=int, default=20)
    args = ap.parse_args()
    out = LAB / "out" / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1))
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    cnn.rng = rng                                     # cnn.batch draws windows and crops from this generator
    t0 = time.time()
    real, synth = load_parts(args, rng)
    pool = lambda parts: {id(p): np.flatnonzero(~p["es"]) for p in parts}  # noqa: E731
    pr_real, pr_syn = pool(real), pool(synth)
    net = cnn.T2Net(real[0]["x"].shape[1], cnn.WIDTH).to(DEV)
    best = {"bce": 1e9, "state": None}
    if args.mode == "real_only":
        train_phase(net, lambda: (real, pr_real), args.steps_pre + args.steps_ft, args.lr_pre, real, t0, "real", best)
    elif args.mode == "synth_only":
        train_phase(net, lambda: (synth, pr_syn), args.steps_pre + args.steps_ft, args.lr_pre, real, t0, "synth", best)
    elif args.mode == "pretrain_finetune":
        train_phase(net, lambda: (synth, pr_syn), args.steps_pre, args.lr_pre, real, t0, "pre", best)
        torch.save(net.state_dict(), out / "pretrained_last.pt")
        best_pre = dict(best)
        best.update(bce=1e9, state=None)              # the fine-tuned model is selected on its own
        train_phase(net, lambda: (real, pr_real), args.steps_ft, args.lr_ft, real, t0, "ft", best)
        print(f"best ES bce: pretraining {best_pre['bce']:.4f}, fine-tuning {best['bce']:.4f}", flush=True)
    else:
        both = lambda: (real, pr_real) if rng.random() < args.real_share else (synth, pr_syn)  # noqa: E731
        train_phase(net, both, args.steps_pre + args.steps_ft, args.lr_pre, real, t0, "mix", best)
    torch.save(best["state"], out / "model.pt")
    print(f"saved {out / 'model.pt'} (best real ES bce {best['bce']:.4f}, {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
