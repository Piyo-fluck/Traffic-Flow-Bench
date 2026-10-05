"""Experiment E: meta-learn the adaptation (first- or second-order MAML) on the TRAIN run.

Episode = (panel, day d of the train month):
    inner  k SGD steps of the self-supervised loss (common.ssl_loss) on history windows sampled from days < d
           of the released (masked) train layer - exactly what happens at test time, no label
    outer  Task 2 loss (BCE + 0.5 Lovasz, as production) of the ADAPTED weights on the labelled train windows of day d
           + rec_w x the self-supervised loss at the unadapted weights (keeps the rec head meaningful)
Task 2 labels appear only in the outer loop, so test-time behaviour is unchanged: evaluate the saved checkpoint with
    python tta/run_adapt.py --name E_eval --ckpt runs/E_meta/meta.pt --opt sgd --lr <alpha> --steps <k> --reset-daily
The train run is a single simulation run, so a run shift is simulated per episode (--aug): the v/vf and q/cap
channels of BOTH inner and outer inputs are multiplied by exp(U(-aug, aug)) (one factor per channel per episode).

    python tta/run_E_meta.py --name E_meta --steps 3000 --inner 5 --alpha 1e-3 --order 1 --aug 0.1
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.func import functional_call

import common as C
from t2_eda02 import candidates  # noqa: E402

CROP = 64


class Both(nn.Module):
    """functional_call target: forward(x, rec=False) -> Task 2 logits, or the reconstruction."""
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x, rec=False):
        if rec:
            return self.net.rec(self.net.blocks(self.net.inp(x)))
        return self.net(x)[:, :C.F]


def f_ssl(model, params, xb, losses, k_tail):
    """common.ssl_loss, but through functional_call so the inner update is differentiable."""
    ok = (xb[:, 2] > 0.5).float().unsqueeze(1)
    tgt = xb[:, 0:2]
    loss = xb.new_zeros(())
    w_mask, w_next, w_delta = losses
    with torch.autocast("cuda", dtype=torch.bfloat16):
        if w_mask:
            hid = torch.rand_like(xb[:, 2]) < 0.2
            xin = xb.clone()
            xin[:, 0:4] = xin[:, 0:4] * (~hid).unsqueeze(1).float()
            rec = functional_call(model, params, (xin,), {"rec": True}).float()
            w = ok * hid.unsqueeze(1).float()
            loss = loss + w_mask * ((rec - tgt) ** 2 * w).sum() / w.sum().clamp(min=1)
        if w_next or w_delta:
            k = k_tail
            xin = xb.clone()
            xin[:, 0:4, -k:] = 0
            rec = functional_call(model, params, (xin,), {"rec": True}).float()
            if w_next:
                w = ok[:, :, -k:]
                loss = loss + w_next * ((rec[:, :, -k:] - tgt[:, :, -k:]) ** 2 * w).sum() / w.sum().clamp(min=1)
            if w_delta:
                dp = rec[:, :, -k:] - rec[:, :, -k - 1:-1]
                dt = tgt[:, :, -k:] - tgt[:, :, -k - 1:-1]
                w = ok[:, :, -k:] * ok[:, :, -k - 1:-1]
                loss = loss + w_delta * ((dp - dt) ** 2 * w).sum() / w.sum().clamp(min=1)
    return loss


def train_windows(ctx, stat, rng, max_og=3000):
    """cnn.train_windows plus the day of every window (needed to pair it with its past days)."""
    z = ctx.z
    d_on, t_on = ctx.eo
    c = candidates(z)
    d_og, t_og = c["d"][c["ongoing"]], c["t0"][c["ongoing"]]
    if len(d_og) > max_og:
        s = rng.choice(len(d_og), max_og, replace=False)
        d_og, t_og = d_og[s], t_og[s]
    d, t0 = np.concatenate([d_on, d_og]), np.concatenate([t_on, t_og])
    hi = t0[:, None] + np.arange(-C.H, 0)
    fi = t0[:, None] + np.arange(1, C.F + 1)
    v, qf, ok = z["true_speed"][d[:, None], hi], z["true_flow"][d[:, None], hi], ctx.el[d[:, None], hi]
    x = C.cnn.make_x(ctx, stat, v, qf, ok, t0, z["weekday"][d] >= 5)
    return dict(x=x, y=ctx.q[d[:, None], fi].astype(np.float16), m=ctx.el[d[:, None], fi], d=d)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--init", default=str(C.MODELS / "t2_cnn_lov_rec_s20.pt"))
    ap.add_argument("--steps", type=int, default=3000, help="outer steps")
    ap.add_argument("--inner", type=int, default=5, help="inner SGD steps (use the same --steps at evaluation)")
    ap.add_argument("--alpha", type=float, default=1e-3, help="inner SGD lr (use as --lr at evaluation)")
    ap.add_argument("--lr", type=float, default=1e-4, help="outer AdamW lr")
    ap.add_argument("--order", type=int, default=1, choices=(1, 2))
    ap.add_argument("--inner-scope", default="full", choices=("full", "norm", "last2"),
                    help="which parameters the inner loop moves (match run_adapt --mode)")
    ap.add_argument("--losses", type=float, nargs=3, default=(1.0, 0.0, 0.0))
    ap.add_argument("--k-tail", type=int, default=1)
    ap.add_argument("--past-days", type=int, default=14, help="inner batch from days [d-N, d)")
    ap.add_argument("--inner-bs", type=int, default=16)
    ap.add_argument("--outer-bs", type=int, default=16)
    ap.add_argument("--rec-w", type=float, default=0.3)
    ap.add_argument("--aug", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = C.RUNS / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    t_all = time.time()
    parts = []
    for i, panel in enumerate(C.T2_PANELS):
        ctx = C.Ctx(panel, i)
        ctx.fitd = np.ones_like(ctx.fitd)
        ctx.loff = 0
        stat = C.cnn.static_channels(ctx, C.load_ramps(panel, "train", ctx.z))
        tw = train_windows(ctx, stat, rng)
        z = ctx.z
        zz = {k: z[k] for k in ("obs_speed", "obs_flow", "elig", "weekday")}   # the released train layer
        parts.append(dict(ctx=ctx, stat=stat, tw=tw, z=zz))
        ctx.z = {k: z[k] for k in ("link_ids", "weekday", "dates")}
        ctx.q = ctx.el = None
        print(f"[{panel}] {len(tw['d'])} train windows ({time.time() - t_all:.0f}s)", flush=True)

    net, _ = C.build_net(args.init, "full", train_rec=True)
    for p in net.parameters():
        p.requires_grad_(True)
    model = Both(net)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    names = [n for n, _ in model.named_parameters()]
    if args.inner_scope == "full":
        inner_keys = [n for n in names if n.startswith(("net.inp.", "net.blocks.", "net.rec."))]
    elif args.inner_scope == "last2":
        inner_keys = [n for n in names if n.startswith(("net.blocks.6.", "net.blocks.7."))]
    else:
        gn = {f"net.blocks.{n}.weight" for n, m in net.blocks.named_modules() if isinstance(m, nn.GroupNorm)}
        inner_keys = [n for n in names if n in gn or n.replace(".bias", ".weight") in gn]
    log = []
    for step in range(1, args.steps + 1):
        P = parts[rng.integers(len(parts))]
        tw, ctx, stat, z = P["tw"], P["ctx"], P["stat"], P["z"]
        d = int(rng.choice(tw["d"][tw["d"] >= 1]))
        pool = np.arange(max(0, d - args.past_days), d)
        xin = C.past_batch(ctx, stat, z, pool, args.inner_bs, rng)
        j = rng.choice(np.flatnonzero(tw["d"] == d), size=args.outer_bs)
        S = tw["x"].shape[-1]
        s0 = rng.integers(0, S - CROP + 1) if S > CROP else 0
        sl = slice(s0, s0 + CROP)
        xo = C.to_dev(tw["x"][j][..., sl])
        y = C.to_dev(tw["y"][j][..., sl])
        m = C.to_dev(tw["m"][j][..., sl])
        scale = torch.from_numpy(np.exp(rng.uniform(-args.aug, args.aug, 2))).to(C.DEV, torch.float32)
        for xx in (xin, xo):
            xx[:, 0:2] = xx[:, 0:2] * scale.view(1, 2, 1, 1)
        params = dict(model.named_parameters())
        fast = {k: params[k] for k in inner_keys}
        for _ in range(args.inner):
            li = f_ssl(model, {**params, **fast}, xin, args.losses, args.k_tail)
            g = torch.autograd.grad(li, list(fast.values()), create_graph=args.order == 2, allow_unused=True)
            fast = {k: (v - args.alpha * (gg if args.order == 2 else gg.detach())) if gg is not None else v
                    for (k, v), gg in zip(fast.items(), g)}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = functional_call(model, {**params, **fast}, (xo,)).float()
        outer = C.cnn.loss_fn(logit, y, m) + 0.5 * C.cnn.lovasz_hinge(logit, y, m)
        l0 = f_ssl(model, params, xin, (1.0, 0.0, 0.0), 1)
        total = outer + args.rec_w * l0
        opt.zero_grad(set_to_none=True)
        total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        log.append((float(outer.item()), float(li.item()) if args.inner else 0.0))
        if step % 100 == 0:
            a = np.mean(log[-100:], 0)
            print(f"step {step}: outer {a[0]:.4f}  inner-last {a[1]:.4f} ({time.time() - t_all:.0f}s)", flush=True)
        if step % 500 == 0 or step == args.steps:
            torch.save(net.state_dict(), out / f"meta_step{step}.pt")
            torch.save(net.state_dict(), out / "meta.pt")
    print(f"saved {out / 'meta.pt'}; evaluate with run_adapt.py --ckpt {out / 'meta.pt'} --opt sgd "
          f"--lr {args.alpha} --steps {args.inner} --reset-daily --mode {args.inner_scope}")


if __name__ == "__main__":
    main()
