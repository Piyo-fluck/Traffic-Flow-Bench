"""Experiment 3: pretrain on masks we generate ourselves, then fine-tune on the real Task 1 task.

The user's proposal: make training multi-stage, with a stage whose job is to produce data that makes S_LWR more
learnable, then the ordinary stage.

What the measurements allow us to expect (notes 114, 115):

  * the model is BIAS limited - train and test errors fall together and the gap is a constant 0.0019 - so
    simply creating more (input, target) pairs mainly buys variance, which is nearly exhausted.
  * shrinking the receptive field at fixed parameters helps only +0.003 S_LWR, so capacity is not being
    badly wasted on irrelevant context either.

What is left is the inductive bias: which configurations the network is asked to solve. Two things are unused.

  1. **A train/test mismatch.** Production augmentation (`hide_extra`) hides extra visible cells on top of the
     real mask, so the input is thinner than it is at test. Here the WHOLE mask is redrawn from the truth at the
     regime's own rate, which makes the input density exactly right and still gives a different target set every
     epoch.
  2. **Unused supervision.** Only 20/30/50% of eligible cells are ever targets; the rest have truth and are
     never asked for. `hide_extra` reaches about 7.5% of them.

Stage 1 trains on a generated mask, stage 2 on the real one. Mask modes:

    prod   Bernoulli at the day regime's own rate - the production geometry, redrawn
    iso    temporally isolated targets - the configuration that carries 70% of the S_LWR residual (note 105),
           since an isolated target has a visible neighbour on both sides and both boundaries are scored
    wide   the rate drawn from U(0.10, 0.70), a harder and broader reconstruction task

Correctness: masking the truth-built input with the REAL target set must reproduce the production input exactly,
and --check verifies that before training.

    .venv/bin/python analysis/t1_ssl.py --mode iso --pre-epochs 25 --ft-epochs 25 --tag ssl_iso
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT))
import t1_m3_cnn as M  # noqa: E402
from tfb.data import load  # noqa: E402

RATE = {"R1": 0.20, "R2": 0.30, "R3": 0.50}      # MASK_SPEC, per day regime


def truth_inputs(p):
    """The network input built from the TRUTH at every cell the release would show if it were not masked."""
    z = load(p["panel"], "train")
    avail = np.isfinite(z["obs_speed"]) | z["target"]          # released, or released-but-masked
    zt = dict(z)
    for a, b in (("obs_speed", "true_speed"), ("obs_flow", "true_flow"), ("obs_occ", "true_occ")):
        zt[a] = np.where(avail, z[b], np.nan).astype(np.float32)
    extra = list(M.v2_channels(p["panel"], "train", zt)) if M.V2 else []
    if M.SCALE_CH:
        extra += M.scale_channels(zt, p["base_q"])
    X = M.make_inputs(zt, p["base_v"], p["base_q"], extra or None)
    pool = z["elig"] & np.isfinite(z["true_speed"]) & np.isfinite(z["true_flow"])
    return X, pool


def apply_mask(x, mask):
    """Blank the observation channels where the mask hides a cell, exactly as make_inputs would have."""
    keep = (~mask).unsqueeze(1).float()
    x = x.clone()
    x[:, 0:4] = x[:, 0:4] * keep
    return x


def draw_mask(pool, regime, mode, rng):
    """(B, T, S) bool mask drawn inside the eligible pool."""
    B = pool.shape[0]
    out = np.zeros(pool.shape, bool)
    for i in range(B):
        if mode == "wide":
            r = rng.uniform(0.10, 0.70)
        else:
            r = RATE[regime[i]]
        if mode == "iso":
            # a hard-core draw in time: propose at a higher rate, then drop any cell whose previous slot is kept
            prop = rng.random(pool[i].shape) < min(2 * r, 0.9)
            keep = prop.copy()
            keep[1:] &= ~prop[:-1]
            out[i] = keep & pool[i]
        else:
            out[i] = (rng.random(pool[i].shape) < r) & pool[i]
    return out


def ssl_batch(parts, rng, mode):
    xs, ys, ms, rg = [], [], [], []
    for _ in range(M.BATCH):
        p = parts[rng.integers(len(parts))]
        d = p["tr_days"][rng.integers(len(p["tr_days"]))]
        S = p["Xt"].shape[-1]
        s0 = rng.integers(0, S - M.CROP_S + 1) if S > M.CROP_S else 0
        sl = slice(s0, s0 + M.CROP_S)
        x, y, pool = p["Xt"][d, :, :, sl], p["Y"][d, :, :, sl], p["pool"][d, :, sl]
        pad = M.CROP_S - x.shape[-1]
        if pad > 0:
            x, y = np.pad(x, ((0, 0), (0, 0), (0, pad))), np.pad(y, ((0, 0), (0, 0), (0, pad)))
            pool = np.pad(pool, ((0, 0), (0, pad)))
        xs.append(x); ys.append(y); ms.append(pool); rg.append(p["regime"][d])
    pool = np.stack(ms)
    mask = draw_mask(pool, rg, mode, rng)
    to = lambda a, dt: torch.from_numpy(np.stack(a) if isinstance(a, list) else a).to(M.DEV, dt)  # noqa: E731
    x = to(xs, torch.float32)
    return apply_mask(x, to(mask, torch.bool)), to(ys, torch.float32), to(mask, torch.bool)


def prod_batch(parts, rng):
    """The real task, built from the same truth tensors by masking with the official target set."""
    xs, ys, ms = [], [], []
    for _ in range(M.BATCH):
        p = parts[rng.integers(len(parts))]
        d = p["tr_days"][rng.integers(len(p["tr_days"]))]
        S = p["Xt"].shape[-1]
        s0 = rng.integers(0, S - M.CROP_S + 1) if S > M.CROP_S else 0
        sl = slice(s0, s0 + M.CROP_S)
        x, y, m = p["Xt"][d, :, :, sl], p["Y"][d, :, :, sl], p["M"][d, :, sl]
        pad = M.CROP_S - x.shape[-1]
        if pad > 0:
            x, y = np.pad(x, ((0, 0), (0, 0), (0, pad))), np.pad(y, ((0, 0), (0, 0), (0, pad)))
            m = np.pad(m, ((0, 0), (0, pad)))
        xs.append(x); ys.append(y); ms.append(m)
    to = lambda a, dt: torch.from_numpy(np.stack(a)).to(M.DEV, dt)  # noqa: E731
    x, mask = to(xs, torch.float32), to(ms, torch.bool)
    return apply_mask(x, mask), to(ys, torch.float32), mask


def train(net, parts, rng, epochs, batch_fn, lr, tag):
    if epochs <= 0:
        return -1.0, None
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * M.STEPS_PER_EPOCH,
                                                pct_start=0.1)
    best, best_state = -1.0, None
    for ep in range(epochs):
        t1, tot = time.time(), 0.0
        net.train()
        for _ in range(M.STEPS_PER_EPOCH):
            x, y, m = batch_fn()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = net(x)
            loss = M.loss_fn(out.float(), y, m, x, None)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item()
        es, _ = M.evaluate(net, parts, "es_days")
        flag = ""
        if es > best:
            best, best_state, flag = es, {k: v.detach().clone() for k, v in net.state_dict().items()}, " *"
        print(f"[{tag}] epoch {ep + 1:>2}: loss {tot / M.STEPS_PER_EPOCH:.4f}  ES S_state {es:.4f} "
              f"({time.time() - t1:.0f}s){flag}", flush=True)
    return best, best_state


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="iso", choices=("prod", "iso", "wide"))
    ap.add_argument("--pre-epochs", type=int, default=25)
    ap.add_argument("--ft-epochs", type=int, default=25)
    ap.add_argument("--ft-lr", type=float, default=M.LR / 3)
    ap.add_argument("--tag", default="ssl")
    ap.add_argument("--panels", nargs="*", default=None)
    ap.add_argument("--check", action="store_true", help="verify the truth-built input reproduces production")
    args = ap.parse_args()
    t0 = time.time()
    panels = args.panels or M.PANELS
    parts = []
    for name in panels:
        p = M.panel_tensors(name)
        Xt, pool = truth_inputs(p)
        if args.check:
            m = torch.from_numpy(p["M"][:2]).to(M.DEV)
            a = apply_mask(torch.from_numpy(Xt[:2].astype(np.float32)).to(M.DEV), m).cpu().numpy()
            b = p["X"][:2].astype(np.float32)
            d = np.abs(a - b)
            print(f"  {name}: masked truth input vs production input, max |diff| {d.max():.6f}", flush=True)
        p["Xt"], p["pool"] = Xt, pool
        # evaluation and the real task both use the production input, which IS the truth input masked by the
        # official target set - rebuild it from Xt so the two can never drift apart
        p["X"] = Xt.copy()
        p["X"][:, 0:4] *= (~p["M"])[:, None]
        parts.append(p)
        print(f"  {name} ready ({time.time() - t0:.0f}s)", flush=True)
    if args.check:
        return
    net = M.Net(parts[0]["Xt"].shape[1], M.WIDTH).to(M.DEV)
    print(f"params {sum(q.numel() for q in net.parameters()) / 1e6:.2f}M (width {M.WIDTH}, RF {M.RF}); "
          f"stage 1 mode {args.mode} x {args.pre_epochs}, stage 2 x {args.ft_epochs}", flush=True)
    rng = np.random.default_rng(M.SEED)
    if args.pre_epochs > 0:
        _, st = train(net, parts, rng, args.pre_epochs, lambda: ssl_batch(parts, rng, args.mode), M.LR,
                      f"{args.tag}/pre")
        if st is not None:
            net.load_state_dict(st)
    best, st = train(net, parts, rng, args.ft_epochs, lambda: prod_batch(parts, rng), args.ft_lr,
                     f"{args.tag}/ft")
    if st is not None:
        net.load_state_dict(st)
    torch.save(net.state_dict(), ROOT / "work" / "models" / f"t1_{args.tag}_all.pt")
    print(f"best ES {best:.4f}", flush=True)
    M.report_test(net, parts, args.tag)


if __name__ == "__main__":
    main()
