"""Candidate A: a conditional diffusion model of the whole future speed field, for joint Task 2 samples.

Literature: CoST (2025, deterministic mean + diffusion for the rest), SpecSTG / DiffSTG (diffusion spatio-temporal
forecasting). Every Task 2 member we ship gives marginal per-cell probabilities; note 121 showed that the
per-window budget (how many cells to call queued) holds a +0.05..+0.076 ongoing oracle that marginals cannot
reach. A generative model gives JOINT samples, so the expected IoU of any candidate answer can be computed under
the correlated distribution instead of under independence.

Model
    condition   history (C, H, S) -> T2Net trunk (inp, blocks, collapse) -> (c, S); initialised from the
                SSL-pretrained width-192 checkpoint of note 123 and trained end to end
    target      x0 = future v/vf at T+5..T+30, (F, S), clipped to [0, 1.6]; ineligible cells masked out
    denoiser    x_t (F ch) + condition (c ch) -> 1x1 -> dilated residual 1-D convs over links, FiLM on the
                diffusion step -> epsilon (F ch). Cosine schedule, 1000 steps, DDIM 50 steps at sampling
    training    the same train windows and 64-link crops as analysis/t2_cnn.py (onset + ongoing), fit days only

Output: K samples per official-like window of both scored months, packed as bits of (x0 <= 0.6), to
gen_lab/cache/diff_<tag>_<split>_<panel>.npz (og_bits (N, K, F, S) packed along S, og_mean (N, F, S) float16,
on_mean (N, S)).

    T2CNN_AUX=1 T2CNN_WIDTH=192 T2CNN_REC=0.3 python gen_lab/t2_diffusion.py --init work/models/t2_cnn_lov_w192_rec_ssl30k_s20.pt
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("T2CNN_AUX", "1")          # t2_cnn.train_windows then also returns future v/vf (ya, ma)
os.environ.setdefault("T2CNN_WIDTH", "192")
os.environ.setdefault("T2CNN_REC", "0.3")
LAB = Path(__file__).resolve().parent
PROJ = LAB.parent
sys.path.insert(0, str(PROJ / "tta_lab" / "tta"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import common as C  # noqa: E402

cnn = C.cnn
F, H = C.F, C.H
DEV = C.DEV
CUT = 0.6
XMAX = 1.6


# ------------------------------------------------------------------------------------------ network
class StepEmb(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.SiLU(), nn.Linear(2 * d, d))

    def forward(self, t):
        half = self.d // 2
        f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        a = t.float()[:, None] * f[None]
        return self.mlp(torch.cat([a.sin(), a.cos()], 1))


class DBlock(nn.Module):
    def __init__(self, c, dil, d_emb):
        super().__init__()
        self.n1, self.n2 = nn.GroupNorm(8, c), nn.GroupNorm(8, c)
        self.c1 = nn.Conv1d(c, c, 3, padding=dil, dilation=dil)
        self.c2 = nn.Conv1d(c, c, 3, padding=1)
        self.film = nn.Linear(d_emb, 2 * c)

    def forward(self, h, e):
        g, b = self.film(e)[:, :, None].chunk(2, 1)
        y = self.c1(nn.functional.gelu(self.n1(h)))
        y = self.n2(y) * (1 + g) + b
        return h + self.c2(nn.functional.gelu(y))


class CondDiffusion(nn.Module):
    def __init__(self, cin, c_cond=192, c=192, d_emb=128):
        super().__init__()
        self.enc = cnn.T2Net(cin, c_cond)                 # only inp / blocks / collapse are used
        self.emb = StepEmb(d_emb)
        self.inp = nn.Conv1d(F + c_cond, c, 1)
        self.blocks = nn.ModuleList([DBlock(c, d, d_emb) for d in (1, 2, 4, 8, 16, 1, 2, 4)])
        self.out = nn.Sequential(nn.GroupNorm(8, c), nn.GELU(), nn.Conv1d(c, F, 1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def cond(self, x):                                    # (B, C, H, S) -> (B, c, S)
        return self.enc.collapse(self.enc.blocks(self.enc.inp(x)))[:, :, 0]

    def forward(self, xt, t, cf):
        e = self.emb(t)
        h = self.inp(torch.cat([xt, cf], 1))
        for b in self.blocks:
            h = b(h, e)
        return self.out(h)


def schedule(T=1000):
    s = 0.008
    x = torch.linspace(0, T, T + 1)
    ac = torch.cos(((x / T) + s) / (1 + s) * math.pi / 2) ** 2
    ac = ac / ac[0]
    return ac[1:].clamp(1e-5, 1.0).to(DEV)                # alpha_bar_t, t = 1..T


def scale(x0):                                            # v/vf in [0, XMAX] -> [-1, 1]
    return x0.clamp(0, XMAX) / XMAX * 2 - 1


def unscale(y):
    return (y.clamp(-1, 1) + 1) / 2 * XMAX


@torch.no_grad()
def sample(net, cf, ab, K, steps=50, gen=None):
    """DDIM (eta 0) from noise: cf (B, c, S) -> (B, K, F, S) samples of v/vf."""
    B, _, S = cf.shape
    cfk = cf.repeat_interleave(K, 0)
    x = torch.randn(B * K, F, S, device=DEV, generator=gen)
    ts = torch.linspace(len(ab) - 1, 0, steps, device=DEV).long()
    for j, t in enumerate(ts):
        a = ab[t]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            eps = net(x, t.repeat(B * K), cfk).float()
        x0 = ((x - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-1, 1)
        if j + 1 < len(ts):
            an = ab[ts[j + 1]]
            x = an.sqrt() * x0 + (1 - an).sqrt() * eps
        else:
            x = x0
    return unscale(x).view(B, K, F, S)


# ------------------------------------------------------------------------------------------ training
def train(args):
    parts, n_links = [], 0
    stats = {}
    for i, panel in enumerate(C.T2_PANELS):
        ctx = C.Ctx(panel, i)
        ctx.loff = n_links
        n_links += ctx.S
        stats[panel] = cnn.static_channels(ctx, C.load_ramps(panel, "train", ctx.z))
        p = cnn.train_windows(ctx, stats[panel])
        parts.append(p)
        print(f"[{panel}] windows {len(p['x']):,}", flush=True)
    pool = {id(p): np.flatnonzero(~p["es"]) for p in parts}
    net = CondDiffusion(parts[0]["x"].shape[1]).to(DEV)
    if args.init:
        sd = torch.load(args.init, map_location=DEV)
        own = net.enc.state_dict()
        n = 0
        for k, v in sd.items():
            if k in own and own[k].shape == v.shape:
                own[k] = v
                n += 1
        net.enc.load_state_dict(own)
        print(f"condition encoder initialised from {args.init}: {n} tensors", flush=True)
    ab = schedule()
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    t0 = time.time()
    run = 0.0
    for step in range(1, args.steps + 1):
        x, y, m, ex = cnn.batch(parts, pool, args.bs, n_links)
        x0 = scale(ex["ya"])                               # (B, F, S)
        mk = ex["ma"]
        t = torch.randint(0, len(ab), (len(x0),), device=DEV)
        a = ab[t][:, None, None]
        noise = torch.randn_like(x0)
        xt = a.sqrt() * x0 + (1 - a).sqrt() * noise
        with torch.autocast("cuda", dtype=torch.bfloat16):
            eps = net(xt, t, net.cond(x)).float()
        loss = (((eps - noise) ** 2) * mk).sum() / mk.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
        run += loss.item()
        if step % 1000 == 0:
            print(f"step {step}: loss {run / 1000:.4f} ({time.time() - t0:.0f}s)", flush=True)
            run = 0.0
    torch.save(net.state_dict(), PROJ / "work" / "models" / f"t2diff_{args.tag}.pt")
    return net, ab


# ------------------------------------------------------------------------------------------ scoring
@torch.no_grad()
def score(net, ab, args):
    net.eval()
    (LAB / "cache").mkdir(exist_ok=True)
    gen = torch.Generator(device=DEV).manual_seed(11)
    res = {}
    for i, panel in enumerate(C.T2_PANELS):
        ctx, stat, W = C.windows(i, panel)
        for s in C.SPLITS:
            w = W[s]
            on, og, z = w["on"], w["og"], w["z"]
            bits, means, cont = [], [], []
            for j in range(0, len(og["d"]), args.score_bs):
                m = np.arange(j, min(j + args.score_bs, len(og["d"])))
                x = C.to_dev(C.make_x(ctx, stat, og, m, z))
                smp = sample(net, net.cond(x), ab, args.k, args.ddim, gen)     # (b, K, F, S)
                q = (smp <= CUT).cpu().numpy()
                bits.append(np.packbits(q, axis=-1))
                if args.save_cont:            # v/vf quantised to 1/100 for the copula decision
                    cont.append((smp.clamp(0, 2.55) * 100).round().to(torch.uint8).cpu().numpy())
                means.append(q.mean(1).astype(np.float16))
            on_mean = np.zeros((len(on["d"]), ctx.S), np.float32)
            for j in range(0, len(on["d"]), args.score_bs):
                m = np.arange(j, min(j + args.score_bs, len(on["d"])))
                x = C.to_dev(C.make_x(ctx, stat, on, m, z))
                smp = sample(net, net.cond(x), ab, args.k, args.ddim, gen)
                on_mean[m] = (smp[:, :, F - 1] <= CUT).float().mean(1).cpu().numpy()
            og_bits = np.concatenate(bits) if bits else np.zeros((0, args.k, F, (ctx.S + 7) // 8), np.uint8)
            og_mean = np.concatenate(means) if means else np.zeros((0, F, ctx.S), np.float16)
            np.savez(LAB / "cache" / f"diff_{args.tag}_{s}_{panel}.npz", og_bits=og_bits, og_mean=og_mean,
                     on_mean=on_mean, S=ctx.S)
            if args.save_cont:
                np.save(LAB / "cache" / f"diffc_{args.tag}_{s}_{panel}.npy",
                        np.concatenate(cont) if cont else np.zeros((0, args.k, F, ctx.S), np.uint8))
            ref = w["ref"]
            old = 0.2 * ref["og_gs1"] + 0.6 * ref["og_tfm8"] + 0.2 * ref["og_cnn3"]
            r = res.setdefault(s, {})
            if len(og["d"]):
                pm = og_mean.astype(np.float32)
                for nm, p in (("ongoing diff mean", pm), ("ongoing old(12)", old),
                              ("ongoing 0.8old+0.2diff", 0.8 * old + 0.2 * pm)):
                    r.setdefault(nm, {})[panel] = C.ongoing_iou(p, og, 0.5).mean()
            if len(on["d"]):
                r.setdefault("onset diff mean", {})[panel] = C.onset_iou(on_mean, on).mean()
            print(f"  scored {panel} {s} ({len(og['d'])} ongoing, {len(on['d'])} onset)", flush=True)
    for s in C.SPLITS:
        print(f"\n=== {s} ===")
        for nm, d in res.get(s, {}).items():
            print(f"  {nm:<28} {C.family_mean(d):.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="a")
    ap.add_argument("--init", default="")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--ddim", type=int, default=50)
    ap.add_argument("--score-bs", type=int, default=8)
    ap.add_argument("--eval-only", default="")
    ap.add_argument("--save-cont", action="store_true", help="also save the continuous samples (uint8, v/vf x 100)")
    args = ap.parse_args()
    torch.manual_seed(0)
    if args.eval_only:
        sd = torch.load(args.eval_only, map_location=DEV)
        net = CondDiffusion(sd["enc.inp.weight"].shape[1]).to(DEV)
        net.load_state_dict(sd)
        ab = schedule()
    else:
        net, ab = train(args)
    score(net, ab, args)


if __name__ == "__main__":
    main()
