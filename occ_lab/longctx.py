"""Task 2 with long context: an STD-MAE-style (IJCAI 2024, arXiv 2312.00516) long-range branch on the SSL w192 CNN.

Every Task 2 member reads only the 60 minutes of `window_history.parquet`, while the organizers allow every released
observation at or before T (2026-09-20). For ONGOING windows two things are invisible in 60 minutes: how long the
queue has already lasted (a queue that began three hours ago is due to dissolve), and how this run behaved at this
place and time on its earlier days (the same-run evidence that made the onset bottleneck classifier work, note 129).
Note 122 refuted the long-context hypothesis only for onset bottleneck choice on the train run.

    long inputs (per window, all at or before T)
        A   the same day, the 60 slots before the short history (T-360 .. T-65), channels v/vf, q/cap, observed,
            queued, weekend, available
        B   the K = 3 earlier days of the same split / run, slots T-180 .. T+115 of THOSE days (whole days in the
            past, so their "future" is released history), same channels
        age the queue's age at T-5 per link over the whole day so far (slots since the last observed free-flow
            cell, /72, capped at 2) and its max over +-2 links
    encoders    two small dilated CNNs (the STD-MAE decoupling): encS pretrained by masked reconstruction with
                whole LINKS hidden (S-mask), encT with time PATCHES of 12 slots hidden (T-mask), mask ratio 0.25,
                MAE loss on masked visible cells, train truth only (--stage pretrain)
    fusion      as STD-MAE eq. 6: H = H_short + MLP([encS(A), encT(A), mean_k encS(B_k), mean_k encT(B_k), age]),
                added to the short CNN's collapsed representation before its head; the MLP's last layer is
                zero-initialised, so training starts exactly at the SSL w192 model
    control     --no-long: the same fine-tuning of the same checkpoint with the long branch off

Stages
    pretrain    python occ_lab/longctx.py --stage pretrain --out work/models/lc_enc.pt
    train       python occ_lab/longctx.py --stage train --ckpt work/models/t2_cnn_lov_w192_rec_ssl30k_s20.pt
                    --enc work/models/lc_enc.pt --tag s20            (or --no-long --tag ctrl_s20)
    lab         python occ_lab/longctx.py --stage lab --model work/models/lc_s20.pt --tag s20
                causal pseudo-supervised adaptation per day (note 106 recipe), official-like pool of both months
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ["T2CNN_ARCH"] = "cnn"
os.environ.setdefault("T2CNN_REC", "0.3")
os.environ.setdefault("T2CNN_LOVASZ", "0.5")
os.environ.setdefault("T2CNN_WIDTH", "192")
LAB = Path(__file__).resolve().parent
PROJ = LAB.parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(PROJ / "tta_lab" / "tta"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import common as C  # noqa: E402

cnn = C.cnn
H, F, T = C.H, C.F, cnn.T
DEV = C.DEV
LS, LB, K, PAD = 60, 60, 3, 128
B_OFF = -(H + 24)               # B window starts 24 slots before the short history: T-180 min
CL = 64                         # long-encoder width
CACHE = LAB / "cache"
MODELS = PROJ / "work" / "models"


# ------------------------------------------------------------------------------------------------ long inputs
def arrays(v, q, ok, vf, cap):
    """(D, T, S) -> (D, T + 2 PAD, 4, S) float16: v/vf, q/cap, observed, queued; zero outside the day."""
    qd = ok & (np.nan_to_num(v, nan=1e9) <= 0.6 * vf)
    a = np.stack([np.where(ok, v / vf, 0.0), np.where(ok, q / cap, 0.0), ok, qd], 2).astype(np.float16)
    return np.pad(a, ((0, 0), (PAD, PAD), (0, 0), (0, 0)))


def queue_age(arr, d, t0):
    """(N, 2, S): slots since the last observed free-flow cell of the day, if the link is queued now; +-2 max."""
    out = np.zeros((len(d), 2, arr.shape[-1]), np.float32)
    for n, (dd, tt) in enumerate(zip(d, t0)):
        seg = arr[dd, PAD:PAD + tt]
        ok, qd = seg[:, 2] > 0.5, seg[:, 3] > 0.5
        idx = np.arange(tt)[:, None]
        last_free = np.where(ok & ~qd, idx, -1).max(0)
        last_q = np.where(qd, idx, -1).max(0)
        age = np.where(last_q > last_free, (tt - 1 - last_free), 0) / 72.0
        age = np.minimum(age, 2.0)
        out[n, 0] = age
        pa = np.pad(age, 2)
        out[n, 1] = np.max(np.stack([pa[k:k + len(age)] for k in range(5)]), 0)
    return out


def long_inputs(arr, wk, d, t0, dmin=0, blank_rng=None, with_age=True):
    """xa (N, 6, LS, S), xb (N, K, 6, LB, S), age (N, 2, S) for windows (d, t0) of one panel and split."""
    N, S = len(d), arr.shape[-1]
    sa = (t0 - H - LS)[:, None] + np.arange(LS) + PAD
    xa = np.zeros((N, 6, LS, S), np.float16)
    xa[:, :4] = arr[d[:, None], sa].transpose(0, 2, 1, 3)
    xa[:, 4] = (wk[d] >= 5)[:, None, None]
    xa[:, 5] = 1
    xb = np.zeros((N, K, 6, LB, S), np.float16)
    sb = (t0 + B_OFF)[:, None] + np.arange(LB) + PAD
    for k in range(1, K + 1):
        dk = d - k
        av = dk >= dmin
        if not av.any():
            continue
        j = np.flatnonzero(av)
        xb[j, k - 1, :4] = arr[dk[j][:, None], sb[j]].transpose(0, 2, 1, 3)
        xb[j, k - 1, 4] = (wk[dk[j]] >= 5)[:, None, None]
        xb[j, k - 1, 5] = 1
    if blank_rng is not None:       # the scored months hide an hour after every official window: simulate it
        for n in range(N):
            for tgt in [xa[n]] + [xb[n, k] for k in range(K)]:
                if blank_rng.random() < 0.5:
                    s0 = int(blank_rng.integers(0, LS - 18))
                    tgt[:4, s0:s0 + 18] = 0
    return xa, xb, (queue_age(arr, d, t0) if with_age else None)


# ------------------------------------------------------------------------------------------------ networks
class Enc(nn.Module):
    def __init__(self, cin=6, c=CL, L=LS):
        super().__init__()
        self.inp = nn.Conv2d(cin, c, 1)
        self.blocks = nn.Sequential(*[cnn.Block(c, d) for d in (1, 2, 4, 8)])
        self.collapse = nn.Sequential(nn.GroupNorm(8, c), nn.GELU(), nn.Conv2d(c, c, (L, 1)))
        self.rec = nn.Conv2d(c, 2, 1)

    def feats(self, x):
        return self.blocks(self.inp(x))

    def forward(self, x):                                   # (B, 6, L, S) -> (B, c, S)
        return self.collapse(self.feats(x))[:, :, 0]


class LCNet(nn.Module):
    def __init__(self, short):
        super().__init__()
        self.short = short
        self.encS, self.encT = Enc(), Enc()
        c = short.inp.weight.shape[0]
        self.fuse = nn.Sequential(nn.Conv1d(4 * CL + 2, c, 1), nn.GELU(), nn.Conv1d(c, c, 1))
        nn.init.zeros_(self.fuse[2].weight)
        nn.init.zeros_(self.fuse[2].bias)

    def forward(self, x, xa=None, xb=None, age=None):       # -> (B, F, S) logits
        s = self.short
        h = s.collapse(s.blocks(s.inp(x)))[:, :, 0]
        if xa is not None:
            B, Kk, Cc, L, S = xb.shape
            xbf = xb.reshape(B * Kk, Cc, L, S)
            w = xb[:, :, 5, 0, :].unsqueeze(2)              # (B, K, 1, S) day available
            pool = lambda e: (e.view(B, Kk, -1, S) * w).sum(1) / w.sum(1).clamp(min=1)  # noqa: E731
            g = torch.cat([self.encS(xa), self.encT(xa), pool(self.encS(xbf)), pool(self.encT(xbf)), age], 1)
            h = h + self.fuse(g.to(h.dtype))
        return s.head(h[:, :, None, :])[:, :F, 0]


def load_short(ckpt):
    sd = torch.load(ckpt, map_location=DEV)
    net = cnn.T2Net(sd["inp.weight"].shape[1], sd["inp.weight"].shape[0]).to(DEV)
    net.load_state_dict(sd)
    return net


def tt(a, dt=torch.float32):
    return torch.from_numpy(np.ascontiguousarray(a)).to(DEV, dt)


# ------------------------------------------------------------------------------------------------ train-run data
def train_panels():
    """Per train panel: ctx, static channels, long arrays (truth), windows (d, t0), ES flags."""
    from t2_eda02 import candidates
    from tfb.t1eval import FIT_END
    rng = np.random.default_rng(0)
    out = []
    for i, panel in enumerate(C.T2_PANELS):
        ctx = C.Ctx(panel, i)
        ctx.loff = 0
        stat = cnn.static_channels(ctx, C.load_ramps(panel, "train", ctx.z))
        z = ctx.z
        ok = ctx.el & np.isfinite(z["true_speed"]) & np.isfinite(z["true_flow"])
        arr = arrays(z["true_speed"], z["true_flow"], ok, ctx.vf, ctx.cap)
        d_on, t_on = ctx.eo
        c = candidates(z)
        d_og, t_og = c["d"][c["ongoing"]], c["t0"][c["ongoing"]]
        if len(d_og) > 3000:
            sel = rng.choice(len(d_og), 3000, replace=False)
            d_og, t_og = d_og[sel], t_og[sel]
        d, t0 = np.concatenate([d_on, d_og]), np.concatenate([t_on, t_og])
        out.append(dict(panel=panel, ctx=ctx, stat=stat, arr=arr, wk=z["weekday"], q=ctx.q, el=ctx.el,
                        d=d, t0=t0, es=z["dates"][d] >= FIT_END,
                        v=z["true_speed"], qf=z["true_flow"], ok=ok))
        print(f"  [{panel}] {len(d):,} windows", flush=True)
    return out


def short_x(p, d, t0):
    hi = t0[:, None] + np.arange(-H, 0)
    return cnn.make_x(p["ctx"], p["stat"], p["v"][d[:, None], hi], p["qf"][d[:, None], hi], p["ok"][d[:, None], hi],
                      t0, p["wk"][d] >= 5)


def crop(a, sl, S):
    a = a[..., sl]
    pad = C.cnn.CROP - a.shape[-1]
    return np.pad(a, [(0, 0)] * (a.ndim - 1) + [(0, pad)]) if pad > 0 else a


def train_batch(parts, bs, rng, with_long):
    xs, xas, xbs, ags, ys, ms = [], [], [], [], [], []
    for _ in range(bs):
        p = parts[rng.integers(len(parts))]
        pool = p["tr"]
        j = pool[rng.integers(len(pool))]
        d, t0 = p["d"][j:j + 1], p["t0"][j:j + 1]
        S = p["arr"].shape[-1]
        s0 = rng.integers(0, S - cnn.CROP + 1) if S > cnn.CROP else 0
        sl = slice(s0, s0 + cnn.CROP)
        fi = t0[:, None] + np.arange(1, F + 1)
        xs.append(crop(short_x(p, d, t0)[0], sl, S))
        ys.append(crop(p["q"][d[:, None], fi][0].astype(np.float32), sl, S))
        ms.append(crop(p["el"][d[:, None], fi][0].astype(np.float32), sl, S))
        if with_long:
            xa, xb, ag = long_inputs(p["arr"], p["wk"], d, t0, 0, rng)
            xas.append(crop(xa[0], sl, S)); xbs.append(crop(xb[0], sl, S)); ags.append(crop(ag[0], sl, S))
    st = lambda a: tt(np.stack(a))  # noqa: E731
    lo = (st(xas), st(xbs), st(ags)) if with_long else (None, None, None)
    return st(xs), lo, st(ys), st(ms)


# ------------------------------------------------------------------------------------------------ stage: pretrain
def stage_pretrain(args):
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    parts = train_panels()
    encS, encT = Enc().to(DEV), Enc().to(DEV)
    params = list(encS.parameters()) + list(encT.parameters())
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=args.steps, pct_start=0.05)
    t_start, run = time.time(), [0.0, 0.0]
    for step in range(1, args.steps + 1):
        xs = []
        for _ in range(args.bs):
            p = parts[rng.integers(len(parts))]
            D, S = p["arr"].shape[0], p["arr"].shape[-1]
            d = np.array([rng.integers(D)])
            t_end = np.array([rng.integers(LS, T + 1)])
            xa, _, _ = long_inputs(p["arr"], p["wk"], d, t_end + H, 0, rng, with_age=False)   # A ends at t_end - 1
            s0 = rng.integers(0, S - cnn.CROP + 1) if S > cnn.CROP else 0
            xs.append(crop(xa[0], slice(s0, s0 + cnn.CROP), S))
        x = tt(np.stack(xs))
        ok = x[:, 2:3]
        losses = []
        for enc, kind in ((encS, "S"), (encT, "T")):
            if kind == "S":
                hid = (torch.rand(x.shape[0], 1, 1, x.shape[-1], device=DEV) < 0.25).expand(-1, 1, LS, -1)
            else:
                patch = torch.rand(x.shape[0], LS // 12, device=DEV) < 0.25
                hid = patch.repeat_interleave(12, 1)[:, None, :, None].expand(-1, 1, -1, x.shape[-1])
            xin = x.clone()
            xin[:, 0:4] = xin[:, 0:4] * (~hid).float()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                rec = enc.rec(enc.feats(xin)).float()
            w = ok * hid.float()
            losses.append((rec - x[:, 0:2]).abs().mul(w).sum() / w.sum().clamp(min=1) / 2)
        loss = losses[0] + losses[1]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        run[0] += losses[0].item(); run[1] += losses[1].item()
        if step % 1000 == 0:
            print(f"pretrain {step}/{args.steps}: S-mask {run[0] / 1000:.4f}  T-mask {run[1] / 1000:.4f} "
                  f"({time.time() - t_start:.0f}s)", flush=True)
            run = [0.0, 0.0]
    torch.save({"encS": encS.state_dict(), "encT": encT.state_dict()}, args.out)
    print(f"saved {args.out}")


# ------------------------------------------------------------------------------------------------ stage: train
def stage_train(args):
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    parts = train_panels()
    for p in parts:
        p["tr"] = np.flatnonzero(~p["es"])
        es = np.flatnonzero(p["es"])[:400]
        p["es_idx"] = es
    net = LCNet(load_short(args.ckpt)).to(DEV)
    if args.enc:
        e = torch.load(args.enc, map_location=DEV)
        net.encS.load_state_dict(e["encS"]); net.encT.load_state_dict(e["encT"])
    long_on = not args.no_long
    short_p = list(net.short.parameters())
    long_p = list(net.encS.parameters()) + list(net.encT.parameters()) + list(net.fuse.parameters())
    groups = [{"params": short_p, "lr": args.lr_short}] + ([{"params": long_p, "lr": args.lr_long}] if long_on else [])
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], total_steps=args.steps,
                                                pct_start=0.05)
    es_data = []
    for p in parts:
        d, t0 = p["d"][p["es_idx"]], p["t0"][p["es_idx"]]
        fi = t0[:, None] + np.arange(1, F + 1)
        lo = long_inputs(p["arr"], p["wk"], d, t0) if long_on else None
        es_data.append((short_x(p, d, t0), lo, p["q"][d[:, None], fi].astype(np.float32),
                        p["el"][d[:, None], fi].astype(np.float32)))
    best, best_state, t_start = 1e9, None, time.time()
    for step in range(1, args.steps + 1):
        net.train()
        x, (xa, xb, ag), y, m = train_batch(parts, args.bs, rng, long_on)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = net(x, xa, xb, ag)
        logit = out.float()
        loss = cnn.loss_fn(logit, y, m) + cnn.LOVASZ * cnn.lovasz_hinge(logit, y, m)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 500 == 0:
            tot, n = 0.0, 0.0
            for xs, lo, yv, mv in es_data:
                pr = predict(net, xs, lo)
                bce = -(yv * np.log(pr + 1e-6) + (1 - yv) * np.log(1 - pr + 1e-6))
                tot += float((bce * mv).sum()); n += float(mv.sum())
            es_loss = tot / max(n, 1)
            flag = ""
            if es_loss < best:
                best, flag = es_loss, " *"
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            print(f"step {step}: train {loss.item():.4f}  ES bce {es_loss:.5f}{flag} ({time.time() - t_start:.0f}s)",
                  flush=True)
    torch.save({"state": best_state, "long": long_on}, MODELS / f"lc_{args.tag}.pt")
    print(f"saved lc_{args.tag}.pt (ES bce {best:.5f})")


@torch.no_grad()
def predict(net, x, lo, bs=48):
    net.eval()
    out = []
    for a in range(0, len(x), bs):
        args = (tt(x[a:a + bs]),) + ((tt(lo[0][a:a + bs]), tt(lo[1][a:a + bs]), tt(lo[2][a:a + bs]))
                                     if lo is not None else ())
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(torch.sigmoid(net(*args).float()).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, F, x.shape[-1]), np.float32)


# ------------------------------------------------------------------------------------------------ stage: lab
def load_lc(path):
    d = torch.load(path, map_location=DEV)
    st = d["state"]
    short = cnn.T2Net(st["short.inp.weight"].shape[1], st["short.inp.weight"].shape[0]).to(DEV)
    net = LCNet(short).to(DEV)
    net.load_state_dict(st)
    return net.eval(), d["long"]


def sup_batch(ctx, stat, z, arr, pool, days, cuts, bs, rng, long_on):
    ov, oq, ok, q = pool
    cand = []
    for _ in range(bs * 8):
        d = int(days[rng.integers(len(days))])
        lim = min(cuts.get(d, 10 ** 6), T - 1)
        if lim <= H + F + 1:
            continue
        t0 = int(rng.integers(H + 1, lim - F))
        if q[d, t0 - H:t0].sum() >= 2 and q[d, t0 + 1:t0 + 1 + F].any():
            cand.append((d, t0))
        if len(cand) >= bs:
            break
    if not cand:
        return None
    d = np.array([c[0] for c in cand]); t0 = np.array([c[1] for c in cand])
    hi, fi = t0[:, None] + np.arange(-H, 0), t0[:, None] + np.arange(1, F + 1)
    x = cnn.make_x(ctx, stat, ov[d[:, None], hi], oq[d[:, None], hi], ok[d[:, None], hi], t0, z["weekday"][d] >= 5)
    lo = long_inputs(arr, z["weekday"], d, t0) if long_on else None
    return x, lo, q[d[:, None], fi].astype(np.float32), ok[d[:, None], fi].astype(np.float32)


def adapt(net, batches, rec_batches, steps, lr, w_rec):
    s = net.short
    params = list(s.inp.parameters()) + list(s.blocks.parameters()) + list(s.rec.parameters()) + \
        list(net.fuse.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    net.train()
    for _ in range(steps):
        b = batches()
        if b is None:
            break
        x, lo, y, m = b
        args = (tt(x),) + ((tt(lo[0]), tt(lo[1]), tt(lo[2])) if lo is not None else ())
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = net(*args)
        logit = out.float()
        yt, mt = tt(y), tt(m)
        loss = cnn.loss_fn(logit, yt, mt) + cnn.LOVASZ * cnn.lovasz_hinge(logit, yt, mt)
        if w_rec:
            xb = rec_batches()
            if xb is not None:
                loss = loss + w_rec * C.ssl_loss(s, xb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    net.eval()


def stage_lab(args):
    t_start = time.time()
    res = {}
    for i, panel in enumerate(C.T2_PANELS):
        ctx, stat, W = C.windows(i, panel)
        for s in C.SPLITS:
            z, og, on = W[s]["z"], W[s]["og"], W[s]["on"]
            ov, oq = z["obs_speed"], z["obs_flow"]
            ok = z["elig"] & np.isfinite(ov) & np.isfinite(oq)
            q = ok & (np.nan_to_num(ov, nan=1e9) <= 0.6 * ctx.vf)
            pool = (ov, oq, ok, q)
            arr = arrays(ov, oq, ok, ctx.vf, ctx.cap)
            days = np.unique(np.concatenate([on["d"], og["d"]]))
            firsts = {int(d): int(min(on["t0"][on["d"] == d].min(initial=10 ** 6),
                                      og["t0"][og["d"] == d].min(initial=10 ** 6))) for d in days}
            for mode in args.modes:
                net, long_on = load_lc(args.model)
                rng = np.random.default_rng(7 + i)
                p_on = np.zeros((len(on["d"]), ctx.S), np.float32)
                p_og = np.zeros(og["fut"].shape, np.float32)
                for d in days:
                    if mode != "none":
                        past = np.arange(d)
                        first = firsts[int(d)]
                        pool_days = np.append(past, d) if first > H + F + 2 else past
                        cuts = {int(k): (first if int(k) == d else 10 ** 6) for k in pool_days}
                        if len(pool_days):
                            adapt(net, lambda: sup_batch(ctx, stat, z, arr, pool, pool_days, cuts, args.bs, rng,
                                                         long_on),
                                  lambda: C.past_batch(ctx, stat, z, past, args.bs, rng),
                                  args.steps, args.lr, args.w_rec if "rec" in mode else 0.0)
                    for W_, key in ((on, "on"), (og, "og")):
                        idx = np.flatnonzero(W_["d"] == d)
                        if not len(idx):
                            continue
                        x = C.make_x(ctx, stat, W_, idx, z)
                        lo = long_inputs(arr, z["weekday"], W_["d"][idx], W_["t0"][idx]) if long_on else None
                        pr = predict(net, x, lo)
                        if key == "on":
                            p_on[idx] = pr[:, F - 1]
                        else:
                            p_og[idx] = pr
                a = res.setdefault((s, mode), {})
                a[panel] = (C.onset_iou(p_on, on).mean(), C.ongoing_iou(p_og, og, 0.5).mean())
                np.savez(CACHE / f"fam_lc_{args.tag}_{mode}_{s}_{panel}.npz", og=p_og.astype(np.float16),
                         on=p_on.astype(np.float32))
            msg = "  ".join(f"{m} og {res[(s, m)][panel][1]:.4f}" for m in args.modes)
            print(f"  {panel} {s}: {msg} ({time.time() - t_start:.0f}s)", flush=True)
    for s in C.SPLITS:
        print(f"\n=== {s}: lc {args.tag} ===")
        for m in args.modes:
            d = res[(s, m)]
            print(f"  {m:<8} onset {C.family_mean({k: v[0] for k, v in d.items()}):.4f}  "
                  f"ongoing {C.family_mean({k: v[1] for k, v in d.items()}):.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("pretrain", "train", "lab"))
    ap.add_argument("--ckpt", type=Path)
    ap.add_argument("--enc", type=Path)
    ap.add_argument("--model", type=Path)
    ap.add_argument("--out", type=Path, default=MODELS / "lc_enc.pt")
    ap.add_argument("--tag", default="s20")
    ap.add_argument("--no-long", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--bs", type=int, default=None)
    ap.add_argument("--lr-short", type=float, default=1e-4)
    ap.add_argument("--lr-long", type=float, default=5e-4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--w-rec", type=float, default=1.0)
    ap.add_argument("--modes", nargs="*", default=["none", "sup+rec"])
    args = ap.parse_args()
    if args.stage == "pretrain":
        args.steps, args.bs = args.steps or 20000, args.bs or 64
        stage_pretrain(args)
    elif args.stage == "train":
        args.steps, args.bs = args.steps or 8000, args.bs or 32
        stage_train(args)
    else:
        args.steps, args.bs = args.steps or 30, args.bs or 16
        stage_lab(args)


if __name__ == "__main__":
    main()
