"""Shared pieces of the test-time-adaptation lab (experiments A-F, see ../EXPERIMENTS.md).

Everything here imports the project's own modules (analysis/, scripts/, tfb/) read-only; nothing in the project is
modified. The kit lives at <project>/tta_lab, so the project root is two levels above this file.

Contents
    windows(panel)          the blank-aware official-like onset / ongoing windows of both scored months, plus every
                            cached probability the reference blends need
    build_net(...)          the reconstruction-head CNN with one of the parameter-efficient adaptation modes (C)
    ssl_loss(...)           masked reconstruction + next-step + delta losses (D)
    ssl_score(...)          per-window self-supervised loss with fixed masks (the gate of F)
    past_batch / adapt      the causal past-day adaptation of idea 9b
    evaluate / report       official-like IoU, family means, paired bootstrap CIs against the production reference
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("T2CNN_REC", "0.3")      # T2Net gets its reconstruction head only when this is set at import
os.environ.setdefault("T2CNN_LOVASZ", "0.5")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

KIT = Path(__file__).resolve().parents[1]
PROJ = KIT.parent
for p in (PROJ, PROJ / "analysis", PROJ / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import t2_cnn as cnn  # noqa: E402
from t2_eda02 import T2_PANELS  # noqa: E402
from t2_eda03 import F  # noqa: E402
from t2_expected_iou import select  # noqa: E402
from t2_first_onset_eval import first_onsets, official_blank  # noqa: E402
from t2_m1_models import Ctx  # noqa: E402
from t2_official_like_eval import ongoing_official  # noqa: E402
from t2_pseudo_eval import iou  # noqa: E402
from tfb.data import CACHE, load  # noqa: E402
from tfb.physics import load_ramps  # noqa: E402
from tfb.t1eval import family_mean  # noqa: E402

H = cnn.H
DEV = cnn.DEV
ASSETS = KIT / "assets"
RUNS = KIT / "runs"
MODELS = PROJ / "work" / "models"
SPLITS = ("validation", "private")
CNN3 = ["t2_cnn_lov", "t2_cnn_lov_s1", "t2_cnn_lov_s2"]
TFM8 = [f"t2_cnn_lov_tfm_s{s}" for s in range(3, 11)]
AD9B = "t2_cnn_lov_adapt"          # idea 9b as shipped in candidate T (seed 20, 30 steps, lr 1e-4)
W_ON = 0.7                         # onset blend: 0.7 LightGBM H + 0.3 CNN
OG_W = (0.20, 0.60, 0.20)          # ongoing blend of candidate T: gs1, tfm x8, CNN
OG_TAU = 0.50


# ------------------------------------------------------------------------------------------------ windows / caches
def _cnn_cache(tag, split, panel, key):
    return np.load(CACHE / f"t2cnn_{tag}_{split}_{panel}.npz")[key].astype(np.float32)


def windows(i, panel, need_train=False):
    """ctx, static channels, and {split: dict} with z, on, og and the reference probabilities."""
    ctx = Ctx(panel, i)
    ctx.fitd = np.ones_like(ctx.fitd)
    ctx.loff = 0
    stat = cnn.static_channels(ctx, load_ramps(panel, "train", ctx.z))
    out = {}
    for split in SPLITS:
        z = load(panel, split)
        blank = official_blank(panel, split, z)
        on = first_onsets(ctx.vf, z, blank=blank)
        og, _ = ongoing_official(ctx, z, blank)
        c = np.load(CACHE / f"t2probs_{split}_{panel}.npz")
        assert (c["on_d"] == on["d"]).all() and (c["og_d"] == og["d"]).all(), "window sets differ from the cache"
        g = np.load(ASSETS / "cache" / f"t2gs_{split}_{panel}.npz")
        ref = dict(
            on_H=c["on_p"].astype(np.float32),
            on_cnn3=np.mean([_cnn_cache(t, split, panel, "on_p") for t in CNN3], 0),
            on_ad9b=_cnn_cache(AD9B, split, panel, "on_p"),
            og_gs1=g["og_p1"].astype(np.float32),
            og_tfm8=np.mean([_cnn_cache(t, split, panel, "og_p") for t in TFM8], 0),
            og_cnn3=np.mean([_cnn_cache(t, split, panel, "og_p") for t in CNN3], 0),
            og_ad9b=_cnn_cache(AD9B, split, panel, "og_p"),
        )
        out[split] = dict(z=z, on=on, og=og, ref=ref)
    if not need_train:
        ctx.z = {k: ctx.z[k] for k in ("link_ids", "weekday", "dates")}
        ctx.q = ctx.el = None
    return ctx, stat, out


def make_x(ctx, stat, w, m, z):
    """CNN input of the windows w[...][m]."""
    return cnn.make_x(ctx, stat, w["v"][m], w["qf"][m], w["ok"][m], w["t0"][m], z["weekday"][w["d"][m]] >= 5)


def to_dev(x):
    return torch.from_numpy(np.ascontiguousarray(x)).to(DEV, torch.float32)


# ------------------------------------------------------------------------------------------------ PEFT modes (C)
class LoRAConv(nn.Module):
    """W' = W + B A for a (dilated) 3x3 conv: A has the conv geometry and rank r, B is 1x1 and starts at zero."""
    def __init__(self, conv: nn.Conv2d, r: int):
        super().__init__()
        self.base = conv
        self.A = nn.Conv2d(conv.in_channels, r, conv.kernel_size, padding=conv.padding, dilation=conv.dilation,
                           bias=False)
        self.B = nn.Conv2d(r, conv.out_channels, 1, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        return self.base(x) + self.B(self.A(x))


class FiLM(nn.Module):
    """h' = gamma * block(h) + beta, per channel, initialised at the identity."""
    def __init__(self, block, c):
        super().__init__()
        self.block = block
        self.g = nn.Parameter(torch.ones(1, c, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        return self.g * self.block(x) + self.b


class Adapter(nn.Module):
    """h' = block(h) + up(gelu(down(block(h)))), bottleneck k, up starts at zero."""
    def __init__(self, block, c, k=16):
        super().__init__()
        self.block = block
        self.down, self.up = nn.Conv2d(c, k, 1), nn.Conv2d(k, c, 1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        h = self.block(x)
        return h + self.up(nn.functional.gelu(self.down(h)))


class InputCal(nn.Module):
    """Input calibration: v/vf -> a0 * v/vf + b0 and q/cap -> a1 * q/cap + b1 on observed cells only
    (the 'speed scale / flow scale of this run' in its most literal form)."""
    def __init__(self, inp):
        super().__init__()
        self.inp = inp
        self.a = nn.Parameter(torch.ones(2))
        self.b = nn.Parameter(torch.zeros(2))

    def forward(self, x):
        ok = x[:, 2:3]
        cal = (x[:, 0:2] * self.a.view(1, 2, 1, 1) + self.b.view(1, 2, 1, 1)) * ok
        return self.inp(torch.cat([cal.to(x.dtype), x[:, 2:]], 1))


MODES = ("full", "last2", "norm", "film", "lora", "adapter", "inp", "incal")


def build_net(ckpt, mode="full", rank=4, train_rec=None):
    """The reconstruction-head T2Net with the chosen trainable subset. Returns (net, trainable params).
    Only inp and blocks change the Task 2 output (the rec head does not), so the rec head is trainable only in
    'full' by default (as in idea 9b); elsewhere it would just absorb the self-supervised loss."""
    sd = torch.load(ckpt, map_location=DEV)
    net = cnn.T2Net(sd["inp.weight"].shape[1], cnn.WIDTH).to(DEV)
    net.load_state_dict(sd)
    for p in net.parameters():
        p.requires_grad_(False)
    c = cnn.WIDTH
    train = []
    if mode == "full":
        train = list(net.inp.parameters()) + list(net.blocks.parameters())
    elif mode == "last2":
        train = list(net.blocks[6:].parameters())
    elif mode == "norm":
        train = [p for m in net.blocks.modules() if isinstance(m, nn.GroupNorm) for p in m.parameters()]
    elif mode == "film":
        net.blocks = nn.Sequential(*[FiLM(b, c) for b in net.blocks]).to(DEV)
        train = [p for b in net.blocks for p in (b.g, b.b)]
    elif mode == "lora":
        for b in net.blocks:
            for j in (2, 5):
                b.net[j] = LoRAConv(b.net[j], rank).to(DEV)
        train = [p for b in net.blocks for j in (2, 5) for p in (b.net[j].A.weight, b.net[j].B.weight)]
    elif mode == "adapter":
        net.blocks = nn.Sequential(*[Adapter(b, c) for b in net.blocks]).to(DEV)
        train = [p for b in net.blocks for p in list(b.down.parameters()) + list(b.up.parameters())]
    elif mode == "inp":
        train = list(net.inp.parameters())
    elif mode == "incal":
        net.inp = InputCal(net.inp).to(DEV)
        train = [net.inp.a, net.inp.b]
    else:
        raise ValueError(mode)
    if train_rec if train_rec is not None else mode == "full":
        train += list(net.rec.parameters())
    for p in train:
        p.requires_grad_(True)
    net.eval()
    return net, train


def snapshot(params):
    return [p.detach().clone() for p in params]


def restore(params, snap):
    with torch.no_grad():
        for p, s in zip(params, snap):
            p.copy_(s)


# ------------------------------------------------------------------------------------------------ losses (D)
def rec_out(net, xin):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return net.rec(net.blocks(net.inp(xin))).float()


def ssl_loss(net, xb, w_mask=1.0, w_next=0.0, w_delta=0.0, k_tail=1, gen=None):
    """L = w_mask * L_mask + w_next * L_nextstep + w_delta * L_delta on (B, C, H, S) inputs.
    L_mask     hide 20% of the cells, reconstruct v/vf and q/cap there (idea 9b)
    L_next     hide the last k_tail history slots entirely, reconstruct them from the earlier slots
    L_delta    on that same input: (x^_t - x^_{t-1}) against (x_t - x_{t-1}) for the hidden tail slots"""
    ok = (xb[:, 2] > 0.5).float().unsqueeze(1)
    tgt = xb[:, 0:2].float()
    loss = xb.new_zeros(())
    if w_mask:
        hid = torch.rand(xb[:, 2].shape, device=xb.device, generator=gen) < 0.2
        xin = xb.clone()
        xin[:, 0:4] = xin[:, 0:4] * (~hid).unsqueeze(1).float()
        w = ok * hid.unsqueeze(1).float()
        loss = loss + w_mask * ((rec_out(net, xin) - tgt) ** 2 * w).sum() / w.sum().clamp(min=1)
    if w_next or w_delta:
        k = k_tail
        xin = xb.clone()
        xin[:, 0:4, -k:] = 0
        rec = rec_out(net, xin)
        if w_next:
            w = ok[:, :, -k:]
            loss = loss + w_next * ((rec[:, :, -k:] - tgt[:, :, -k:]) ** 2 * w).sum() / w.sum().clamp(min=1)
        if w_delta:
            dp = rec[:, :, -k:] - rec[:, :, -k - 1:-1]
            dt = tgt[:, :, -k:] - tgt[:, :, -k - 1:-1]
            w = ok[:, :, -k:] * ok[:, :, -k - 1:-1]
            loss = loss + w_delta * ((dp - dt) ** 2 * w).sum() / w.sum().clamp(min=1)
    return loss


@torch.no_grad()
def ssl_score(net, x, n_masks=4, seed=12345):
    """(N,) masked-reconstruction loss of each window with FIXED masks, so different models are comparable (F)."""
    out = []
    for j in range(0, len(x), 64):
        xb = to_dev(x[j:j + 64])
        gen = torch.Generator(device=DEV).manual_seed(seed)
        ok = (xb[:, 2] > 0.5).float().unsqueeze(1)
        num = torch.zeros(len(xb), device=DEV)
        den = torch.zeros(len(xb), device=DEV)
        for _ in range(n_masks):
            hid = torch.rand(xb[:, 2].shape, device=DEV, generator=gen) < 0.2
            xin = xb.clone()
            xin[:, 0:4] = xin[:, 0:4] * (~hid).unsqueeze(1).float()
            w = ok * hid.unsqueeze(1).float()
            num += ((rec_out(net, xin) - xb[:, 0:2]) ** 2 * w).sum((1, 2, 3))
            den += w.sum((1, 2, 3))
        out.append((num / den.clamp(min=1)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, np.float32)


# ------------------------------------------------------------------------------------------------ adaptation
def past_batch(ctx, stat, z, days_pool, bs, rng):
    """A batch of history windows from the given (strictly earlier) days of the released layer; no label used."""
    if len(days_pool) == 0:
        return None
    days = rng.choice(days_pool, size=bs)
    t0 = rng.integers(H + 1, z["obs_speed"].shape[1] - F - 1, size=bs)
    hi = t0[:, None] + np.arange(-H, 0)
    v, q = z["obs_speed"][days[:, None], hi], z["obs_flow"][days[:, None], hi]
    ok = z["elig"][days[:, None], hi] & np.isfinite(v) & np.isfinite(q)
    return to_dev(cnn.make_x(ctx, stat, v, q, ok, t0, z["weekday"][days] >= 5))


def day_batch(ctx, stat, z, day, t_max, bs, rng):
    """History windows from EARLIER TODAY: origins t0 with t0 < t_max, so every cell is before t_max - H.
    The organizers confirmed (2026-09-20) that data timestamped at or before the window origin may be used."""
    if t_max <= H + 1:
        return None
    t0 = rng.integers(H + 1, t_max, size=bs)
    hi = t0[:, None] + np.arange(-H, 0)
    d = np.full(bs, day)
    v, q = z["obs_speed"][d[:, None], hi], z["obs_flow"][d[:, None], hi]
    ok = z["elig"][d[:, None], hi] & np.isfinite(v) & np.isfinite(q)
    return to_dev(cnn.make_x(ctx, stat, v, q, ok, t0, z["weekday"][d] >= 5))


def make_opt(params, lr, opt="adamw"):
    if opt == "sgd":
        return torch.optim.SGD(params, lr=lr)
    return torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)


def adapt(net, params, batches, steps, lr, opt="adamw", losses=(1.0, 0.0, 0.0), k_tail=1):
    """steps optimiser steps; batches is a callable returning a fresh (B, C, H, S) tensor (or a fixed tensor)."""
    o = make_opt(params, lr, opt)
    net.train()
    last = float("nan")
    for _ in range(steps):
        xb = batches() if callable(batches) else batches
        if xb is None:
            break
        loss = ssl_loss(net, xb, *losses, k_tail=k_tail)
        o.zero_grad(set_to_none=True)
        loss.backward()
        o.step()
        last = float(loss.item())
    net.eval()
    return last


# ------------------------------------------------------------------------------------------------ evaluation
def onset_iou(p, on):
    pr = np.zeros(on["fut"].shape, bool)
    if len(p):
        pr[:, F - 1] = np.stack([select(q, "ratio", 1.0) for q in p])
    return iou(pr, on["fut"], on["efut"])


def ongoing_iou(p, og, tau):
    return iou(p >= tau, og["fut"], og["efut"])


def blends(ref, on_p=None, og_p=None):
    """Named (kind, probability) variants of one run, next to the production reference of candidate T."""
    out = {}
    if on_p is not None:
        out["onset alone"] = on_p
        out["onset T-blend"] = W_ON * ref["on_H"] + (1 - W_ON) * on_p            # replaces the 9b CNN of T
    if og_p is not None:
        out["ongoing alone"] = og_p
        out["ongoing T-blend (x replaces cnn3)"] = OG_W[0] * ref["og_gs1"] + OG_W[1] * ref["og_tfm8"] + OG_W[2] * og_p
    return out


def reference(ref):
    return {"onset alone": ref["on_ad9b"],
            "onset T-blend": W_ON * ref["on_H"] + (1 - W_ON) * ref["on_ad9b"],
            "ongoing alone": ref["og_ad9b"],
            "ongoing T-blend (x replaces cnn3)": OG_W[0] * ref["og_gs1"] + OG_W[1] * ref["og_tfm8"] + OG_W[2] * ref["og_cnn3"]}


def score(name, p, w, kind_tau=None):
    if name.startswith("onset"):
        return onset_iou(p, w)
    return ongoing_iou(p, w, OG_TAU if kind_tau is None else kind_tau)


_boot = np.random.default_rng(7)


def ci(d, n=2000):
    idx = _boot.integers(0, len(d), (n, len(d)))
    m = d[idx].mean(1)
    return f"{d.mean():+.4f} [{np.percentile(m, 2.5):+.4f}, {np.percentile(m, 97.5):+.4f}]"


def save_run(run, split, panel, **arrays):
    d = RUNS / run
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / f"probs_{split}_{panel}.npz", **arrays)


def load_run(run, split, panel):
    f = RUNS / run / f"probs_{split}_{panel}.npz"
    return np.load(f) if f.exists() else None


def report(runs, tau=None, file=None):
    """Official-like table of saved runs against candidate T's reference (idea 9b in the same slot).
    'delta' is the paired per-window difference to the reference, with a 95% bootstrap CI; 'fam' is the family
    mean over panels, which is what the official score averages (0.15 x fam delta = total delta)."""
    lines = []
    acc: dict = {}
    for i, panel in enumerate(T2_PANELS):
        ctx, stat, W = windows(i, panel)
        for split in SPLITS:
            w = W[split]
            ref = reference(w["ref"])
            for kind, p in ref.items():
                s = score(kind, p, w["on"] if kind.startswith("onset") else w["og"], tau)
                acc.setdefault((split, kind, "REF (candidate T)"), []).append((panel, s))
            for run in runs:
                r = load_run(run, split, panel)
                if r is None:
                    continue
                bl = blends(w["ref"], r["on_p"] if "on_p" in r and len(r["on_p"]) == len(w["on"]["d"]) else None,
                            r["og_p"] if "og_p" in r and len(r["og_p"]) == len(w["og"]["d"]) else None)
                for kind, p in bl.items():
                    s = score(kind, p, w["on"] if kind.startswith("onset") else w["og"], tau)
                    acc.setdefault((split, kind, run), []).append((panel, s))
        print(f"  scored {panel}", flush=True)
    for split in SPLITS:
        for kind in ("onset alone", "onset T-blend", "ongoing alone", "ongoing T-blend (x replaces cnn3)"):
            base = acc.get((split, kind, "REF (candidate T)"))
            if not base:
                continue
            bcat = np.concatenate([s for _, s in base])
            lines.append(f"\n=== {split} | {kind} | {len(bcat)} windows ===")
            for run in ["REF (candidate T)"] + list(runs):
                v = acc.get((split, kind, run))
                if not v or len(v) != len(base):
                    continue
                cat = np.concatenate([s for _, s in v])
                fam = family_mean({p: s.mean() for p, s in v if len(s)})
                famb = family_mean({p: s.mean() for p, s in base if len(s)})
                lines.append(f"  {run:<34} IoU {cat.mean():.4f}  fam {fam:.4f} ({fam - famb:+.4f})  "
                             f"delta {ci(cat - bcat)}")
    txt = "\n".join(lines)
    print(txt)
    if file:
        Path(file).write_text(txt + "\n", encoding="utf-8")
    return acc
