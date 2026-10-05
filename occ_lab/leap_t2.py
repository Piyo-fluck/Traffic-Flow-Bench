"""LEAP ClimSim 1st-place tricks (Kaggle 2024) on the SSL w192 Task 2 CNN: multiple input representations and a
confidence head.

    multirep   the same history in several representations, concatenated before the input conv (ASLFR / LEAP):
               speed km/h / 100, flow per lane / 2000, density proxy q/cap / max(v/vf, 0.05) (= k / k_crit on a
               triangular FD), log1p(flow per lane) / 8, per-link standardisation of v/vf and q/cap over the hour
               (RevIN-style change from the window's own level), first differences in time of v/vf and q/cap;
               all masked by the observed flag. The new input weights start at zero, so training starts exactly
               at the checkpoint.
    conf       a second head predicting each cell's own BCE (detached), MAE loss, weight 1; training only.
    control    the same fine-tuning without either.

Fine-tuning: whole panels, bs 16, 6k steps, BCE + 0.5 Lovasz, lr 1e-4 (1e-3 for the new parts), ES on the train
tail. Lab: note 106's causal per-day sup+rec adaptation, ongoing / onset IoU, all days and days 1-5 (d <= 4).

    python occ_lab/leap_t2.py --stage train --ckpt work/models/t2_cnn_lov_w192_rec_ssl30k_s20.pt --multirep --conf --tag both_s20
    python occ_lab/leap_t2.py --stage lab --tag both_s20
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
import longctx as LC  # noqa: E402  (T2CNN environment: CNN, rec head, Lovasz, width 192)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import pseudo_adapt as PA  # noqa: E402

C, cnn = LC.C, LC.cnn
H, F, DEV = LC.H, LC.F, LC.DEV
CACHE, MODELS = LC.CACHE, LC.MODELS
NREP = 8


class MultiRep(nn.Module):
    """x (B, 12, H, S): v/vf, q/cap, observed, queued, lanes/5, cap/lane/2500, vf/120, on, off, sin, cos, weekend."""
    def forward(self, x):
        vr, qc, ok = x[:, 0], x[:, 1], x[:, 2]
        lanes = (x[:, 4] * 5).clamp(min=1)
        cpl = x[:, 5] * 2500
        vf = x[:, 6] * 120
        qpl = qc * cpl                                         # veh/h/lane (cap = cpl * lanes)
        rep = [vr * vf / 100, qpl / 2000, qc / vr.clamp(min=0.05) / 4, torch.log1p(qpl.clamp(min=0)) / 8]
        n = ok.sum(1, keepdim=True).clamp(min=1)
        for a in (vr, qc):
            mu = (a * ok).sum(1, keepdim=True) / n
            sd = (((a - mu) ** 2 * ok).sum(1, keepdim=True) / n).sqrt().clamp(min=0.02)
            rep.append((a - mu) / sd / 4)
        for a in (vr, qc):
            rep.append(torch.diff(a, dim=1, prepend=a[:, :1]) * 4)
        R = torch.stack(rep, 1) * ok.unsqueeze(1)
        return torch.cat([x, R.to(x.dtype)], 1)


class Flip(nn.Module):
    """Reverses the link axis when the current panel runs downstream -> upstream (S / W panels, note 49)."""
    def __init__(self, state):
        super().__init__()
        self.state = state

    def forward(self, x):
        return x.flip(-1) if self.state["flip"] else x


FLIP_PANEL = lambda panel: panel.endswith(("_S", "_W"))  # noqa: E731


class LeapNet(nn.Module):
    """T2Net with optional multirep input and a confidence head; exposes inp / blocks / rec for the adaptation kit."""
    def __init__(self, short, multirep, conf, orient=False):
        super().__init__()
        c = short.inp.weight.shape[0]
        self.orient = {"flip": False}
        self.hyper = None                                          # optional E modules (instance attributes, so
        self.moe = None                                            # nn.Module registers them when assigned)
        self.rctx = None
        if multirep:
            conv = nn.Conv2d(short.inp.in_channels + NREP, c, 1)
            with torch.no_grad():
                conv.weight.zero_()
                conv.weight[:, :short.inp.in_channels] = short.inp.weight
                conv.bias.copy_(short.inp.bias)
            self.inp = nn.Sequential(MultiRep(), conv)
        else:
            self.inp = short.inp
        self.blocks, self.collapse, self.head, self.rec = short.blocks, short.collapse, short.head, short.rec
        self.conf = nn.Sequential(nn.GELU(), nn.Conv2d(c, F, 1)) if conf else None
        self.flip = Flip(self.orient) if orient else None
        if orient:                                             # every panel seen upstream -> downstream
            self.inp = nn.Sequential(Flip(self.orient), self.inp)
            self.rec = nn.Sequential(self.rec, Flip(self.orient))

    def set_panel(self, panel):
        self.orient["flip"] = FLIP_PANEL(panel)

    def features(self, x):
        return self.collapse(self.blocks(self.inp(x)))

    def forward(self, x, lid=None):
        h = self.features(x)
        if self.hyper is not None and self.rctx is not None:
            h = self.hyper(h, self.rctx[:len(x)])
        out = self.head(h)[:, :F, 0]
        if self.moe is not None:
            out = self.moe(x, out)
        return self.flip(out) if self.flip is not None else out


# ------------------------------------------------------------------------------------------------ boundary losses
KB = 8


def _edges(q):
    """(B, F, S) 0/1 -> cells where the queue flag changes against a link neighbour (both sides of the edge)."""
    e = torch.zeros_like(q)
    d = (q[..., 1:] != q[..., :-1]).float()
    e[..., 1:] = torch.maximum(e[..., 1:], d)
    e[..., :-1] = torch.maximum(e[..., :-1], d)
    return e


def _dilate(b, k):
    B, Fh, S = b.shape
    return (nn.functional.max_pool1d(b.reshape(B * Fh, 1, S), 2 * k + 1, 1, k).reshape(B, Fh, S) > 0).float()


def edge_distance(q):
    """Distance in links to the nearest queue edge of the same step, capped at KB + 1."""
    e = _edges(q)
    dist = torch.full_like(q, KB + 1.0)
    for k in range(KB, -1, -1):
        dist = torch.where(_dilate(e, k) > 0, torch.full_like(dist, float(k)), dist)
    return dist


def history_edge_band(x, k=2):
    """The edge of the last observed queue in the history (known at inference), broadcast over the horizon."""
    ok, qd = x[:, 2] > 0.5, x[:, 3] > 0.5                       # (B, H, S)
    Hh = ok.shape[1]
    idx = torch.where(ok, torch.arange(Hh, device=x.device)[None, :, None], torch.full_like(ok, -1, dtype=torch.long)).max(1).values
    last = torch.gather(qd.float(), 1, idx.clamp(min=0)[:, None, :])[:, 0] * (idx >= 0).float()   # (B, S)
    e = _edges(last[:, None, :].expand(-1, F, -1).contiguous())
    return _dilate(e, k)


def boundary_extra(kind, logit, y, m, x, step_frac):
    """Extra loss terms; kind in {sdf, band, blov, band+sdf}."""
    out = logit.new_zeros(())
    yb = (y > 0.5).float()
    if "band" in kind or kind == "blov":
        band = torch.clamp(_dilate(_edges(yb), 2) + history_edge_band(x), max=1.0) * m
    if "band" in kind:                                           # BCE re-weighted x3 on the edge band
        bce = nn.functional.binary_cross_entropy_with_logits(logit, y, reduction="none")
        out = out + 2.0 * (bce * band).sum() / m.sum().clamp(min=1)
    if kind == "blov":                                           # Lovasz on the band cells only
        out = out + 0.5 * cnn.lovasz_hinge(logit, yb, band)
    if "sdf" in kind:                                            # Kervadec boundary loss, weight ramped 0.01 -> 0.3
        phi = torch.where(yb > 0.5, -edge_distance(yb), edge_distance(yb)) / (KB + 1)
        alpha = 0.01 + 0.29 * step_frac
        out = out + alpha * (torch.sigmoid(logit) * phi * m).sum() / m.sum().clamp(min=1)
    return out


# ------------------------------------------------------------------------------------------------ E: shift-robust
def time_warp(x, a):
    """Monotone warp of the history axis with both ends fixed: u(t) = t + a t (H-1-t) / (H-1), per window a."""
    B, Cc, Hh, S = x.shape
    t = torch.arange(Hh, device=x.device, dtype=torch.float32)[None, :]
    u = (t + a[:, None] * t * (Hh - 1 - t) / (Hh - 1)).clamp(0, Hh - 1)          # (B, H)
    lo = u.floor().long(); hi = (lo + 1).clamp(max=Hh - 1); w = (u - lo.float())[:, None, :, None]
    gl = lambda idx: torch.gather(x[:, :4], 2, idx[:, None, :, None].expand(B, 4, Hh, S))  # noqa: E731
    dyn = (1 - w) * gl(lo) + w * gl(hi)
    dyn = torch.cat([dyn[:, :2], (dyn[:, 2:3] > 0.5).float(), (dyn[:, 3:4] > 0.5).float()], 1)
    return torch.cat([dyn, x[:, 4:]], 1)


def run_context(z_v, z_ok, vf, d, t0, H_=12):
    """Per window, 4 run-level scalars from released data before the history: the panel's queued share and mean
    v/vf over the 3 h before the history on the same day, and over the same clock hours on the previous day."""
    out = np.zeros((len(d), 4), np.float32)
    for n, (dd, tt) in enumerate(zip(d, t0)):
        for k, (day, a, b) in enumerate(((dd, tt - H_ - 36, tt - H_), (dd - 1, tt - H_ - 36, tt + 6))):
            if day < 0 or b <= 0:
                continue
            a = max(a, 0)
            v, ok = z_v[day, a:b], z_ok[day, a:b]
            if ok.sum() == 0:
                continue
            vr = np.where(ok, v / vf, np.nan)
            out[n, 2 * k] = np.nanmean(vr <= 0.6)
            out[n, 2 * k + 1] = np.nanmean(vr)
    return out


class Hyper(nn.Module):
    """Channel-wise FiLM of the collapsed features from the run context (identity at start)."""
    def __init__(self, c):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(4, 64), nn.GELU(), nn.Linear(64, 2 * c))
        nn.init.zeros_(self.mlp[2].weight); nn.init.zeros_(self.mlp[2].bias)

    def forward(self, h, ctxv):                                  # h (B, c, 1, S)
        g, b = self.mlp(ctxv).chunk(2, -1)
        return h * (1 + g[:, :, None, None]) + b[:, :, None, None]


class TemporalExpert(nn.Module):
    """TESTAM's identity expert: each link's own 12-slot history only (no spatial mixing), plus a per-link gate."""
    def __init__(self, c_in=4 * 12):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(c_in + 3, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU())
        self.out = nn.Linear(128, F)
        self.gate = nn.Linear(128, 1)
        nn.init.zeros_(self.gate.weight); nn.init.constant_(self.gate.bias, 4.0)   # start on the CNN

    def forward(self, x, logit_cnn):
        B, Cc, Hh, S = x.shape
        inp = torch.cat([x[:, :4].permute(0, 3, 1, 2).reshape(B, S, 4 * Hh), x[:, 9:12, -1].permute(0, 2, 1)], -1)
        hdn = self.f(inp.float())
        g = torch.sigmoid(self.gate(hdn))[..., 0][:, None, :]                      # (B, 1, S)
        return g * logit_cnn + (1 - g) * self.out(hdn).permute(0, 2, 1)


def build(ckpt, multirep, conf, orient=False):
    return LeapNet(LC.load_short(ckpt), multirep, conf, orient).to(DEV)


# ------------------------------------------------------------------------------------------------ train
def stage_train(args):
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    parts = LC.train_panels()
    gparts = []
    if args.gen:                                               # note 169: DLSim-generated runs mixed in
        import genparts as GP
        gparts = [g for g in GP.gen_parts(args.gen, parts) if g["panel"] not in args.gen_skip]
        for p in gparts:
            p["tr"], p["pw"] = np.arange(len(p["d"])), None
    net = build(args.ckpt, args.multirep, args.conf, args.orient)
    new = [p for n, p in net.named_parameters() if n.startswith("conf.") or n == "inp.1.weight"]
    ids = {id(p) for p in new}
    old = [p for p in net.parameters() if id(p) not in ids]
    groups = [{"params": old, "lr": 1e-4}] + ([{"params": new, "lr": 1e-3}] if new else [])
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], total_steps=args.steps,
                                                pct_start=0.05)
    es = []
    for p in parts:
        p["es_idx"] = np.flatnonzero(p["es"])[:400]
        p["tr"] = np.arange(len(p["d"])) if args.full else np.flatnonzero(~p["es"])
        p["pw"] = None
        if args.adv:                                           # adversarial-validation sampling weights
            a = np.load(CACHE / f"advw_{p['panel']}.npz")
            assert (a["d"] == p["d"]).all() and (a["t0"] == p["t0"]).all()
            w = a["w"][p["tr"]]
            p["pw"] = w / w.sum()
        e = np.flatnonzero(p["es"])[:400]
        d, t0 = p["d"][e], p["t0"][e]
        fi = t0[:, None] + np.arange(1, F + 1)
        es.append((p["panel"], LC.short_x(p, d, t0), p["q"][d[:, None], fi].astype(np.float32),
                   p["el"][d[:, None], fi].astype(np.float32)))
    hyper = Hyper(net.inp.weight.shape[0] if hasattr(net.inp, "weight") else net.inp[1].weight.shape[0]).to(DEV) \
        if args.hyper else None
    moe = TemporalExpert().to(DEV) if args.moe else None
    net.hyper, net.moe = hyper, moe
    if hyper is not None or moe is not None:
        extra = list(hyper.parameters()) if hyper is not None else list(moe.parameters())
        opt.add_param_group({"params": extra, "lr": 1e-3})
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in opt.param_groups],
                                                    total_steps=args.steps, pct_start=0.05)
    for p in parts:
        p["month"] = np.array([str(x)[:7] for x in p["ctx"].z["dates"]])[p["d"]]
        if args.hyper:
            p["rctx"] = run_context(p["v"], p["ok"], np.asarray(p["ctx"].vf, np.float32), p["d"], p["t0"])
    teachers = [LC.load_short(Path(str(args.ckpt).replace(f"_s{args.seed_ckpt}", f"_s{k}"))).eval()
                for k in (20, 21, 22, 23)] if args.distill else []
    ema = None
    if args.ema:
        ema = {k: v.detach().clone().float() for k, v in net.state_dict().items()}
    best, best_state, t_start = 1e9, None, time.time()
    for step in range(1, args.steps + 1):
        net.train()
        gfrac = args.gen_frac * (1 - step / args.steps) if args.gen_decay else args.gen_frac
        if gparts and rng.random() < gfrac:
            p = gparts[rng.integers(len(gparts))]
        else:
            p = parts[rng.integers(len(parts))]
        net.set_panel(p["panel"])
        if p["pw"] is not None:
            j = p["tr"][rng.choice(len(p["tr"]), size=args.bs, p=p["pw"])]
        elif args.vrex > 0:                                        # 4 months x bs/4 windows: the environments
            mon = p["month"][p["tr"]]
            ms = rng.choice(np.unique(mon), size=4, replace=False)
            j = np.concatenate([rng.choice(p["tr"][mon == mm], size=args.bs // 4) for mm in ms])
        else:
            j = p["tr"][rng.integers(len(p["tr"]), size=args.bs)]
        d, t0 = p["d"][j], p["t0"][j]
        fi = t0[:, None] + np.arange(1, F + 1)
        x = LC.tt(LC.short_x(p, d, t0))
        y, m = LC.tt(p["q"][d[:, None], fi].astype(np.float32)), LC.tt(p["el"][d[:, None], fi].astype(np.float32))
        soft = None
        if teachers:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                soft = torch.stack([torch.sigmoid(t_(x)[:, :F].float()) for t_ in teachers]).mean(0)
        if args.mixup > 0:                                      # pairs within the batch (one panel, same links)
            lam = float(np.random.default_rng(step).beta(args.mixup, args.mixup))
            perm = torch.randperm(len(x), device=x.device)
            x = lam * x + (1 - lam) * x[perm]
            y = lam * y + (1 - lam) * y[perm]
            m = m * m[perm]
        yb = y * (1 - args.ls) + 0.5 * args.ls if args.ls > 0 else y
        if args.tada:                                              # worst of 3 monotone time warps (no grad pick)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                cands = [time_warp(x, torch.empty(len(x), device=x.device).uniform_(-0.4, 0.4)) for _ in range(3)]
                ls_ = [cnn.loss_fn(net.head(net.features(xc))[:, :F, 0].float(), yb, m) for xc in cands]
            x = torch.cat([x, cands[int(torch.stack(ls_).argmax())]], 0)
            y, yb, m = torch.cat([y, y], 0), torch.cat([yb, yb], 0), torch.cat([m, m], 0)
        if args.engr > 0:                                          # pre-additive noise on observed v, q
            x = torch.cat([x, x], 0)
            y, yb, m = torch.cat([y, y], 0), torch.cat([yb, yb], 0), torch.cat([m, m], 0)
            noise = torch.randn_like(x[:, :2]) * args.engr * x[:, 2:3]
            x = torch.cat([x[:, :2] + noise, x[:, 2:]], 1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = net.features(x)
            if hyper is not None:
                h = hyper(h, LC.tt(np.concatenate([p["rctx"][j]] * (len(x) // len(j)))))
            logit = net.head(h)[:, :F, 0].float()
            if moe is not None:
                logit = moe(x, logit).float()
            if net.flip is not None:
                logit = net.flip(logit)
            cf = net.conf(h)[:, :, 0].float() if net.conf is not None else None
            if cf is not None and net.flip is not None:
                cf = net.flip(cf)
        if args.engr > 0:                                          # engression: score the two-sample mixture
            half = len(logit) // 2
            pm = 0.5 * (torch.sigmoid(logit[:half]) + torch.sigmoid(logit[half:])).clamp(1e-5, 1 - 1e-5)
            yh, mh = yb[:half], m[:half]
            loss = (nn.functional.binary_cross_entropy(pm, yh, reduction="none") * mh).sum() / mh.sum().clamp(min=1) \
                + cnn.LOVASZ * cnn.lovasz_hinge(0.5 * (logit[:half] + logit[half:]), (yh > 0.5).float(), mh)
        elif args.vrex > 0:                                        # V-REx over the 4 months of the batch
            q = len(logit) // 4
            risks = torch.stack([cnn.loss_fn(logit[k * q:(k + 1) * q], yb[k * q:(k + 1) * q], m[k * q:(k + 1) * q])
                                 + cnn.LOVASZ * cnn.lovasz_hinge(logit[k * q:(k + 1) * q], (y[k * q:(k + 1) * q] > 0.5).float(),
                                                                 m[k * q:(k + 1) * q]) for k in range(4)])
            lam = args.vrex if step > args.steps // 6 else 0.0
            loss = risks.mean() + lam * risks.var()
        else:
            loss = cnn.loss_fn(logit, yb, m) + cnn.LOVASZ * cnn.lovasz_hinge(logit, (y > 0.5).float(), m)
        if args.bloss != "none":
            loss = loss + boundary_extra(args.bloss, logit, y, m, x, step / args.steps)
        if soft is not None:
            loss = 0.5 * loss + 0.5 * cnn.loss_fn(logit, soft, m)
        if cf is not None:
            bce = nn.functional.binary_cross_entropy_with_logits(logit, y, reduction="none").detach()
            loss = loss + ((cf - bce).abs() * m).sum() / m.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.awp > 0 and step > args.steps // 6:            # adversarial weight perturbation (one ascent step)
            backup = {}
            for n_, q_ in net.named_parameters():
                if q_.grad is not None and q_.dim() > 1 and "rec" not in n_:
                    backup[n_] = q_.data.clone()
                    q_.data.add_(args.awp * q_.data.norm() * q_.grad / (q_.grad.norm() + 1e-12))
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg2 = net.head(net.features(x))[:, :F, 0].float()
                if net.flip is not None:
                    lg2 = net.flip(lg2)
            (cnn.loss_fn(lg2, yb, m) + cnn.LOVASZ * cnn.lovasz_hinge(lg2, (y > 0.5).float(), m)).backward()
            for n_, q_ in net.named_parameters():
                if n_ in backup:
                    q_.data.copy_(backup[n_])
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
        if ema is not None:
            with torch.no_grad():
                for k, v in net.state_dict().items():
                    if v.dtype.is_floating_point:
                        ema[k].mul_(args.ema).add_(v.float(), alpha=1 - args.ema)
                    else:
                        ema[k].copy_(v)
        live = None
        if step % 500 == 0:
            if ema is not None:
                live = {k: v.detach().clone() for k, v in net.state_dict().items()}
                net.load_state_dict({k: ema[k].to(live[k].dtype) for k in live})
            tot, n = 0.0, 0.0
            for (pn, xs, yv, mv), p_ in zip(es, parts):
                net.set_panel(pn)
                pr = predict_ctx(net, xs, p_["rctx"][p_["es_idx"]] if hyper is not None else None)
                bce = -(yv * np.log(pr + 1e-6) + (1 - yv) * np.log(1 - pr + 1e-6))
                tot += float((bce * mv).sum()); n += float(mv.sum())
            v = tot / max(n, 1)
            flag = ""
            if args.full:                                       # no early stopping: keep the final weights
                best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
                best, flag = v, " (last)"
            elif v < best:
                best, flag = v, " *"
                best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
            print(f"step {step}: train {loss.item():.4f}  ES bce {v:.5f}{flag} ({time.time() - t_start:.0f}s)", flush=True)
            if live is not None:
                net.load_state_dict(live)
    torch.save({"state": best_state, "multirep": args.multirep, "conf": args.conf, "orient": args.orient,
                "bloss": args.bloss, "hyper": hyper.state_dict() if hyper is not None else None,
                "moe": moe.state_dict() if moe is not None else None,
                "ckpt": str(args.ckpt)},
               MODELS / f"leap_{args.tag}.pt")
    print(f"saved leap_{args.tag}.pt (ES bce {best:.5f})")


# ------------------------------------------------------------------------------------------------ lab
def predict_ctx(net, x, rctx, bs=64):
    """cnn.predict with the run context handed to the hypernetwork batch by batch."""
    if rctx is None or getattr(net, "hyper", None) is None:
        return cnn.predict(net, x)
    net.eval()
    out = []
    with torch.no_grad():
        for a in range(0, len(x), bs):
            net.rctx = LC.tt(rctx[a:a + bs])
            xb = LC.tt(x[a:a + bs])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out.append(torch.sigmoid(net(xb).float()).cpu().numpy())
    net.rctx = None
    return np.concatenate(out)


def load_leap(tag):
    d = torch.load(MODELS / f"leap_{tag}.pt", map_location=DEV)
    net = build(d["ckpt"], d["multirep"], d["conf"], d.get("orient", False))
    if d.get("hyper") is not None:
        net.hyper = Hyper(net.inp.weight.shape[0] if hasattr(net.inp, "weight") else net.inp[1].weight.shape[0]).to(DEV)
        net.hyper.load_state_dict(d["hyper"])
    if d.get("moe") is not None:
        net.moe = TemporalExpert().to(DEV)
        net.moe.load_state_dict(d["moe"])
    net.load_state_dict(d["state"], strict=False)
    return net.eval()


def sup_batch_ctx(ctx, stat, z, pool, days, cuts, bs, rng, vf):
    """PA.sup_batch (loose ongoing selector) that also returns each pseudo window's run context."""
    ov, oq, ok, q = pool
    cand = []
    for _ in range(bs * 8):
        d = int(days[rng.integers(len(days))])
        lim = min(cuts.get(d, 10 ** 6), q.shape[1] - 1)
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
    return (LC.tt(x), LC.tt(q[d[:, None], fi].astype(np.float32)), LC.tt(ok[d[:, None], fi].astype(np.float32)),
            LC.tt(run_context(ov, ok, vf, d, t0)))


def tta_predict(net, x, k, rng):
    """Mean over k copies with 10% of the observed history cells hidden (the SSL masking the backbone learned)."""
    if k <= 1:
        return cnn.predict(net, x)
    out = []
    for _ in range(k):
        xx = x.copy()
        hid = (rng.random(xx[:, 2].shape) < 0.10) & (xx[:, 2] > 0.5)
        xx[:, 0:4] = np.where(hid[:, None], 0, xx[:, 0:4])
        out.append(cnn.predict(net, xx))
    return np.mean(out, 0)


def adapt_sup_ex(net, params, batches, steps, lr, w_rec, rec_batches, kind):
    """PA.adapt_sup with the boundary term of the training added (the adaptation optimises what training did)."""
    opt = C.make_opt(params, lr)
    net.train()
    for _ in range(steps):
        b = batches()
        if b is None:
            break
        x, y, m = b
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = net(x)[:, :F].float()
        loss = cnn.loss_fn(logit, y, m) + cnn.LOVASZ * cnn.lovasz_hinge(logit, y, m) + \
            boundary_extra(kind, logit, y, m, x, 1.0)
        xb = rec_batches()
        if w_rec and xb is not None:
            loss = loss + w_rec * C.ssl_loss(net, xb)
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
            pool = PA.pseudo_pool(ctx, z)
            days = np.unique(np.concatenate([on["d"], og["d"]]))
            firsts = {int(d): int(min(on["t0"][on["d"] == d].min(initial=10 ** 6),
                                      og["t0"][og["d"] == d].min(initial=10 ** 6))) for d in days}
            for mode in ("none", "sup+rec"):
                net = load_leap(args.tag)
                net.set_panel(panel)
                params = list(net.inp.parameters()) + list(net.blocks.parameters()) + list(net.rec.parameters())
                rng = np.random.default_rng(7 + i)
                p_on = np.zeros((len(on["d"]), ctx.S), np.float32)
                p_og = np.zeros(og["fut"].shape, np.float32)
                for d in days:
                    d = int(d)
                    if mode != "none":
                        past = np.arange(d)
                        pool_days = np.append(past, d) if firsts[d] > H + F + 2 else past
                        cuts = {int(k): (firsts[d] if int(k) == d else 10 ** 6) for k in pool_days}
                        if len(pool_days):
                            kind = torch.load(MODELS / f"leap_{args.tag}.pt", map_location="cpu").get("bloss", "none") \
                                if not hasattr(args, "_kind") else args._kind
                            args._kind = kind
                            sup = lambda: PA.sup_batch(ctx, stat, z, pool, pool_days, cuts, 16, rng, "ongoing")  # noqa: E731
                            rec = lambda: C.past_batch(ctx, stat, z, past, 16, rng)  # noqa: E731
                            if getattr(net, "hyper", None) is not None:
                                vfa = np.asarray(ctx.vf, np.float32)
                                sb = lambda: sup_batch_ctx(ctx, stat, z, pool, pool_days, cuts, 16, rng, vfa)  # noqa: E731
                                opt_ = C.make_opt(params, 1e-4)
                                net.train()
                                for _ in range(30):
                                    b = sb()
                                    if b is None:
                                        break
                                    xb_, yb_, mb_, cb_ = b
                                    net.rctx = cb_
                                    with torch.autocast("cuda", dtype=torch.bfloat16):
                                        lg_ = net(xb_)[:, :F].float()
                                    loss_ = cnn.loss_fn(lg_, yb_, mb_) + cnn.LOVASZ * cnn.lovasz_hinge(lg_, yb_, mb_)
                                    r_ = rec()
                                    if r_ is not None:
                                        loss_ = loss_ + C.ssl_loss(net, r_)
                                    opt_.zero_grad(set_to_none=True); loss_.backward(); opt_.step()
                                net.rctx = None
                                net.eval()
                            elif kind == "none":
                                PA.adapt_sup(net, params, sup, 30, 1e-4, 1.0, rec)
                            else:
                                adapt_sup_ex(net, params, sup, 30, 1e-4, 1.0, rec, kind)
                    io = np.flatnonzero(on["d"] == d)
                    if len(io):
                        rc = run_context(z["obs_speed"], z["elig"] & np.isfinite(z["obs_speed"]),
                                         np.asarray(ctx.vf, np.float32), on["d"][io], on["t0"][io]) \
                            if getattr(net, "hyper", None) is not None else None
                        p_on[io] = (predict_ctx(net, C.make_x(ctx, stat, on, io, z), rc) if rc is not None else
                                    tta_predict(net, C.make_x(ctx, stat, on, io, z), args.tta, rng))[:, F - 1]
                    ig = np.flatnonzero(og["d"] == d)
                    if len(ig):
                        rc = run_context(z["obs_speed"], z["elig"] & np.isfinite(z["obs_speed"]),
                                         np.asarray(ctx.vf, np.float32), og["d"][ig], og["t0"][ig]) \
                            if getattr(net, "hyper", None) is not None else None
                        p_og[ig] = predict_ctx(net, C.make_x(ctx, stat, og, ig, z), rc) if rc is not None else \
                            tta_predict(net, C.make_x(ctx, stat, og, ig, z), args.tta, rng)
                sg = C.ongoing_iou(p_og, og, 0.5)
                res.setdefault((s, mode), {})[panel] = (C.onset_iou(p_on, on).mean(), sg.mean(),
                                                         sg[og["d"] <= 4].mean() if (og["d"] <= 4).any() else np.nan)
                out_tag = args.tag + (f"_tta{args.tta}" if args.tta > 1 else "")
                np.savez(CACHE / f"fam_leap_{out_tag}_{mode}_{s}_{panel}.npz", og=p_og.astype(np.float16), on=p_on)
            print(f"  {panel} {s} ({time.time() - t_start:.0f}s)", flush=True)
    for s in C.SPLITS:
        print(f"\n=== {s}: leap {args.tag} ===")
        for m in ("none", "sup+rec"):
            d = res[(s, m)]
            fm = lambda j: C.family_mean({k: v[j] for k, v in d.items() if np.isfinite(v[j])})  # noqa: E731
            print(f"  {m:<8} onset {fm(0):.4f}  ongoing {fm(1):.4f}  ongoing days 1-5 {fm(2):.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("train", "lab"))
    ap.add_argument("--ckpt", type=Path)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--multirep", action="store_true")
    ap.add_argument("--conf", action="store_true")
    ap.add_argument("--orient", action="store_true", help="flip S / W panels to upstream -> downstream")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--adv", action="store_true", help="sample train windows by adversarial-validation weights")
    ap.add_argument("--full", action="store_true", help="train on every train window incl. the ES tail, keep last")
    ap.add_argument("--distill", action="store_true", help="0.5 x BCE to the mean of the four SSL w192 seeds")
    ap.add_argument("--seed-ckpt", type=int, default=20, help="the seed in --ckpt's name (to find the teachers)")
    ap.add_argument("--ema", type=float, default=0.0)
    ap.add_argument("--awp", type=float, default=0.0, help="relative weight perturbation size, e.g. 1e-3")
    ap.add_argument("--ls", type=float, default=0.0, help="label smoothing on the BCE targets")
    ap.add_argument("--mixup", type=float, default=0.0, help="Beta(a, a) mixup within the batch")
    ap.add_argument("--tta", type=int, default=1, help="lab: average over k masked copies")
    ap.add_argument("--vrex", type=float, default=0.0, help="V-REx penalty over the 4 months of each batch")
    ap.add_argument("--tada", action="store_true", help="worst-of-3 monotone time-warp augmentation")
    ap.add_argument("--engr", type=float, default=0.0, help="engression pre-additive noise sd on v/vf, q/cap")
    ap.add_argument("--hyper", action="store_true", help="run-context FiLM (hypernetwork) on the collapsed features")
    ap.add_argument("--moe", action="store_true", help="TESTAM-style temporal expert with a per-link gate")
    ap.add_argument("--gen", default="", help="directory of DLSim-generated runs to mix into training (note 169)")
    ap.add_argument("--gen-frac", type=float, default=0.5, help="share of batches drawn from generated runs")
    ap.add_argument("--gen-decay", action="store_true", help="the generated share decays linearly to 0 over training")
    ap.add_argument("--gen-skip", nargs="*", default=[], help="panels whose generated runs are not used")
    ap.add_argument("--bloss", default="none", choices=("none", "sdf", "band", "blov", "band+sdf"),
                    help="boundary-aware extra loss, used in training and in the daily adaptation")
    args = ap.parse_args()
    stage_train(args) if args.stage == "train" else stage_lab(args)


if __name__ == "__main__":
    main()
