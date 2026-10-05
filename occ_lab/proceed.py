"""Proceed (Zhao & Shen, KDD 2025, arXiv 2412.08435) on the SSL w192 Task 2 CNN: proactive adaptation to the drift
between the latest labelled window and the window being predicted.

Our causal adaptation (note 106) fine-tunes once per day on windows whose labels are complete before the day's
first origin, so by T the model is up to a day behind. Proceed's point is exactly this gap: labels arrive a horizon
late, so the model is adapted to an outdated concept. Its fix, reproduced here:

    concept encoders   E (reference: history X_r and its labels Y_r) and E' (target: history X_t only); each a
                       per-link MLP averaged over the panel's links (paper eq. 1-2); drift delta = E'(X_t) - E(X_r, Y_r)
    generator          bottleneck r = 32: [alpha, beta] = W2_l sigma(W1 delta + b_l) + 1 per layer l, W2_l zero-init
                       (eq. 3); alpha scales a conv's input channels, beta its output channels, i.e.
                       W_l' = (beta alpha^T) * W_l, bias * beta (eq. 4); applied to the 16 block convs, the history
                       collapse and the head
    training           on the train run, forecaster and adapter jointly, over pairs (reference, target) of one panel:
                       half the references are the target day's latest labelled window (origin T-35 min, labels up to
                       T-5), half a random window of the panel - the paper's shuffled "synthetic concept drifts"
    online (lab)       step 1, the usual causal per-day sup+rec fine-tuning of the forecaster (adapter frozen, the
                       generator off, as Proceed's online fine-tuning); step 3, each window predicted with the
                       coefficients generated from its own drift: reference = the same day's window at origin T-35 min,
                       history and labels from the released layer (all at or before T-5)
    control            --no-adapter: the same joint training with the generator off

    python occ_lab/proceed.py --stage train --ckpt work/models/t2_cnn_lov_w192_rec_ssl30k_s20.pt --tag s20
    python occ_lab/proceed.py --stage lab --model work/models/proceed_s20.pt --tag s20
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
import longctx as LC  # noqa: E402  (sets the T2CNN environment, imports the lab kit)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

C, cnn = LC.C, LC.cnn
H, F, T, DEV = LC.H, LC.F, LC.T, LC.DEV
CACHE, MODELS = LC.CACHE, LC.MODELS
LAG = F + 1                     # reference origin T - 35 min: its labels T-30 .. T-5 are released before T
DZ, R = 64, 32


class Proceed(nn.Module):
    def __init__(self, short):
        super().__init__()
        self.short = short
        c = short.inp.weight.shape[0]
        self.enc_r = nn.Sequential(nn.Linear(4 * H + 2 * F, 128), nn.GELU(), nn.Linear(128, DZ))
        self.enc_t = nn.Sequential(nn.Linear(4 * H, 128), nn.GELU(), nn.Linear(128, DZ))
        self.layers = [(c, c)] * 17 + [(c, F)]       # 16 block convs, collapse, head
        self.w1 = nn.Linear(DZ, R, bias=False)
        self.b = nn.Parameter(torch.zeros(len(self.layers), R))
        self.w2 = nn.ModuleList([nn.Linear(R, i + o) for i, o in self.layers])
        for l in self.w2:
            nn.init.zeros_(l.weight); nn.init.zeros_(l.bias)

    @staticmethod
    def _links(x):                                   # (B, C, H, S) -> (B, S, 4H), validity (B, S)
        B, _, Hh, S = x.shape
        return x[:, :4].permute(0, 3, 1, 2).reshape(B, S, 4 * Hh), (x[:, 2].sum(1) > 0).float()

    def concept(self, enc, feats, valid):
        z = enc(feats.float())
        return (z * valid[..., None]).sum(1) / valid.sum(1, keepdim=True).clamp(min=1)

    def coefs(self, x, xr, yr, mr):
        ft, vt = self._links(x)
        fr, vr = self._links(xr)
        fr = torch.cat([fr, (yr * mr).permute(0, 2, 1), mr.permute(0, 2, 1)], 2)
        delta = self.concept(self.enc_t, ft, vt) - self.concept(self.enc_r, fr, vr)
        out = []
        for l, (i, o) in enumerate(self.layers):
            g = self.w2[l](torch.sigmoid(self.w1(delta) + self.b[l])) + 1
            out.append((g[:, :i], g[:, i:]))
        return out

    def forward(self, x, ref=None):                  # ref = (xr, yr, mr) or None -> (B, F, S) logits
        s = self.short
        co = self.coefs(x, *ref) if ref is not None else None

        def conv(m, u, k):
            if co is None:
                return m(u)
            a, b = co[k]
            return m(u * a[:, :, None, None].to(u.dtype)) * b[:, :, None, None].to(u.dtype)

        h = s.inp(x)
        k = 0
        for blk in s.blocks:
            n = blk.net
            u = conv(n[2], n[1](n[0](h)), k); k += 1
            u = conv(n[5], n[4](n[3](u)), k); k += 1
            h = h + u
        h = conv(s.collapse[2], s.collapse[1](s.collapse[0](h)), k); k += 1
        return conv(s.head[1], s.head[0](h), k)[:, :F, 0]


# ------------------------------------------------------------------------------------------------ train
def windows_xy(p, d, t0):
    fi = t0[:, None] + np.arange(1, F + 1)
    return LC.short_x(p, d, t0), p["q"][d[:, None], fi].astype(np.float32), p["el"][d[:, None], fi].astype(np.float32)


def ref_origin(p, d, t0, rng):
    """Half: the same day's latest labelled window (T-35 min); half: a random window of the panel."""
    dr, tr = d.copy(), t0 - LAG
    rnd = (rng.random(len(d)) < 0.5) | (tr < H + 1)
    j = rng.integers(len(p["d"]), size=int(rnd.sum()))
    dr[rnd], tr[rnd] = p["d"][j], p["t0"][j]
    return dr, tr


def stage_train(args):
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    parts = LC.train_panels()
    net = Proceed(LC.load_short(args.ckpt)).to(DEV)
    on = not args.no_adapter
    ad_p = [q for n, q in net.named_parameters() if not n.startswith("short.")]
    groups = [{"params": list(net.short.parameters()), "lr": args.lr_short}] + \
        ([{"params": ad_p, "lr": args.lr_ad}] if on else [])
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], total_steps=args.steps,
                                                pct_start=0.05)
    es_data = []
    for p in parts:
        p["tr"] = np.flatnonzero(~p["es"])
        e = np.flatnonzero(p["es"])[:400]
        d, t0 = p["d"][e], p["t0"][e]
        ok = t0 - LAG >= H + 1
        d, t0 = d[ok], t0[ok]
        es_data.append((windows_xy(p, d, t0), windows_xy(p, d, t0 - LAG)))
    best, best_state, t_start = 1e9, None, time.time()
    for step in range(1, args.steps + 1):
        net.train()
        p = parts[rng.integers(len(parts))]
        j = p["tr"][rng.integers(len(p["tr"]), size=args.bs)]
        d, t0 = p["d"][j], p["t0"][j]
        x, y, m = windows_xy(p, d, t0)
        ref = None
        if on:
            dr, tr = ref_origin(p, d, t0, rng)
            ref = tuple(LC.tt(a) for a in windows_xy(p, dr, tr))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = net(LC.tt(x), ref)
        logit = out.float()
        yt, mt = LC.tt(y), LC.tt(m)
        loss = cnn.loss_fn(logit, yt, mt) + cnn.LOVASZ * cnn.lovasz_hinge(logit, yt, mt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 500 == 0:
            tot, n = 0.0, 0.0
            for (xs, yv, mv), rf in es_data:
                pr = predict(net, xs, rf if on else None)
                bce = -(yv * np.log(pr + 1e-6) + (1 - yv) * np.log(1 - pr + 1e-6))
                tot += float((bce * mv).sum()); n += float(mv.sum())
            es_loss = tot / max(n, 1)
            flag = ""
            if es_loss < best:
                best, flag = es_loss, " *"
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            print(f"step {step}: train {loss.item():.4f}  ES bce {es_loss:.5f}{flag} ({time.time() - t_start:.0f}s)",
                  flush=True)
    torch.save({"state": best_state, "adapter": on}, MODELS / f"proceed_{args.tag}.pt")
    print(f"saved proceed_{args.tag}.pt (ES bce {best:.5f})")


@torch.no_grad()
def predict(net, x, ref, bs=32):
    net.eval()
    out = []
    for a in range(0, len(x), bs):
        r = tuple(LC.tt(q[a:a + bs]) for q in ref) if ref is not None else None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(torch.sigmoid(net(LC.tt(x[a:a + bs]), r).float()).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, F, x.shape[-1]), np.float32)


# ------------------------------------------------------------------------------------------------ lab
def load_proceed(path):
    d = torch.load(path, map_location=DEV)
    st = d["state"]
    short = cnn.T2Net(st["short.inp.weight"].shape[1], st["short.inp.weight"].shape[0]).to(DEV)
    net = Proceed(short).to(DEV)
    net.load_state_dict(st)
    return net.eval(), d["adapter"]


def adapt_short(short, batches, rec_batches, steps, lr, w_rec):
    params = list(short.inp.parameters()) + list(short.blocks.parameters()) + list(short.rec.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    short.train()
    for _ in range(steps):
        b = batches()
        if b is None:
            break
        x, _, y, m = b
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = short(LC.tt(x))[:, :F].float()
        yt, mt = LC.tt(y), LC.tt(m)
        loss = cnn.loss_fn(logit, yt, mt) + cnn.LOVASZ * cnn.lovasz_hinge(logit, yt, mt)
        if w_rec:
            xb = rec_batches()
            if xb is not None:
                loss = loss + w_rec * C.ssl_loss(short, xb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    short.eval()


def released_ref(ctx, stat, z, pool, d, t0):
    """Reference windows at origin T-35 min from the released layer: history, labels, label mask."""
    ov, oq, ok, q = pool
    tr = np.maximum(t0 - LAG, H + 1)
    hi, fi = tr[:, None] + np.arange(-H, 0), tr[:, None] + np.arange(1, F + 1)
    x = cnn.make_x(ctx, stat, ov[d[:, None], hi], oq[d[:, None], hi], ok[d[:, None], hi], tr, z["weekday"][d] >= 5)
    m = ok[d[:, None], fi] & (fi < t0[:, None])[:, :, None]
    return x, q[d[:, None], fi].astype(np.float32) * m, m.astype(np.float32)


def stage_lab(args):
    t_start = time.time()
    res = {}
    for i, panel in enumerate(C.T2_PANELS):
        ctx, stat, W = C.windows(i, panel)
        for s in C.SPLITS:
            z, og, on = W[s]["z"], W[s]["og"], W[s]["on"]
            ov, oq = z["obs_speed"], z["obs_flow"]
            ok = z["elig"] & np.isfinite(ov) & np.isfinite(oq)
            pool = (ov, oq, ok, ok & (np.nan_to_num(ov, nan=1e9) <= 0.6 * ctx.vf))
            days = np.unique(np.concatenate([on["d"], og["d"]]))
            firsts = {int(d): int(min(on["t0"][on["d"] == d].min(initial=10 ** 6),
                                      og["t0"][og["d"] == d].min(initial=10 ** 6))) for d in days}
            for mode in args.modes:
                net, use_ad = load_proceed(args.model)
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
                            adapt_short(net.short,
                                        lambda: LC.sup_batch(ctx, stat, z, None, pool, pool_days, cuts, args.bs, rng,
                                                             False),
                                        lambda: C.past_batch(ctx, stat, z, past, args.bs, rng),
                                        args.steps, args.lr, args.w_rec)
                    for W_, key in ((on, "on"), (og, "og")):
                        idx = np.flatnonzero(W_["d"] == d)
                        if not len(idx):
                            continue
                        x = C.make_x(ctx, stat, W_, idx, z)
                        ref = released_ref(ctx, stat, z, pool, W_["d"][idx], W_["t0"][idx]) if use_ad else None
                        pr = predict(net, x, ref)
                        if key == "on":
                            p_on[idx] = pr[:, F - 1]
                        else:
                            p_og[idx] = pr
                res.setdefault((s, mode), {})[panel] = (C.onset_iou(p_on, on).mean(),
                                                        C.ongoing_iou(p_og, og, 0.5).mean())
                np.savez(CACHE / f"fam_proceed_{args.tag}_{mode}_{s}_{panel}.npz", og=p_og.astype(np.float16),
                         on=p_on.astype(np.float32))
            msg = "  ".join(f"{m} og {res[(s, m)][panel][1]:.4f}" for m in args.modes)
            print(f"  {panel} {s}: {msg} ({time.time() - t_start:.0f}s)", flush=True)
    for s in C.SPLITS:
        print(f"\n=== {s}: proceed {args.tag} ===")
        for m in args.modes:
            d = res[(s, m)]
            print(f"  {m:<8} onset {C.family_mean({k: v[0] for k, v in d.items()}):.4f}  "
                  f"ongoing {C.family_mean({k: v[1] for k, v in d.items()}):.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("train", "lab"))
    ap.add_argument("--ckpt", type=Path)
    ap.add_argument("--model", type=Path)
    ap.add_argument("--tag", default="s20")
    ap.add_argument("--no-adapter", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--bs", type=int, default=None)
    ap.add_argument("--lr-short", type=float, default=1e-4)
    ap.add_argument("--lr-ad", type=float, default=1e-3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--w-rec", type=float, default=1.0)
    ap.add_argument("--modes", nargs="*", default=["none", "sup+rec"])
    args = ap.parse_args()
    if args.stage == "train":
        args.steps, args.bs = args.steps or 6000, args.bs or 16
        stage_train(args)
    else:
        args.steps, args.bs = args.steps or 30, args.bs or 16
        stage_lab(args)


if __name__ == "__main__":
    main()
