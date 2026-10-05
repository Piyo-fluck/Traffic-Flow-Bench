"""Ways of adapting to the scored run that are not note 106's recipe (A of the 2026-09-27 survey), on p_PS's SSL w192 CNN.

The shipped recipe: per panel and split, per day before its first origin, 30 steps of sup (BCE + 0.5 Lovasz) + rec on
the released pseudo windows, updating inp + blocks + rec, lr 1e-4, weights carried forward. Every method below runs
the same days, pools, batches and seeds (the baseline is re-run inside the same script):

    base          the shipped recipe
    wise          WiSE-FT (Wortsman et al. 2022): predict with (1 - a) theta_0 + a theta_adapted, a = .25 / .5 / .75,
                  the adaptation trajectory unchanged ("wise_a"); and a variant that also CARRIES the interpolated
                  weights to the next day ("wisecarry_a")
    surgical      Lee et al. (ICLR 2023): only a subset adapts - inp | inp + blocks 0-1 | blocks 6-7 + collapse + head |
                  GroupNorm affine only
    solid         SOLID (KDD 2024): after the day's adaptation, each window gets its own head: the 32 pseudo windows of
                  the pool whose last-15-min v/vf pattern is closest (cosine, same hour +-2 h preferred) fine-tune
                  collapse + head for 5 steps (lr 5e-4), predict, restore
    gcm           TAFAS (AAAI 2025): gated calibration modules, identity at start - input: per dynamic channel
                  x + tanh(g) (a x + b); output: per horizon step z + tanh(g) (a z + b). "gcm" adapts only the modules
                  (lr 1e-2, no rec); "gcm+full" adapts modules and the shipped subset together

    python occ_lab/adapt_variants.py --ckpt work/models/t2_cnn_lov_w192_rec_ssl30k_s20.pt --tag s20 --method wise
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from pathlib import Path

os.environ["T2CNN_ARCH"] = "cnn"
os.environ.setdefault("T2CNN_REC", "0.3")
os.environ.setdefault("T2CNN_LOVASZ", "0.5")
os.environ.setdefault("T2CNN_WIDTH", "192")
LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import common as C  # noqa: E402
import pseudo_adapt as PA  # noqa: E402

cnn = C.cnn
H, F, DEV = C.H, C.F, C.DEV
CACHE = LAB / "cache"


# ------------------------------------------------------------------------------------------------ helpers
def lossf(logit, y, m):
    return cnn.loss_fn(logit, y, m) + cnn.LOVASZ * cnn.lovasz_hinge(logit, y, m)


def subset(net, name):
    if name == "full":
        return list(net.inp.parameters()) + list(net.blocks.parameters()) + list(net.rec.parameters())
    if name == "inp":
        return list(net.inp.parameters())
    if name == "early":
        return list(net.inp.parameters()) + list(net.blocks[0].parameters()) + list(net.blocks[1].parameters())
    if name == "late":
        return list(net.blocks[6].parameters()) + list(net.blocks[7].parameters()) + \
            list(net.collapse.parameters()) + list(net.head.parameters())
    if name == "norm":
        return [p for m in net.modules() if isinstance(m, nn.GroupNorm) for p in m.parameters()]
    raise ValueError(name)


def set_trainable(net, params):
    for p in net.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)


def adapt(net, params, sup, rec, steps, lr, w_rec):
    set_trainable(net, params)
    PA.adapt_sup(net, params, sup, steps, lr, w_rec, rec)


class GCM(nn.Module):
    """Gated calibration around a T2Net: input per dynamic channel, output per horizon step (identity at start)."""
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.ia = nn.Parameter(torch.zeros(4)); self.ib = nn.Parameter(torch.zeros(4)); self.ig = nn.Parameter(torch.zeros(4))
        self.oa = nn.Parameter(torch.zeros(F)); self.ob = nn.Parameter(torch.zeros(F)); self.og = nn.Parameter(torch.zeros(F))
        # the adaptation kit reads inp / blocks / rec for the reconstruction loss
        self.inp, self.blocks, self.rec = net.inp, net.blocks, net.rec

    def gcm_params(self):
        return [self.ia, self.ib, self.ig, self.oa, self.ob, self.og]

    def forward(self, x, lid=None):
        g = torch.tanh(self.ig)[None, :, None, None]
        dyn = x[:, :4] + g * (self.ia[None, :, None, None] * x[:, :4] + self.ib[None, :, None, None]) * (x[:, 2:3] > 0.5)
        z = self.net(torch.cat([dyn, x[:, 4:]], 1))[:, :F]
        go = torch.tanh(self.og)[None, :, None]
        return z + go * (self.oa[None, :, None] * z + self.ob[None, :, None])


def mix_state(s0, s1, a):
    return {k: ((1 - a) * s0[k].float() + a * s1[k].float()).to(s0[k].dtype) if s0[k].dtype.is_floating_point else s1[k]
            for k in s0}


# ------------------------------------------------------------------------------------------------ SOLID pool
def pool_windows(ctx, stat, z, pool, days, cuts, rng, cap=600):
    """All pseudo ongoing windows (2-slot grid) of the pool days before their cut-offs, subsampled to cap."""
    ov, oq, ok, q = pool
    dd, tt = [], []
    for d in days:
        lim = min(cuts.get(int(d), 10 ** 6), q.shape[1] - 1)
        for t0 in range(H + 1, lim - F, 2):
            if q[d, t0 - H:t0].sum() >= 2 and q[d, t0 + 1:t0 + 1 + F].any():
                dd.append(int(d)); tt.append(t0)
    if not dd:
        return None
    dd, tt = np.array(dd), np.array(tt)
    if len(dd) > cap:
        k = np.sort(rng.choice(len(dd), cap, replace=False))
        dd, tt = dd[k], tt[k]
    hi, fi = tt[:, None] + np.arange(-H, 0), tt[:, None] + np.arange(1, F + 1)
    x = cnn.make_x(ctx, stat, ov[dd[:, None], hi], oq[dd[:, None], hi], ok[dd[:, None], hi], tt, z["weekday"][dd] >= 5)
    return dict(x=x, y=q[dd[:, None], fi].astype(np.float32), m=ok[dd[:, None], fi].astype(np.float32), t0=tt,
                d=dd, key=signature(x))


def signature(x):
    v = x[:, 0, -3:, :].astype(np.float32).reshape(len(x), -1)
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-6)


@torch.no_grad()
def embed(net, x, bs=48):
    """The adapted backbone's collapsed features (N, c, S), float16 on the GPU."""
    net.eval()
    out = []
    for a in range(0, len(x), bs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(net.collapse(net.blocks(net.inp(C.to_dev(x[a:a + bs]))))[:, :, 0].half())
    return torch.cat(out)


def solid2_predict(net, x, t0, P, kmax=32, steps=5, lr=5e-4, radius=8, day=None):
    """SOLID v2: neighbours by the backbone's own features on the target's queue neighbourhood (+-radius links
    around the links queued in its last 15 min), minus a time-of-day distance; similarity-weighted head update."""
    head = list(net.collapse.parameters()) + list(net.head.parameters())
    snap = [p.detach().clone() for p in head]
    fe_t = embed(net, x)
    if "feat" not in P:
        P["feat"] = embed(net, P["x"])
    out = np.zeros((len(x), F, x.shape[-1]), np.float32)
    k = int(min(kmax, max(8, len(P["x"]) // 4)))
    for n in range(len(x)):
        q = x[n, 3, -3:].max(0) > 0.5                              # links queued in the last 15 min
        nb = np.zeros(x.shape[-1], bool)
        for j in np.flatnonzero(q):
            nb[max(0, j - radius):j + radius + 1] = True
        if not nb.any():
            nb[:] = True
        nbt = torch.from_numpy(nb).to(DEV)
        a = fe_t[n][:, nbt].float().reshape(-1)
        b = P["feat"][:, :, nbt].float().reshape(len(P["x"]), -1)
        sim = (b @ a) / (b.norm(dim=1) * a.norm() + 1e-6)
        sim = sim.cpu().numpy() - 0.02 * np.abs(P["t0"] - t0[n]) / 12.0
        if day is not None:                                         # v3: same-day windows released by this origin
            ok_ = (P["d"] < day) | (P["t0"] + F < t0[n])
            sim = np.where(ok_, sim, -1e9)
        j = np.argsort(-sim)[:k]
        j = j[sim[j] > -1e8]
        if not len(j):
            out[n] = cnn.predict(net, x[n:n + 1])[0]
            continue
        w = np.exp((sim[j] - sim[j].max()) / 0.05)
        w = torch.from_numpy((w / w.sum() * len(j)).astype(np.float32)).to(DEV)
        set_trainable(net, head)
        opt = torch.optim.AdamW(head, lr=lr, weight_decay=0)
        xb, yb, mb = C.to_dev(P["x"][j]), torch.from_numpy(P["y"][j]).to(DEV), torch.from_numpy(P["m"][j]).to(DEV)
        net.train()
        for _ in range(steps):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lgt = net(xb)[:, :F].float()
            bce = nn.functional.binary_cross_entropy_with_logits(lgt, yb, reduction="none")
            loss = (bce * mb * w[:, None, None]).sum() / mb.sum().clamp(min=1) + cnn.LOVASZ * cnn.lovasz_hinge(lgt, yb, mb)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        out[n] = cnn.predict(net, x[n:n + 1])[0]
        with torch.no_grad():
            for p_, s_ in zip(head, snap):
                p_.copy_(s_)
    net.eval()
    return out


@torch.no_grad()
def embed_tfm(net, x, bs=48):
    """T2Tfm: the link tokens after the transformer blocks and the final norm, as (N, c, S)."""
    net.eval()
    out = []
    for a_ in range(0, len(x), bs):
        xb = C.to_dev(x[a_:a_ + bs])
        B, Cc, Hh, S = xb.shape
        seq = xb.permute(0, 3, 2, 1).reshape(B * S, Hh, Cc).float()
        with torch.autocast("cuda", enabled=False):
            _, hN = net.gru(seq)
        tok = hN[-1].view(B, S, -1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            f = net.norm(net.blocks(tok))
        out.append(f.permute(0, 2, 1).half())
    return torch.cat(out)


def solid_generic(net, x, t0, P, head, embed_fn, day=None, kmax=32, steps=5, lr=5e-4, radius=8):
    """solid2_predict for any member: head = the parameters fine-tuned per window, embed_fn = the features the
    neighbours are found with (N, c, S)."""
    snap = [p.detach().clone() for p in head]
    fe_t = embed_fn(net, x)
    if "feat_g" not in P:
        P["feat_g"] = embed_fn(net, P["x"])
    out = np.zeros((len(x), F, x.shape[-1]), np.float32)
    k = int(min(kmax, max(8, len(P["x"]) // 4)))
    for n in range(len(x)):
        q = x[n, 3, -3:].max(0) > 0.5
        nb = np.zeros(x.shape[-1], bool)
        for j in np.flatnonzero(q):
            nb[max(0, j - radius):j + radius + 1] = True
        if not nb.any():
            nb[:] = True
        nbt = torch.from_numpy(nb).to(DEV)
        a = fe_t[n][:, nbt].float().reshape(-1)
        b = P["feat_g"][:, :, nbt].float().reshape(len(P["x"]), -1)
        sim = ((b @ a) / (b.norm(dim=1) * a.norm() + 1e-6)).cpu().numpy() - 0.02 * np.abs(P["t0"] - t0[n]) / 12.0
        if day is not None:
            sim = np.where((P["d"] < day) | (P["t0"] + F < t0[n]), sim, -1e9)
        j = np.argsort(-sim)[:k]
        j = j[sim[j] > -1e8]
        if not len(j):
            out[n] = cnn.predict(net, x[n:n + 1])[0]
            continue
        w = np.exp((sim[j] - sim[j].max()) / 0.05)
        w = torch.from_numpy((w / w.sum() * len(j)).astype(np.float32)).to(DEV)
        set_trainable(net, head)
        opt = torch.optim.AdamW(head, lr=lr, weight_decay=0)
        xb, yb, mb = C.to_dev(P["x"][j]), torch.from_numpy(P["y"][j]).to(DEV), torch.from_numpy(P["m"][j]).to(DEV)
        net.train()
        for _ in range(steps):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lgt = net(xb)[:, :F].float()
            bce = nn.functional.binary_cross_entropy_with_logits(lgt, yb, reduction="none")
            loss = (bce * mb * w[:, None, None]).sum() / mb.sum().clamp(min=1) + cnn.LOVASZ * cnn.lovasz_hinge(lgt, yb, mb)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        out[n] = cnn.predict(net, x[n:n + 1])[0]
        with torch.no_grad():
            for p_, s_ in zip(head, snap):
                p_.copy_(s_)
    for p_ in net.parameters():
        p_.requires_grad_(True)
    net.eval()
    return out


def solid_predict(net, x, t0, P, k=32, steps=5, lr=5e-4):
    head = list(net.collapse.parameters()) + list(net.head.parameters())
    snap = [p.detach().clone() for p in head]
    sig = signature(x)
    out = np.zeros((len(x), F, x.shape[-1]), np.float32)
    for n in range(len(x)):
        sim = P["key"] @ sig[n] - 0.05 * (np.abs(P["t0"] - t0[n]) > 24)
        j = np.argsort(-sim)[:k]
        set_trainable(net, head)
        opt = torch.optim.AdamW(head, lr=lr, weight_decay=0)
        xb, yb, mb = C.to_dev(P["x"][j]), torch.from_numpy(P["y"][j]).to(DEV), torch.from_numpy(P["m"][j]).to(DEV)
        net.train()
        for _ in range(steps):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = net(xb)[:, :F].float()
            loss = lossf(lg, yb, mb)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        out[n] = cnn.predict(net, x[n:n + 1])[0]
        with torch.no_grad():
            for p, s in zip(head, snap):
                p.copy_(s)
    net.eval()
    return out


# ------------------------------------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--method", required=True,
                    choices=("base", "wise", "surgical", "solid", "gcm", "lagged", "solid2", "solid3"))
    args = ap.parse_args()
    t_start = time.time()
    if args.method == "wise":
        variants = [f"wise_{a}" for a in (0.25, 0.5, 0.75)] + [f"wisecarry_{a}" for a in (0.5, 0.75)] + ["base"]
    elif args.method == "surgical":
        variants = ["surg_inp", "surg_early", "surg_late", "surg_norm"]
    elif args.method == "solid":
        variants = ["solid"]
    elif args.method == "solid2":
        variants = ["solid2"]
    elif args.method == "solid3":
        variants = ["solid3"]
    elif args.method == "gcm":
        variants = ["gcm", "gcm+full"]
    elif args.method == "lagged":                               # C: time-lagged forecasts, base trajectory
        variants = ["base", "lag1", "lag2"]
    else:
        variants = ["base"]
    res = {}
    for i, panel in enumerate(C.T2_PANELS):
        ctx, stat, W = C.windows(i, panel)
        for s in C.SPLITS:
            z, og, on = W[s]["z"], W[s]["og"], W[s]["on"]
            pool = PA.pseudo_pool(ctx, z)
            days = np.unique(np.concatenate([on["d"], og["d"]]))
            firsts = {int(d): int(min(on["t0"][on["d"] == d].min(initial=10 ** 6),
                                      og["t0"][og["d"] == d].min(initial=10 ** 6))) for d in days}
            # one adaptation trajectory per "trajectory family"; wise_* share the base trajectory
            trajs = sorted({("base" if (v.startswith("wise_") or v.startswith("lag")) else v) for v in variants})
            preds = {v: np.zeros(og["fut"].shape, np.float32) for v in variants}
            for tr in trajs:
                net, _ = C.build_net(args.ckpt, "full")
                theta0 = {k: v.detach().clone() for k, v in net.state_dict().items()}
                model = GCM(net).to(DEV) if tr.startswith("gcm") else net
                if tr.startswith("surg_"):
                    params = subset(net, tr[5:])
                elif tr == "gcm":
                    params = model.gcm_params()
                elif tr == "gcm+full":
                    params = model.gcm_params() + subset(net, "full")
                else:
                    params = subset(net, "full")
                rng = np.random.default_rng(7 + i)
                mixer = copy.deepcopy(net) if tr == "base" and any(v.startswith("wise_") for v in variants) else None
                for d in days:
                    d = int(d)
                    past = np.arange(d)
                    pool_days = np.append(past, d) if firsts[d] > H + F + 2 else past
                    cuts = {int(k): (firsts[d] if int(k) == d else 10 ** 6) for k in pool_days}
                    if len(pool_days):
                        sup = lambda: PA.sup_batch(ctx, stat, z, pool, pool_days, cuts, 16, rng, "ongoing")  # noqa: E731
                        rec = lambda: C.past_batch(ctx, stat, z, past, 16, rng)  # noqa: E731
                        lr = 1e-2 if tr == "gcm" else 1e-4
                        w_rec = 0.0 if tr == "gcm" else 1.0
                        if tr == "gcm+full":                      # two groups: modules fast, backbone as shipped
                            set_trainable(net, params)
                            for p in model.gcm_params():
                                p.requires_grad_(True)
                            opt = torch.optim.AdamW([{"params": model.gcm_params(), "lr": 1e-2},
                                                     {"params": subset(net, "full"), "lr": 1e-4}], weight_decay=1e-4)
                            model.train()
                            for _ in range(30):
                                b = sup()
                                if b is None:
                                    break
                                xb, yb, mb = b
                                with torch.autocast("cuda", dtype=torch.bfloat16):
                                    lg = model(xb)[:, :F].float()
                                loss = lossf(lg, yb, mb)
                                r = rec()
                                if r is not None:
                                    loss = loss + C.ssl_loss(net, r)
                                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                            model.eval()
                        else:
                            adapt(model, params, sup, rec, 30, lr, w_rec)
                        if tr.startswith("wisecarry_"):
                            a = float(tr.split("_")[1])
                            net.load_state_dict(mix_state(theta0, net.state_dict(), a))
                    ig = np.flatnonzero(og["d"] == d)
                    if not len(ig):
                        continue
                    x = C.make_x(ctx, stat, og, ig, z)
                    if tr in ("solid", "solid2"):
                        P = pool_windows(ctx, stat, z, pool, pool_days, cuts, rng) if len(pool_days) else None
                        fn = solid_predict if tr == "solid" else solid2_predict
                        preds[tr][ig] = fn(net, x, og["t0"][ig], P) if P is not None else cnn.predict(net, x)
                    elif tr == "solid3":
                        days3 = np.append(np.arange(d), d)             # the whole day, filtered per window
                        cuts3 = {int(k): (int(og["t0"][ig].max()) if int(k) == d else 10 ** 6) for k in days3}
                        P = pool_windows(ctx, stat, z, pool, days3, cuts3, rng, cap=900)
                        preds[tr][ig] = solid2_predict(net, x, og["t0"][ig], P, day=d) if P is not None \
                            else cnn.predict(net, x)
                    else:
                        if tr in preds:
                            preds[tr][ig] = cnn.predict(model, x)
                        for L in (1, 2):                         # forecasts issued L slots before the origin
                            if f"lag{L}" not in preds:
                                continue
                            ov, oq, okk, _ = pool
                            dd, tl = og["d"][ig], og["t0"][ig] - L
                            hi = tl[:, None] + np.arange(-H, 0)
                            xl = cnn.make_x(ctx, stat, ov[dd[:, None], hi], oq[dd[:, None], hi], okk[dd[:, None], hi],
                                            tl, z["weekday"][dd] >= 5)
                            pl = cnn.predict(model, xl)          # steps tl+1 .. tl+6 = targets t0+1-L .. t0+6-L
                            out = np.full((len(ig), F, x.shape[-1]), np.nan, np.float32)
                            out[:, :F - L] = pl[:, L:]           # target step h (1-based) <- lagged step h + L
                            preds[f"lag{L}"][ig] = out
                        if mixer is not None:
                            for v in variants:
                                if v.startswith("wise_"):
                                    mixer.load_state_dict(mix_state(theta0, net.state_dict(), float(v.split("_")[1])))
                                    preds[v][ig] = cnn.predict(mixer, x)
            for v in variants:
                sc = C.ongoing_iou(np.nan_to_num(preds[v], nan=0.0), og, 0.5)
                res.setdefault((s, v), {})[panel] = (sc.mean(), sc[og["d"] <= 4].mean() if (og["d"] <= 4).any() else np.nan)
                np.savez(CACHE / f"fam_av_{args.tag}_{v}_{s}_{panel}.npz", og=preds[v].astype(np.float16))
                if v.startswith("lag"):
                    continue
            print(f"  {panel} {s} ({time.time() - t_start:.0f}s)", flush=True)
    for s in C.SPLITS:
        print(f"\n=== {s}: {args.tag} {args.method} (ongoing, single member) ===")
        for v in variants:
            d = res[(s, v)]
            fm = lambda j: C.family_mean({k: x[j] for k, x in d.items() if np.isfinite(x[j])})  # noqa: E731
            print(f"  {v:<16} all {fm(0):.4f}   days 1-5 {fm(1):.4f}")


if __name__ == "__main__":
    main()
