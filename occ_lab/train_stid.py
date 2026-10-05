"""A fourth family the user asked for: STID (Spatial-Temporal IDentity) with the occupancy channels.

Shao et al. 2022 showed that most of what a spatio-temporal graph network buys can be had from identity
information alone: tell a plain MLP *which link* and *which time of day and day of week* a series belongs to, and
it matches the graph models. That is the representation the three production members do not have -

    gs1   a gradient-boosted snapshot of hand-made features, no identity
    cnn3  dilated convolutions over (history x links), translation invariant along the corridor
    tfm8  a GRU per link then attention across links, permutation invariant apart from what the channels say

- none of them can learn "link 47 at 08:35 on a Tuesday behaves like this". STID can only do that, which is why
  its errors have a chance of sitting somewhere else (note 83's criterion: D_m, not its own IoU).

    embed    per link the 15 x 12 history is flattened and projected to D, then concatenated with a link-identity
             embedding, a time-of-day embedding and a day-of-week embedding
    body     3 residual MLP blocks over the concatenation (no mixing across links: identity carries the space)
    head     linear to the 6 horizons, emitted at once
    loss     BCE + 0.5 Lovasz, the production recipe; 30k steps, OneCycle 1e-3

    CUDA_VISIBLE_DEVICES=0 python occ_lab/train_stid.py --seed 0
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("T2CNN_LOVASZ", "0.5")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ["T2CNN_ARCH"] = "cnn"
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

LAB = Path(__file__).resolve().parent
PROJ = LAB.parent
for p in (PROJ, PROJ / "analysis", PROJ / "scripts", LAB):
    sys.path.insert(0, str(p))
import t2_cnn as cnn  # noqa: E402
from t2_eda02 import T2_PANELS, candidates  # noqa: E402
from t2_eda03 import F, H, earliest_onset  # noqa: E402
from t2_m1_models import Ctx  # noqa: E402
from tfb.physics import load_ramps  # noqa: E402
from tfb.t1eval import FIT_END  # noqa: E402
from train_occ import occ_channels  # noqa: E402

MODELS = LAB / "models"
EMB_DROP = 0.10


class STID(nn.Module):
    def __init__(self, cin, n_links, n_slots, d=96, ds=32, blocks=3):
        super().__init__()
        self.series = nn.Linear(cin * H, d)
        self.e_link = nn.Embedding(n_links + 1, ds, padding_idx=n_links)   # the last id is "unknown link"
        self.e_tod = nn.Embedding(n_slots, ds)
        self.e_dow = nn.Embedding(7, ds)
        c = d + 3 * ds
        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(c, c), nn.GELU(), nn.Linear(c, c)) for _ in range(blocks)])
        self.norm = nn.LayerNorm(c)
        self.head = nn.Linear(c, F)

    def forward(self, x, lid, tod, dow):                     # (B, C, H, S), (B, S), (B,), (B,) -> (B, F, S)
        B, C, Hh, S = x.shape
        h = self.series(x.permute(0, 3, 2, 1).reshape(B, S, Hh * C))
        e = torch.cat([self.e_link(lid),
                       self.e_tod(tod)[:, None].expand(-1, S, -1),
                       self.e_dow(dow)[:, None].expand(-1, S, -1)], -1)
        h = torch.cat([h, e], -1)
        for blk in self.blocks:
            h = h + blk(h)
        return self.head(self.norm(h)).permute(0, 2, 1)


class STIDLH(STID):
    """STID + a low-rank link x horizon term: logit[h, i] += sum_r A[i, r] B[h, r] ("link 52 keeps its queue at
    +5 and loses it by +30"). A has n_links x r parameters, so it can be re-fitted on the scored run alone."""
    def __init__(self, cin, n_links, n_slots, rank=4, **kw):
        super().__init__(cin, n_links, n_slots, **kw)
        self.lh_a = nn.Embedding(n_links + 1, rank, padding_idx=n_links)
        self.lh_b = nn.Parameter(torch.randn(F, rank) * 0.1)
        nn.init.zeros_(self.lh_a.weight)

    def forward(self, x, lid, tod, dow):
        base = super().forward(x, lid, tod, dow)                          # (B, F, S)
        return base + torch.einsum("bsr,fr->bfs", self.lh_a(lid).to(base.dtype), self.lh_b.to(base.dtype))


def windows(ctx, stat, rng, occ_prof, max_og=3000):
    """The production window set plus what STID needs: global link ids, the origin slot and the weekday."""
    z = ctx.z
    d_on, t_on = np.nonzero(earliest_onset(ctx.q))
    c = candidates(z)
    d_og, t_og = c["d"][c["ongoing"]], c["t0"][c["ongoing"]]
    if len(d_og) > max_og:
        s = rng.choice(len(d_og), max_og, replace=False)
        d_og, t_og = d_og[s], t_og[s]
    d, t0 = np.concatenate([d_on, d_og]), np.concatenate([t_on, t_og])
    hi, fi = t0[:, None] + np.arange(-H, 0), t0[:, None] + np.arange(1, F + 1)
    v, qf = z["true_speed"][d[:, None], hi], z["true_flow"][d[:, None], hi]
    ok = ctx.el[d[:, None], hi]
    x = cnn.make_x(ctx, stat, v, qf, ok, t0, z["weekday"][d] >= 5)
    x = np.concatenate([x, occ_channels(z["true_occ"][d[:, None], hi], ok, occ_prof)], 1)
    return dict(x=x, y=ctx.q[d[:, None], fi].astype(np.float16), m=ctx.el[d[:, None], fi],
                es=z["dates"][d] >= FIT_END, lid=np.arange(ctx.S) + ctx.loff,
                t0=t0.astype(np.int64), dow=z["weekday"][d].astype(np.int64))


def batch(parts, pool, bs, rng, n_links, train=True):
    """One panel per batch, all its links (STID has no spatial mixing, so a crop would only lose context)."""
    p = parts[rng.integers(len(parts))]
    j = np.sort(pool[id(p)][rng.integers(len(pool[id(p)]), size=bs)])
    tt = lambda a, dt: torch.from_numpy(np.ascontiguousarray(a)).to(cnn.DEV, dt)  # noqa: E731
    lid = np.broadcast_to(p["lid"], (bs, len(p["lid"]))).copy()
    if train and rng.random() < EMB_DROP:           # identity dropout, so the MLP cannot lean on it alone
        lid[:] = n_links
    return (tt(p["x"][j], torch.float32), tt(p["y"][j], torch.float32), tt(p["m"][j], torch.float32),
            tt(lid, torch.long), tt(p["t0"][j], torch.long), tt(p["dow"][j], torch.long))


@torch.no_grad()
def predict(net, x, lid, t0, dow, bs=64):
    net.eval()
    out = []
    tt = lambda a, dt: torch.from_numpy(np.ascontiguousarray(a)).to(cnn.DEV, dt)  # noqa: E731
    for a in range(0, len(x), bs):
        sl = slice(a, a + bs)
        n = len(x[sl])
        out.append(torch.sigmoid(net(tt(x[sl], torch.float32),
                                     tt(np.broadcast_to(lid, (n, len(lid))).copy(), torch.long),
                                     tt(t0[sl], torch.long), tt(dow[sl], torch.long))).float().cpu().numpy())
    net.train()
    return np.concatenate(out) if out else np.zeros((0, F, x.shape[-1]), np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lh-rank", type=int, default=0, help="> 0: STIDLH with this rank")
    args = ap.parse_args()
    MODELS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    cnn.rng = rng
    torch.manual_seed(args.seed)
    t_start = time.time()
    profs = np.load(MODELS / "occ_profiles.npz")
    parts, n_links, n_slots = [], 0, 0
    for i, panel in enumerate(T2_PANELS):
        ctx = Ctx(panel, i)
        ctx.loff = n_links
        n_links += ctx.S
        n_slots = max(n_slots, ctx.z["true_speed"].shape[1])
        stat = cnn.static_channels(ctx, load_ramps(panel, "train", ctx.z))
        parts.append(windows(ctx, stat, rng, profs[panel]))
        print(f"[{panel}] windows {len(parts[-1]['x']):,}, links {ctx.S}, channels {parts[-1]['x'].shape[1]} "
              f"({time.time() - t_start:.0f}s)", flush=True)
        ctx.z = ctx.q = ctx.el = None
    pool = {id(p): np.flatnonzero(~p["es"]) for p in parts}
    net = (STIDLH(parts[0]["x"].shape[1], n_links, n_slots, rank=args.lh_rank) if args.lh_rank
           else STID(parts[0]["x"].shape[1], n_links, n_slots)).to(cnn.DEV)
    print(f"STID: {n_links} links, {n_slots} slots, {sum(p.numel() for p in net.parameters()):,} parameters",
          flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    best, best_state = 1e9, None
    for step in range(1, args.steps + 1):
        x, y, m, lid, tod, dow = batch(parts, pool, cnn.BATCH, rng, n_links)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = net(x, lid, tod, dow)
        logit = logit.float()
        loss = cnn.loss_fn(logit, y, m) + cnn.LOVASZ * cnn.lovasz_hinge(logit, y, m)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0:
            tot, n = 0.0, 0.0
            for p in parts:
                es = np.flatnonzero(p["es"])[:400]
                if not len(es):
                    continue
                pr = predict(net, p["x"][es], p["lid"], p["t0"][es], p["dow"][es])
                yv, mv = p["y"][es].astype(np.float32), p["m"][es].astype(np.float32)
                bce = -(yv * np.log(pr + 1e-6) + (1 - yv) * np.log(1 - pr + 1e-6))
                tot += float((bce * mv).sum()); n += float(mv.sum())
            es_loss = tot / max(n, 1)
            flag = ""
            if es_loss < best:
                best, best_state, flag = es_loss, {k: v.detach().clone() for k, v in net.state_dict().items()}, " *"
            print(f"step {step}: train loss {loss.item():.4f}  ES bce {es_loss:.4f}{flag} "
                  f"({time.time() - t_start:.0f}s)", flush=True)
    name = f"stid_lh{args.lh_rank}_s{args.seed}.pt" if args.lh_rank else f"stid_s{args.seed}.pt"
    torch.save({"state": best_state, "n_links": n_links, "n_slots": n_slots, "cin": parts[0]["x"].shape[1],
                "lh_rank": args.lh_rank}, MODELS / name)
    print(f"saved {name} (ES bce {best:.4f}, {time.time() - t_start:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
