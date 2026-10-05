"""A heterogeneous onset model: STID at T+30 only.

Now that the organizers have confirmed the onset answer is always at T+30 (notes/92), onset is exactly "which
link breaks down first", a per-link propensity problem. That is the one thing a link-identity embedding is for,
and none of the production onset members has one: LightGBM H sees hand-made features with no identity, the CNN is
translation invariant along the corridor. Note 85 found STID to be the least correlated member ever measured on
ongoing; on onset its bias should fit even better.

    input     per link the 15 x 12 history (the note-84 channels, occupancy included), flattened and projected,
              concatenated with link-identity, time-of-day and day-of-week embeddings
    body      3 residual MLP blocks, no mixing across links
    head      ONE map: the T+30 queue logit
    windows   onset windows only (`earliest_onset`, the official selector condition)
    loss      BCE + 0.5 Lovasz at T+30

    CUDA_VISIBLE_DEVICES=0 python occ_lab/train_onset_stid.py --seed 0
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
from t2_eda02 import T2_PANELS  # noqa: E402
from t2_eda03 import F, H, earliest_onset  # noqa: E402
from t2_m1_models import Ctx  # noqa: E402
from tfb.physics import load_ramps  # noqa: E402
from tfb.t1eval import FIT_END  # noqa: E402
from train_occ import occ_channels  # noqa: E402
from graph_ident import adj as chain_adj, lappe  # noqa: E402
from train_stid import STID  # noqa: E402

MODELS = LAB / "models"
EMB_DROP = 0.10


class OnsetSTID(STID):
    """STID with a single output map instead of F, and an optional structured identity beside the free one.

    ident "id"     the free per-link embedding only (candidate AN)
          "lappe"  + a projection of the corridor's Laplacian positional encoding (STEMLP)
          "ab"     + the neighbourhood-averaged identity A E_link (ST-MLP)
          "mult"   a per-link MULTIPLICATIVE weight on every (history slot x channel) input before the series
                   projection: link x lag identity ("at link 37 the T-15 state matters most"). An additive
                   link x lag embedding would collapse into the link embedding after the linear projection.
          "pm"     + a pattern-memory identity: the link's history encoded and attended over K learned prototypes
                   (PM-MemNet / MegaCRN's memory idea), "which pre-onset pattern this link is in"
    """
    def __init__(self, cin, n_links, n_slots, d=96, ds=32, blocks=3, ident="id", k_lap=12, dropout=0.0):
        super().__init__(cin, n_links, n_slots, d, ds, blocks)
        self.ident, self.k_lap = ident, k_lap
        self.dropout = dropout                        # note 169 size study: residual-branch dropout (no new keys)
        extra = 0 if ident in ("id", "mult") else ds
        if ident == "lappe":
            self.to_lap = nn.Linear(k_lap, ds)
        if ident == "mult":
            self.g_link = nn.Embedding(n_links + 1, cin * H, padding_idx=n_links)
            nn.init.zeros_(self.g_link.weight)
        if ident == "pm":
            K, dq = 48, 32
            self.pm_q = nn.Sequential(nn.Linear(cin * H, dq), nn.GELU(), nn.Linear(dq, dq))
            self.pm_k = nn.Parameter(torch.randn(K, dq) / dq ** 0.5)
            self.pm_v = nn.Parameter(torch.randn(K, ds) * 0.1)
        c = d + 3 * ds + extra
        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(c, c), nn.GELU(), nn.Linear(c, c))
                                     for _ in range(blocks)])
        self.norm = nn.LayerNorm(c)
        self.head = nn.Linear(c, 1)

    def forward(self, x, lid, tod, dow, lap=None, adj=None):     # (B, C, H, S) -> (B, 1, S)
        B, C, Hh, S = x.shape
        flat = x.permute(0, 3, 2, 1).reshape(B, S, Hh * C)
        if self.ident == "mult":
            flat = flat * (1 + torch.tanh(self.g_link(lid))).to(flat.dtype)
        h = self.series(flat)
        e_link = self.e_link(lid)
        parts = [h, e_link]
        if self.ident == "pm":
            q = self.pm_q(flat)
            a = torch.softmax(q @ self.pm_k.t().to(q.dtype) / self.pm_k.shape[1] ** 0.5, -1)
            parts.append(a @ self.pm_v.to(a.dtype))
        if self.ident == "lappe":
            parts.append(self.to_lap(lap))
        elif self.ident == "ab":
            parts.append(torch.einsum("st,btc->bsc", adj, e_link))
        parts += [self.e_tod(tod)[:, None].expand(-1, S, -1), self.e_dow(dow)[:, None].expand(-1, S, -1)]
        h = torch.cat(parts, -1)
        for blk in self.blocks:
            h = h + nn.functional.dropout(blk(h), self.dropout, self.training)
        return self.head(self.norm(h)).permute(0, 2, 1)


def windows(ctx, stat, occ_prof):
    z = ctx.z
    d, t0 = np.nonzero(earliest_onset(ctx.q))
    hi = t0[:, None] + np.arange(-H, 0)
    v, qf = z["true_speed"][d[:, None], hi], z["true_flow"][d[:, None], hi]
    ok = ctx.el[d[:, None], hi]
    x = cnn.make_x(ctx, stat, v, qf, ok, t0, z["weekday"][d] >= 5)
    x = np.concatenate([x, occ_channels(z["true_occ"][d[:, None], hi], ok, occ_prof)], 1)
    tgt = t0 + F                                                     # T+30, the only step that can be queued
    return dict(x=x, y=ctx.q[d, tgt].astype(np.float16), m=ctx.el[d, tgt],
                es=z["dates"][d] >= FIT_END, lid=np.arange(ctx.S) + ctx.loff,
                t0=t0.astype(np.int64), dow=z["weekday"][d].astype(np.int64))


ID_DROP_LINK = 0.0            # note 169 regularization study: per-link identity dropout (replaces the batch-level one)


def batch(parts, pool, bs, rng, n_links, train=True):
    p = parts[rng.integers(len(parts))]
    j = np.sort(pool[id(p)][rng.integers(len(pool[id(p)]), size=bs)])
    tt = lambda a, dt: torch.from_numpy(np.ascontiguousarray(a)).to(cnn.DEV, dt)  # noqa: E731
    lid = np.broadcast_to(p["lid"], (bs, len(p["lid"]))).copy()
    if train and ID_DROP_LINK > 0:
        lid[rng.random(lid.shape) < ID_DROP_LINK] = n_links
    elif train and rng.random() < EMB_DROP:
        lid[:] = n_links
    lap = p.get("lap")
    if lap is not None:
        lap = np.broadcast_to(lap, (bs,) + lap.shape).copy()
        if train:                     # eigenvector signs are arbitrary, so make the net ignore them
            lap = lap * rng.choice([-1.0, 1.0], size=(bs, 1, lap.shape[-1])).astype(np.float32)
    return (tt(p["x"][j], torch.float32), tt(p["y"][j], torch.float32), tt(p["m"][j], torch.float32),
            tt(lid, torch.long), tt(p["t0"][j], torch.long), tt(p["dow"][j], torch.long),
            None if lap is None else tt(lap, torch.float32),
            None if p.get("adj") is None else tt(p["adj"], torch.float32))


@torch.no_grad()
def predict(net, x, lid, t0, dow, bs=64, lap=None, adj=None):
    net.eval()
    out = []
    tt = lambda a, dt: torch.from_numpy(np.ascontiguousarray(a)).to(cnn.DEV, dt)  # noqa: E731
    adj_t = None if adj is None else tt(adj, torch.float32)
    for a in range(0, len(x), bs):
        sl = slice(a, a + bs)
        n = len(x[sl])
        lap_t = None if lap is None else tt(np.broadcast_to(lap, (n,) + lap.shape).copy(), torch.float32)
        logit = net(tt(x[sl], torch.float32), tt(np.broadcast_to(lid, (n, len(lid))).copy(), torch.long),
                    tt(t0[sl], torch.long), tt(dow[sl], torch.long), lap_t, adj_t)
        out.append(torch.sigmoid(logit[:, 0]).float().cpu().numpy())
    net.train()
    return np.concatenate(out) if out else np.zeros((0, x.shape[-1]), np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ident", default="id", choices=("id", "lappe", "ab", "mult", "pm"))
    ap.add_argument("--k-lap", type=int, default=12)
    ap.add_argument("--d", type=int, default=96, help="note 169 size study: history projection width")
    ap.add_argument("--ds", type=int, default=32, help="identity / time embedding width")
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--size-tag", default="", help="suffix of the saved file, e.g. _M (never overwrite production)")
    ap.add_argument("--id-drop-link", type=float, default=0.0, help="per-link identity dropout (else batch-level 0.10)")
    ap.add_argument("--id-consist", type=float, default=0.0,
                    help="penalty on the gap between predictions with and without the link identity")
    ap.add_argument("--tod-tv", type=float, default=0.0, help="smoothness penalty on neighbouring time-of-day embeddings")
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--emb-drop", type=float, default=0.10, help="batch-level identity dropout (production 0.10)")
    args = ap.parse_args()
    global ID_DROP_LINK, EMB_DROP
    ID_DROP_LINK = args.id_drop_link
    EMB_DROP = args.emb_drop
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
        parts.append(windows(ctx, stat, profs[panel]))
        if args.ident == "lappe":
            parts[-1]["lap"] = lappe(ctx, args.k_lap)
        elif args.ident == "ab":
            parts[-1]["adj"] = chain_adj(ctx)
        print(f"[{panel}] onset windows {len(parts[-1]['x']):,}, links {ctx.S} ({time.time() - t_start:.0f}s)",
              flush=True)
        ctx.z = ctx.q = ctx.el = None
    pool = {id(p): np.flatnonzero(~p["es"]) for p in parts}
    net = OnsetSTID(parts[0]["x"].shape[1], n_links, n_slots, d=args.d, ds=args.ds, blocks=args.blocks,
                    ident=args.ident, k_lap=args.k_lap, dropout=args.dropout).to(cnn.DEV)
    print(f"OnsetSTID: {n_links} links, {sum(p.numel() for p in net.parameters()):,} parameters", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    best, best_state = 1e9, None
    for step in range(1, args.steps + 1):
        x, y, m, lid, tod, dow, lap, adjm = batch(parts, pool, cnn.BATCH, rng, n_links)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = net(x, lid, tod, dow, lap, adjm)[:, 0]
        logit = logit.float()
        loss = cnn.loss_fn(logit[:, None], y[:, None], m[:, None]) \
            + cnn.LOVASZ * cnn.lovasz_hinge(logit[:, None], y[:, None], m[:, None])
        if args.id_consist > 0:                    # the same windows with every link identity unknown
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg0 = net(x, torch.full_like(lid, n_links), tod, dow, lap, adjm)[:, 0].float()
            gap = (torch.sigmoid(logit) - torch.sigmoid(lg0)) ** 2
            loss = loss + args.id_consist * (gap * m).sum() / m.sum().clamp(min=1) * 100.0
        if args.tod_tv > 0:
            e_t = net.e_tod.weight
            loss = loss + args.tod_tv * ((e_t[1:] - e_t[:-1]) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 250 == 0:
            tot, n = 0.0, 0.0
            for p in parts:
                es = np.flatnonzero(p["es"])[:400]
                if not len(es):
                    continue
                pr = predict(net, p["x"][es], p["lid"], p["t0"][es], p["dow"][es],
                             lap=p.get("lap"), adj=p.get("adj"))
                yv, mv = p["y"][es].astype(np.float32), p["m"][es].astype(np.float32)
                bce = -(yv * np.log(pr + 1e-6) + (1 - yv) * np.log(1 - pr + 1e-6))
                tot += float((bce * mv).sum()); n += float(mv.sum())
            es_loss = tot / max(n, 1)
            flag = ""
            if es_loss < best:
                best, best_state, flag = es_loss, {k: v.detach().clone() for k, v in net.state_dict().items()}, " *"
            print(f"step {step}: train {loss.item():.4f}  ES bce(T+30) {es_loss:.5f}{flag} "
                  f"({time.time() - t_start:.0f}s)", flush=True)
    tag = ("" if args.ident == "id" else f"_{args.ident}") + args.size_tag
    torch.save({"state": best_state, "n_links": n_links, "n_slots": n_slots, "cin": parts[0]["x"].shape[1],
                "ident": args.ident, "k_lap": args.k_lap, "d": args.d, "ds": args.ds, "blocks": args.blocks,
                "dropout": args.dropout, "es_bce": best,
                "reg": dict(id_drop_link=args.id_drop_link, id_consist=args.id_consist, tod_tv=args.tod_tv, wd=args.wd)},
               MODELS / f"onset_stid{tag}_s{args.seed}.pt")
    print(f"saved onset_stid{tag}_s{args.seed}.pt (ES {best:.5f}, {time.time() - t_start:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
