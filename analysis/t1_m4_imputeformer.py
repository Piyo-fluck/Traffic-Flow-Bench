"""Task 1, a second family: an ImputeFormer-style low-rank bidirectional imputer.

Note 96 closed the structural arguments - the mask is i.i.d. per (day, slot, link), and cells with no temporal
neighbour carry error in proportion to their number - so this is not here to fill long gaps. It is here because
90% of the remaining Task 1 error is the free-flow flow bulk, the production family is three dilated CNNs, and a
different function class over the same inputs may fit that bulk differently.

What is taken from ImputeFormer (KDD 2024):

    projected temporal attention   the day's 288 slots are summarised onto k learnable prototypes and read back,
                                   which induces a low-rank temporal structure instead of free T x T attention
    embedded spatial attention     attention across the corridor's links whose queries and keys come from a
                                   learned link-identity embedding, not from the noisy values
    bidirectional by construction  Task 1 is offline, so a slot attends to the whole day, future included

Everything else matches `t1_m3_cnn`: the same (D, C, T, S) inputs, the same profile-deviation targets, the same
masked loss, the same fit/ES/test day split. The self-supervised hiding now draws the official regime rates
(0.20 / 0.30 / 0.50, MASK_SPEC) at (slot, link) level instead of a generic 0-30%.

    M4_EPOCHS=15 py analysis/t1_m4_imputeformer.py            # all panels
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
import t1_m3_cnn as m3  # noqa: E402
from tfb.data import PANELS  # noqa: E402
from tfb.metrics import state_score  # noqa: E402
from tfb.t1eval import family_mean  # noqa: E402

DEV = "cuda"
D_MODEL = int(os.environ.get("M4_WIDTH", 96))
N_PROTO = int(os.environ.get("M4_PROTO", 16))
BLOCKS = int(os.environ.get("M4_BLOCKS", 3))
HEADS = 4
BATCH = int(os.environ.get("M4_BATCH", 6))
CROP_S = 64
STEPS = int(os.environ.get("M4_STEPS", 400))
EPOCHS = int(os.environ.get("M4_EPOCHS", 15))
LR = float(os.environ.get("M4_LR", 1e-3))
SEED = int(os.environ.get("M4_SEED", 0))
RATES = (0.20, 0.30, 0.50)          # the official regime rates


class ProjTemporal(nn.Module):
    """Low-rank temporal attention: tokens -> k prototypes -> tokens."""
    def __init__(self, d, k, heads):
        super().__init__()
        self.proto = nn.Parameter(torch.randn(k, d) * 0.02)
        self.down = nn.MultiheadAttention(d, heads, batch_first=True)
        self.up = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):                                   # (N, T, d)
        h = self.norm(x)
        p = self.proto.unsqueeze(0).expand(h.shape[0], -1, -1)
        summ, _ = self.down(p, h, h, need_weights=False)     # k summaries of the whole day
        back, _ = self.up(h, summ, summ, need_weights=False)
        return x + back


class EmbSpatial(nn.Module):
    """Attention across links whose queries and keys carry the link-identity embedding."""
    def __init__(self, d, heads):
        super().__init__()
        self.att = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm = nn.LayerNorm(d)

    def forward(self, x, e):                                # (N, S, d), (N, S, d)
        h = self.norm(x)
        q = h + e
        out, _ = self.att(q, q, h, need_weights=False)
        return x + out


class FFN(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, x):
        return x + self.net(x)


class ImputeFormer(nn.Module):
    def __init__(self, cin, n_links, d=D_MODEL, k=N_PROTO, blocks=BLOCKS):
        super().__init__()
        self.inp = nn.Linear(cin, d)
        self.e_link = nn.Embedding(n_links + 1, d, padding_idx=n_links)
        self.temporal = nn.ModuleList([ProjTemporal(d, k, HEADS) for _ in range(blocks)])
        self.spatial = nn.ModuleList([EmbSpatial(d, HEADS) for _ in range(blocks)])
        self.ffn = nn.ModuleList([FFN(d) for _ in range(blocks)])
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, m3.N_OUT))

    def forward(self, x, lid):                              # (B, C, T, S), (B, S) -> (B, N_OUT, T, S)
        B, C, T, S = x.shape
        h = self.inp(x.permute(0, 2, 3, 1))                 # (B, T, S, d)
        e = self.e_link(lid)                                # (B, S, d)
        h = h + e.unsqueeze(1)
        for tm, sp, ff in zip(self.temporal, self.spatial, self.ffn):
            h = tm(h.permute(0, 2, 1, 3).reshape(B * S, T, -1)).reshape(B, S, T, -1).permute(0, 2, 1, 3)
            h = sp(h.reshape(B * T, S, -1), e.repeat_interleave(T, 0)).reshape(B, T, S, -1)
            h = ff(h)
        return self.head(h).permute(0, 3, 1, 2)


def sample_batch(parts, rng):
    xs, ys, ms, es, ls = [], [], [], [], []
    for _ in range(BATCH):
        p = parts[rng.integers(len(parts))]
        d = p["tr_days"][rng.integers(len(p["tr_days"]))]
        S = p["X"].shape[-1]
        s0 = rng.integers(0, max(S - CROP_S, 0) + 1)
        sl = slice(s0, s0 + CROP_S)
        x, y, m, e = p["X"][d, :, :, sl], p["Y"][d, :, :, sl], p["M"][d, :, sl], p["E"][d, :, sl]
        lid = p["lid"][sl]
        pad = CROP_S - x.shape[-1]
        if pad > 0:
            x, y = np.pad(x, ((0, 0), (0, 0), (0, pad))), np.pad(y, ((0, 0), (0, 0), (0, pad)))
            m, e = np.pad(m, ((0, 0), (0, pad))), np.pad(e, ((0, 0), (0, pad)))
            lid = np.pad(lid, (0, pad), constant_values=p["n_links"])
        xs.append(x); ys.append(y); ms.append(m); es.append(e); ls.append(lid)
    to = lambda a, dt: torch.from_numpy(np.stack(a)).to(DEV, dt)  # noqa: E731
    return (to(xs, torch.float32), to(ys, torch.float32), to(ms, torch.bool), to(es, torch.bool),
            to(ls, torch.long))


def hide_official(x, m, e, gen):
    """Hide visible eligible cells at one of the official regime rates, per (slot, link) as the release does."""
    B = x.shape[0]
    pick = torch.randint(0, len(RATES), (B, 1, 1), device=DEV, generator=gen)
    rate = torch.tensor(RATES, device=DEV)[pick]
    hide = (x[:, 2] > 0.5) & e & (torch.rand(x[:, 2].shape, device=DEV, generator=gen) < rate)
    keep = (~hide).unsqueeze(1).float()
    x = x.clone()
    x[:, 0:4] = x[:, 0:4] * keep
    return x, m | hide


@torch.no_grad()
def predict_days(net, p, days, chunk=128):
    pv, pq = p["base_v"].copy(), p["base_q"].copy()
    lanes = p["z"]["lanes"]
    lid = torch.from_numpy(p["lid"][None]).to(DEV, torch.long)
    for d in days:
        x = torch.from_numpy(p["X"][d:d + 1].astype(np.float32)).to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = net(x, lid)
        out = out.float()[0].cpu().numpy()
        pv[d] = p["base_v"][d] + out[0] * m3.SCALE_V
        pq[d] = p["base_q"][d] + out[1] * m3.SCALE_Q * lanes
    return pv, pq


def evaluate(net, parts, key):
    net.eval()
    scores = {}
    for p in parts:
        days = p[key]
        pv, pq = predict_days(net, p, days)
        cells = np.zeros(len(p["z"]["dates"]), bool)
        cells[days] = True
        scores[p["panel"]] = state_score(p["z"], pv, pq, cells=cells[:, None, None])["mean"]["S"]
    net.train()
    return family_mean(scores), scores


def main() -> None:
    panels = sys.argv[1:] or PANELS
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    gen = torch.Generator(device=DEV).manual_seed(SEED)
    t0 = time.time()
    parts, n_links = [], 0
    for panel in panels:
        p = m3.panel_tensors(panel)
        p["lid"] = np.arange(p["X"].shape[-1]) + n_links
        n_links += p["X"].shape[-1]
        parts.append(p)
    for p in parts:
        p["n_links"] = n_links
    print(f"loaded {len(parts)} panels in {time.time() - t0:.0f}s; channels {parts[0]['X'].shape[1]}; "
          f"links {n_links}", flush=True)
    net = ImputeFormer(parts[0]["X"].shape[1], n_links).to(DEV)
    print(f"params {sum(q.numel() for q in net.parameters()) / 1e6:.2f}M "
          f"(d {D_MODEL}, {N_PROTO} prototypes, {BLOCKS} blocks)", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=EPOCHS * STEPS, pct_start=0.1)
    best, best_state = -1.0, None
    ckpt = ROOT / "work" / "models"
    ckpt.mkdir(parents=True, exist_ok=True)
    for ep in range(EPOCHS):
        t1, tot = time.time(), 0.0
        for _ in range(STEPS):
            x, y, m, e, lid = sample_batch(parts, rng)
            x, m = hide_official(x, m, e, gen)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = net(x, lid)
            loss = m3.loss_fn(out.float(), y, m)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item()
        es_score, _ = evaluate(net, parts, "es_days")
        flag = ""
        if es_score > best:
            best, best_state, flag = es_score, {k: v.detach().clone() for k, v in net.state_dict().items()}, " *"
        print(f"epoch {ep + 1:>2}: loss {tot / STEPS:.4f}  ES S_state {es_score:.4f} "
              f"({time.time() - t1:.0f}s){flag}", flush=True)
    net.load_state_dict(best_state)
    name = "_".join(panels) if len(panels) <= 2 else "all"
    tag = os.environ.get("M4_TAG", "m4")
    torch.save(best_state, ckpt / f"t1_{tag}_{name}.pt")
    te, per = evaluate(net, parts, "te_days")
    print(f"\nbest ES {best:.4f}; TEST S_state {te:.4f}")
    for k, v in per.items():
        print(f"   {k:<12} {v:.4f}")
    preds = ROOT / "work" / "preds"
    preds.mkdir(parents=True, exist_ok=True)
    net.eval()
    for p in parts:
        pv, pq = predict_days(net, p, p["te_days"])
        np.savez_compressed(preds / f"t1_{tag}_{p['panel']}_test.npz", days=p["te_days"],
                            pv=pv[p["te_days"]].astype(np.float32), pq=pq[p["te_days"]].astype(np.float32))
    print(f"saved test predictions to work/preds/t1_{tag}_*", flush=True)


if __name__ == "__main__":
    main()
