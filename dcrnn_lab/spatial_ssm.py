"""A fourth ongoing family, second attempt: a bidirectional SELECTIVE SCAN ALONG THE CORRIDOR (spatial Mamba).

Note 83 rejected the DCRNN version: its autoregressive decoder carried its own error from horizon to horizon and
doubled the error on persistence cells, and its errors correlated with the members exactly as much as they
correlate with each other. Two corrections here, following the user's reasoning:

  * the state space runs over LINKS, not over time. The history is 12 slots, where a state-space model has no
    advantage, while the corridor has 80-350 links with a physical direction (upstream -> downstream). The
    transformer attends to all links without order or decay, the CNN mixes them locally with dilated convolutions,
    and the LightGBM sees a snapshot, so a directed scan with an input-dependent decay - "how far downstream does
    this disturbance reach" - is the bias none of them has.
  * the six horizons are emitted at once, as the existing members do, so nothing feeds a prediction back.

    embed     per link: the 12 history slots x channels flattened through a 1x1 conv -> token (B, S, c)
    scan      h_i = a_i * h_(i-1) + b_i * x_i with a_i = sigmoid(...) per link and channel, run upstream and
              downstream and summed; computed in log space with a cumulative sum, so there is no Python loop
    head      two scan blocks with residual connections and LayerNorm, then a linear head to the 6 horizons
    loss      BCE + 0.5 Lovasz, as in production

    CUDA_VISIBLE_DEVICES=0 python dcrnn_lab/spatial_ssm.py --seed 0 --steps 30000
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ["T2CNN_ARCH"] = "tfm"
os.environ.setdefault("T2CNN_LOVASZ", "0.5")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

LAB = Path(__file__).resolve().parent
PROJ = LAB.parent
for p in (PROJ, PROJ / "analysis", PROJ / "scripts"):
    sys.path.insert(0, str(p))
import t2_cnn as cnn  # noqa: E402
from t2_eda02 import T2_PANELS  # noqa: E402
from t2_eda03 import F, H  # noqa: E402
from t2_m1_models import Ctx  # noqa: E402
from tfb.physics import load_ramps  # noqa: E402

MODELS = LAB / "models"


CHUNK = 8


def scan(a, b):
    """h_i = a_i h_(i-1) + b_i along dim 1 (links), a in (0, 1).

    Chunked: inside a chunk of CHUNK links the recurrence is solved with a cumulative sum in log space (the
    exponent stays bounded), and only the chunk carries are passed sequentially. A plain cumulative sum over 350
    links overflows, which is what produced NaNs in the first version.
    """
    a = a.float().clamp(min=0.05, max=0.999)
    b = b.float()
    B, S, C = a.shape
    pad = (-S) % CHUNK
    if pad:
        a = torch.cat([a, torch.ones(B, pad, C, device=a.device)], 1)
        b = torch.cat([b, torch.zeros(B, pad, C, device=b.device)], 1)
    n = a.shape[1] // CHUNK
    a, b = a.view(B, n, CHUNK, C), b.view(B, n, CHUNK, C)
    A = torch.cumsum(torch.log(a), 2)
    eA = torch.exp(A)
    local = eA * torch.cumsum(b * torch.exp(-A), 2)          # the chunk's own contribution
    outs, carry = [], torch.zeros(B, C, device=a.device)      # a list, not in-place writes, so autograd works
    for k in range(n):                                        # n = S / CHUNK sequential steps, not S
        blk = local[:, k] + eA[:, k] * carry[:, None]
        outs.append(blk)
        carry = blk[:, -1]
    return torch.stack(outs, 1).reshape(B, -1, C)[:, :S]


class ScanBlock(nn.Module):
    """One selective scan upstream and downstream, gated, with a residual connection."""
    def __init__(self, c):
        super().__init__()
        self.norm = nn.LayerNorm(c)
        self.to_a = nn.Linear(c, 2 * c)
        self.to_b = nn.Linear(c, 2 * c)
        self.gate = nn.Linear(c, c)
        self.out = nn.Linear(2 * c, c)

    def forward(self, x):                                    # (B, S, c)
        h = self.norm(x)
        a = torch.sigmoid(self.to_a(h))
        b = self.to_b(h)
        c = h.shape[-1]
        fwd = scan(a[..., :c], b[..., :c])                                        # upstream -> downstream
        bwd = scan(a[..., c:].flip(1), b[..., c:].flip(1)).flip(1)                # downstream -> upstream
        y = self.out(torch.cat([fwd, bwd], -1)) * torch.sigmoid(self.gate(h))
        return x + y


class SpatialSSM(nn.Module):
    def __init__(self, cin, c=64, blocks=4):
        super().__init__()
        self.embed = nn.Linear(cin * H, c)
        self.blocks = nn.ModuleList([ScanBlock(c) for _ in range(blocks)])
        self.norm = nn.LayerNorm(c)
        self.head = nn.Linear(c, F)

    def forward(self, x, lid=None):                          # (B, C, H, S) -> (B, F, S)
        B, C, Hh, S = x.shape
        tok = self.embed(x.permute(0, 3, 2, 1).reshape(B, S, Hh * C))
        for blk in self.blocks:
            tok = blk(tok)
        return self.head(self.norm(tok)).permute(0, 2, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--blocks", type=int, default=4)
    args = ap.parse_args()
    MODELS.mkdir(parents=True, exist_ok=True)
    cnn.rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    t0 = time.time()
    parts = []
    for i, panel in enumerate(T2_PANELS):
        ctx = Ctx(panel, i)
        ctx.loff = 0
        stat = cnn.static_channels(ctx, load_ramps(panel, "train", ctx.z))
        parts.append(cnn.train_windows(ctx, stat))
        print(f"[{panel}] windows {len(parts[-1]['x']):,} ({time.time() - t0:.0f}s)", flush=True)
    pool = {id(p): np.flatnonzero(~p["es"]) for p in parts}
    net = SpatialSSM(parts[0]["x"].shape[1], cnn.WIDTH, args.blocks).to(cnn.DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    best, best_state = 1e9, None
    for step in range(1, args.steps + 1):
        x, y, m, _ = cnn.batch(parts, pool, cnn.BATCH)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = net(x)
        logit = logit.float()
        loss = cnn.loss_fn(logit, y, m) + cnn.LOVASZ * cnn.lovasz_hinge(logit, y, m)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0:
            with torch.no_grad():
                tot, n = 0.0, 0.0
                for p in parts:
                    es = np.flatnonzero(p["es"])[:400]
                    if not len(es):
                        continue
                    pr = cnn.predict(net, p["x"][es])
                    yv, mv = p["y"][es].astype(np.float32), p["m"][es].astype(np.float32)
                    bce = -(yv * np.log(pr + 1e-6) + (1 - yv) * np.log(1 - pr + 1e-6))
                    tot += float((bce * mv).sum()); n += float(mv.sum())
            es_loss = tot / max(n, 1)
            flag = ""
            if es_loss < best:
                best, best_state, flag = es_loss, {k: v.detach().clone() for k, v in net.state_dict().items()}, " *"
            print(f"step {step}: train loss {loss.item():.4f}  ES bce {es_loss:.4f}{flag} ({time.time() - t0:.0f}s)",
                  flush=True)
    torch.save(best_state, MODELS / f"ssm_s{args.seed}.pt")
    print(f"saved ssm_s{args.seed}.pt (ES bce {best:.4f}, {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
