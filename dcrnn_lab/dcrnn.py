"""A fourth ongoing model family: diffusion-GRU encoder + autoregressive 6-step decoder (DCRNN-style).

Why this one: note 80 measured that the ongoing blend wins on the 2.9% of cells its three members dispute, i.e.
on decorrelated errors, and note 52 put the error mass on the queue front extending at T+25/T+30. The three
members all emit the six horizons at once; none of them carries a state forward from one horizon to the next.
That recurrence is what this model adds. (The corridor is a path graph, so the diffusion convolution itself
overlaps with the CNN's dilated link convolutions; the decoder is the genuinely new part.)

    encoder   per-link GRU over the 12 history slots, with the link states mixed after each step by a
              bidirectional diffusion convolution: shifts of 1, 2 and 3 links upstream and downstream, learned
              weights (a truncated random walk on the directed path graph, as in DCRNN)
    decoder   a GRU cell unrolled 6 times; at each step it takes the previous step's predicted queue probability
              (scheduled sampling: the truth early in training, its own prediction later) plus the static link
              features, and emits the queue logit of that horizon
    loss      BCE + 0.5 Lovasz, exactly the production recipe, so the comparison is about the architecture only

    CUDA_VISIBLE_DEVICES=0 python dcrnn_lab/dcrnn.py --seed 0 --steps 30000
Writes dcrnn_lab/models/dcrnn_s<seed>.pt (best early-stopping BCE on the 2031-01/02 windows).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ["T2CNN_ARCH"] = "tfm"        # whole-panel batches, as for the transformers
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
SHIFTS = (1, 2, 3)


class Diffusion(nn.Module):
    """h (B, S, c) -> mixed states: the link itself plus shifted copies from up- and downstream."""
    def __init__(self, c):
        super().__init__()
        self.lin = nn.Linear(c * (1 + 2 * len(SHIFTS)), c)

    def forward(self, h):
        parts = [h]
        for k in SHIFTS:
            up = torch.cat([h[:, :k] * 0, h[:, :-k]], 1)        # from upstream (smaller index)
            dn = torch.cat([h[:, k:], h[:, -k:] * 0], 1)        # from downstream
            parts += [up, dn]
        return self.lin(torch.cat(parts, -1))


class DCRNN(nn.Module):
    def __init__(self, cin, c=64):
        super().__init__()
        self.enc = nn.GRUCell(cin, c)
        self.mix_e = Diffusion(c)
        self.dec = nn.GRUCell(cin + 1, c)                        # + the previous horizon's queue probability
        self.mix_d = Diffusion(c)
        self.head = nn.Linear(c, 1)
        self.c = c

    def forward(self, x, y_teacher=None, p_teacher=0.0):
        """x (B, C, H, S) -> (B, F, S) logits. y_teacher (B, F, S) feeds scheduled sampling."""
        B, C, Hh, S = x.shape
        h = x.new_zeros(B * S, self.c)
        seq = x.permute(0, 3, 2, 1).reshape(B * S, Hh, C)
        for t in range(Hh):
            h = self.enc(seq[:, t], h)
            h = self.mix_e(h.view(B, S, self.c)).reshape(B * S, self.c)
        last = seq[:, -1]
        prev = (x[:, 3, -1] > 0.5).float().reshape(B * S, 1)     # the queued flag at the end of the history
        out = []
        for k in range(F):
            h = self.dec(torch.cat([last, prev], -1), h)
            h = self.mix_d(h.view(B, S, self.c)).reshape(B * S, self.c)
            logit = self.head(h)
            out.append(logit.view(B, S))
            p = torch.sigmoid(logit.detach())
            if y_teacher is not None and p_teacher > 0:
                use = (torch.rand_like(p) < p_teacher).float()
                p = use * y_teacher[:, k].reshape(B * S, 1) + (1 - use) * p
            prev = p
        return torch.stack(out, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--teacher-end", type=float, default=0.5,
                    help="share of training over which the teacher probability decays from 1 to 0")
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
    net = DCRNN(parts[0]["x"].shape[1], cnn.WIDTH).to(cnn.DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    best, best_state = 1e9, None
    for step in range(1, args.steps + 1):
        x, y, m, _ = cnn.batch(parts, pool, cnn.BATCH)
        p_teacher = max(0.0, 1.0 - step / (args.teacher_end * args.steps))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = net(x, y, p_teacher)
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
                    es = np.flatnonzero(p["es"])[:200]
                    if not len(es):
                        continue
                    pr = []
                    for a in range(0, len(es), 32):
                        xb = torch.from_numpy(np.ascontiguousarray(p["x"][es[a:a + 32]])).to(cnn.DEV, torch.float32)
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            pr.append(torch.sigmoid(net(xb)).float().cpu().numpy())
                    pr = np.concatenate(pr)
                    yv, mv = p["y"][es].astype(np.float32), p["m"][es].astype(np.float32)
                    bce = -(yv * np.log(pr + 1e-6) + (1 - yv) * np.log(1 - pr + 1e-6))
                    tot += float((bce * mv).sum()); n += float(mv.sum())
            es_loss = tot / max(n, 1)
            flag = ""
            if es_loss < best:
                best, best_state, flag = es_loss, {k: v.detach().clone() for k, v in net.state_dict().items()}, " *"
            print(f"step {step}: train loss {loss.item():.4f}  teacher {p_teacher:.2f}  ES bce {es_loss:.4f}{flag} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    torch.save(best_state, MODELS / f"dcrnn_s{args.seed}.pt")
    print(f"saved dcrnn_s{args.seed}.pt (ES bce {best:.4f}, {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
