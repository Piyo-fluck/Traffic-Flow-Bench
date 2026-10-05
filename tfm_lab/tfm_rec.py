"""Plan B: idea 9b (past-day self-supervised adaptation) for the ongoing TRANSFORMER ensemble.

The production transformers (t2_cnn_lov_tfm_s3..s10, weight 0.60 of the ongoing blend) have no reconstruction
head, so they cannot be adapted. This module adds one and retrains them with the same recipe.

  T2TfmRec   analysis/t2_cnn.py:T2Tfm + rec = Linear(c, 2 x H): from each link's final token, reconstruct the v/vf
             and q/cap of all 12 history slots of that link. Training hides 20% of the history cells from the
             input (as the CNN rec head, note 67) and scores the reconstruction on hidden observed cells.
  train      the tfm recipe of t2_cnn.py (whole-panel batches of 32, BCE + 0.5 Lovasz, OneCycle lr 1e-3, 30k steps,
             ES on the 2031-01/02 windows) + 0.3 x the reconstruction loss.

    CUDA_VISIBLE_DEVICES=0 python tfm_lab/tfm_rec.py --seed 3        # -> tfm_lab/models/t2_tfm_rec_s3.pt
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ["T2CNN_ARCH"] = "tfm"          # cnn.batch then serves whole panels, as for the production transformers
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

REC_W = 0.3
MODELS = LAB / "models"


class T2TfmRec(cnn.T2Tfm):
    def __init__(self, cin, c=64):
        super().__init__(cin, c)
        self.rec = nn.Linear(c, 2 * H)

    def tokens(self, x):                                    # (B, C, H, S) -> (B, S, c)
        B, C, Hh, S = x.shape
        seq = x.permute(0, 3, 2, 1).reshape(B * S, Hh, C).float()
        with torch.autocast("cuda", enabled=False):
            _, hN = self.gru(seq)
        return self.norm(self.blocks(hN[-1].view(B, S, -1)))

    def forward(self, x, lid=None):                         # (B, F, S) queue logits, as T2Tfm
        return self.head(self.tokens(x)).permute(0, 2, 1)

    def reconstruct(self, x):                               # (B, 2, H, S): v/vf and q/cap of every history cell
        B, _, Hh, S = x.shape
        return self.rec(self.tokens(x)).view(B, S, 2, Hh).permute(0, 2, 3, 1)


def rec_loss(net, x, gen=None):
    """20% of the cells hidden from the input; MSE on the hidden observed cells (channels 0:2)."""
    hid = torch.rand(x[:, 2].shape, device=x.device, generator=gen) < 0.2
    xin = x.clone()
    xin[:, 0:4] = xin[:, 0:4] * (~hid).unsqueeze(1).float()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        rec = net.reconstruct(xin).float()
    w = (x[:, 2] > 0.5).float().unsqueeze(1) * hid.unsqueeze(1).float()
    return ((rec - x[:, 0:2].float()) ** 2 * w).sum() / w.sum().clamp(min=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--lr", type=float, default=1e-3)
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
    net = T2TfmRec(parts[0]["x"].shape[1], cnn.WIDTH).to(cnn.DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    best, best_state = 1e9, None
    for step in range(1, args.steps + 1):
        x, y, m, _ = cnn.batch(parts, pool, cnn.BATCH)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = net(x)
        logit = out[:, :F].float()
        loss = cnn.loss_fn(logit, y, m) + cnn.LOVASZ * cnn.lovasz_hinge(logit, y, m) + REC_W * rec_loss(net, x)
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
                    if len(es):
                        pr = cnn.predict(net, p["x"][es])
                        yv, mv = p["y"][es].astype(np.float32), p["m"][es].astype(np.float32)
                        bce = -(yv * np.log(pr + 1e-6) + (1 - yv) * np.log(1 - pr + 1e-6))
                        tot += float((bce * mv).sum())
                        n += float(mv.sum())
            es_loss = tot / max(n, 1)
            flag = ""
            if es_loss < best:
                best, best_state, flag = es_loss, {k: v.detach().clone() for k, v in net.state_dict().items()}, " *"
            print(f"step {step}: train loss {loss.item():.4f}  ES bce {es_loss:.4f}{flag} ({time.time() - t0:.0f}s)",
                  flush=True)
    torch.save(best_state, MODELS / f"t2_tfm_rec_s{args.seed}.pt")
    print(f"saved t2_tfm_rec_s{args.seed}.pt (ES bce {best:.4f}, {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
