"""Idea 9b in production: adapt the CNN on past days and predict the OFFICIAL Task 2 windows.

Note 67/68: adapting the trunk on the released layer of days strictly before the window day, with the
reconstruction loss only and no Task 2 label, lifts onset by +0.0485 alone and by +0.0079 / +0.0189 inside the
production blend. This script produces the probabilities for the real official windows so a submission can use it.

For each panel and scored split, over the official window days in ascending order:
    1. predict that day windows with the current weights
    2. adapt the weights on windows sampled from days strictly before it (30 steps, lr 1e-4, reconstruction only)
The weights carry forward, so the run is causal and never sees the day it is predicting.

Writes prob_<split>_<panel>.npz with prob (N, F, S), is_onset, window_id, t0 - the layout
scripts/t2_cnn_blend_predict.py reads.

    T2CNN_REC=0.3 T2CNN_LOVASZ=0.5 .venv/Scripts/python.exe scripts/t2_adapt_official.py --out-dir work/sub_t2adaptcnn
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "scripts"))
import t2_cnn as cnn  # noqa: E402
from t2_eda02 import T2_PANELS  # noqa: E402
from t2_eda03 import F  # noqa: E402
from t2_m1_models import Ctx  # noqa: E402
from t2_m1_predict import hist_from_release  # noqa: E402
from tfb.data import RELEASE, load  # noqa: E402
from tfb.physics import load_ramps  # noqa: E402

DEFAULT_CKPT = ROOT / "work" / "models_remote" / "t2_cnn_lov_rec_s20.pt"
ADAPT_STEPS, ADAPT_LR, ADAPT_BS = 30, 1e-4, 16
H = cnn.H
rng = np.random.default_rng(3)


def past_batch(ctx, stat, z, d_max, bs):
    if d_max <= 0:
        return None
    days = rng.integers(0, d_max, size=bs)
    t0 = rng.integers(H + 1, z["obs_speed"].shape[1] - F - 1, size=bs)
    hi = t0[:, None] + np.arange(-H, 0)
    v, q = z["obs_speed"][days[:, None], hi], z["obs_flow"][days[:, None], hi]
    ok = z["elig"][days[:, None], hi] & np.isfinite(v) & np.isfinite(q)
    x = cnn.make_x(ctx, stat, v, q, ok, t0, z["weekday"][days] >= 5)
    return torch.from_numpy(np.ascontiguousarray(x)).to(cnn.DEV, torch.float32)


NEXT_W = 0.0   # note 72 D: weight of the next-step loss (hide the last history slot, reconstruct it)


def adapt(net, xb, steps, lr):
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    net.train()
    for _ in range(steps):
        hid = torch.rand_like(xb[:, 2]) < 0.2
        xin = xb.clone()
        xin[:, 0:4] = xin[:, 0:4] * (~hid).unsqueeze(1).float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            rec = net.rec(net.blocks(net.inp(xin))).float()
        w = (xb[:, 2] > 0.5).float().unsqueeze(1) * hid.unsqueeze(1).float()
        loss = ((rec - xb[:, 0:2].float()) ** 2 * w).sum() / w.sum().clamp(min=1)
        if NEXT_W > 0:
            xn = xb.clone()
            xn[:, 0:4, -1:] = 0
            with torch.autocast("cuda", dtype=torch.bfloat16):
                rn = net.rec(net.blocks(net.inp(xn))).float()
            wn = (xb[:, 2:3, -1:] > 0.5).float()
            loss = loss + NEXT_W * ((rn[:, :, -1:] - xb[:, 0:2, -1:].float()) ** 2 * wn).sum() / wn.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    net.eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT,
                    help="a checkpoint trained with T2CNN_REC>0, so the trunk carries the reconstruction head")
    ap.add_argument("--steps", type=int, default=ADAPT_STEPS,
                    help="adaptation steps per day; 0 = the unadapted checkpoint (the base of the delta correction)")
    ap.add_argument("--next-w", type=float, default=0.0, help="note 72 D: next-step loss weight (k=1)")
    args = ap.parse_args()
    global NEXT_W
    NEXT_W = args.next_w
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0_all = time.time()
    sd = torch.load(args.ckpt, map_location=cnn.DEV)
    cin = sd["inp.weight"].shape[1]
    for i, panel in enumerate(T2_PANELS):
        ctx = Ctx(panel, i)
        ctx.fitd = np.ones_like(ctx.fitd)
        ctx.loff = 0
        stat = cnn.static_channels(ctx, load_ramps(panel, "train", ctx.z))
        for split in ("validation", "private"):
            z = load(panel, split)
            base = RELEASE / "task2" / panel / split
            W = pd.read_csv(base / "window_index.csv")
            Hh = pd.read_parquet(base / "window_history.parquet")
            v, qf, ok, t0, grp = hist_from_release(ctx, Hh, W)
            d_of = {str(x): k for k, x in enumerate(z["dates"])}
            d = np.array([d_of[str(x)] for x in W.date], int)
            net = cnn.T2Net(cin, cnn.WIDTH).to(cnn.DEV)
            net.load_state_dict(sd)
            net.eval()
            prob = np.zeros((len(W), F, ctx.S), np.float32)
            for day in sorted(set(d.tolist())):
                m = d == day
                x = cnn.make_x(ctx, stat, v[m], qf[m], ok[m], t0[m], grp[m].astype(bool))
                prob[m] = cnn.predict(net, x, None)
                xb = past_batch(ctx, stat, z, int(day), ADAPT_BS)
                if xb is not None and args.steps > 0:
                    adapt(net, xb, args.steps, ADAPT_LR)
            np.savez(args.out_dir / f"prob_{split}_{panel}.npz", prob=prob,
                     is_onset=(W.condition == "queue_onset").to_numpy(),
                     window_id=W.window_id.to_numpy().astype(str), t0=t0)
            print(f"[{split} {panel}] {len(W)} windows over {len(set(d.tolist()))} days "
                  f"({time.time() - t0_all:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
