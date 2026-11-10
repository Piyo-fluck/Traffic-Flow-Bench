"""Give the existing ongoing families a channel they have never had: OCCUPANCY.

Two architectures (DCRNN-style recurrence and a directed spatial scan) landed on the same error correlation with
the current members (0.43-0.57) and the same error profile by horizon and cell type (note 83), while the cells the
members dispute carry calibrated coin-flip probabilities (note 82). What did move the needle was an input:
occupancy raised the AUC on those cells from 0.772 to 0.785 (`analysis/t2_disputed_occ_signal.py`). The Task 2
models see v/vf, q/cap, the observed flag and the queued flag - never occupancy.

This trains the production CNN and transformer with three extra channels: occupancy, occupancy over the link's
weekday x tod mean, and the change of occupancy across the history. Everything else (windows, losses, steps,
schedule) is the production recipe, so the comparison is about the input only.

    CUDA_VISIBLE_DEVICES=0 python occ_lab/train_occ.py --arch cnn --seed 0
    CUDA_VISIBLE_DEVICES=1 python occ_lab/train_occ.py --arch tfm --seed 0
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("T2CNN_LOVASZ", "0.5")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

LAB = Path(__file__).resolve().parent
PROJ = LAB.parent
for p in (PROJ, PROJ / "analysis", PROJ / "scripts"):
    sys.path.insert(0, str(p))
ARCH = os.environ.get("OCC_ARCH", "cnn")
os.environ["T2CNN_ARCH"] = ARCH
import t2_cnn as cnn  # noqa: E402
from t2_eda02 import T2_PANELS, candidates  # noqa: E402
from t2_eda03 import F, H, earliest_onset  # noqa: E402
from t2_m1_models import Ctx  # noqa: E402
from tfb.physics import load_ramps  # noqa: E402
from tfb.t1eval import FIT_END  # noqa: E402

MODELS = LAB / "models"


def occ_channels(occ, ok, occ_prof):
    """(N, 3, H, S): occupancy, occupancy / profile, and its change over the history, zero where unseen."""
    o = np.where(ok, np.nan_to_num(occ, nan=0.0), 0.0).astype(np.float32)
    rel = o / np.maximum(occ_prof[None, None, :], 1e-3)
    d = o - np.concatenate([o[:, :1], o[:, :-1]], 1)
    return np.stack([o * 10, np.clip(rel, 0, 5), d * 10], 1).astype(np.float16)


def windows(ctx, stat, rng, max_og=3000):
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
    with np.errstate(all="ignore"):
        occ_prof = np.nan_to_num(np.nanmean(np.where(ctx.el, z["true_occ"], np.nan), (0, 1)), nan=0.05)
    x = cnn.make_x(ctx, stat, v, qf, ok, t0, z["weekday"][d] >= 5)
    x = np.concatenate([x, occ_channels(z["true_occ"][d[:, None], hi], ok, occ_prof)], 1)
    return dict(x=x, y=ctx.q[d[:, None], fi].astype(np.float16), m=ctx.el[d[:, None], fi],
                es=z["dates"][d] >= FIT_END, lid=np.arange(ctx.S),
                is_on=np.r_[np.ones(len(d_on), bool), np.zeros(len(d_og), bool)]), occ_prof


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default=ARCH, choices=("cnn", "tfm"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    assert args.arch == ARCH, "set OCC_ARCH before running (it configures t2_cnn at import)"
    MODELS.mkdir(parents=True, exist_ok=True)
    cnn.rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    t0 = time.time()
    parts, profs = [], {}
    for i, panel in enumerate(T2_PANELS):
        ctx = Ctx(panel, i)
        ctx.loff = 0
        stat = cnn.static_channels(ctx, load_ramps(panel, "train", ctx.z))
        p, profs[panel] = windows(ctx, stat, cnn.rng)
        parts.append(p)
        print(f"[{panel}] windows {len(p['x']):,}, channels {p['x'].shape[1]} ({time.time() - t0:.0f}s)", flush=True)
        ctx.z = ctx.q = ctx.el = None
    np.savez(MODELS / "occ_profiles.npz", **profs)
    pool = {id(p): np.flatnonzero(~p["es"]) for p in parts}
    cin = parts[0]["x"].shape[1]
    net = (cnn.T2Tfm if args.arch == "tfm" else cnn.T2Net)(cin, cnn.WIDTH).to(cnn.DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    best, best_state = 1e9, None
    for step in range(1, args.steps + 1):
        x, y, m, _ = cnn.batch(parts, pool, cnn.BATCH)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = net(x)
        logit = out[:, :F].float()
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
    torch.save(best_state, MODELS / f"occ_{args.arch}_s{args.seed}.pt")
    print(f"saved occ_{args.arch}_s{args.seed}.pt (ES bce {best:.4f}, {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
