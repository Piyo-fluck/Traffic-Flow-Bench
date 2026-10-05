"""Onset as a discrete-time hazard: train on every origin that could have broken down, not one per day.

`t2_cnn.train_windows` takes `ctx.eo` and `train_occ.windows` takes `earliest_onset(ctx.q)`, so every onset
model we have is fitted on the first queue of each morning: 1,722 windows over the whole train period, against
24,087 free-to-queued transitions and 28.1M at-risk intervals in the same data. Ninety-three percent of the
breakdown events are never seen.

That restriction matches the EVALUATION distribution - official onset windows are first-of-morning - but it is
not a reason to withhold the other events from LEARNING. The quantity to learn is the hazard

    h(s) = P( link s is queued at T+30 | free-flowing through the history and at T, state )

which every clean-history origin observes, whether or not a queue happened to follow. The first-of-morning
structure belongs at prediction time, as a condition, not in the training set.

`occ_lab/onset_learning_curve.py` says the restriction binds: onset IoU on held-out days climbs by about +0.015
to +0.024 per doubling of the fit windows and is still climbing at 1,783 (0.8240 -> 0.8389 -> 0.8625 -> 0.8775),
and the train-run location prior loses its value as the data grows (its edge falls 0.158 -> 0.072 -> 0.017 ->
0.008), which is what a memorised prior should do.

    --mode earliest   the first clean origin of each day whose T+30 is queued     (reproduces production)
    --mode queued     every clean origin whose T+30 is queued                     (about 14x)
    --mode all        every clean origin                                          (about 120x, a true hazard)

Windows are indexed, never materialised: batches are built from the raw arrays, so `all` fits in memory.
Evaluation is identical in all three modes - the held-out days' first-of-morning windows, scored with the
production expected-IoU ratio rule - so only the training set differs.

    CUDA_VISIBLE_DEVICES=0 python occ_lab/train_onset_hazard.py --mode queued --steps 12000
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("T2CNN_ARCH", "cnn")
os.environ.setdefault("OMP_NUM_THREADS", "8")
LAB = Path(__file__).resolve().parent
PROJ = LAB.parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(PROJ / "analysis"))
sys.path.insert(0, str(PROJ / "scripts"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import t2_cnn as cnn  # noqa: E402
from t2_eda02 import T2_PANELS  # noqa: E402
from t2_eda03 import F, H  # noqa: E402
from t2_expected_iou import select  # noqa: E402
from t2_m1_models import Ctx  # noqa: E402
from tfb.physics import load_ramps  # noqa: E402
from tfb.t1eval import FIT_END  # noqa: E402
from train_occ import occ_channels  # noqa: E402
from train_onset_stid import OnsetSTID  # noqa: E402

MODELS = LAB / "models"
LO, HI = 48, 168                                             # 04:00 .. 14:00, the official origin band


def index_windows(ctx, mode):
    """(d, t0) with a clean history and slot T. `mode` decides which of them are kept."""
    q = ctx.q
    D, T, S = q.shape
    lo, hi = max(LO, H + 1), min(HI, T - F - 1)
    clean = np.ones((D, hi - lo), bool)
    fut = np.zeros((D, hi - lo), bool)
    for k, t0 in enumerate(range(lo, hi)):
        clean[:, k] = ~q[:, t0 - H:t0 + 1].any((1, 2))
        fut[:, k] = q[:, t0 + F].any(1)
    keep = clean if mode == "all" else (clean & fut)
    if mode == "earliest":
        first = np.where(keep.any(1), keep.argmax(1), -1)
        keep = np.zeros_like(keep)
        ok = first >= 0
        keep[np.flatnonzero(ok), first[ok]] = True
    d, k = np.nonzero(keep)
    return d.astype(np.int32), (k + lo).astype(np.int32)


def build(part, j):
    """Materialise the batch's windows from the panel's raw arrays."""
    ctx, stat, prof = part["ctx"], part["stat"], part["prof"]
    z = ctx.z
    d, t0 = part["d"][j], part["t0"][j]
    hi = t0[:, None] + np.arange(-H, 0)
    ok = ctx.el[d[:, None], hi]
    x = cnn.make_x(ctx, stat, z["true_speed"][d[:, None], hi], z["true_flow"][d[:, None], hi], ok, t0,
                   z["weekday"][d] >= 5)
    x = np.concatenate([x, occ_channels(z["true_occ"][d[:, None], hi], ok, prof)], 1)
    tgt = t0 + F
    return (x, ctx.q[d, tgt].astype(np.float32), ctx.el[d, tgt].astype(np.float32),
            part["lid"], t0.astype(np.int64), z["weekday"][d].astype(np.int64))


def to_dev(a, dt):
    return torch.from_numpy(np.ascontiguousarray(a)).to(cnn.DEV, dt)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="queued", choices=("earliest", "queued", "all"))
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--emb-drop", type=float, default=0.5)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--init", type=Path, default=None,
                    help="start from a checkpoint trained on a larger, off-distribution window set. The extra "
                         "windows failed as a TARGET (queued 0.8424, all 0.6899 against earliest's 0.8462) "
                         "because a second onset of the day is a different conditional distribution; this uses "
                         "them as a representation instead")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    MODELS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    cnn.rng = rng
    torch.manual_seed(args.seed)
    t0_ = time.time()
    profs = np.load(MODELS / "occ_profiles.npz")
    parts, n_links, n_slots = [], 0, 0
    for i, panel in enumerate(T2_PANELS):
        ctx = Ctx(panel, i)
        ctx.loff = n_links
        n_links += ctx.S
        n_slots = max(n_slots, ctx.z["true_speed"].shape[1])
        stat = cnn.static_channels(ctx, load_ramps(panel, "train", ctx.z))
        d, t0 = index_windows(ctx, args.mode)
        es = ctx.z["dates"][d] >= FIT_END
        de, te = index_windows(ctx, "earliest")
        ese = ctx.z["dates"][de] >= FIT_END
        # the held-out days are split in two: odd days pick the checkpoint, even days are reported, so the
        # number that gets quoted was never used to stop on
        half = (de[ese] % 2).astype(bool)
        p = {"ctx": ctx, "stat": stat, "prof": profs[panel], "d": d, "t0": t0,
             "lid": np.arange(ctx.S) + ctx.loff,
             "fit": np.flatnonzero(~es),
             "ev_d": de[ese][half], "ev_t0": te[ese][half],
             "rp_d": de[ese][~half], "rp_t0": te[ese][~half]}
        parts.append(p)
        print(f"[{panel}] {args.mode}: {len(p['fit']):,} fit windows "
              f"({len(d):,} total), {int(ese.sum())} held-out first-of-morning windows, links {ctx.S} "
              f"({time.time() - t0_:.0f}s)", flush=True)
    print(f"total fit windows: {sum(len(p['fit']) for p in parts):,}", flush=True)

    net = OnsetSTID(15 + 0, n_links, n_slots, ident="id").to(cnn.DEV)
    if args.init is not None:
        d_ = torch.load(args.init, map_location=cnn.DEV)
        net.load_state_dict(d_["state"] if "state" in d_ else d_)
        print(f"initialised from {args.init.name}", flush=True)
    cin = None
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    best, best_state, best_step = -1.0, None, 0

    def evaluate(which="ev"):
        net.eval()
        out = []
        with torch.no_grad():
            for p in parts:
                ev = p[which + "_d"]
                if not len(ev):
                    continue
                q = dict(p)
                q["d"], q["t0"] = p[which + "_d"], p[which + "_t0"]
                for a in range(0, len(ev), 64):
                    sl = slice(a, a + 64)
                    qq = dict(q)
                    qq["d"], qq["t0"] = q["d"][sl], q["t0"][sl]
                    x, y, m, lid, tod, dow = build(qq, np.arange(len(qq["d"])))
                    lg = net(to_dev(x, torch.float32),
                             to_dev(np.broadcast_to(lid, (len(x), len(lid))).copy(), torch.long),
                             to_dev(tod, torch.long), to_dev(dow, torch.long))[:, 0]
                    P = torch.sigmoid(lg).float().cpu().numpy()
                    for j in range(len(P)):
                        mm = m[j] > 0.5
                        if not mm.any() or not (y[j][mm] > 0.5).any():
                            continue
                        pr = select(P[j][mm], "ratio", 1.0)
                        t = y[j][mm] > 0.5
                        out.append(int((pr & t).sum()) / max(int((pr | t).sum()), 1))
        net.train()
        return float(np.mean(out)) if out else 0.0

    for step in range(1, args.steps + 1):
        p = parts[rng.integers(len(parts))]
        j = p["fit"][rng.integers(len(p["fit"]), size=args.bs)]
        x, y, m, lid, tod, dow = build(p, j)
        lid_b = np.broadcast_to(lid, (args.bs, len(lid))).copy()
        if rng.random() < args.emb_drop:
            lid_b[:] = n_links
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = net(to_dev(x, torch.float32), to_dev(lid_b, torch.long),
                     to_dev(tod, torch.long), to_dev(dow, torch.long))[:, 0]
        lg = lg.float()
        yt, mt = to_dev(y, torch.float32), to_dev(m, torch.float32)
        loss = (nn.functional.binary_cross_entropy_with_logits(lg, yt, reduction="none") * mt).sum() \
            / mt.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % args.eval_every == 0:
            iou = evaluate("ev")
            flag = ""
            if iou > best:
                best, best_step, flag = iou, step, " *"
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            print(f"step {step}: loss {loss.item():.4f}  select IoU {iou:.4f}{flag} "
                  f"({time.time() - t0_:.0f}s)", flush=True)
    _ = cin
    name = f"onset_hazard_{args.mode}{args.tag}_s{args.seed}.pt"
    torch.save({"state": best_state, "cin": 15, "n_links": n_links, "n_slots": n_slots, "ident": "id"},
               MODELS / name)
    net.load_state_dict(best_state)
    rep = evaluate("rp")
    print(f"saved {name}: select IoU {best:.4f} at step {best_step}; "
          f"REPORTED IoU on the untouched half {rep:.4f} ({time.time() - t0_:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
