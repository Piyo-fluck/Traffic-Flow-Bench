"""The transformer family with every panel oriented upstream -> downstream (idea A; Contrails 1st place's lesson).

`lwr_mainline_topology.csv` fixes the direction exactly: in the four S / W panels the next link in order_index is
the INCOMING link of the previous one, so our link axis runs downstream -> upstream there (note 49). The transformer
carries a learned bias per signed link distance (up- vs downstream), so it has been learning two opposite meanings
of "upstream" at once. Here the S / W panels are flipped on the way in and the output flipped back.

    train   fine-tune a shipped tfm8 checkpoint on whole panels, 4k steps, BCE + 0.5 Lovasz, lr 1e-4,
            --orient (flip S / W) or not (the control: identical fine-tuning, original order)
    lab     AU's recipe (note 125): per day, 30 steps of the supervised loss on the run's released pseudo windows,
            lr 1e-4; caches fam_otfm_<tag>_{none,sup}_<split>_<panel>.npz

    python occ_lab/orient_tfm.py --stage train --seed 3 --orient --tag o_s3
    python occ_lab/orient_tfm.py --stage lab --tag o_s3
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
import adapt_family as AF  # noqa: E402  (tfm loading, pseudo batches, adaptation, prediction)
import longctx as LC  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

C, cnn = AF.C, AF.cnn
H, F, DEV = AF.H, AF.F, C.DEV
CACHE, MODELS = LC.CACHE, LC.MODELS
FLIP_PANEL = lambda panel: panel.endswith(("_S", "_W"))  # noqa: E731


class Oriented(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net, self.flip = net, False

    def forward(self, x, lid=None):
        if not self.flip:
            return self.net(x)
        return self.net(x.flip(-1)).flip(-1)


def stage_train(args):
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    parts = LC.train_panels()
    base = AF.load_model("tfm", MODELS / f"t2_cnn_lov_tfm_s{args.seed}.pt")
    if args.scratch:                                          # same architecture, fresh weights
        torch.manual_seed(1000 + args.seed)
        base = cnn.T2Tfm(base.gru.input_size, base.gru.hidden_size).to(DEV)
    net = Oriented(base).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    es = []
    for p in parts:
        p["tr"] = np.flatnonzero(~p["es"])
        e = np.flatnonzero(p["es"])[:400]
        d, t0 = p["d"][e], p["t0"][e]
        fi = t0[:, None] + np.arange(1, F + 1)
        es.append((p["panel"], LC.short_x(p, d, t0), p["q"][d[:, None], fi].astype(np.float32),
                   p["el"][d[:, None], fi].astype(np.float32)))

    def set_flip(panel):
        net.flip = args.orient and FLIP_PANEL(panel)

    best, best_state, t_start = 1e9, None, time.time()
    for step in range(1, args.steps + 1):
        net.train()
        p = parts[rng.integers(len(parts))]
        set_flip(p["panel"])
        j = p["tr"][rng.integers(len(p["tr"]), size=args.bs)]
        d, t0 = p["d"][j], p["t0"][j]
        fi = t0[:, None] + np.arange(1, F + 1)
        x = LC.tt(LC.short_x(p, d, t0))
        y, m = LC.tt(p["q"][d[:, None], fi].astype(np.float32)), LC.tt(p["el"][d[:, None], fi].astype(np.float32))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = net(x)[:, :F].float()
        loss = cnn.loss_fn(logit, y, m) + cnn.LOVASZ * cnn.lovasz_hinge(logit, y, m)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 500 == 0:
            tot, n = 0.0, 0.0
            for pn, xs, yv, mv in es:
                set_flip(pn)
                pr = cnn.predict(net, xs)
                bce = -(yv * np.log(pr + 1e-6) + (1 - yv) * np.log(1 - pr + 1e-6))
                tot += float((bce * mv).sum()); n += float(mv.sum())
            v = tot / max(n, 1)
            flag = ""
            if v < best:
                best, flag = v, " *"
                best_state = {k: t.detach().clone() for k, t in net.net.state_dict().items()}
            print(f"step {step}: train {loss.item():.4f}  ES bce {v:.5f}{flag} ({time.time() - t_start:.0f}s)", flush=True)
    torch.save({"state": best_state, "orient": args.orient, "seed": args.seed, "scratch": args.scratch}, MODELS / f"otfm_{args.tag}.pt")
    print(f"saved otfm_{args.tag}.pt (ES bce {best:.5f})")


def stage_lab(args):
    d0 = torch.load(MODELS / f"otfm_{args.tag}.pt", map_location=DEV)
    t_start = time.time()
    off = AF.link_offsets()
    res = {}
    for i, panel in enumerate(C.T2_PANELS):
        for s, ctx, stat, z, X, d_all, t0_all, firsts, extra in AF.windows_for("lab", i, panel, "tfm", 0):
            lid = np.arange(ctx.S) + off[panel]
            dow_all = z["weekday"][d_all].astype(np.int64)
            lab = AF.released_labels(ctx, z)
            og = extra["og"]
            for mode in ("none", "sup"):
                base = AF.load_model("tfm", MODELS / f"t2_cnn_lov_tfm_s{d0['seed']}.pt")
                base.load_state_dict(d0["state"])
                net = Oriented(base).to(DEV)
                net.flip = d0["orient"] and FLIP_PANEL(panel)
                rng = np.random.default_rng(7 + i)
                prob = np.zeros((len(d_all), F, ctx.S), np.float32)
                for d in sorted(firsts):
                    m = d_all == d
                    if mode == "sup":
                        past = np.arange(d)
                        first = firsts[d]
                        pool_days = np.append(past, d) if first > H + F + 2 else past
                        cuts = {int(k): (first if int(k) == d else 10 ** 6) for k in pool_days}
                        if len(pool_days):
                            AF.adapt("tfm", net, lid,
                                     lambda: AF.sup_batch("tfm", ctx, stat, z, lab, pool_days, cuts, 16, rng, None),
                                     30, 1e-4)
                    prob[m] = AF.predict("tfm", net, X[m], lid, t0_all[m].astype(np.int64), dow_all[m])
                sc = C.ongoing_iou(prob, og, 0.5)
                res.setdefault((s, mode), {})[panel] = (sc.mean(), sc[og["d"] <= 4].mean() if (og["d"] <= 4).any()
                                                        else np.nan)
                np.savez(CACHE / f"fam_otfm_{args.tag}_{mode}_{s}_{panel}.npz", og=prob.astype(np.float16))
            print(f"  {panel} {s} ({time.time() - t_start:.0f}s)", flush=True)
    for s in C.SPLITS:
        print(f"\n=== {s}: otfm {args.tag} ===")
        for m in ("none", "sup"):
            d = res[(s, m)]
            fm = lambda j: C.family_mean({k: v[j] for k, v in d.items() if np.isfinite(v[j])})  # noqa: E731
            print(f"  {m:<5} ongoing {fm(0):.4f}  days 1-5 {fm(1):.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("train", "lab"))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--orient", action="store_true")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--scratch", action="store_true", help="train from random weights (lr 1e-3 recommended)")
    args = ap.parse_args()
    stage_train(args) if args.stage == "train" else stage_lab(args)


if __name__ == "__main__":
    main()
