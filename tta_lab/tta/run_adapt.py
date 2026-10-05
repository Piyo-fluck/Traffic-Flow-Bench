"""One runner for experiments B, C, D (and the evaluation half of E, and the raw material of F).

Per panel and scored month, over the days that carry official-like windows, in ascending order:
    [adapt-then-predict]  adapt on past days (< d), then predict day d
    [predict-then-adapt]  predict day d, then adapt on past days (< d)          <- idea 9b exactly (default)
Past-day adaptation is cumulative (weights carry forward) unless --reset-daily, which restarts every day from the
checkpoint and adapts on a batch drawn from all days < d (the setting meta-learning E is trained for).
--micro K adds experiment B: before predicting a window, K extra steps on that window's own history
(replicated --micro-bs times with different masks); the weights are restored after the window.

Saved per split/panel in runs/<name>/probs_*.npz: on_p (N,S) at T+30, og_p (N,F,S), on_ssl/og_ssl (N,) the
fixed-mask reconstruction loss of each window under the weights that predicted it (the gate of F).

Examples (see ../EXPERIMENTS.md for the full grid):
    python tta/run_adapt.py --name rep9b                                   # replication of 9b
    python tta/run_adapt.py --name C_norm --mode norm --lr 1e-3            # C
    python tta/run_adapt.py --name D_nd --losses 1 0.5 0.5 --k-tail 2      # D
    python tta/run_adapt.py --name B_k3 --micro 3 --micro-lr 1e-4          # B
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

import common as C


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="run name -> tta_lab/runs/<name>/")
    ap.add_argument("--ckpt", default=str(C.MODELS / "t2_cnn_lov_rec_s20.pt"))
    ap.add_argument("--mode", default="full", choices=C.MODES, help="C: which parameters adapt")
    ap.add_argument("--rank", type=int, default=4, help="LoRA rank")
    ap.add_argument("--train-rec", type=int, default=None, help="1/0: force the rec head (default: only in full)")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--opt", default="adamw", choices=("adamw", "sgd"))
    ap.add_argument("--losses", type=float, nargs=3, default=(1.0, 0.0, 0.0), metavar=("MASK", "NEXT", "DELTA"),
                    help="D: weights of masked reconstruction, next-step, delta")
    ap.add_argument("--k-tail", type=int, default=1, help="D: hidden tail slots for next-step / delta")
    ap.add_argument("--order", default="predict-then-adapt", choices=("predict-then-adapt", "adapt-then-predict"))
    ap.add_argument("--reset-daily", action="store_true", help="E-style: restart from the checkpoint every day")
    ap.add_argument("--window-days", type=int, default=0, help="adapt only on the last N past days (0 = all)")
    ap.add_argument("--no-adapt", action="store_true", help="the unadapted checkpoint (control)")
    ap.add_argument("--today-steps", type=int, default=0,
                    help="extra adaptation steps on windows from EARLIER THE SAME DAY, before that day's first "
                         "window history (allowed since the 2026-09-20 clarification); 0 = off")
    ap.add_argument("--micro", type=int, default=0, help="B: steps on the window's own history")
    ap.add_argument("--micro-lr", type=float, default=1e-4)
    ap.add_argument("--micro-bs", type=int, default=8)
    ap.add_argument("--micro-kinds", default="on", choices=("on", "og", "both"))
    ap.add_argument("--micro-losses", type=float, nargs=3, default=(1.0, 0.0, 0.0))
    ap.add_argument("--kinds", default="both", choices=("on", "og", "both"), help="which windows to score")
    ap.add_argument("--panels", nargs="*", default=None)
    ap.add_argument("--splits", nargs="*", default=list(C.SPLITS))
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--no-report", action="store_true")
    args = ap.parse_args()
    (C.RUNS / args.name).mkdir(parents=True, exist_ok=True)
    (C.RUNS / args.name / "args.json").write_text(json.dumps(vars(args), indent=1))
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    t_all = time.time()
    kinds = ("on", "og") if args.kinds == "both" else (args.kinds,)
    micro_kinds = ("on", "og") if args.micro_kinds == "both" else (args.micro_kinds,)
    losses = tuple(args.losses)
    for i, panel in enumerate(C.T2_PANELS):
        if args.panels and panel not in args.panels:
            continue
        ctx, stat, W = C.windows(i, panel)
        for split in args.splits:
            w = W[split]
            z = w["z"]
            net, params = C.build_net(args.ckpt, args.mode, args.rank,
                                      None if args.train_rec is None else bool(args.train_rec))
            init = C.snapshot(params)
            res = {}
            for k in kinds:
                n = len(w[k]["d"])
                res[f"{k}_p"] = np.zeros((n, C.F, ctx.S), np.float32)
                res[f"{k}_ssl"] = np.zeros(n, np.float32)
            days = sorted(set().union(*[set(w[k]["d"].tolist()) for k in kinds]))
            last_loss = []

            def past(day):
                lo = 0 if not args.window_days else max(0, day - args.window_days)
                pool = np.arange(lo, day)
                return lambda: C.past_batch(ctx, stat, z, pool, args.bs, rng)

            first_t0 = {int(d): int(min(w[k]["t0"][w[k]["d"] == d].min() for k in kinds
                                        if (w[k]["d"] == d).any()))
                        for d in set().union(*[set(w[k]["d"].tolist()) for k in kinds])}

            def do_today(day):
                if args.no_adapt or not args.today_steps:
                    return
                t_max = first_t0[int(day)] - C.H          # strictly before that day's earliest window history
                C_adapt = (lambda: C.day_batch(ctx, stat, z, int(day), t_max, args.bs, rng))
                if C_adapt() is not None:
                    C.adapt(net, params, C_adapt, args.today_steps, args.lr, args.opt, losses, args.k_tail)

            def do_adapt(day):
                if args.no_adapt or day <= 0:
                    return
                if args.reset_daily:
                    C.restore(params, init)
                last_loss.append(C.adapt(net, params, past(day), args.steps, args.lr, args.opt, losses, args.k_tail))

            for day in days:
                if args.order == "adapt-then-predict":
                    do_adapt(day)
                do_today(day)
                for k in kinds:
                    m = np.flatnonzero(w[k]["d"] == day)
                    if not len(m):
                        continue
                    x = C.make_x(ctx, stat, w[k], m, z)
                    if args.micro and k in micro_kinds:
                        state = C.snapshot(params)
                        for j, idx in enumerate(m):
                            xb = C.to_dev(np.repeat(x[j:j + 1], args.micro_bs, 0))
                            C.adapt(net, params, xb, args.micro, args.micro_lr, "adamw", tuple(args.micro_losses),
                                    args.k_tail)
                            res[f"{k}_p"][idx] = cnn_predict(net, x[j:j + 1])[0]
                            res[f"{k}_ssl"][idx] = C.ssl_score(net, x[j:j + 1])[0]
                            C.restore(params, state)
                    else:
                        res[f"{k}_p"][m] = cnn_predict(net, x)
                        res[f"{k}_ssl"][m] = C.ssl_score(net, x)
                if args.order == "predict-then-adapt":
                    do_adapt(day)
            out = {}
            if "on" in kinds:
                out.update(on_p=res["on_p"][:, C.F - 1], on_ssl=res["on_ssl"], on_d=w["on"]["d"])
            if "og" in kinds:
                out.update(og_p=res["og_p"], og_ssl=res["og_ssl"], og_d=w["og"]["d"])
            C.save_run(args.name, split, panel, **out)
            ll = f"{np.mean(last_loss):.4f}" if last_loss else "-"
            print(f"[{panel} {split}] days {len(days)}, mean last adapt loss {ll} ({time.time() - t_all:.0f}s)",
                  flush=True)
    if not args.no_report:
        C.report([args.name], file=C.RUNS / args.name / "report.txt")
    print(f"total {time.time() - t_all:.0f}s")


def cnn_predict(net, x):
    return C.cnn.predict(net, x, None)


if __name__ == "__main__":
    main()
