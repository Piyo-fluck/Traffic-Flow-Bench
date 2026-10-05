"""Same-day causal adaptation: use every released cell before each window's own origin (note 124).

The pseudo-supervised adaptation behind p_PS (note 106) cuts same-day data at the day's FIRST official origin.
62 of the 80 official ongoing windows are not the first window of their day and have on average 9.1 h of same-day
released data before their own T that no member uses - today's own queues, in the hours just before the window.
The rules allow every released observation timestamped at or before T.

Two modes, same initial checkpoint and seeds, paired on the official-like pool of both scored months:

    base      the production recipe: per day, --steps of sup+rec on earlier days and the same day before the day's
              first origin; weights carried forward (occ_lab/pseudo_adapt.py, mode sup+rec)
    sameday   base, then the day's windows in time order: at every --bucket slots, --inc-steps more sup+rec steps
              on pseudo windows whose horizon ends before the bucket start (half of them from today, half from
              earlier days), weights carried forward within the day and to the next day

Labels come from the released layer (obs == true at visible cells); nothing at or after a window's origin enters
its own prediction. Caches occ_lab/cache/pseudo_<save>_<mode>_<split>_<panel>.npz for occ_lab/ongoing_ablation.py.

    T2CNN_WIDTH=192 python occ_lab/pseudo_adapt_sameday.py --ckpt work/models/t2_cnn_lov_w192_rec_ssl30k_s20.pt --save sd_s20
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import pseudo_adapt as PA  # noqa: E402

C = PA.C


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--inc-steps", type=int, default=10)
    ap.add_argument("--bucket", type=int, default=12, help="slots between same-day updates (12 = 1 h)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--w-rec", type=float, default=1.0)
    ap.add_argument("--save", default="")
    ap.add_argument("--panels", nargs="*", default=None)
    args = ap.parse_args()
    t_start = time.time()
    res = {}
    for i, panel in enumerate(C.T2_PANELS):
        if args.panels and panel not in args.panels:
            continue
        ctx, stat, W = C.windows(i, panel)
        for s in C.SPLITS:
            on, og, z = W[s]["on"], W[s]["og"], W[s]["z"]
            pool = PA.pseudo_pool(ctx, z)
            days = np.unique(np.concatenate([on["d"], og["d"]]))
            for mode in ("base", "sameday"):
                net, params = C.build_net(args.ckpt, "full")
                rng = np.random.default_rng(7 + i)
                p_on = np.zeros((len(on["d"]), ctx.S), np.float32)
                p_og = np.zeros(og["fut"].shape, np.float32)
                for d in days:
                    d = int(d)
                    io, ig = np.flatnonzero(on["d"] == d), np.flatnonzero(og["d"] == d)
                    first = int(min(on["t0"][io].min(initial=10 ** 6), og["t0"][ig].min(initial=10 ** 6)))
                    past = np.arange(d)
                    # start of day: the production recipe (earlier days + today before the first origin)
                    pool_days = np.append(past, d) if first > C.H + C.F + 2 else past
                    cuts = {int(k): (first if int(k) == d else 10 ** 6) for k in pool_days}
                    if len(pool_days):
                        PA.adapt_sup(net, params,
                                     lambda: PA.sup_batch(ctx, stat, z, pool, pool_days, cuts, args.bs, rng, "ongoing"),
                                     args.steps, args.lr, args.w_rec,
                                     lambda: C.past_batch(ctx, stat, z, past, args.bs, rng))
                    t_all = np.r_[on["t0"][io], og["t0"][ig]]
                    kind = np.r_[np.zeros(len(io), int), np.ones(len(ig), int)]
                    idx = np.r_[io, ig]
                    order = np.argsort(t_all, kind="stable")
                    last_cut = first
                    for b in np.unique(t_all[order] // args.bucket):
                        sel = order[(t_all[order] // args.bucket) == b]
                        cut = int(t_all[sel].min())            # everything strictly before the bucket's first origin
                        if mode == "sameday" and cut - last_cut >= args.bucket and cut > C.H + C.F + 2:
                            # half today (up to the cut), half earlier days; the day list is weighted accordingly
                            today = np.full(max(len(past), 1), d)
                            dl = np.r_[past, today]
                            cuts2 = {int(k): (cut if int(k) == d else 10 ** 6) for k in dl}
                            PA.adapt_sup(net, params,
                                         lambda: PA.sup_batch(ctx, stat, z, pool, dl, cuts2, args.bs, rng, "ongoing"),
                                         args.inc_steps, args.lr, args.w_rec,
                                         lambda: C.day_batch(ctx, stat, z, d, cut, args.bs, rng))
                            last_cut = cut
                        so = idx[sel][kind[sel] == 0]
                        sg = idx[sel][kind[sel] == 1]
                        if len(so):
                            p_on[so] = C.cnn.predict(net, C.make_x(ctx, stat, on, so, z))[:, C.F - 1]
                        if len(sg):
                            p_og[sg] = C.cnn.predict(net, C.make_x(ctx, stat, og, sg, z))
                a = res.setdefault((s, mode), {})
                a[panel] = (C.onset_iou(p_on, on).mean(), C.ongoing_iou(p_og, og, 0.5).mean())
                if args.save:
                    np.savez(LAB / "cache" / f"pseudo_{args.save}_{mode}_{s}_{panel}.npz",
                             on=p_on.astype(np.float32), og=p_og.astype(np.float16))
            print(f"  {panel} {s}: base {res[(s, 'base')][panel]}  sameday {res[(s, 'sameday')][panel]} "
                  f"({time.time() - t_start:.0f}s)", flush=True)
    for s in C.SPLITS:
        print(f"\n=== {s} (single CNN, its own IoU) ===")
        for mode in ("base", "sameday"):
            d = res.get((s, mode))
            if d:
                print(f"{mode:<9} onset {C.family_mean({k: v[0] for k, v in d.items()}):.4f}  "
                      f"ongoing {C.family_mean({k: v[1] for k, v in d.items()}):.4f}")


if __name__ == "__main__":
    main()
