"""Causal pseudo-SUPERVISED adaptation of the ongoing transformer family (T2Tfm), the recipe of note 106.

p_PS (the adapted CNN) is 0.60 of AT's ongoing blend; the transformer family (tfm8, 0.60 of the other 0.40) has never
been adapted to the scored run. The one earlier attempt used the reconstruction loss only, and it hurt because the
reconstruction and queue heads share the final token (note 75). Here the adaptation is supervised only: real Task 2
windows built from the scored run's released layer strictly before the day's first origin (earlier days, and the
same day before that origin), labels from the released layer (obs == true at visible cells), BCE + Lovasz. The
weights carry forward from day to day, exactly as p_PS.

Per checkpoint and learning rate: the transformer frozen ('none') and adapted, on the official-like pool of both
scored months; caches occ_lab/cache/tfmps_<tag>_<lr>_<split>_<panel>.npz (og probabilities) for the stack.

    .venv/Scripts/python.exe occ_lab/pseudo_adapt_tfm.py --ckpt work/models_remote/t2_cnn_lov_tfm_s3.pt --tag s3
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
cnn = C.cnn


def build_tfm(ckpt):
    sd = torch.load(ckpt, map_location=C.DEV)
    cin = sd["gru.weight_ih_l0"].shape[1]
    width = sd["gru.weight_hh_l0"].shape[1]
    net = cnn.T2Tfm(cin, width).to(C.DEV)
    net.load_state_dict(sd)
    net.eval()
    return net, list(net.parameters())


def adapt(net, params, batches, steps, lr):
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    net.train()
    for _ in range(steps):
        b = batches()
        if b is None:
            break
        x, y, m = b
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = net(x)
        logit = out[:, :C.F].float()
        loss = cnn.loss_fn(logit, y, m) + cnn.LOVASZ * cnn.lovasz_hinge(logit, y, m)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    net.eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--lrs", type=float, nargs="+", default=[1e-4, 3e-4])
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--panels", nargs="*", default=None)
    ap.add_argument("--solid3", action="store_true",
                    help="note 141 for the transformer: after the day's adaptation (lr = the first of --lrs), each "
                         "window's norm + head is fitted on its 32 closest released pseudo windows (earlier days and "
                         "the same day with the horizon ended before its origin), by the post-block link tokens")
    args = ap.parse_args()
    t_start = time.time()
    res = {}
    modes = ["none"] + [f"sup{lr:g}" for lr in args.lrs]
    if args.solid3:
        import adapt_variants as AV                                   # noqa: E402
        modes = [f"sup{args.lrs[0]:g}", "solid3"]
    for i, panel in enumerate(C.T2_PANELS):
        if args.panels and panel not in args.panels:
            continue
        ctx, stat, W = C.windows(i, panel)
        for s in C.SPLITS:
            og, z = W[s]["og"], W[s]["z"]
            if not len(og["d"]):
                continue
            pool = PA.pseudo_pool(ctx, z)
            on = W[s]["on"]
            days = np.unique(np.concatenate([on["d"], og["d"]]))
            for mode in modes:
                net, params = build_tfm(args.ckpt)
                lr = args.lrs[0] if mode == "solid3" else (float(mode[3:]) if mode != "none" else 0.0)
                rng = np.random.default_rng(7 + i)
                p_og = np.zeros(og["fut"].shape, np.float32)
                for d in days:
                    d = int(d)
                    ig = np.flatnonzero(og["d"] == d)
                    first = int(min(on["t0"][on["d"] == d].min(initial=10 ** 6), og["t0"][ig].min(initial=10 ** 6)))
                    if mode != "none":
                        past = np.arange(d)
                        pool_days = np.append(past, d) if first > C.H + C.F + 2 else past
                        cuts = {int(k): (first if int(k) == d else 10 ** 6) for k in pool_days}
                        if len(pool_days):
                            adapt(net, params,
                                  lambda: PA.sup_batch(ctx, stat, z, pool, pool_days, cuts, args.bs, rng, "ongoing"),
                                  args.steps, lr)
                    if len(ig) and mode == "solid3":
                        days3 = np.append(np.arange(d), d)
                        cuts3 = {int(k): (int(og["t0"][ig].max()) if int(k) == d else 10 ** 6) for k in days3}
                        P3 = AV.pool_windows(ctx, stat, z, pool, days3, cuts3, rng, cap=900)
                        xg = C.make_x(ctx, stat, og, ig, z)
                        p_og[ig] = AV.solid_generic(net, xg, og["t0"][ig], P3,
                                                    list(net.norm.parameters()) + list(net.head.parameters()),
                                                    AV.embed_tfm, day=d) if P3 is not None else cnn.predict(net, xg)
                    elif len(ig):
                        p_og[ig] = cnn.predict(net, C.make_x(ctx, stat, og, ig, z))
                res.setdefault((s, mode), {})[panel] = C.ongoing_iou(p_og, og, 0.5).mean()
                np.savez(LAB / "cache" / f"tfmps_{args.tag}_{mode}_{s}_{panel}.npz", og=p_og.astype(np.float16))
            print(f"  {panel} {s}: " + "  ".join(f"{m} {res[(s, m)][panel]:.4f}" for m in modes)
                  + f" ({time.time() - t_start:.0f}s)", flush=True)
    for s in C.SPLITS:
        print(f"\n=== {s}: single transformer {args.tag}, ongoing family IoU (tau 0.50) ===")
        for m in modes:
            if (s, m) in res:
                print(f"  {m:<10} {C.family_mean(res[(s, m)]):.4f}")


if __name__ == "__main__":
    main()
