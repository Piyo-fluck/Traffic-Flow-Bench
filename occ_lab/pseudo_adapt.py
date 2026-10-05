"""Causal pseudo-SUPERVISED adaptation for Task 2: teach the model the actual task on the scored run itself.

Idea 9b adapts on the scored month with a masked-reconstruction loss and never touches a Task 2 label (note 67).
But the organizers' clarification (2026-09-20) allows every released observation timestamped at or before a
window's origin T - including earlier hours of the same day. From that we can build REAL task windows: an origin
tau with tau + 30 min <= T, its history from the released layer, and its labels from the released layer at
tau+5 .. tau+30.

Those labels are not a proxy in the usual sense: `obs == true` to the last bit wherever a cell is visible
(note 103), so at visible future cells the pseudo label IS the label, and only the invisible cells are missing
(they are masked out of the loss).

Three adaptations are compared, all causal, all starting from the same production checkpoint, all 30 steps:

    rec        the production recipe: masked reconstruction only (idea 9b)
    sup        the Task 2 loss on pseudo windows built from data at or before T
    sup+rec    both

Per panel and split, the days that carry official-like windows are walked in order; before predicting a day, the
weights are adapted on windows drawn from strictly earlier days and from earlier hours of that same day, and the
weights carry forward.

    python occ_lab/pseudo_adapt.py --ckpt work/models/t2_cnn_lov_rec_s20.pt
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ["T2CNN_ARCH"] = "cnn"
os.environ.setdefault("T2CNN_REC", "0.3")
os.environ.setdefault("T2CNN_LOVASZ", "0.5")
LAB = Path(__file__).resolve().parent
PROJ = LAB.parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(PROJ / "tta_lab" / "tta"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import common as C  # noqa: E402

MODES = ("none", "rec", "sup", "sup+rec")


def pseudo_pool(ctx, z):
    """Queue labels and history validity on the released layer, for every (day, origin) of the split."""
    ov, oq = z["obs_speed"], z["obs_flow"]
    ok = z["elig"] & np.isfinite(ov) & np.isfinite(oq)
    q = ok & (np.nan_to_num(ov, nan=1e9) <= 0.6 * ctx.vf)     # exact at visible cells: obs == true (note 103)
    return ov, oq, ok, q


def official_ongoing(z, ok, q, d, t0):
    """The organizers' own ongoing selector (`config/synthetic_release_v1.json`, notes 24-25, and the rule
    `t2_official_like_eval.ongoing_official` reconstructs): origins 07-21 h, eligible history coverage >= 0.7,
    and persistence IoU <= 0.9 - the last observed queue held constant must NOT already solve the window.

    `sup_batch`'s own selector is only "two queued history cells and a queued horizon", which admits exactly the
    trivially persistent windows the official set excludes, so the adaptation spends its steps on a task the
    scored windows are chosen not to be."""
    hour = t0 / 12
    if not (7 <= hour < 21):
        return False
    if z["elig"][d, t0 - C.H:t0].mean() < 0.7:
        return False
    okh, vq = ok[d, t0 - C.H:t0], q[d, t0 - C.H:t0]
    S = vq.shape[1]
    idx = np.where(okh[::-1], np.arange(C.H)[::-1][:, None], -1).max(0)
    last = np.zeros(S, bool)
    m = idx >= 0
    last[m] = vq[idx[m], np.arange(S)[m]]
    fut, efut = q[d, t0 + 1:t0 + 1 + C.F], ok[d, t0 + 1:t0 + 1 + C.F]
    p = np.broadcast_to(last[None, :], fut.shape) & efut
    union = int((p | fut).sum())
    return union == 0 or int((p & fut).sum()) / union <= 0.9


def sup_batch(ctx, stat, z, pool, days, t_max_by_day, bs, rng, kind, sel="loose"):
    """A batch of pseudo Task 2 windows whose horizon ends at or before the day's cut-off."""
    ov, oq, ok, q = pool
    cand = []
    for _ in range(bs * (8 if sel == "loose" else 48)):       # the official selector rejects far more draws
        d = days[rng.integers(len(days))]
        hi_lim = min(t_max_by_day.get(int(d), 10 ** 6), z["obs_speed"].shape[1] - 1)
        if hi_lim <= C.H + C.F + 1:
            continue
        t0 = int(rng.integers(C.H + 1, hi_lim - C.F))
        hist = q[d, t0 - C.H:t0]
        fut = q[d, t0 + 1:t0 + 1 + C.F]
        on_ok = (not hist.any()) and (not fut[:C.F - 1].any()) and fut[C.F - 1].any()
        og_ok = hist.sum() >= 2 and fut.any()
        if og_ok and sel == "official" and not official_ongoing(z, ok, q, d, t0):
            og_ok = False
        if (kind == "onset" and on_ok) or (kind == "ongoing" and og_ok):
            cand.append((d, t0))
        if len(cand) >= bs:
            break
    if not cand:
        return None
    d = np.array([c[0] for c in cand]); t0 = np.array([c[1] for c in cand])
    hi = t0[:, None] + np.arange(-C.H, 0)
    fi = t0[:, None] + np.arange(1, C.F + 1)
    w = dict(v=ov[d[:, None], hi], qf=oq[d[:, None], hi], ok=ok[d[:, None], hi], t0=t0, d=d)
    x = C.make_x(ctx, stat, w, np.arange(len(d)), z)
    y = q[d[:, None], fi].astype(np.float32)
    m = ok[d[:, None], fi].astype(np.float32)
    return (C.to_dev(x), torch.from_numpy(y).to(C.DEV), torch.from_numpy(m).to(C.DEV))


def adapt_sup(net, params, batches, steps, lr, w_rec, rec_batches):
    opt = C.make_opt(params, lr)
    net.train()
    for _ in range(steps):
        loss = None
        b = batches()
        if b is not None:
            x, y, m = b
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = net(x)
            logit = out[:, :C.F].float()
            loss = C.cnn.loss_fn(logit, y, m) + C.cnn.LOVASZ * C.cnn.lovasz_hinge(logit, y, m)
        if w_rec:
            xb = rec_batches()
            if xb is not None:
                r = C.ssl_loss(net, xb)
                loss = r * w_rec if loss is None else loss + w_rec * r
        if loss is None:
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    net.eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=PROJ / "work" / "models" / "t2_cnn_lov_rec_s20.pt")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--w-rec", type=float, default=1.0)
    ap.add_argument("--modes", nargs="*", default=list(MODES))
    ap.add_argument("--save", default="", help="tag: also write the probabilities to occ_lab/cache")
    ap.add_argument("--hist", type=int, default=-1,
                    help="note 169: >= 0 resets the model per day and adapts only over the last HIST days (the "
                         "official onset windows' condition on day HIST); -1 = cumulative from day 0 (as before)")
    ap.add_argument("--sel", default="loose", choices=("loose", "official"),
                    help="item 3: which windows the supervision is drawn from. 'official' applies the "
                         "organizers' own ongoing selector (07-21 h, history coverage >= 0.7, persistence "
                         "IoU <= 0.9) so the adaptation trains on the task the scored windows actually are")
    args = ap.parse_args()
    t_start = time.time()
    res = {}
    for i, panel in enumerate(C.T2_PANELS):
        ctx, stat, W = C.windows(i, panel)
        for s in C.SPLITS:
            on, og, z = W[s]["on"], W[s]["og"], W[s]["z"]
            pool = pseudo_pool(ctx, z)
            days = np.unique(np.concatenate([on["d"], og["d"]]))
            first_t0 = {int(d): int(min(on["t0"][on["d"] == d].min(initial=10**6),
                                        og["t0"][og["d"] == d].min(initial=10**6))) for d in days}
            for mode in args.modes:
                net, params = C.build_net(args.ckpt, "full")
                rng = np.random.default_rng(7 + i)
                p_on = np.zeros((len(on["d"]), ctx.S), np.float32)
                p_og = np.zeros(og["fut"].shape, np.float32)
                for d in days:
                    if args.hist >= 0 and mode != "none":
                        # note 169: the official onset windows sit on days 0-4 of the scored month, i.e. after only
                        # 0-4 days of adaptation. Reset the model and adapt day by day over the K days before d
                        # (the official recipe: each day's pool = the earlier days of the window + that day before
                        # the first origin), so a day-d lab window sees what an official day-K window sees.
                        net, params = C.build_net(args.ckpt, "full")
                        for e in range(max(0, int(d) - args.hist), int(d) + 1):
                            past = np.arange(max(0, int(d) - args.hist), e)
                            if e == int(d):
                                pool_days = np.append(past, e) if first_t0[e] > C.H + C.F + 2 else past
                                cuts = {int(k): (first_t0[e] if int(k) == e else 10**6) for k in pool_days}
                            else:
                                pool_days, cuts = np.append(past, e), {int(k): 10**6 for k in np.append(past, e)}
                            if not len(pool_days):
                                continue
                            sup = (lambda: sup_batch(ctx, stat, z, pool, pool_days, cuts, args.bs, rng,
                                                     "ongoing", args.sel)) if "sup" in mode else (lambda: None)
                            recp = past                                    # strictly earlier days, as the official rec_batch
                            rec = (lambda: C.past_batch(ctx, stat, z, recp, args.bs, rng))                                 if ("rec" in mode and len(recp)) else (lambda: None)
                            adapt_sup(net, params, sup, args.steps, args.lr, args.w_rec if "rec" in mode else 0.0, rec)
                    elif mode != "none":
                        past = np.arange(d)
                        cut = {int(d): first_t0[int(d)]}          # same day, only before the first origin
                        pool_days = np.append(past, d) if first_t0[int(d)] > C.H + C.F + 2 else past
                        cuts = {int(k): (cut[int(d)] if int(k) == int(d) else 10**6) for k in pool_days}
                        if len(pool_days):
                            sup = (lambda: sup_batch(ctx, stat, z, pool, pool_days, cuts, args.bs, rng,
                                                     "ongoing", args.sel)) if "sup" in mode \
                                else (lambda: None)
                            rec = (lambda: C.past_batch(ctx, stat, z, past, args.bs, rng)) \
                                if "rec" in mode else (lambda: None)
                            adapt_sup(net, params, sup, args.steps, args.lr,
                                      args.w_rec if "rec" in mode else 0.0, rec)
                    io = np.flatnonzero(on["d"] == d)
                    if len(io):
                        p_on[io] = C.cnn.predict(net, C.make_x(ctx, stat, on, io, z))[:, C.F - 1]
                    ig = np.flatnonzero(og["d"] == d)
                    if len(ig):
                        p_og[ig] = C.cnn.predict(net, C.make_x(ctx, stat, og, ig, z))
                a = res.setdefault((s, mode), {})
                a[panel] = (C.onset_iou(p_on, on).mean(), C.ongoing_iou(p_og, og, 0.5).mean())
                if args.save and mode == args.modes[-1]:
                    np.savez(LAB / "cache" / f"pseudo_{args.save}_{s}_{panel}.npz",
                             on=p_on.astype(np.float32), og=p_og.astype(np.float16))
        print(f"  {panel} ({time.time() - t_start:.0f}s)", flush=True)

    for s in C.SPLITS:
        print(f"\n=== {s} (single CNN, its own IoU) ===")
        print(f"{'adaptation':<12} {'onset':>9} {'ongoing':>9}")
        for mode in args.modes:
            d = res.get((s, mode))
            if d:
                print(f"{mode:<12} {C.family_mean({k: v[0] for k, v in d.items()}):9.4f} "
                      f"{C.family_mean({k: v[1] for k, v in d.items()}):9.4f}")


if __name__ == "__main__":
    main()
