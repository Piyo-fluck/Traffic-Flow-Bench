"""Adapt the single model to the scored run, the way the member that already works is adapted.

The stack has exactly one adapted member, and it is a width-64 CNN: `p_PS` scores 0.8291 alone where twelve
frozen models score 0.8295. The new single model is stronger but frozen - 0.7970 alone - so the operation that
produced the only real gain in the ongoing half has never been applied to it.

Same recipe as `scripts/t2_pseudo_adapt_official.py`: for each day of the scored month in order, fine-tune on
pseudo Task 2 windows whose whole horizon is at or before that day's first official origin, drawn from strictly
earlier days and from the earlier hours of that day, with labels from the released layer; then predict that
day's windows. Weights carry forward, so the run is causal. The reconstruction half of the loss is off - the
transformer has no reconstruction head - so this is the supervised half only, which note 106 measured as the
part that carries the gain.

    T2CNN_WIDTH=128 python occ_lab/adapt_one.py --ckpt occ_lab/models/one_tfm_w128_ema_s0.pt
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("T2CNN_ARCH", "tfm")
os.environ.setdefault("T2CNN_LOVASZ", "0.5")
LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta"))
sys.path.insert(0, str(LAB.parent / "analysis"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import common as C  # noqa: E402
from online_adapter import load_split  # noqa: E402
from residual_diag import TAU  # noqa: E402

CACHE = LAB / "cache"


def predict(net, x, bs=32):
    net.eval()
    out = []
    with torch.no_grad():
        for a in range(0, len(x), bs):
            t = torch.from_numpy(np.ascontiguousarray(x[a:a + bs])).to(C.DEV, torch.float32)
            out.append(torch.sigmoid(net(t)[:, :C.F]).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, C.F, x.shape[-1]), np.float32)


def sup_batch(ctx, stat, z, pool, days, cuts, bs, rng):
    """Pseudo ongoing windows whose horizon ends at or before the day's cut-off."""
    ov, oq, ok, q = pool
    T = z["obs_speed"].shape[1]
    cand = []
    for _ in range(bs * 8):
        d = int(days[rng.integers(len(days))])
        lim = min(cuts.get(d, 10 ** 6), T - 1)
        if lim <= C.H + C.F + 1:
            continue
        t0 = int(rng.integers(C.H + 1, lim - C.F))
        if q[d, t0 - C.H:t0].sum() >= 2 and q[d, t0 + 1:t0 + 1 + C.F].any():
            cand.append((d, t0))
        if len(cand) >= bs:
            break
    if not cand:
        return None
    d = np.array([c[0] for c in cand])
    t0 = np.array([c[1] for c in cand])
    hi = t0[:, None] + np.arange(-C.H, 0)
    fi = t0[:, None] + np.arange(1, C.F + 1)
    w = dict(v=ov[d[:, None], hi], qf=oq[d[:, None], hi], ok=ok[d[:, None], hi], t0=t0, d=d)
    x = C.make_x(ctx, stat, w, np.arange(len(d)), z)
    tt = lambda a, dt: torch.from_numpy(np.ascontiguousarray(a)).to(C.DEV, dt)  # noqa: E731
    return (tt(x, torch.float32), tt(q[d[:, None], fi].astype(np.float32), torch.float32),
            tt(ok[d[:, None], fi].astype(np.float32), torch.float32))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--save", default="one")
    ap.add_argument("--tag", default="s20")
    ap.add_argument("--w-ps", type=float, nargs="+", default=[0.0, 0.3, 0.5, 0.6])
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    sd = torch.load(args.ckpt, map_location=C.DEV)
    is_tfm = "gru.weight_ih_l0" in sd
    cin = sd["gru.weight_ih_l0"].shape[1] if is_tfm else sd["inp.weight"].shape[1]
    mk = (lambda: C.cnn.T2Tfm(cin, C.cnn.WIDTH)) if is_tfm else (lambda: C.cnn.T2Net(cin, C.cnn.WIDTH))
    print(f"{args.ckpt.name}: {'tfm' if is_tfm else 'cnn'} width {C.cnn.WIDTH}, {cin} channels, "
          f"{args.steps} steps/day, lr {args.lr}", flush=True)
    fam = {}
    t_all = time.time()
    for i, panel in enumerate(C.T2_PANELS):
        ctx, stat, Wd = C.windows(i, panel, need_train=True)
        sp = load_split(i, panel, args.tag)
        for s, D in sp.items():
            og, z = Wd[s]["og"], Wd[s]["z"]
            ok_all = z["elig"] & np.isfinite(z["obs_speed"]) & np.isfinite(z["obs_flow"])
            pool = (z["obs_speed"], z["obs_flow"], ok_all,
                    ok_all & (np.nan_to_num(z["obs_speed"], nan=1e9) <= 0.6 * ctx.vf))
            X = C.make_x(ctx, stat, og, np.arange(len(og["d"])), z)
            net = mk().to(C.DEV)
            net.load_state_dict(sd)
            p_fr = predict(net, X)                            # frozen, for the before/after pair
            opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
            rng = np.random.default_rng(7 + i)
            p_ad = np.zeros_like(p_fr)
            days = np.unique(og["d"])
            first = {int(d): int(og["t0"][og["d"] == d].min()) for d in days}
            for d in days:
                past = np.arange(int(d))
                cut = first[int(d)]
                pool_days = np.append(past, int(d)) if cut > C.H + C.F + 2 else past
                if len(pool_days):
                    cuts = {int(k): (cut if int(k) == int(d) else 10 ** 6) for k in pool_days}
                    net.train()
                    for _ in range(args.steps):
                        b = sup_batch(ctx, stat, z, pool, pool_days, cuts, args.bs, rng)
                        if b is None:
                            break
                        x, y, m = b
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            out = net(x)
                        logit = out[:, :C.F].float()
                        loss = C.cnn.loss_fn(logit, y, m) + C.cnn.LOVASZ * C.cnn.lovasz_hinge(logit, y, m)
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        opt.step()
                m = np.flatnonzero(og["d"] == d)
                if len(m):
                    p_ad[m] = predict(net, X[m])
            np.savez(CACHE / f"onead_{args.save}_{s}_{panel}.npz", og=p_ad.astype(np.float16),
                     frozen=p_fr.astype(np.float16))
            y, e, ps = D["y"], D["e"], D["ps"]
            for nm, p in (("one frozen", p_fr), ("one adapted", p_ad), ("p_PS", ps)):
                fam.setdefault((s, nm), {})[panel] = C.iou(p >= TAU, y, e).mean()
            for w in args.w_ps:
                fam.setdefault((s, f"one adapted + p_PS {w:g}"), {})[panel] = C.iou(
                    ((1 - w) * p_ad + w * ps) >= TAU, y, e).mean()
            fam.setdefault((s, "adapted + frozen 0.5"), {})[panel] = C.iou(
                (0.5 * p_ad + 0.5 * p_fr) >= TAU, y, e).mean()
        ctx.z = ctx.q = ctx.el = None
        print(f"  {panel} ({time.time() - t_all:.0f}s)", flush=True)

    names = (["one frozen", "one adapted", "p_PS", "adapted + frozen 0.5"]
             + [f"one adapted + p_PS {w:g}" for w in args.w_ps])
    print(f"\nongoing family IoU at tau 0.50, official-like pool of the scored months\n")
    print(f"{'setting':<26}" + "".join(f"{s:>13}" for s in C.SPLITS) + f"{'mean':>10}")
    for nm in names:
        v = [C.family_mean(fam[(s, nm)]) for s in C.SPLITS]
        print(f"{nm:<26}" + "".join(f"{x:>13.4f}" for x in v) + f"{float(np.mean(v)):>10.4f}")


if __name__ == "__main__":
    main()
