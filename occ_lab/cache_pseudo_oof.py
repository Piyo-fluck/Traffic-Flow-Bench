"""The pseudo-supervised member's probabilities along its own CAUSAL TRAJECTORY, on the held-out tail.

Item 5 of the user's Task 2 plan, and it is open. The residual corrector was fitted against

    p0 = 0.20 gs1 + 0.60 tfm8 + 0.20 cnn3          (occ_lab/residual_oof.py:41)

and the OOF cache holds no pseudo-supervised member at all. Candidate AR applies that same corrector to

    p0 = 0.40 (0.20 gs1 + 0.60 tfm8 + 0.20 cnn3) + 0.60 p_PS

so five of the fifteen features - base, logit(base), |base - 0.5|, logit(stid) - logit(base),
logit(occ_cnn) - logit(base) - are computed from a p0 the corrector has never seen.

Worse, p_PS is not a fixed model. It is adapted before each day from that day's strictly earlier days, so its
feature distribution moves along the month. Refitting on a fixed model's predictions would still be wrong. This
reproduces the trajectory: for each tail day in order, adapt on the earlier days (and the earlier hours of the
same day, up to the day's first window origin, exactly as production does), then predict that day's windows.

Windows are built exactly as `cache_oof.py` builds them, with the same seed, so the caches align; `d` and `t0`
are saved so alignment can be checked rather than assumed.

    python occ_lab/cache_pseudo_oof.py --max-windows 4000
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
sys.path.insert(0, str(PROJ / "analysis"))
import numpy as np  # noqa: E402

import common as C  # noqa: E402
from pseudo_adapt import adapt_sup, pseudo_pool, sup_batch  # noqa: E402
from t2_official_like_eval import ongoing_official  # noqa: E402
from tfb.t1eval import FIT_END  # noqa: E402

CACHE = LAB / "cache"
REMOTE = PROJ / "work" / ("models_remote" if (PROJ / "work" / "models_remote").exists() else "models")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=REMOTE / "t2_cnn_lov_rec_s20.pt")
    ap.add_argument("--max-windows", type=int, default=4000)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--w-rec", type=float, default=1.0)
    ap.add_argument("--tag", default="", help="written to pseudo_oof<tag>_<panel>.npz, so the production cache stays")
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    t0_ = time.time()
    for i, panel in enumerate(C.T2_PANELS):
        ctx = C.Ctx(panel, i)
        ctx.loff = 0
        stat = C.cnn.static_channels(ctx, C.load_ramps(panel, "train", ctx.z))
        z = ctx.z
        og, _ = ongoing_official(ctx, z, None)
        rng_w = np.random.default_rng(0)                      # cache_oof.py's seed, so the windows match
        keep = np.flatnonzero(z["dates"][og["d"]] >= FIT_END)
        if len(keep) > args.max_windows:
            keep = np.sort(rng_w.choice(keep, args.max_windows, replace=False))
        og = {k: (v[keep] if isinstance(v, np.ndarray) and len(v) == len(og["d"]) else v) for k, v in og.items()}
        n = len(og["d"])
        pool = pseudo_pool(ctx, z)
        days = np.unique(og["d"])
        first_t0 = {int(d): int(og["t0"][og["d"] == d].min()) for d in days}
        net, params = C.build_net(args.ckpt, "full")
        rng = np.random.default_rng(7 + i)
        P = np.zeros(og["fut"].shape, np.float16)
        for d in days:
            past = np.arange(d)
            cut = first_t0[int(d)]
            pool_days = np.append(past, d) if cut > C.H + C.F + 2 else past
            if len(pool_days):
                cuts = {int(k): (cut if int(k) == int(d) else 10 ** 6) for k in pool_days}
                adapt_sup(net, params,
                          lambda: sup_batch(ctx, stat, z, pool, pool_days, cuts, args.bs, rng, "ongoing"),
                          args.steps, args.lr, args.w_rec,
                          lambda: C.past_batch(ctx, stat, z, past, args.bs, rng))
            m = np.flatnonzero(og["d"] == d)
            if len(m):
                P[m] = C.cnn.predict(net, C.make_x(ctx, stat, og, m, z)).astype(np.float16)
        np.savez(CACHE / f"pseudo_oof{args.tag}_{panel}.npz", pseudo=P, d=og["d"], t0=og["t0"],
                 y=og["fut"], e=og["efut"])
        print(f"  {panel}: {n:,} windows over {len(days)} tail days ({time.time() - t0_:.0f}s)", flush=True)
        ctx.z = ctx.q = ctx.el = None


if __name__ == "__main__":
    main()
