"""Task 2 ongoing: re-test ongoing-model choices on the official-like ongoing windows.

Official-like ongoing windows (t2_official_like_eval.ongoing_official): a visible history queue and a queue in the
horizon, persistence IoU <= 0.9, eligible history coverage >= 0.7, origins at 07-21 h. Four no-prior ongoing
models are trained on identical sampled rows (the sampling mask comes from the eligible-history version):
    hist    el  : only eligible history cells (current) | fin : every valued history cell (about 7% more; note 01)
    stress  off | on (demand-stress features; note 22)
and each is scored with the threshold tau in {0.55, 0.65, 0.75}. The reference is el/on/0.65, J's ongoing.
The score is the ongoing IoU, validation and private, days 1-7 and 1-14, family mean.

    py analysis/t2_ongoing_variants.py
"""
from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "scripts"))
from t2_eda02 import T2_PANELS, candidates  # noqa: E402
from t2_eda03 import F, H  # noqa: E402
from t2_m1_models import Ctx  # noqa: E402
from t2_m1_predict import feats_og, zero_prior  # noqa: E402
from t2_official_like_eval import ongoing_official  # noqa: E402
from t2_pseudo_eval import iou  # noqa: E402
from t2_shift_remedies import PARAMS  # noqa: E402
from tfb.data import load  # noqa: E402
from tfb.t1eval import family_mean  # noqa: E402

MAX_ROWS = 400_000
TAUS = (0.55, 0.65, 0.75)
DAY_SETS = {"d1-7": 7, "d1-14": 14}
rng = np.random.default_rng(5)


def fmean(d):
    d = {k: v for k, v in d.items() if np.isfinite(v)}
    return family_mean(d) if d else float("nan")


def main() -> None:
    t_start = time.time()
    configs = list(itertools.product(("el", "fin"), (False, True)))
    Xs = {c: [] for c in configs}
    ys, names, ctxs = [], {}, []
    for i, p in enumerate(T2_PANELS):
        ctx = Ctx(p, i)
        ctx.fitd = np.ones_like(ctx.fitd)
        z = ctx.z
        c = candidates(z)
        d, t0 = c["d"][c["ongoing"]], c["t0"][c["ongoing"]]
        parts = {cf: [] for cf in configs}
        yl = []
        for j in range(0, len(d), 400):
            dd, tt = d[j:j + 400], t0[j:j + 400]
            hi = tt[:, None] + np.arange(-H, 0)
            v, qf = z["true_speed"][dd[:, None], hi], z["true_flow"][dd[:, None], hi]
            oks = {"el": ctx.el[dd[:, None], hi], "fin": np.isfinite(v) & np.isfinite(qf)}
            y = ctx.q[dd[:, None], tt[:, None] + np.arange(1, F + 1)]
            e = ctx.el[dd[:, None], tt[:, None] + np.arange(1, F + 1)]
            keep = None
            for cf in configs:
                h, st = cf
                n, Xh, base = feats_og(ctx, st, v, qf, oks[h], tt, ctx.g[dd], zero_prior(ctx), dd, z["weekday"][dd])
                names[cf] = n
                if keep is None:  # the sampling mask comes from the first config (el, no stress) and is reused
                    near = np.minimum(base["dist_slow_up"], base["dist_slow_dn"]) <= 10
                    keep = np.flatnonzero(e & (np.broadcast_to(near[:, None, :], y.shape) | y | (rng.random(y.shape) < 0.02)))
                parts[cf].append(Xh.reshape(-1, Xh.shape[-1])[keep])
            yl.append(y.reshape(-1)[keep])
        yp = np.concatenate(yl)
        sub = rng.choice(len(yp), MAX_ROWS, replace=False) if len(yp) > MAX_ROWS else np.arange(len(yp))
        ys.append(yp[sub])
        for cf in configs:
            Xs[cf].append(np.concatenate(parts[cf])[sub])
        ctx.z = {k: z[k] for k in ("link_ids", "weekday")}
        ctx.q = ctx.el = None
        ctxs.append(ctx)
        print(f"[{p}] ongoing rows {len(sub):,} ({time.time() - t_start:.0f}s)", flush=True)
    y = np.concatenate(ys)
    models = {}
    for cf in configs:
        models[cf] = lgb.train(dict(PARAMS, num_leaves=127), lgb.Dataset(np.concatenate(Xs[cf]), y, feature_name=names[cf]), 3300)
        Xs[cf] = None
        print(f"trained {cf} ({time.time() - t_start:.0f}s)", flush=True)
    res: dict = {}
    for ctx in ctxs:
        for split in ("validation", "private"):
            z = load(ctx.panel, split)
            og, _ = ongoing_official(ctx, z)
            if not len(og["d"]):
                continue
            grp = (z["weekday"][og["d"]] >= 5).astype(int)
            key = np.stack([og["d"], grp], 1)
            wdo = z["weekday"][og["d"]].astype(int)
            okf = np.isfinite(og["v"]) & np.isfinite(og["qf"])
            for cf, m in models.items():
                h, st = cf
                ious = {t: [] for t in TAUS}
                for j in range(0, len(og["d"]), 200):
                    b = slice(j, j + 200)
                    ok = og["ok"][b] if h == "el" else okf[b]
                    _, Xh, _ = feats_og(ctx, st, og["v"][b], og["qf"][b], ok, og["t0"][b], grp[b], zero_prior(ctx), key[b], wdo[b])
                    p = m.predict(Xh.reshape(-1, Xh.shape[-1])).reshape(Xh.shape[:3])
                    for t in TAUS:
                        ious[t].append(iou(p >= t, og["fut"][b], og["efut"][b]))
                for t in TAUS:
                    s = np.concatenate(ious[t])
                    for ds, dmax in DAY_SETS.items():
                        msk = og["d"] < dmax
                        res.setdefault((split, ds), {}).setdefault((h, "stress" if st else "nostress", t), {})[ctx.panel] = \
                            s[msk].mean() if msk.any() else np.nan
        print(f"  scored {ctx.panel} ({time.time() - t_start:.0f}s)", flush=True)
    ref = ("el", "stress", 0.65)
    for (split, ds), r in res.items():
        print(f"\n== {split} {ds}: ongoing IoU on official-like windows, family mean (ref el/stress/0.65 = {fmean(r[ref]):.3f}) ==")
        for cfg, d in sorted(r.items(), key=lambda kv: -fmean(kv[1])):
            diffs = [d[p] - r[ref][p] for p in d if np.isfinite(d[p]) and np.isfinite(r[ref][p])]
            print(f"  {'/'.join(map(str, cfg)):<22} {fmean(d):.3f}  (vs ref {fmean(d) - fmean(r[ref]):+.3f}; "
                  f"panels better/worse {sum(x > 1e-9 for x in diffs)}/{sum(x < -1e-9 for x in diffs)})")
    print(f"total {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
