"""Task 2: make the models robust to the run shift WITHOUT using scored-month data (notes 19-20).

The official validation/private windows sit in the first days of the month, so a past-days-only adaptation has
almost nothing to learn from. The remedy has to be built into training instead:
  E          production models (train priors)
  noprior    trained with every prior feature = 0
  dropout    domain randomisation: every training window gets zero priors with prob 0.5 (a window's prior is
             either intact or absent)
  shuffled   domain randomisation: priors built from a random 50% subset of train days and a time shift of
             up to +-2 slots per window, so the model sees wrong-but-plausible priors (as in another run)
Scored with train priors (noprior: zeros) on pseudo windows of the scored months restricted to days < EVAL_DAYS,
the same days as the official windows.

    py analysis/t2_prior_robust.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "scripts"))
from t2_eda02 import T2_PANELS  # noqa: E402
from t2_eda04 import prior_at  # noqa: E402
from t2_m1_models import Ctx  # noqa: E402
from t2_m1_predict import table_prior, train_prior  # noqa: E402
from t2_pseudo_eval import pseudo_windows  # noqa: E402
from t2_shift_remedies import fit, score, train_rows, zero_prior  # noqa: E402
from tfb.data import load  # noqa: E402
from tfb.t1eval import family_mean  # noqa: E402

EVAL_DAYS = 14
rng = np.random.default_rng(1)


def dropout_prior(ctx: Ctx, p: float = 0.5):
    base = train_prior(ctx)
    cache: dict = {}

    def pri(name, key, t):
        k = id(key)
        # The same windows keep the same on/off draw across all prior features of one feature build.
        # Also check the length, since a freed array's id can be reused by the next chunk.
        if k not in cache or len(cache[k]) != len(key):
            cache.clear()
            cache[k] = rng.random(len(key)) < p
        out = base(name, key, t)
        out[cache[k]] = 0.0
        return out
    return pri


def shuffled_prior(ctx: Ctx):
    use = rng.random(len(ctx.fitd)) < 0.5
    cache: dict = {}

    def pri(name, key, t):
        k = id(key)
        if k not in cache or len(cache[k]) != len(key):
            cache.clear()
            cache[k] = rng.integers(-2, 3, len(key))
        xs = ctx.xs_q if name == "q" else ctx.xs_on
        return prior_at(xs, ctx.g, use, key, np.clip(t + cache[k], 0, xs.shape[1] - 1), loo=True)
    return pri


def main() -> None:
    t_start = time.time()
    ctxs = []
    rows = {"noprior": [], "dropout": [], "shuffled": []}
    for i, p in enumerate(T2_PANELS):
        ctx = Ctx(p, i)
        ctx.fitd = np.ones_like(ctx.fitd)
        for name, pri in (("noprior", zero_prior(ctx)), ("dropout", dropout_prior(ctx)), ("shuffled", shuffled_prior(ctx))):
            names_on, on, names_og, og = train_rows(ctx, pri)
            rows[name].append((on, og))
        # Scoring needs only link ids, static attributes and the smoothed prior arrays (table_prior).
        ctx.z = {"link_ids": ctx.z["link_ids"]}
        ctx.q = ctx.el = None
        ctxs.append(ctx)
        print(f"[{p}] train rows ({time.time() - t_start:.0f}s)", flush=True)
    models = {"E": (lgb.Booster(model_file=str(ROOT / "work" / "sub_t2m1" / "t2_onset.txt")),
                    lgb.Booster(model_file=str(ROOT / "work" / "sub_t2m1" / "t2_ongoing.txt")))}
    for name, rs in rows.items():
        models[name] = (fit(names_on, [(r[0][0], r[0][1], 1.0) for r in rs], 600, 63),
                        fit(names_og, [(r[1][0], r[1][1], 1.0) for r in rs], 2000, 127))
        print(f"trained {name} ({time.time() - t_start:.0f}s)", flush=True)
    del rows

    for split in ("validation", "private"):
        res = {m: {"onset": {}, "ongoing": {}} for m in models}
        for ctx in ctxs:
            z = load(ctx.panel, split)
            w = pseudo_windows(ctx, z)
            grp = (z["weekday"][w["d"]] >= 5).astype(int)
            key = np.stack([w["d"], grp], 1)
            ev = w["d"] < EVAL_DAYS
            for m, (mo, mg) in models.items():
                pri = zero_prior(ctx) if m == "noprior" else table_prior(ctx)
                r = score(ctx, w, ev, grp, pri, key, mo, mg)
                for k2, val in r.items():
                    if not np.isnan(val):
                        res[m][k2][ctx.panel] = val
        print(f"\n== {split}: pseudo windows on days < {EVAL_DAYS} (official-window days), family mean ==")
        for m, r in res.items():
            a, b = family_mean(r["onset"]), family_mean(r["ongoing"])
            print(f"  {m:<9} onset {a:.3f}  ongoing {b:.3f}  S_queue {(a + b) / 2:.3f}", flush=True)
    print(f"total {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
