"""Task 2, option B (measurement only): a queue map from the OTHER days of the same month.

Official onset windows sit at the morning-peak onset (origins 04-14 h) and 97.5% of them fall on days 1-7. So a
past-days-only month prior has 0-6 days to work with, while the run shift (note 17) makes the train priors
mislead. The masked Task 1 layer of the month shows most cells on every day. The window's own day is left out,
and the horizon plus buffer of official windows are blank anyway.

In-month pseudo windows with days < 14 (tfb t2_pseudo_eval) are scored with the saved models:
    E  : sub_t2m1 (models trained with train priors), onset and ongoing
    R  : sub_t2robust (onset trained with prior dropout; the ongoing model uses no prior, so it is left out)
under these scoring priors (P_q and P_on):
    train   the train table
    past4   the month's days < d, kappa 4 (the current rule)
    loo1 / loo4   every other day of the month, kappa 1 / 4 (MonthLOOPrior)
Also a model-free rule: the onset top-3 links by P_on + 0.5 P_q at T+30.

    py analysis/t2_month_loo_eval.py
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
from t2_eda03 import F, H  # noqa: E402
from t2_m1_models import Ctx, onset_decide  # noqa: E402
from t2_m1_predict import ongoing_X, onset_X, table_prior, train_tables  # noqa: E402
from t2_pseudo_eval import iou, month_prior_both, pseudo_windows  # noqa: E402
from tfb.data import T, load  # noqa: E402
from tfb.month import MonthLOOPrior  # noqa: E402
from tfb.t1eval import family_mean  # noqa: E402

EVAL_DAYS = 14


def onset_cells(ctx: Ctx, z: dict):
    ov = z["obs_speed"]
    vis = z["elig"] & np.isfinite(ov)
    q = vis & (np.nan_to_num(ov, nan=1e9) <= 0.6 * ctx.vf)
    on = np.zeros_like(q)
    for t0 in range(H, T - F):
        hist = q[:, t0 - H:t0]
        ongo = (hist.sum((1, 2)) >= 2) & (hist.sum(1).max(1) >= 2)
        ok = ~ongo & ~q[:, t0 + 1:t0 + F].any((1, 2)) & q[:, t0 + F].any(1)
        on[ok, t0 + F] |= q[ok, t0 + F]
    return vis, on


def loo_both(ctx: Ctx, z: dict, kappa: float, vis, on_cell):
    tabs = train_tables(ctx)
    mq = MonthLOOPrior(z, ctx.vf, tabs["q"], kappa)
    zon = dict(z, obs_speed=np.where(on_cell, 0.0, np.where(vis, 999.0, np.nan)).astype(np.float32))
    mon = MonthLOOPrior(zon, ctx.vf, tabs["on"], kappa)  # "queued" in zon == onset cell

    def pri(name, key, t):
        return (mq if name == "q" else mon)(key[:, 0], key[:, 1], t)
    return pri


def main() -> None:
    t_start = time.time()
    models = {name: ([lgb.Booster(model_file=str(f)) for f in sorted((ROOT / "work" / d).glob("t2_onset*.txt"))],
                     lgb.Booster(model_file=str(ROOT / "work" / d / "t2_ongoing.txt")))
              for name, d in (("E", "sub_t2m1"), ("R", "sub_t2robust"))}
    variants = ("train", "past4", "loo1", "loo4")
    res: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for i, panel in enumerate(T2_PANELS):
        ctx = Ctx(panel, i)
        ctx.fitd = np.ones_like(ctx.fitd)
        for split in ("validation", "private"):
            z = load(panel, split)
            w = pseudo_windows(ctx, z)
            grp = (z["weekday"][w["d"]] >= 5).astype(int)
            key = np.stack([w["d"], grp], 1)
            sel = w["d"] < EVAL_DAYS
            vis, on_cell = onset_cells(ctx, z)
            pris = {"train": table_prior(ctx), "past4": month_prior_both(ctx, z, 4.0),
                    "loo1": loo_both(ctx, z, 1.0, vis, on_cell), "loo4": loo_both(ctx, z, 4.0, vis, on_cell)}
            on = sel & w["onset"]
            og = np.flatnonzero(sel & w["ongoing"])
            for v in variants:
                pri = pris[v]
                r = res.setdefault(split, {}).setdefault(v, {})
                if on.any():
                    t30 = w["t0"][on] + F
                    s = pri("on", key[on], t30) + 0.5 * pri("q", key[on], t30)
                    pr = np.zeros((on.sum(), F, ctx.S), bool)
                    np.put_along_axis(pr[:, F - 1], np.argsort(-s, 1)[:, :3], True, 1)
                    r.setdefault("rule_onset", {})[panel] = iou(pr, w["fut"][on], w["efut"][on]).mean()
                    _, X = onset_X(ctx, w["v"][on], w["qf"][on], w["ok"][on], w["t0"][on], grp[on], pri, key[on])
                    for mname, (m_on, _) in models.items():
                        p = np.mean([m.predict(X.reshape(-1, X.shape[-1])) for m in m_on], 0).reshape(X.shape[:2])
                        pr = np.zeros((on.sum(), F, ctx.S), bool)
                        pr[:, F - 1] = onset_decide(p, ("sum", 1.2))
                        r.setdefault(f"{mname}_onset", {})[panel] = iou(pr, w["fut"][on], w["efut"][on]).mean()
                ious = []
                for j in range(0, len(og), 200):  # E's ongoing model uses priors; R's uses none
                    b = og[j:j + 200]
                    _, Xh, _ = ongoing_X(ctx, w["v"][b], w["qf"][b], w["ok"][b], w["t0"][b], grp[b], pri, key[b])
                    p = models["E"][1].predict(Xh.reshape(-1, Xh.shape[-1])).reshape(Xh.shape[:3])
                    ious.append(iou(p >= 0.65, w["fut"][b], w["efut"][b]))
                if ious:
                    r.setdefault("E_ongoing", {})[panel] = np.concatenate(ious).mean()
            print(f"[{panel} {split}] ({time.time() - t_start:.0f}s) " + "  ".join(
                f"{v}: rule {res[split][v]['rule_onset'].get(panel, np.nan):.3f} E {res[split][v]['E_onset'].get(panel, np.nan):.3f} "
                f"R {res[split][v]['R_onset'].get(panel, np.nan):.3f}" for v in variants), flush=True)
    for split, rv in res.items():
        print(f"\n== {split}: pseudo windows on days < {EVAL_DAYS}, family mean ==")
        for v, r in rv.items():
            fm = {k: family_mean(x) for k, x in r.items()}
            print(f"  {v:<6} onset: rule {fm['rule_onset']:.3f}  E {fm['E_onset']:.3f}  R {fm['R_onset']:.3f} | "
                  f"E ongoing {fm['E_ongoing']:.3f} | E S_queue {(fm['E_onset'] + fm['E_ongoing']) / 2:.3f}")
    print(f"total {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
