"""Task 2 pseudo-evaluation INSIDE the validation/private months (the other simulation run, note 17).

The scored months' masked layers still show most cells. Pseudo windows are built from them:
  history : visible eligible cells at T-60..T-5 (Task 1 masking makes them sparser than a real window_history)
  labels  : queued (speed <= 0.6 vf) at visible eligible cells of T+5..T+30; other cells are not scored
  onset   : history not ongoing, nothing visibly queued at T+5..T+25, something queued at T+30
  ongoing : history ongoing (release rule), something visibly queued in the horizon
Official Task 2 horizons and buffers are blank in the release, so they never act as labels.
Compared: rules v2 (C'), and the learned models (E) with train priors or month-past priors (kappa grid,
optionally also for the onset-start prior P_on). The models are the saved production models.
The priors for a window on day d use only days < d of the month (the user's decision).

    py analysis/t2_pseudo_eval.py validation
"""
from __future__ import annotations

import sys
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
from t2_m1_predict import month_prior, ongoing_X, onset_X, table_prior, train_tables  # noqa: E402
from tfb.data import T, load  # noqa: E402
from tfb.month import MonthPastPrior  # noqa: E402
from tfb.t1eval import family_mean  # noqa: E402

STEP = 3  # pseudo origins every 15 min


def month_prior_both(ctx: Ctx, z_split: dict, kappa: float):
    """Like month_prior, but P_on is also adapted (from the month's visible earliest-onset cells, past days only)."""
    tabs = train_tables(ctx)
    mq = MonthPastPrior(z_split, ctx.vf, tabs["q"], kappa)
    ov = z_split["obs_speed"]
    vis = z_split["elig"] & np.isfinite(ov)
    q = vis & (np.nan_to_num(ov, nan=1e9) <= 0.6 * ctx.vf)
    on_cell = np.zeros_like(q)
    for t0 in range(H, T - F):
        hist = q[:, t0 - H:t0]
        ongo = (hist.sum((1, 2)) >= 2) & (hist.sum(1).max(1) >= 2)
        ok = ~ongo & ~q[:, t0 + 1:t0 + F].any((1, 2)) & q[:, t0 + F].any(1)
        on_cell[ok, t0 + F] |= q[ok, t0 + F]
    zon = dict(z_split, obs_speed=np.where(on_cell, 0.0, np.where(vis, 999.0, np.nan)).astype(np.float32))
    mon = MonthPastPrior(zon, ctx.vf, tabs["on"], kappa)  # "queued" in zon == onset cell

    def pri(name, key, t):
        return (mq if name == "q" else mon)(key[:, 0], key[:, 1], t)
    return pri


def pseudo_windows(ctx: Ctx, z: dict):
    ov, of_ = z["obs_speed"], z["obs_flow"]
    ok_all = z["elig"] & np.isfinite(ov) & np.isfinite(of_)
    q = ok_all & (np.nan_to_num(ov, nan=1e9) <= 0.6 * ctx.vf)
    D = len(z["dates"])
    t0s = np.arange(H, T - F, STEP)
    d, t0 = np.repeat(np.arange(D), len(t0s)), np.tile(t0s, D)
    hi = t0[:, None] + np.arange(-H, 0)
    fi = t0[:, None] + np.arange(1, F + 1)
    hist = q[d[:, None], hi]
    ongoing = (hist.sum((1, 2)) >= 2) & (hist.sum(1).max(1) >= 2)
    fut, efut = q[d[:, None], fi], ok_all[d[:, None], fi]
    onset = ~ongoing & ~fut[:, :F - 1].any((1, 2)) & fut[:, F - 1].any(1)
    og = ongoing & fut.any((1, 2))
    return dict(d=d, t0=t0, v=ov[d[:, None], hi], qf=of_[d[:, None], hi], ok=ok_all[d[:, None], hi],
                fut=fut, efut=efut, onset=onset, ongoing=og)


def iou(pred, truth, elig):
    axes = tuple(range(1, pred.ndim))
    p = pred & elig
    inter, union = (p & truth).sum(axes), (p | truth).sum(axes)
    return np.where(union == 0, 1.0, inter / np.maximum(union, 1))


def main() -> None:
    split = sys.argv[1] if len(sys.argv) > 1 else "validation"
    m_on = lgb.Booster(model_file=str(ROOT / "work" / "sub_t2m1" / "t2_onset.txt"))
    m_og = lgb.Booster(model_file=str(ROOT / "work" / "sub_t2m1" / "t2_ongoing.txt"))
    variants = {"model/train-prior": None, "model/month k8": ("q", 8.0), "model/month k4": ("q", 4.0),
                "model/month k2": ("q", 2.0), "model/month k4 +Pon": ("both", 4.0), "model/month k2 +Pon": ("both", 2.0)}
    res: dict[str, dict[str, dict[str, float]]] = {}
    for i, panel in enumerate(T2_PANELS):
        ctx = Ctx(panel, i)
        ctx.fitd = np.ones_like(ctx.fitd)
        z = load(panel, split)
        w = pseudo_windows(ctx, z)
        grp = (z["weekday"][w["d"]] >= 5).astype(int)
        key = np.stack([w["d"], grp], 1)
        tabs = train_tables(ctx)
        # rules v2 (onset: top-3 of P_on + 0.5 P_q at T+30 from train; ongoing: min v/vf last 3 < 0.8)
        on, og = w["onset"], w["ongoing"]
        sc = tabs["on"][grp[on], w["t0"][on] + F] + 0.5 * tabs["q"][grp[on], w["t0"][on] + F]
        pr = np.zeros((on.sum(), F, ctx.S), bool)
        pr[:, F - 1] = onset_decide(sc, ("k", 3))
        res.setdefault("rules v2", {}).setdefault("onset", {})[panel] = iou(pr, w["fut"][on], w["efut"][on]).mean()
        with np.errstate(invalid="ignore", all="ignore"):
            vr = np.where(w["ok"][og][:, -3:], w["v"][og][:, -3:] / ctx.vf, np.nan)
            link = np.nan_to_num(np.nanmin(vr, 1), nan=9) < 0.8
        res["rules v2"].setdefault("ongoing", {})[panel] = iou(np.repeat(link[:, None], F, 1), w["fut"][og], w["efut"][og]).mean()
        for name, spec in variants.items():
            if spec is None:
                pri = table_prior(ctx)
            elif spec[0] == "q":
                pri = month_prior(ctx, z, spec[1])
            else:
                pri = month_prior_both(ctx, z, spec[1])
            _, X = onset_X(ctx, w["v"][on], w["qf"][on], w["ok"][on], w["t0"][on], grp[on], pri, key[on])
            p = m_on.predict(X.reshape(-1, X.shape[-1])).reshape(X.shape[:2])
            pr = np.zeros((on.sum(), F, ctx.S), bool)
            pr[:, F - 1] = onset_decide(p, ("sum", 1.2))
            res.setdefault(name, {}).setdefault("onset", {})[panel] = iou(pr, w["fut"][on], w["efut"][on]).mean()
            ious = []
            idx = np.flatnonzero(og)
            for j in range(0, len(idx), 200):
                b = idx[j:j + 200]
                _, Xh, _ = ongoing_X(ctx, w["v"][b], w["qf"][b], w["ok"][b], w["t0"][b], grp[b], pri, key[b])
                p = m_og.predict(Xh.reshape(-1, Xh.shape[-1])).reshape(Xh.shape[:3])
                ious.append(iou(p >= 0.65, w["fut"][b], w["efut"][b]))
            res[name].setdefault("ongoing", {})[panel] = np.concatenate(ious).mean()
        print(f"{panel}: pseudo onset {int(on.sum())}, ongoing {int(og.sum())} windows", flush=True)
        del ctx
    print(f"\n== pseudo-evaluation inside {split} (family mean over 8 panels) ==")
    for name, r in res.items():
        a, b = family_mean(r["onset"]), family_mean(r["ongoing"])
        print(f"  {name:<22} onset {a:.3f}  ongoing {b:.3f}  S_queue {(a + b) / 2:.3f}")


if __name__ == "__main__":
    main()
