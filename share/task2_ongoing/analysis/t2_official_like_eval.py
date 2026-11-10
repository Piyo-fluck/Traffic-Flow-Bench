"""Task 2: pseudo-evaluation that mimics the official window selection (the generic pseudo windows do not).

The generic in-month pseudo windows (t2_pseudo_eval) take onsets at any time of day. The official ones do not
(notes 24-25). They follow `config/synthetic_release_v1.json` and the released window_index:
  onset   : the first queue of the morning. Origins at 04-14 h, 97.5% on days 1-7, no queue in the history, the
            queue first at T+30. Mimicked by t2_first_onset_eval.first_onsets (one event per day).
  ongoing : a queue visible in the history and a queue in the horizon, persistence IoU <= 0.9
            (maximum_persistence_iou), eligible history coverage >= 0.7, origins at 07-21 h, 83% on days 1-7.
Both are scored on days 1-7 (the official days) and on days 1-14 (more samples), for:
    E : sub_t2m1 (train-prior onset and ongoing models; LB 0.82542)
    G : sub_t2robust (prior-dropout onset, no-prior ongoing)
and S_queue = (onset + ongoing) / 2 is turned into an estimated total change against E (x 0.30).

    py analysis/t2_official_like_eval.py
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
from t2_eda03 import F  # noqa: E402
from t2_first_onset_eval import first_onsets, official_blank, smooth_links  # noqa: E402
from t2_m1_models import Ctx, nanlast, onset_decide  # noqa: E402
from t2_m1_predict import feats_og, feats_on, table_prior, zero_prior  # noqa: E402
from t2_pseudo_eval import iou, pseudo_windows  # noqa: E402
from tfb.data import load  # noqa: E402
from tfb.t1eval import family_mean  # noqa: E402

DAY_SETS = {"days1-7": 7, "days1-14": 14, "all": 31}  # official-blank days are excluded (note 28)
PAST_BETA = 0.5  # onset_past: p * (1 + beta * past-day first-onset location frequency), days < d only
# name: (dir, prior used by the ongoing model, demand-stress features); a model is scored only if its dir exists
MODELS = {"E": ("sub_t2m1", "train", False), "G": ("sub_t2robust", "zero", False),
          "H": ("sub_t2robust_bag5", "zero", False),  # G with 5 onset bags
          "I": ("sub_t2robust_bag5_stress", "zero", True)}  # H + demand-stress features


def fmean(d: dict) -> float:
    d = {k: v for k, v in d.items() if np.isfinite(v)}
    return family_mean(d) if d else float("nan")


def ongoing_official(ctx: Ctx, z: dict, blank=None):
    w = pseudo_windows(ctx, z)
    hour = w["t0"] / 12
    hi = w["t0"][:, None] + np.arange(-12, 0)
    cov = z["elig"][w["d"][:, None], hi].mean((1, 2))
    if blank is not None:  # drop windows whose history, slot T or horizon touches an official Task 2 blank
        span = w["t0"][:, None] + np.arange(-12, F + 1)
        cov = np.where(blank[0][w["d"][:, None], np.clip(span, 0, blank[0].shape[1] - 1)].any(1), -1.0, cov)
    vis_q = w["ok"] & (np.nan_to_num(w["v"], nan=1e9) <= 0.6 * ctx.vf)
    last = nanlast(vis_q.astype(np.float32), w["ok"]) > 0.5
    pers = iou(np.broadcast_to(last[:, None], w["fut"].shape), w["fut"], w["efut"])
    keep = w["ongoing"] & (hour >= 7) & (hour < 21) & (cov >= 0.7) & (pers <= 0.9)
    return {k: (v[keep] if isinstance(v, np.ndarray) and len(v) == len(keep) else v) for k, v in w.items()}, pers[keep]


def main() -> None:
    t_start = time.time()
    boosters = {m: ([lgb.Booster(model_file=str(f)) for f in sorted((ROOT / "work" / d).glob("t2_onset*.txt"))],
                    lgb.Booster(model_file=str(ROOT / "work" / d / "t2_ongoing.txt"))) for m, (d, *_) in MODELS.items()
                if (ROOT / "work" / d / "t2_ongoing.txt").exists()}
    res: dict = {}
    for i, panel in enumerate(T2_PANELS):
        ctx = Ctx(panel, i)
        ctx.fitd = np.ones_like(ctx.fitd)
        for split in ("validation", "private"):
            z = load(panel, split)
            blank = official_blank(panel, split, z)
            on = first_onsets(ctx.vf, z, blank=blank)
            og, pers = ongoing_official(ctx, z, blank)
            line = []
            for mname, (m_on, m_og) in boosters.items():
                # onset
                if len(on["d"]):
                    grp = (z["weekday"][on["d"]] >= 5).astype(int)
                    key = np.stack([on["d"], grp], 1)
                    _, X = feats_on(ctx, MODELS[mname][2], on["v"], on["qf"], on["ok"], on["t0"], grp, table_prior(ctx), key,
                                    z["weekday"][on["d"]].astype(int))
                    p = np.mean([m.predict(X.reshape(-1, X.shape[-1])) for m in m_on], 0).reshape(X.shape[:2])
                    pr = np.zeros((len(on["d"]), F, ctx.S), bool)
                    pr[:, F - 1] = onset_decide(p, ("sum", 1.2))
                    s_on = iou(pr, on["fut"], on["efut"])
                    cells = smooth_links(on["cells"].astype(float))
                    past = np.array([cells[on["d"] < dd].sum(0) for dd in on["d"]])
                    past = past / np.maximum(past.max(1, keepdims=True), 1e-9)
                    k = np.clip(np.round(1.2 * p.sum(1)), 1, 6).astype(int)
                    order = np.argsort(-(p * (1 + PAST_BETA * past)), 1)
                    prp = np.zeros_like(pr)
                    prp[:, F - 1] = np.argsort(order, 1) < k[:, None]
                    s_onp = iou(prp, on["fut"], on["efut"])
                else:
                    s_on = s_onp = np.zeros(0)
                # ongoing
                ious = []
                pri = table_prior(ctx) if MODELS[mname][1] == "train" else zero_prior(ctx)
                grp = (z["weekday"][og["d"]] >= 5).astype(int)
                key = np.stack([og["d"], grp], 1)
                wdo = z["weekday"][og["d"]].astype(int)
                for j in range(0, len(og["d"]), 200):
                    b = slice(j, j + 200)
                    _, Xh, _ = feats_og(ctx, MODELS[mname][2], og["v"][b], og["qf"][b], og["ok"][b], og["t0"][b], grp[b],
                                        pri, key[b], wdo[b])
                    p = m_og.predict(Xh.reshape(-1, Xh.shape[-1])).reshape(Xh.shape[:3])
                    ious.append(iou(p >= 0.65, og["fut"][b], og["efut"][b]))
                s_og = np.concatenate(ious) if ious else np.zeros(0)
                for ds, dmax in DAY_SETS.items():
                    a = s_on[on["d"] < dmax] if len(s_on) else s_on
                    b = s_og[og["d"] < dmax] if len(s_og) else s_og
                    ap = s_onp[on["d"] < dmax] if len(s_onp) else s_onp
                    r = res.setdefault(split, {}).setdefault(ds, {}).setdefault(mname, {"onset": {}, "ongoing": {}, "onset_past": {}})
                    r["onset"][panel] = a.mean() if len(a) else np.nan
                    r["onset_past"][panel] = ap.mean() if len(ap) else np.nan
                    r["ongoing"][panel] = b.mean() if len(b) else np.nan
                line.append(f"{mname}: onset {res[split]['days1-14'][mname]['onset'][panel]:.3f} "
                            f"ongoing {res[split]['days1-14'][mname]['ongoing'][panel]:.3f}")
            print(f"[{panel} {split}] onset events {len(on['d'])}, official-like ongoing windows {len(og['d'])} "
                  f"(persistence IoU mean {pers.mean() if len(pers) else np.nan:.2f}) | " + " | ".join(line)
                  + f" ({time.time() - t_start:.0f}s)", flush=True)
    for split, rs in res.items():
        for ds, rm in rs.items():
            print(f"\n== {split}, {ds}: official-like pseudo windows, family mean ==")
            base = None
            for mname, r in rm.items():
                a, b = fmean(r["onset"]), fmean(r["ongoing"])
                sq = (a + b) / 2
                base = sq if base is None else base
                ap = fmean(r["onset_past"])
                print(f"  {mname}: onset {a:.3f}  ongoing {b:.3f}  S_queue {sq:.3f}  "
                      f"(vs E: {sq - base:+.3f} S_queue = {0.30 * (sq - base):+.4f} total) | "
                      f"onset+past {ap:.3f} -> S_queue {(ap + b) / 2:.3f} ({0.30 * ((ap + b) / 2 - base):+.4f} total)")
    print(f"total {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
