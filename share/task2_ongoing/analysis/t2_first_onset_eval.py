"""Task 2 onset, evaluated on windows that look like the official ones: the first queue of the morning.

Official onset windows (note 24) have origins at 04-14 h, 97.5% on days 1-7, no visible queue in the history,
and the queue first appears at T+30. Our generic pseudo windows take onsets at any time of day, which may rank
methods differently. Here each day contributes at most one event: the first slot t in [04:00, 14:00) where a
visible eligible cell is queued and none was queued in the previous 90 min (t-18 .. t-1). The window has
T = t - 6: history T-60 .. T-5, and the queue at T+30.

Predictors (the IoU of the onset cells at T+30, scored on days < 14):
    E, R          saved models (sub_t2m1, sub_t2robust), train table priors, k = round(1.2 sum p)
    rule_train    top-3 by P_on + 0.5 P_q from the train table
    loc_train     top-k links by the train split's first-onset location frequency (+-1 link smoothing)
    loc_past      the same, from the month's first-onset events on days < d (train fallback on day 0)
    loc_loo       the same, from every other day of the month (option B, measurement only)
    yesterday     the onset links of the most recent earlier event of the month
Also reported: how often an event hits (+-1 link) the modal first-onset link of the other days (consistency).

    py analysis/t2_first_onset_eval.py
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
from t2_m1_predict import onset_X, table_prior  # noqa: E402
from t2_pseudo_eval import iou  # noqa: E402
from tfb.data import load  # noqa: E402
from tfb.t1eval import family_mean  # noqa: E402

LO, HI = 48, 168  # 04:00 .. 14:00
EVAL_DAYS = 14


def official_blank(panel: str, split: str, z: dict):
    """(D, T) cells blanked for the official Task 2 windows (horizon + the following hour), and (D,) days that hold
    an official onset window (their real first morning queue is hidden, so they are skipped)."""
    import pandas as pd
    W = pd.read_csv(ROOT / "data" / "kaggle_public" / "task2" / panel / split / "window_index.csv")
    d_of = {d: i for i, d in enumerate(z["dates"])}
    blank = np.zeros(z["obs_speed"].shape[:2], bool)
    onset_day = np.zeros(len(z["dates"]), bool)
    for w in W.itertuples():
        d = d_of[str(w.date)]
        t = pd.Timestamp(w.forecast_origin)
        t0 = t.hour * 12 + t.minute // 5
        blank[d, t0 + 1:min(t0 + 1 + F + 12, blank.shape[1])] = True
        onset_day[d] |= w.condition == "queue_onset"
    return blank, onset_day


def first_onsets(vf, z, truth: bool = False, blank=None):
    """One event per day. With blank=(blank, onset_day) from official_blank: skip days holding an official onset
    window and any candidate whose history, slot T or horizon touches an official blank (their history is empty)."""
    v = z["true_speed"] if truth else z["obs_speed"]
    qf = z["true_flow"] if truth else z["obs_flow"]
    ok = z["elig"] & np.isfinite(v) & np.isfinite(qf)
    q = ok & (np.nan_to_num(v, nan=1e9) <= 0.6 * vf)
    ds, ts = [], []
    for d in range(q.shape[0]):
        if blank is not None and blank[1][d]:
            continue
        for t in range(max(LO, H + F), HI):
            if blank is not None and blank[0][d, t - H - F:t + 1].any():
                continue
            if q[d, t].any() and not q[d, t - H - F:t].any():
                ds.append(d); ts.append(t)
                break
    d, t0 = np.array(ds, int), np.array(ts, int) - F
    hi = t0[:, None] + np.arange(-H, 0)
    fi = t0[:, None] + np.arange(1, F + 1)
    return dict(d=d, t0=t0, v=v[d[:, None], hi], qf=qf[d[:, None], hi], ok=ok[d[:, None], hi],
                fut=q[d[:, None], fi], efut=ok[d[:, None], fi], cells=q[d, t0 + F])


def smooth_links(x: np.ndarray) -> np.ndarray:
    return x + 0.5 * (np.roll(x, 1, -1) + np.roll(x, -1, -1))


def top_k_pred(score: np.ndarray, k: int, n: int, S: int) -> np.ndarray:
    pr = np.zeros((n, F, S), bool)
    np.put_along_axis(pr[:, F - 1], np.argsort(-score, 1)[:, :k], True, 1)
    return pr


def main() -> None:
    t_start = time.time()
    models = {name: ([lgb.Booster(model_file=str(f)) for f in sorted((ROOT / "work" / d).glob("t2_onset*.txt"))])
              for name, d in (("E", "sub_t2m1"), ("R", "sub_t2robust"))}
    res: dict[str, dict[str, dict[str, float]]] = {}
    cons: dict[str, dict[str, float]] = {}
    for i, panel in enumerate(T2_PANELS):
        ctx = Ctx(panel, i)
        ctx.fitd = np.ones_like(ctx.fitd)
        tr = first_onsets(ctx.vf, ctx.z, truth=True)
        loc_train = smooth_links(tr["cells"].sum(0).astype(float))
        for split in ("validation", "private"):
            z = load(panel, split)
            w = first_onsets(ctx.vf, z)
            n, S = len(w["d"]), ctx.S
            if n == 0:
                continue
            sel = w["d"] < EVAL_DAYS
            cells = smooth_links(w["cells"].astype(float))
            grp = (z["weekday"][w["d"]] >= 5).astype(int)
            key = np.stack([w["d"], grp], 1)
            pri = table_prior(ctx)
            scores = {}
            _, X = onset_X(ctx, w["v"], w["qf"], w["ok"], w["t0"], grp, pri, key)
            for mname, ms in models.items():
                p = np.mean([m.predict(X.reshape(-1, X.shape[-1])) for m in ms], 0).reshape(X.shape[:2])
                pr = np.zeros((n, F, S), bool)
                pr[:, F - 1] = onset_decide(p, ("sum", 1.2))
                scores[mname] = pr
            t30 = w["t0"] + F
            scores["rule_train"] = top_k_pred(pri("on", key, t30) + 0.5 * pri("q", key, t30), 3, n, S)
            past = np.array([cells[w["d"] < d].sum(0) if (w["d"] < d).any() else loc_train / max(loc_train.sum(), 1)
                             for d in w["d"]])
            loo = np.array([cells[w["d"] != d].sum(0) for d in w["d"]])
            yest = np.array([w["cells"][np.flatnonzero(w["d"] < d)[-1]] if (w["d"] < d).any() else np.zeros(S, bool)
                             for d in w["d"]]).astype(float) + 1e-3 * loc_train  # tie-break by train
            for k in (1, 2, 3):
                scores[f"loc_train_k{k}"] = top_k_pred(np.broadcast_to(loc_train, (n, S)), k, n, S)
                scores[f"loc_past_k{k}"] = top_k_pred(past, k, n, S)
                scores[f"loc_loo_k{k}"] = top_k_pred(loo, k, n, S)
            scores["yesterday"] = top_k_pred(yest, max(1, int(np.median(w["cells"].sum(1)))), n, S)
            for name, pr in scores.items():
                res.setdefault(split, {}).setdefault(name, {})[panel] = iou(pr[sel], w["fut"][sel], w["efut"][sel]).mean()
            modal = np.array([np.argmax(loo[j]) for j in range(n)])
            hit = np.array([w["cells"][j, max(0, modal[j] - 1):modal[j] + 2].any() for j in range(n)])
            cons.setdefault(split, {})[panel] = hit.mean()
            print(f"[{panel} {split}] events {n} (days<14: {sel.sum()}), onset cells/event {w['cells'].sum(1).mean():.1f}, "
                  f"hit modal(+-1) {hit.mean():.2f} | E {res[split]['E'][panel]:.3f} R {res[split]['R'][panel]:.3f} "
                  f"rule {res[split]['rule_train'][panel]:.3f} loc_train1 {res[split]['loc_train_k1'][panel]:.3f} "
                  f"loc_past1 {res[split]['loc_past_k1'][panel]:.3f} loc_loo1 {res[split]['loc_loo_k1'][panel]:.3f} "
                  f"yest {res[split]['yesterday'][panel]:.3f} ({time.time() - t_start:.0f}s)", flush=True)
    for split, r in res.items():
        print(f"\n== {split}: first-morning-onset windows, days < {EVAL_DAYS}, family mean IoU ==")
        print("  " + "  ".join(f"{k} {family_mean(v):.3f}" for k, v in r.items()))
        print(f"  consistency: event hits the other days' modal onset link (+-1): {family_mean(cons[split]):.3f}")


if __name__ == "__main__":
    main()
