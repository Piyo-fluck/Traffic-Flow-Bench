"""Task 2 EDA 05: do the validation/private windows look like the train official windows?

No labels on validation/private, so compare what can be seen: the history
(slow links, trends), the origin time, and how strongly the recurrence prior
points at T+30 for onset windows. The priors are the rule-v2 score
(P_on + 0.5 P_q, weekday/weekend, +-1 slot), fitted on every train day.

    py analysis/t2_eda05_split_shift.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "scripts"))
from t2_eda02 import T2_PANELS  # noqa: E402
from t2_predict_rules import group_of, slot, train_priors  # noqa: E402
from tfb.data import RELEASE, load  # noqa: E402

F = 6


def main() -> None:
    rows = []
    for panel in T2_PANELS:
        z = load(panel, "train")
        pr = train_priors(z, 1, "wt2")
        score = pr["P_on"] + 0.5 * pr["P_q"]
        s_of = {l: i for i, l in enumerate(z["link_ids"])}
        vf = z["free_speed"]
        S = len(s_of)
        for split in ("train", "validation", "private"):
            base = RELEASE / "task2" / panel / split
            W = pd.read_csv(base / "window_index.csv")
            H = pd.read_parquet(base / "window_history.parquet")
            H["t"] = slot(H.timestamp)
            H["s"] = H.link_id.astype(str).map(s_of)
            for w in W.itertuples(index=False):
                t0 = int(slot(pd.Series([w.forecast_origin]))[0])
                h = H[(H.window_id == w.window_id) & H.is_score_eligible.astype(bool) & H.speed_kmh.notna()]
                ratio = h.speed_kmh.to_numpy(float) / vf[h.s.to_numpy()]
                ht = h.t.to_numpy() - t0
                slow_last3 = np.unique(h.s.to_numpy()[(ht >= -3) & (ratio < 0.8)])
                slow_early = np.unique(h.s.to_numpy()[(ht <= -10) & (ratio < 0.8)])
                g = int(group_of(pd.Timestamp(str(w.date)).weekday(), "wt2"))
                sc = np.sort(score[g, t0 + F])[::-1]
                rows.append(dict(panel=panel, split=split, cond=w.condition, hour=t0 / 12,
                                 weekend=g, min_ratio=float(ratio.min()) if len(ratio) else np.nan,
                                 slow_last3=len(slow_last3), slow_early=len(slow_early),
                                 top1=sc[0], top3=sc[2], top3_share=sc[:3].sum() / max(sc.sum(), 1e-9)))
    r = pd.DataFrame(rows)
    r.to_csv(ROOT / "work" / "t2_eda05_windows.csv", index=False)
    pd.set_option("display.width", 200)
    for cond in ("queue_onset", "queue_ongoing"):
        g = r[r.cond == cond].groupby("split")
        print(f"\n== {cond}: mean over windows (8 panels x 5 per split) ==")
        print(g[["hour", "weekend", "min_ratio", "slow_last3", "slow_early", "top1", "top3", "top3_share"]].mean().round(3).to_string())
    on = r[r.cond == "queue_onset"]
    print("\nonset: top-1 prior score at T+30 by panel and split")
    print(on.pivot_table(index="panel", columns="split", values="top1", aggfunc="mean").round(3).to_string())
    og = r[r.cond == "queue_ongoing"]
    print("\nongoing: slow links now (last 3) vs early (T-60..T-50), growth ratio by split")
    print((og.groupby("split").slow_last3.sum() / og.groupby("split").slow_early.sum().clip(lower=1)).round(3).to_string())


if __name__ == "__main__":
    main()
