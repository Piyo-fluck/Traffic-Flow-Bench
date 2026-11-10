"""Task 2 rule-based predictor for validation and private (notes 09-11).

  ongoing windows: a link is queued over the whole horizon when its lowest
                   eligible v/vf in the last LAST history steps is below VCUT.
  onset windows:   only T+30 can be queued (the official origins sit 30 min before
                   onset); predict the top-k links of  P_on + alpha * P_q  at T+30.

P_q is P(link queued at slot | day group), and P_on is P(link starts a queue
at slot | day group). Both are learned from train truth over every train day
and smoothed over +-smooth slots. Inputs per window are only window_history
(T-60..T-5) and those priors. The condition comes from window_index.csv.

    py scripts/t2_predict_rules.py --k 3 --alpha 0.5 --smooth 1 --group wt2 --out-dir work/sub_t2rules_v2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
from t2_eda02 import T2_PANELS  # noqa: E402
from t2_eda03 import F, earliest_onset, labels  # noqa: E402
from tfb.data import RELEASE, load  # noqa: E402

N_GROUPS = {"wt2": 2, "dow7": 7, "all": 1}


def group_of(weekday: np.ndarray | int, mode: str) -> np.ndarray:
    wd = np.asarray(weekday).astype(int)
    return {"wt2": (wd >= 5).astype(int), "dow7": wd, "all": np.zeros_like(wd)}[mode]


def slot(ts: pd.Series) -> np.ndarray:
    s = ts.astype(str)
    return (s.str.slice(11, 13).astype(int) * 12 + s.str.slice(14, 16).astype(int) // 5).to_numpy()


def train_priors(z: dict, smooth: int, mode: str) -> dict[str, np.ndarray]:
    """(G, T, S) priors per day group over every train day, smoothed in time."""
    q, _ = labels(z)
    g = group_of(z["weekday"], mode)
    eo = earliest_onset(q)
    onset_cell = np.zeros_like(q)
    d_i, t_i = np.nonzero(eo)
    onset_cell[d_i, t_i + F] = q[d_i, t_i + F]
    out = {}
    for name, x in (("P_q", q), ("P_on", onset_cell)):
        xs = sum(np.roll(x.astype(np.float32), k, axis=1) for k in range(-smooth, smooth + 1)) / (2 * smooth + 1)
        out[name] = np.stack([xs[g == k].mean(0) for k in range(N_GROUPS[mode])])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.5, help="weight of P_q added to P_on in the onset score")
    ap.add_argument("--smooth", type=int, default=1)
    ap.add_argument("--group", choices=list(N_GROUPS), default="wt2")
    ap.add_argument("--vcut", type=float, default=0.8)
    ap.add_argument("--last", type=int, default=3)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"config: k={args.k} alpha={args.alpha} smooth={args.smooth} group={args.group} vcut={args.vcut} last={args.last}")
    ctx = {}
    for panel in T2_PANELS:
        z = load(panel, "train")
        pr = train_priors(z, args.smooth, args.group)
        ctx[panel] = (pr["P_on"] + args.alpha * pr["P_q"], {l: i for i, l in enumerate(z["link_ids"])}, z["free_speed"])
    for split in ("validation", "private"):
        frames = []
        for panel in T2_PANELS:
            score, s_of, vf = ctx[panel]
            base = RELEASE / "task2" / panel / split
            W = pd.read_csv(base / "window_index.csv")
            Hh = pd.read_parquet(base / "window_history.parquet")
            Q = pd.read_csv(base / "sample_submission_queue.csv", usecols=["window_id", "timestamp", "link_id"])
            Q["queue_pred"] = 0
            q_slot = slot(Q.timestamp)
            q_s = Q.link_id.astype(str).map(s_of).to_numpy()
            n_on = n_og = 0
            for w in W.itertuples(index=False):
                t0 = int(slot(pd.Series([w.forecast_origin]))[0])
                rows = (Q.window_id == w.window_id).to_numpy()
                if w.condition == "queue_ongoing":
                    h = Hh[Hh.window_id == w.window_id]
                    h = h[(slot(h.timestamp) >= t0 - args.last) & h.is_score_eligible.astype(bool) & h.speed_kmh.notna()]
                    hs = h.link_id.astype(str).map(s_of).to_numpy()
                    queued = np.zeros(len(s_of), bool)
                    queued[hs[h.speed_kmh.to_numpy(float) / vf[hs] < args.vcut]] = True
                    sel = rows & queued[q_s]
                    n_og += int(queued.sum())
                else:
                    g = int(group_of(pd.Timestamp(str(w.date)).weekday(), args.group))
                    top = np.argsort(-score[g, t0 + F])[:args.k]
                    sel = rows & (q_slot == t0 + F) & np.isin(q_s, top)
                    n_on += int(sel.sum())
                Q.loc[sel, "queue_pred"] = 1
            frames.append(Q)
            print(f"[{split} {panel}] windows {len(W)}, onset cells predicted {n_on}, ongoing queued links {n_og}", flush=True)
        out = pd.concat(frames, ignore_index=True)
        out.to_csv(args.out_dir / f"queue_{split}.csv", index=False)
        print(f"wrote {args.out_dir / f'queue_{split}.csv'}: {len(out):,} rows, share 1 = {out.queue_pred.mean():.4f}")


if __name__ == "__main__":
    main()
