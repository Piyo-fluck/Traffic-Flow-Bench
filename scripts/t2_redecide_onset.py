"""Re-decide the onset windows of a Task 2 queue file from saved onset probabilities (no retraining).

Production (t2_m1_predict.py) saves prob_<split>_<panel>.npz (prob (N, F, S), is_onset, window_id, t0). This script
takes a base queue file (e.g. work/sub_t2J), keeps every ongoing row, and replaces the onset rows with an
expected-IoU decision on the T+30 probabilities (analysis/t2_expected_iou.select; note 31):
    ratio : k* = argmax_k S_k / (k + P - S_k) on p' = min(1, c p)
    mc    : exact E[IoU_k] by Monte Carlo, conditioned on a non-empty truth

    py scripts/t2_redecide_onset.py --probs-dir work/sub_t2robust_bag5 --base-dir work/sub_t2J --rule ratio --c 1.2 --out-dir work/sub_t2K
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "analysis"), str(ROOT / "scripts")]
from t2_eda02 import T2_PANELS  # noqa: E402
from t2_eda03 import F  # noqa: E402
from t2_expected_iou import select  # noqa: E402
from t2_predict_rules import slot  # noqa: E402
from tfb.data import load  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs-dir", type=Path, required=True)
    ap.add_argument("--base-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--rule", choices=["ratio", "mc"], default="ratio")
    ap.add_argument("--c", type=float, default=1.2)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("validation", "private"):
        q = pd.read_csv(args.base_dir / f"queue_{split}.csv", dtype={"window_id": str, "timestamp": str, "link_id": str})
        new = q.queue_pred.to_numpy().copy()
        wid = q.window_id.to_numpy()
        n_on = 0
        for panel in T2_PANELS:
            f = np.load(args.probs_dir / f"prob_{split}_{panel}.npz")
            on_ids = f["window_id"][f["is_onset"]]
            rows = np.flatnonzero(np.isin(wid, on_ids))
            if not len(rows):
                continue
            s_of = {l: i for i, l in enumerate(load(panel, split)["link_ids"])}
            w_of = {w: i for i, w in enumerate(f["window_id"])}
            wi = np.array([w_of[w] for w in wid[rows]])
            h = np.asarray(slot(q.timestamp.iloc[rows])) - f["t0"][wi] - 1
            si = q.link_id.iloc[rows].map(s_of).to_numpy()
            if np.isnan(si.astype(float)).any() or (h < 0).any() or (h >= F).any():
                raise ValueError(f"{panel}/{split}: unmapped links or timestamps outside T+5..T+30")
            chosen = {i: select(f["prob"][i, F - 1], args.rule, args.c) for i in np.unique(wi)}
            new[rows] = [int(hh == F - 1 and chosen[ww][int(ss)]) for ww, hh, ss in zip(wi, h, si)]
            n_on += len(np.unique(wi))
        changed = int((new != q.queue_pred.to_numpy()).sum())
        q["queue_pred"] = new
        q.to_csv(args.out_dir / f"queue_{split}.csv", index=False)
        print(f"{split}: onset windows re-decided {n_on}, rows changed {changed}, share 1 {new.mean():.4f} "
              f"-> {args.out_dir / f'queue_{split}.csv'}", flush=True)


if __name__ == "__main__":
    main()
