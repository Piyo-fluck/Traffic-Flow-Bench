"""Task 1 production run for M2 (LightGBM): fit on every train day, predict validation and private.

The profile is the weekday x tod table over all of train, as in the official
baseline. The models are pooled over the panels and trained for a fixed
number of rounds, taken from the holdout run.

    py scripts/t1_m2_predict.py --rounds-v 6000 --rounds-q 6000 --out-dir work/sub_m2
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
from t1_m2_lgbm import features  # noqa: E402
from tfb.data import PANELS, load  # noqa: E402
from tfb.features import profile_table, truth_valid  # noqa: E402
from tfb.submit import append_csv, check_against_template, state_frame  # noqa: E402

SEED = 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", nargs="+", default=PANELS)
    ap.add_argument("--rounds-v", type=int, required=True)
    ap.add_argument("--rounds-q", type=int, required=True)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--rows-per-panel", type=int, default=1_500_000)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tables, X, yv, yq, names = {}, [], [], [], None
    for panel in args.panels:
        t0 = time.time()
        z = load(panel, "train")
        alldays = np.ones(len(z["dates"]), bool)
        valid = truth_valid(z)
        tv = profile_table(z["true_speed"], valid, z["weekday"], alldays)
        tq = profile_table(z["true_flow"], valid, z["weekday"], alldays)
        tables[panel] = (tv, tq)
        base_v, base_q = tv[z["weekday"]], tq[z["weekday"]]
        F = features(z, base_v, base_q)
        names = list(F)
        cells = z["target"] & np.isfinite(z["true_speed"]) & np.isfinite(z["true_flow"])
        idx = np.flatnonzero(cells)
        if idx.size > args.rows_per_panel:
            keep = np.zeros(cells.size, bool)
            keep[rng.choice(idx, args.rows_per_panel, replace=False)] = True
            cells = keep.reshape(cells.shape)
        X.append(np.stack([F[k][cells] for k in names], 1).astype(np.float32))
        yv.append((z["true_speed"] - base_v)[cells].astype(np.float32))
        yq.append(((z["true_flow"] - base_q) / z["lanes"])[cells].astype(np.float32))
        del F
        print(f"[train {panel}] {int(cells.sum()):,} rows ({time.time() - t0:.0f}s)", flush=True)

    params = dict(objective="regression", learning_rate=args.lr, num_leaves=255, min_data_in_leaf=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  verbose=-1, seed=SEED, num_threads=8)
    Xa = np.concatenate(X)
    models = {}
    for ch, y, rounds in (("v", np.concatenate(yv), args.rounds_v), ("q", np.concatenate(yq), args.rounds_q)):
        t0 = time.time()
        models[ch] = lgb.train(params, lgb.Dataset(Xa, y, feature_name=names), num_boost_round=rounds)
        models[ch].save_model(str(args.out_dir / f"t1_m2_{ch}.txt"))
        print(f"model {ch}: {rounds} rounds on {len(y):,} rows ({time.time() - t0:.0f}s)", flush=True)
    del Xa, X

    for split in ("validation", "private"):
        out = args.out_dir / f"state_{split}.csv"
        for i, panel in enumerate(args.panels):
            z = load(panel, split)
            tv, tq = tables[panel]
            base_v, base_q = tv[z["weekday"]], tq[z["weekday"]]
            F = features(z, base_v, base_q)
            cells = z["target"]
            Xs = np.stack([F[k][cells] for k in names], 1).astype(np.float32)
            del F
            pv, pq = base_v.copy(), base_q.copy()
            pv[cells] += models["v"].predict(Xs)
            pq[cells] += models["q"].predict(Xs) * np.broadcast_to(z["lanes"], cells.shape)[cells]
            frame = state_frame(panel, z, pv, pq)
            check_against_template(panel, split, frame)
            append_csv(frame, out, first=(i == 0))
            print(f"[{split} {panel}] {len(frame):,} rows, speed mean {frame.speed_kmh.mean():.1f}", flush=True)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
