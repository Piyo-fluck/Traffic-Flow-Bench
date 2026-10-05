"""Production onset probabilities from the H model PLUS the Task 1 residual block (ideas 2 and 3, scale-free half).

Note 64: the scale-free half of the Task 1 residual features (each divided by the median over the links of the
window, and the within-window ranks) lifts onset by +0.0084…+0.0196 alone and about +0.004 inside M blend, with the
private month carrying most of it. This script trains the production onset model with those features and writes the
probabilities for the REAL official windows.

Training: H configuration - prior dropout, 5 bags, 600 rounds, 63 leaves - on the earliest-onset train windows.
Features: the usual onset set plus t1_av3_rel, t1_mix_rel, t1_av3_rk, t1_mix_rk built from the three cached Task 1
reconstructions (`work/sub_m3v1`, `sub_m3v1_s1`, `sub_m3w128`), which exist for train and for both scored months.

Writes prob_<split>_<panel>.npz (prob (N, F, S) with the onset row at T+30, is_onset, window_id, t0) in
window_index.csv order, the layout scripts/t2_cnn_blend_predict.py reads with --onset-probs-dir.

    .venv/Scripts/python.exe scripts/t2_onset_t1res_predict.py --out-dir work/sub_t2onset_t1res
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "scripts"))
import t2_prior_robust  # noqa: E402
from t2_eda02 import T2_PANELS  # noqa: E402
from t2_eda03 import F  # noqa: E402
from t2_m1_models import Ctx, hist_from_truth  # noqa: E402
from t2_m1_predict import feats_on, hist_from_release, table_prior  # noqa: E402
from t2_onset_t1res_feats import NAMES, t1_feats  # noqa: E402
from t2_shift_remedies import PARAMS  # noqa: E402
from tfb.data import RELEASE, load  # noqa: E402

COLS = slice(8, 12)          # the scale-free half only (note 64)
SEL = NAMES[COLS]
N_BAGS, ROUNDS, LEAVES, MAX_WIN = 5, 600, 63, 2000
rng = np.random.default_rng(29)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    Xb, ys, ctxs, names_on = [[] for _ in range(N_BAGS)], [], [], None
    for i, panel in enumerate(T2_PANELS):
        ctx = Ctx(panel, i)
        ctx.fitd = np.ones_like(ctx.fitd)
        z = ctx.z
        d, t0 = ctx.eo
        if len(d) > MAX_WIN:
            s = np.sort(rng.choice(len(d), MAX_WIN, replace=False))
            d, t0 = d[s], t0[s]
        v, qf, ok = hist_from_truth(ctx, d, t0)
        E, _ = t1_feats(panel, "train", d, t0, v, qf, ok)
        t2_prior_robust.rng = np.random.default_rng(1000 + i)
        for b in range(N_BAGS):
            names_on, X = feats_on(ctx, False, v, qf, ok, t0, ctx.g[d], t2_prior_robust.dropout_prior(ctx), d,
                                   z["weekday"][d])
            Xb[b].append(np.concatenate([X.reshape(-1, X.shape[-1]), E[..., COLS].reshape(-1, len(SEL))], 1))
        ys.append(ctx.q[d, t0 + F].reshape(-1))
        ctx.z = {k: z[k] for k in ("link_ids", "weekday", "dates")}
        ctx.q = ctx.el = None
        ctxs.append(ctx)
        print(f"[{panel}] train windows {len(d)} ({time.time() - t_start:.0f}s)", flush=True)
    y = np.concatenate(ys)
    names = names_on + SEL
    models = [lgb.train(dict(PARAMS, num_leaves=LEAVES, seed=b), lgb.Dataset(np.concatenate(Xb[b]), y,
                                                                            feature_name=names), ROUNDS)
              for b in range(N_BAGS)]
    print(f"trained {N_BAGS} bags ({time.time() - t_start:.0f}s)", flush=True)
    del Xb
    for ctx in ctxs:
        for split in ("validation", "private"):
            z = load(ctx.panel, split)
            base = RELEASE / "task2" / ctx.panel / split
            W = pd.read_csv(base / "window_index.csv")
            Hh = pd.read_parquet(base / "window_history.parquet")
            v, qf, ok, t0, grp = hist_from_release(ctx, Hh, W)
            d_of = {str(x): k for k, x in enumerate(z["dates"])}
            d = np.array([d_of[str(x)] for x in W.date], int)
            key = np.stack([d, grp.astype(int)], 1)
            _, X = feats_on(ctx, False, v, qf, ok, t0, grp.astype(int), table_prior(ctx), key,
                            z["weekday"][d].astype(int))
            E, _ = t1_feats(ctx.panel, split, d, t0, v, qf, ok)
            Xo = np.concatenate([X, E[..., COLS]], -1)
            p = np.mean([m.predict(Xo.reshape(-1, Xo.shape[-1])) for m in models], 0).reshape(Xo.shape[:2])
            prob = np.zeros((len(W), F, ctx.S), np.float32)
            prob[:, F - 1] = p
            np.savez(args.out_dir / f"prob_{split}_{ctx.panel}.npz", prob=prob,
                     is_onset=(W.condition == "queue_onset").to_numpy(),
                     window_id=W.window_id.to_numpy().astype(str), t0=t0)
            print(f"[{split} {ctx.panel}] {len(W)} windows ({time.time() - t_start:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
