"""Production Task 2: the state-conditioned analog onset map (note 47) on the OFFICIAL windows.

It uses exactly the functions and configuration that were evaluated (analysis/t2_analog_tune.py, variant 'K40'):
state = the last-30-min log(flow / train profile) over all links; key = its 8 principal components (fitted on the
library) plus the time of day; library = the train earliest-onset windows of the same weekday type with their T+30
onset sets; K = 40 nearest, Gaussian weights, shrunk to the library frequency (alpha 2).
For every official window of the validation and private splits, the state comes from its window_history.parquet
(history only). Writes prob_<split>_<panel>.npz (prob (N, F, S): the map at T+30 for onset windows, 0 elsewhere;
is_onset, window_id, t0) in window_index.csv order, for scripts/t2_cnn_blend_predict.py --analog-dir.

    .venv/Scripts/python.exe scripts/t2_analog_probs.py --out-dir work/sub_t2analog
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "analysis"), str(ROOT / "scripts")]
from t2_analog_prior import train_profile  # noqa: E402
from t2_analog_tune import REF, amap, build, state  # noqa: E402
from t2_eda02 import T2_PANELS  # noqa: E402
from t2_eda03 import F  # noqa: E402
from t2_m1_models import Ctx, hist_from_truth  # noqa: E402
from t2_m1_predict import hist_from_release  # noqa: E402
from tfb.data import RELEASE  # noqa: E402

CFG = dict(REF, k=40)   # the evaluated 'K40' map (note 47)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"config {CFG}", flush=True)
    for i, panel in enumerate(T2_PANELS):
        ctx = Ctx(panel, i)
        d, t0 = ctx.eo
        v, qf, ok = hist_from_truth(ctx, d, t0)
        y = ctx.q[d, t0 + F].astype(np.float32)
        grp = ctx.g[d].astype(int)
        prof = train_profile(ctx)
        raw = state(ctx, prof, v, qf, ok, t0, grp, CFG["key"])
        kf = build(CFG, raw, t0, CFG["npc"], lib_y=y)
        L = dict(raw=raw, t0=t0, y=y, grp=grp)
        pg = y.mean(0)
        for split in ("validation", "private"):
            base = RELEASE / "task2" / panel / split
            W = pd.read_csv(base / "window_index.csv")
            Hh = pd.read_parquet(base / "window_history.parquet")
            vv, qq, oo, tt, gg = hist_from_release(ctx, Hh, W)
            is_on = (W.condition == "queue_onset").to_numpy()
            prob = np.zeros((len(W), F, ctx.S), np.float32)
            if is_on.any():
                g_on = gg[is_on].astype(int)
                q_raw = state(ctx, prof, vv[is_on], qq[is_on], oo[is_on], tt[is_on], g_on, CFG["key"])
                shrink = np.broadcast_to(pg, (int(is_on.sum()), ctx.S))
                mp = amap(CFG, kf, L, q_raw, tt[is_on], g_on, shrink)
                prob[is_on, F - 1] = mp
                print(f"[{split} {panel}] onset windows {int(is_on.sum())}, observed history share {oo[is_on].mean():.3f}, "
                      f"map max per window {mp.max(1).round(2).tolist()}", flush=True)
            np.savez(args.out_dir / f"prob_{split}_{panel}.npz", prob=prob, is_onset=is_on,
                     window_id=W.window_id.to_numpy().astype(str), t0=tt)


if __name__ == "__main__":
    main()
