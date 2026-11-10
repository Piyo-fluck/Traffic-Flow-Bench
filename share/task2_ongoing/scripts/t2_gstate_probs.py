"""Production Task 2 ongoing probabilities with the global network-state features (notes 42-43).

Uses the ongoing model trained by analysis/t2_gstate.py on every train day (I's configuration: no priors,
demand-stress features, plus the 12 global-state features) and its per-panel PCA:
    work/models_gstate/ongoing_g1.txt, work/models_gstate/pca_<panel>.npz
Features come only from the official window histories (task2/<panel>/<split>/window_history.parquet), exactly as
scripts/t2_m1_predict.py builds them. It writes prob_<split>_<panel>.npz in the same format and window order, with
the onset rows left at 0 (the blend takes onset from --onset-probs-dir). Feed the directory to
scripts/t2_cnn_blend_predict.py as --ongoing-probs-dir.

    .venv/Scripts/python.exe scripts/t2_gstate_probs.py --out-dir work/sub_t2gs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "analysis"), str(ROOT / "scripts")]
from t2_eda02 import T2_PANELS  # noqa: E402
from t2_eda03 import F  # noqa: E402
from t2_gstate import G_NAMES, gstate  # noqa: E402
from t2_m1_models import Ctx  # noqa: E402
from t2_m1_predict import feats_og, hist_from_release, zero_prior  # noqa: E402
from tfb.data import RELEASE, load  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path, default=ROOT / "work" / "models_gstate")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    m_og = lgb.Booster(model_file=str(args.models / "ongoing_g1.txt"))
    for i, panel in enumerate(T2_PANELS):
        ctx = Ctx(panel, i)
        ctx.fitd = np.ones_like(ctx.fitd)
        pz = np.load(args.models / f"pca_{panel}.npz")
        pca = (pz["mu"], pz["vt"])
        for split in ("validation", "private"):
            base = RELEASE / "task2" / panel / split
            W = pd.read_csv(base / "window_index.csv")
            Hh = pd.read_parquet(base / "window_history.parquet")
            v, qf, ok, t0, grp = hist_from_release(ctx, Hh, W)
            z_split = load(panel, split)
            d_of = {d: k for k, d in enumerate(z_split["dates"])}
            key = np.stack([W.date.astype(str).map(d_of).to_numpy(), grp], 1).astype(int)
            wd = np.array([pd.Timestamp(str(x)).weekday() for x in W.date])
            is_on = (W.condition == "queue_onset").to_numpy()
            prob = np.zeros((len(W), F, ctx.S), np.float32)
            og = ~is_on
            if og.any():
                _, Xh, _ = feats_og(ctx, True, v[og], qf[og], ok[og], t0[og], grp[og], zero_prior(ctx), key[og], wd[og])
                g = np.broadcast_to(gstate(ctx, pca, v[og], qf[og], ok[og])[:, None, None], Xh.shape[:3] + (len(G_NAMES),))
                X = np.concatenate([Xh, g], -1)
                assert X.shape[-1] == m_og.num_feature(), "feature count differs from the trained model"
                prob[og] = m_og.predict(X.reshape(-1, X.shape[-1])).reshape(Xh.shape[:3])
            np.savez(args.out_dir / f"prob_{split}_{panel}.npz", prob=prob, is_onset=is_on,
                     window_id=W.window_id.to_numpy().astype(str), t0=t0)
            print(f"[{split} {panel}] ongoing windows {int(og.sum())}, mean p {prob[og].mean() if og.any() else 0:.4f}", flush=True)


if __name__ == "__main__":
    main()
