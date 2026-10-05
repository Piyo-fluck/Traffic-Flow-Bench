"""Adversarial validation weights for the Task 2 train windows (Kaggle playbook: train-vs-test classifier).

Per panel, a LightGBM separates the train run's windows (the ones longctx.train_panels samples: every first onset and
up to 3,000 ongoing candidates) from the scored run's windows (the official-like onset and ongoing pools of both
months, histories from the released layer - inputs only, no label). Features are window-level summaries of the
released history (the same 60 minutes every member reads): per slot the panel mean of v/vf, q/cap, queued share and
observed share, their ranges over the hour, the queued-link count and extent, time of day and weekend.
The weight of a train window is the density ratio p / (1 - p) from 5-fold out-of-fold probabilities, clipped to
[0.2, 5] and normalised to mean 1 per panel. The run shift is corridor-specific (note 134), hence per panel.

    python occ_lab/adv_weights.py      -> occ_lab/cache/advw_<panel>.npz (d, t0, w, auc)
"""
from __future__ import annotations

import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
import longctx as LC  # noqa: E402

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

C, cnn = LC.C, LC.cnn
H = LC.H


def summarise(x):
    """make_x output (N, 12, H, S) -> (N, K) window-level features."""
    x = x.astype(np.float32)
    vr, qc, ok, qd = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    n = ok.sum(2).clip(min=1)
    # observed-cell averages only: the scored months carry Task 1 masks, so the observed SHARE (or anything that
    # counts cells) separates the runs trivially (AUC 1.000 in the first version) and says nothing about traffic
    f = [(vr * ok).sum(2) / n, (qc * ok).sum(2) / n, (qd * ok).sum(2) / n]                # (N, H) each
    out = [np.concatenate(f, 1)]
    out.append(np.stack([f[0].max(1) - f[0].min(1), f[1].max(1) - f[1].min(1), f[2].max(1),
                         x[:, 9, -1, 0], x[:, 10, -1, 0], x[:, 11, -1, 0]], 1))
    return np.concatenate(out, 1)


def main() -> None:
    parts = LC.train_panels()
    par = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 50,
           "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "num_threads": 8}
    for i, (p, panel) in enumerate(zip(parts, C.T2_PANELS)):
        assert p["panel"] == panel
        Xtr = np.concatenate([summarise(LC.short_x(p, p["d"][a:a + 512], p["t0"][a:a + 512]))
                              for a in range(0, len(p["d"]), 512)])
        ctx, stat, W = C.windows(i, panel)
        Xsc = []
        for s in C.SPLITS:
            z = W[s]["z"]
            for w in (W[s]["og"], W[s]["on"]):
                if len(w["d"]):
                    Xsc.append(summarise(C.make_x(ctx, stat, w, np.arange(len(w["d"])), z)))
        Xsc = np.concatenate(Xsc)
        X = np.concatenate([Xtr, Xsc]); y = np.r_[np.zeros(len(Xtr)), np.ones(len(Xsc))]
        oof = np.zeros(len(y))
        fold = np.random.default_rng(0).integers(5, size=len(y))
        for k in range(5):
            m = lgb.train(par, lgb.Dataset(X[fold != k], y[fold != k]), 300)
            oof[fold == k] = m.predict(X[fold == k])
        r = np.argsort(np.argsort(oof)) + 1                     # rank AUC (no sklearn on the Ada venv)
        auc = (r[y == 1].sum() - (y == 1).sum() * ((y == 1).sum() + 1) / 2) / ((y == 1).sum() * (y == 0).sum())
        pt = np.clip(oof[:len(Xtr)], 1e-3, 1 - 1e-3)
        w = np.clip(pt / (1 - pt) * (len(Xtr) / len(Xsc)), 0.2, 5.0)
        w = w / w.mean()
        np.savez(LC.CACHE / f"advw_{panel}.npz", d=p["d"], t0=p["t0"], w=w.astype(np.float32), auc=auc)
        print(f"  {panel}: train {len(Xtr):,} vs scored {len(Xsc):,}  AUC {auc:.3f}  weight p10/p50/p90 "
              f"{np.percentile(w, 10):.2f}/{np.percentile(w, 50):.2f}/{np.percentile(w, 90):.2f}", flush=True)


if __name__ == "__main__":
    main()
