"""Task 1 method M2: LightGBM on neighbourhood features, one pooled model per channel.

Targets are anomalies relative to the fit-day weekday x tod profile: speed in
km/h, flow per lane. Features come only from the masked layer (what the
validation and private months also provide) plus static station attributes.

Days: Jun–Nov 2030 train the model, Dec 2030 early-stops it, Jan–Feb 2031 test it.

    py analysis/t1_m2_lgbm.py D12_I5_N D7_I210_W
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tfb.data import PANELS, T, load  # noqa: E402
from tfb.features import interp_along, neighbors_along, profile, shift, truth_valid  # noqa: E402
from tfb.metrics import fmt, state_score  # noqa: E402
from tfb.t1eval import family_mean, split_days  # noqa: E402

ES_START = "2030-12-01"
MAX_TRAIN_ROWS_PER_PANEL = int(os.environ.get("M2_ROWS", 1_500_000))
ROUNDS = int(os.environ.get("M2_ROUNDS", 3000))
LR = float(os.environ.get("M2_LR", 0.05))
SEED = 0


def features(z: dict, base_v: np.ndarray, base_q: np.ndarray) -> dict[str, np.ndarray]:
    lanes = z["lanes"]
    ov, oq, oo = z["obs_speed"], z["obs_flow"], z["obs_occ"]
    vis = np.isfinite(ov) & np.isfinite(oq)
    F: dict[str, np.ndarray] = {}
    for ch, an in (("v", ov - base_v), ("q", (oq - base_q) / lanes)):
        for ax, tag in ((1, "t"), (2, "s")):
            xp, xn, dp, dn = neighbors_along(an, vis, ax)
            F[f"{ch}_{tag}prev"], F[f"{ch}_{tag}next"] = xp, xn
            if ch == "v":
                F[f"d{tag}prev"], F[f"d{tag}next"] = dp, dn
            F[f"{ch}_{tag}interp"] = interp_along(an, vis, ax)[0]
        ti = F[f"{ch}_tinterp"]
        F[f"{ch}_tinterp_up"] = shift(ti, 1, 2)
        F[f"{ch}_tinterp_dn"] = shift(ti, -1, 2)
        F[f"{ch}_tinterp_up2"] = shift(ti, 2, 2)
        F[f"{ch}_tinterp_dn2"] = shift(ti, -2, 2)
    # Raw state context: congestion is a level, not only a deviation.
    F["v_raw_tprev"], F["v_raw_tnext"], _, _ = neighbors_along(ov, vis, 1)
    F["occ_tprev"], F["occ_tnext"], _, _ = neighbors_along(oo, vis, 1)
    F["base_v"] = base_v
    F["base_ql"] = base_q / lanes
    shape = ov.shape
    F["tod"] = np.broadcast_to(np.arange(T, dtype=np.float32)[None, :, None], shape)
    F["weekday"] = np.broadcast_to(z["weekday"].astype(np.float32)[:, None, None], shape)
    S = shape[2]
    for name, vec in (("lanes", lanes), ("free_speed", z["free_speed"]),
                      ("pos", np.arange(S) / max(S - 1, 1)), ("sensor", z["has_sensor"])):
        F[name] = np.broadcast_to(np.asarray(vec, np.float32)[None, None, :], shape)
    return F


def rows(F: dict, cells: np.ndarray, names: list[str]) -> np.ndarray:
    return np.stack([F[k][cells] for k in names], axis=1).astype(np.float32)


def prepare(panel: str, rng: np.random.Generator) -> dict:
    t0 = time.time()
    z = load(panel, "train")
    fit, test = split_days(z)
    ts, tq, lanes = z["true_speed"], z["true_flow"], z["lanes"]
    valid = truth_valid(z)
    base_v = profile(ts, valid, z["weekday"], fit)
    base_q = profile(tq, valid, z["weekday"], fit)
    F = features(z, base_v, base_q)
    names = list(F)
    tgt = z["target"] & np.isfinite(ts) & np.isfinite(tq)
    es = z["dates"] >= ES_START
    tr_cells = tgt & (fit & ~es)[:, None, None]
    idx = np.flatnonzero(tr_cells)
    if idx.size > MAX_TRAIN_ROWS_PER_PANEL:
        keep = np.zeros(tr_cells.size, bool)
        keep[rng.choice(idx, MAX_TRAIN_ROWS_PER_PANEL, replace=False)] = True
        tr_cells = keep.reshape(tr_cells.shape)
    es_cells = tgt & (fit & es)[:, None, None]
    te_cells = tgt & test[:, None, None]
    yv = ts - base_v
    yq = (tq - base_q) / lanes
    out = {"panel": panel, "names": names, "z": z, "base_v": base_v, "base_q": base_q,
           "te_cells": te_cells, "test_days": test}
    for tag, cells in (("tr", tr_cells), ("es", es_cells), ("te", te_cells)):
        out[f"X_{tag}"] = rows(F, cells, names)
        out[f"yv_{tag}"] = yv[cells].astype(np.float32)
        out[f"yq_{tag}"] = yq[cells].astype(np.float32)
    del F
    print(f"[{panel}] rows tr {len(out['yv_tr']):,} es {len(out['yv_es']):,} te {len(out['yv_te']):,} "
          f"features {len(names)} ({time.time() - t0:.0f}s)", flush=True)
    return out


def main() -> None:
    panels = sys.argv[1:] or PANELS
    rng = np.random.default_rng(SEED)
    parts = [prepare(p, rng) for p in panels]
    names = parts[0]["names"]
    params = dict(objective="regression", learning_rate=LR, num_leaves=255, min_data_in_leaf=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  verbose=-1, seed=SEED, num_threads=8)
    models = {}
    for ch in ("v", "q"):
        t0 = time.time()
        dtr = lgb.Dataset(np.concatenate([p["X_tr"] for p in parts]),
                          np.concatenate([p[f"y{ch}_tr"] for p in parts]), feature_name=names)
        des = lgb.Dataset(np.concatenate([p["X_es"] for p in parts]),
                          np.concatenate([p[f"y{ch}_es"] for p in parts]), reference=dtr)
        models[ch] = lgb.train(params, dtr, num_boost_round=ROUNDS, valid_sets=[des],
                               callbacks=[lgb.early_stopping(100, verbose=False)])
        imp = models[ch].feature_importance("gain")
        top = np.argsort(imp)[::-1][:10]
        print(f"model {ch}: best_iter {models[ch].best_iteration}, {time.time() - t0:.0f}s; top gain: "
              + ", ".join(f"{names[i]} {imp[i] / imp.sum():.2f}" for i in top), flush=True)

    scores = {}
    preds_dir = Path(__file__).resolve().parent.parent / "work" / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)
    for p in parts:
        z, cells = p["z"], p["te_cells"]
        pv, pq = p["base_v"].copy(), p["base_q"].copy()
        pv[cells] += models["v"].predict(p["X_te"], num_iteration=models["v"].best_iteration)
        pq[cells] += models["q"].predict(p["X_te"], num_iteration=models["q"].best_iteration) * np.broadcast_to(z["lanes"], cells.shape)[cells]
        days = np.flatnonzero(p["test_days"])
        np.savez(preds_dir / f"t1_m2_{p['panel']}_test.npz", days=days, pv=pv[days], pq=pq[days])
        r = state_score(z, pv, pq, cells=p["test_days"][:, None, None])
        scores[p["panel"]] = r["mean"]["S"]
        print(f"[{p['panel']}] M2 LightGBM  {fmt(r)}")
    if len(parts) > 1:
        print(f"family-weighted mean M2: {family_mean(scores):.4f}")


if __name__ == "__main__":
    main()
