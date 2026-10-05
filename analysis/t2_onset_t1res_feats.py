"""Task 2 onset: the Task 1 residual / ensemble-spread block as FEATURES (ideas 2 and 3 of the user, tested paired).

Note 63 measured them as a standalone ranker: the signed residual is useless, but |residual| x spread reaches 0.48
top-1 over all windows (the best non-M ranker found here) and 0.35 / 0.72 top-1/top-3 on the windows that M misses.
That is candidate-narrowing information which Task 2 has never been given, so here it enters the H onset model as
features and is measured paired, exactly as notes 44 and 50 did.

Features per link, from the three Task 1 CNN reconstructions (their MEAN is the prediction, because only the scored
months have the test-time-trained version, while sub_m3v1, sub_m3v1_s1 and sub_m3w128 exist for train as well):
    raw          t1_rv3 t1_rv12 (signed speed residual over the last 15 / 60 min), t1_av3 t1_av12 (absolute),
                 t1_rq3 (signed flow residual), t1_sd6 (dispersion of the residual), t1_sig (ensemble spread),
                 t1_mix = |t1_rv3| * t1_sig
    scale-free   t1_av3_rel, t1_mix_rel (divided by the median over the links of that window) and the
                 within-window ranks t1_av3_rk, t1_mix_rk
The scale-free half matters because Task 1 was TRAINED on the train month, so its residuals there are smaller than
on the scored months. The run prints both levels so that the gap is visible.

Paired LightGBM in the H configuration (prior dropout, 5 bags), 3 seeds, identical dropout draws across variants.
Scored on the blank-aware official-like onset windows: alone and in the M blend (0.7 LGBM + 0.3 CNN x3, ratio rule).

    .venv/Scripts/python.exe analysis/t2_onset_t1res_feats.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "scripts"))
import t2_prior_robust  # noqa: E402
from t2_eda02 import T2_PANELS  # noqa: E402
from t2_eda03 import F  # noqa: E402
from t2_expected_iou import select  # noqa: E402
from t2_first_onset_eval import first_onsets, official_blank  # noqa: E402
from t2_m1_models import Ctx, hist_from_truth  # noqa: E402
from t2_m1_predict import feats_on, table_prior  # noqa: E402
from t2_pseudo_eval import iou  # noqa: E402
from t2_shift_remedies import PARAMS  # noqa: E402
from tfb.data import CACHE, load  # noqa: E402
from tfb.t1eval import family_mean  # noqa: E402

TAGS = ["t2_cnn_lov", "t2_cnn_lov_s1", "t2_cnn_lov_s2"]
MODELS = ["m3v1", "m3v1_s1", "m3w128"]
SEEDS, N_BAGS, MAX_WIN, H = (0, 1, 2), 5, 2000, 12
NAMES = ["t1_rv3", "t1_rv12", "t1_av3", "t1_av12", "t1_rq3", "t1_sd6", "t1_sig", "t1_mix",
         "t1_av3_rel", "t1_mix_rel", "t1_av3_rk", "t1_mix_rk"]
CONDS = [("validation", 14), ("validation", 31), ("private", 14), ("private", 31)]
# T1RES_COLS=sf keeps only the scale-free half (ratios to the window median and within-window ranks), which is the
# check for the train/scored residual level gap: Task 1 was trained on the train month, so its raw residuals there
# are about 1.4x smaller than on the scored months.
COLS = slice(8, 12) if os.environ.get("T1RES_COLS", "all") == "sf" else slice(0, len(NAMES))
SEL = NAMES[COLS]
rng = np.random.default_rng(29)


def fmean(d):
    d = {k: v for k, v in d.items() if np.isfinite(v)}
    return family_mean(d) if d else float("nan")


def wrank(x):
    S = x.shape[1]
    r = np.argsort(np.argsort(np.where(np.isfinite(x), x, -np.inf), 1), 1) / max(S - 1, 1)
    return np.where(np.isfinite(x), r, 0.5).astype(np.float32)


def t1_feats(panel, split, d, t0, obs_v, obs_q, ok):
    hi = t0[:, None] + np.arange(-H, 0)
    pv, pq = [], []
    for m in MODELS:
        f = np.load(ROOT / "work" / f"sub_{m}" / f"pred_{split}_{panel}.npz")
        pv.append(f["pv"][d[:, None], hi]); pq.append(f["pq"][d[:, None], hi])
    PV, PQ = np.stack(pv), np.stack(pq)
    sig = PV.std(0)
    rv, rq = obs_v - PV.mean(0), obs_q - PQ.mean(0)
    with np.errstate(all="ignore"):
        m3, m12, m6 = ok[:, -3:], ok, ok[:, -6:]
        rv3 = np.nanmean(np.where(m3, rv[:, -3:], np.nan), 1)
        rv12 = np.nanmean(np.where(m12, rv, np.nan), 1)
        av3 = np.nanmean(np.where(m3, np.abs(rv[:, -3:]), np.nan), 1)
        av12 = np.nanmean(np.where(m12, np.abs(rv), np.nan), 1)
        rq3 = np.nanmean(np.where(m3, rq[:, -3:], np.nan), 1)
        sd6 = np.nanstd(np.where(m6, rv[:, -6:], np.nan), 1)
        sg3 = np.nanmean(np.where(m3, sig[:, -3:], np.nan), 1)
        mix = np.abs(rv3) * sg3
        med_a = np.nanmedian(av3, 1, keepdims=True)
        med_m = np.nanmedian(mix, 1, keepdims=True)
        cols = [rv3, rv12, av3, av12, rq3, sd6, sg3, mix,
                av3 / np.maximum(med_a, 1e-3), mix / np.maximum(med_m, 1e-6), wrank(av3), wrank(mix)]
    return np.stack([np.nan_to_num(np.asarray(c, np.float32), nan=0.0) for c in cols], -1), float(np.nanmean(av3))


def main() -> None:
    t0_all = time.time()
    Xb = {(s, b): [] for s in SEEDS for b in range(N_BAGS)}
    Es, ys, ctxs, names_on = [], [], [], None
    lvl = {}
    for i, panel in enumerate(T2_PANELS):
        ctx = Ctx(panel, i)
        ctx.fitd = np.ones_like(ctx.fitd)
        z = ctx.z
        d, t0 = ctx.eo
        if len(d) > MAX_WIN:
            s = np.sort(rng.choice(len(d), MAX_WIN, replace=False))
            d, t0 = d[s], t0[s]
        v, qf, ok = hist_from_truth(ctx, d, t0)
        E, lv = t1_feats(panel, "train", d, t0, v, qf, ok)
        lvl[("train", panel)] = lv
        Es.append(E.reshape(-1, E.shape[-1]))
        ys.append(ctx.q[d, t0 + F].reshape(-1))
        for sd in SEEDS:
            t2_prior_robust.rng = np.random.default_rng(1000 * (sd + 1) + i)
            for b in range(N_BAGS):
                names_on, X = feats_on(ctx, False, v, qf, ok, t0, ctx.g[d], t2_prior_robust.dropout_prior(ctx), d,
                                       z["weekday"][d])
                Xb[(sd, b)].append(X.reshape(-1, X.shape[-1]))
        ctx.z = {k: z[k] for k in ("link_ids", "weekday", "dates")}
        ctx.q = ctx.el = None
        ctxs.append(ctx)
        print(f"[{panel}] windows {len(d)}, train abs residual {lv:.2f} km/h ({time.time() - t0_all:.0f}s)", flush=True)
    y_all, E_all = np.concatenate(ys), np.concatenate(Es)
    del Es, ys
    models = {}
    for sd in SEEDS:
        for var in ("base", "t1res"):
            ms = []
            for b in range(N_BAGS):
                X = np.concatenate(Xb[(sd, b)])
                if var == "t1res":
                    X = np.concatenate([X, E_all[:, COLS]], 1)
                nm = names_on + (SEL if var == "t1res" else [])
                ms.append(lgb.train(dict(PARAMS, num_leaves=63, seed=b + 100 * sd),
                                    lgb.Dataset(X, y_all, feature_name=nm), 600))
            models[(sd, var)] = ms
            if var == "t1res":
                imp = ms[0].feature_importance("gain")
                print(f"  seed {sd}: T1 block ({len(SEL)} cols) share of gain {imp[-len(SEL):].sum() / imp.sum():.3f}; top: "
                      + ", ".join(f"{nm[j]} {imp[j] / imp.sum():.3f}" for j in np.argsort(-imp)[:5]), flush=True)
    del Xb
    res: dict = {}
    for ctx in ctxs:
        for split in ("validation", "private"):
            z = load(ctx.panel, split)
            on = first_onsets(ctx.vf, z, blank=official_blank(ctx.panel, split, z))
            if not len(on["d"]):
                continue
            c = np.load(CACHE / f"t2probs_{split}_{ctx.panel}.npz")
            assert (c["on_d"] == on["d"]).all()
            cn = np.mean([np.load(CACHE / f"t2cnn_{t}_{split}_{ctx.panel}.npz")["on_p"] for t in TAGS], 0)
            grp = (z["weekday"][on["d"]] >= 5).astype(int)
            key = np.stack([on["d"], grp], 1)
            wd = z["weekday"][on["d"]].astype(int)
            _, X = feats_on(ctx, False, on["v"], on["qf"], on["ok"], on["t0"], grp, table_prior(ctx), key, wd)
            E, lv = t1_feats(ctx.panel, split, on["d"], on["t0"], on["v"], on["qf"], on["ok"])
            lvl[(split, ctx.panel)] = lv
            for (sd, var), ms in models.items():
                Xo = np.concatenate([X, E[..., COLS]], -1) if var == "t1res" else X
                p = np.mean([m.predict(Xo.reshape(-1, Xo.shape[-1])) for m in ms], 0).reshape(Xo.shape[:2])
                for mode, pp in (("alone", p), ("M-blend", 0.7 * p + 0.3 * cn)):
                    pr = np.zeros(on["fut"].shape, bool)
                    pr[:, F - 1] = np.stack([select(q, "ratio", 1.0) for q in pp])
                    s = iou(pr, on["fut"], on["efut"])
                    for sp, dmax in CONDS:
                        if sp == split:
                            m = on["d"] < dmax
                            res.setdefault((mode, var, sd), {}).setdefault((sp, dmax), {})[ctx.panel] = (
                                s[m].mean() if m.any() else np.nan)
        print(f"  scored {ctx.panel} ({time.time() - t0_all:.0f}s)", flush=True)
    tr = np.mean([v for (s, _), v in lvl.items() if s == "train"])
    sc = np.mean([v for (s, _), v in lvl.items() if s != "train"])
    print(f"\nmean abs speed residual: train {tr:.2f} km/h vs scored months {sc:.2f} km/h (ratio {sc / max(tr, 1e-6):.2f})")
    for mode in ("alone", "M-blend"):
        print(f"\n== onset, {mode}: val d1-14 | val all | priv d1-14 | priv all (mean of 3 seeds) ==")
        for var in ("base", "t1res"):
            cells = []
            for cnd in CONDS:
                vals = [fmean(res[(mode, var, sd)][cnd]) for sd in SEEDS]
                base = [fmean(res[(mode, "base", sd)][cnd]) for sd in SEEDS]
                dl = [a - b for a, b in zip(vals, base)]
                cells.append(f"{np.mean(vals):.3f}" + ("" if var == "base" else
                                                       f" ({np.mean(dl):+.4f}, {sum(x > 0 for x in dl)}/3)"))
            print(f"  {var:<6} " + " | ".join(cells))
    print(f"total {time.time() - t0_all:.0f}s")


if __name__ == "__main__":
    main()
