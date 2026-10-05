"""Task 2 onset: pre-breakdown instability features (the user's second proposal; Tak & Yeo 2026, "section-based
transitional instability": short-term speed fluctuation x upstream/downstream imbalance before a breakdown).

None of the current onset features is a temporal dispersion, a second difference or an explicit spatial speed gradient
(t2_m1_models.link_features holds last/min/mean/trend and plain neighbour shifts). Added per link, from the history:
    i_vsd3/6/12        std of v/vf over the last 15 / 30 / 60 min
    i_d1, i_d2         last first difference and second difference of v/vf
    i_mad6, i_drop6    mean |first difference| and largest single drop over the last 30 min
    i_sp_a1, i_sp_b1   v/vf (last 15 min) minus each neighbour's
    i_grad1, i_grad2   neighbour-to-neighbour gradient at +-1 and +-2 links, i_grad1_tr its change over the hour
    i_nbsd_a, i_nbsd_b the neighbours' 30-min std
    i_dis, i_dis_rank  temporal fluctuation x spatial imbalance (i_vsd6 * |i_grad1|) and its percentile in the window
    i_qsd6, i_qgrad1   the same dispersion / gradient for q/cap
Train truth equals the observed values in the train split, so train and scoring features are alike.
1) Diagnostic: per-window AUC of each feature for "this link is in the T+30 set", over all eligible links and over
   the candidate links only (within +-1 of train's 10 most frequent run ends), next to vr_min3 and M's probability.
2) Paired LightGBM (H's onset configuration), 3 seeds x 5 bags: base / +all 18 features / +SUBSET (the features whose
   TRAIN AUC is furthest from 0.5 in run 1; chosen on train only). Scored alone and in M's blend at w 0.7 and 0.85.
Run 1 (all 18 only, w 0.7) is in note 44.

    .venv/Scripts/python.exe analysis/t2_onset_instability.py
"""
from __future__ import annotations

import sys
import time
from collections import Counter
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
from t2_onset_geometry import runs  # noqa: E402
from t2_pseudo_eval import iou  # noqa: E402
from t2_shift_remedies import PARAMS  # noqa: E402
from tfb.data import CACHE, load  # noqa: E402
from tfb.t1eval import family_mean  # noqa: E402

SEEDS, N_BAGS, MAX_WIN = (0, 1, 2), 5, 2000
I_NAMES = ["i_vsd3", "i_vsd6", "i_vsd12", "i_d1", "i_d2", "i_mad6", "i_drop6", "i_sp_a1", "i_sp_b1", "i_grad1", "i_grad2",
           "i_grad1_tr", "i_nbsd_a", "i_nbsd_b", "i_dis", "i_dis_rank", "i_qsd6", "i_qgrad1"]
SUBSET = ["i_qgrad1", "i_qsd6", "i_vsd12", "i_vsd6", "i_nbsd_a", "i_nbsd_b", "i_dis"]  # train AUC 0.51-0.59 (0.41 qsd6)
VARS = {"base": [], "inst": list(range(len(I_NAMES))), "sub": [I_NAMES.index(n) for n in SUBSET]}
W_BLEND = (0.7, 0.85)
TAGS = ["t2_cnn_lov", "t2_cnn_lov_s1", "t2_cnn_lov_s2"]
DAY_SETS = {"d1-14": 14, "all": 31}
rng = np.random.default_rng(29)


def fmean(d):
    d = {k: v for k, v in d.items() if np.isfinite(v)}
    return family_mean(d) if d else float("nan")


def nb(x, k):
    out = np.full_like(x, np.nan)
    if k > 0:
        out[:, :-k] = x[:, k:]
    else:
        out[:, -k:] = x[:, :k]
    return out


def wrank(x):
    S = x.shape[1]
    r = np.argsort(np.argsort(np.where(np.isfinite(x), x, -np.inf), 1), 1) / max(S - 1, 1)
    return np.where(np.isfinite(x), r, 0.5).astype(np.float32)


def inst_feats(ctx, v, qf, ok) -> np.ndarray:
    vr = np.where(ok, v / ctx.vf, np.nan)
    qc = np.where(ok, qf / ctx.cap, np.nan)
    with np.errstate(all="ignore"):
        sd3, sd6, sd12 = (np.nanstd(vr[:, -k:], 1) for k in (3, 6, 12))
        dv = np.diff(vr, axis=1)
        d1, d2 = dv[:, -1], dv[:, -1] - dv[:, -2]
        mad6, drop6 = np.nanmean(np.abs(dv[:, -5:]), 1), np.nanmax(-dv[:, -5:], 1)
        v3, v0 = np.nanmean(vr[:, -3:], 1), np.nanmean(vr[:, :3], 1)
        g1, g2 = nb(v3, 1) - nb(v3, -1), nb(v3, 2) - nb(v3, -2)
        g1_tr = g1 - (nb(v0, 1) - nb(v0, -1))
        dis = sd6 * np.abs(g1)
        q3 = np.nanmean(qc[:, -3:], 1)
        cols = [sd3, sd6, sd12, d1, d2, mad6, drop6, v3 - nb(v3, 1), v3 - nb(v3, -1), g1, g2, g1_tr, nb(sd6, 1),
                nb(sd6, -1), dis, wrank(dis), np.nanstd(qc[:, -6:], 1), nb(q3, 1) - nb(q3, -1)]
    return np.stack([np.asarray(c, np.float32) for c in cols], -1)                 # (N, S, len(I_NAMES))


def win_auc(f, y, valid) -> float:
    out = []
    for fi, yi, vi in zip(f, y, valid):
        m = vi & np.isfinite(fi)
        pos, neg = fi[m & yi], fi[m & ~yi]
        if len(pos) and len(neg):
            out.append(np.mean(pos[:, None] > neg[None, :]) + 0.5 * np.mean(pos[:, None] == neg[None, :]))
    return float(np.mean(out)) if out else np.nan


def candidates_of(train_cells: np.ndarray, S: int) -> np.ndarray:
    ends = Counter()
    for c in train_cells:
        for s, e in runs(c):
            ends[s] += 1; ends[e] += 1
    cand = np.zeros(S, bool)
    for e, _ in ends.most_common(10):
        cand[max(0, e - 1):e + 2] = True
    return cand


def main() -> None:
    t_start = time.time()
    diag: dict = {}
    Xb = {(s, b): [] for s in SEEDS for b in range(N_BAGS)}
    Is, ys, ctxs, names_on = [], [], [], None
    for i, p in enumerate(T2_PANELS):
        ctx = Ctx(p, i)
        ctx.fitd = np.ones_like(ctx.fitd)
        z = ctx.z
        # ---- diagnostic on train first-morning events
        tr = first_onsets(ctx.vf, z, truth=True)
        ctx.cand = candidates_of(tr["cells"], ctx.S)
        valid = ctx.el[tr["d"], tr["t0"] + F]
        It = inst_feats(ctx, tr["v"], tr["qf"], tr["ok"])
        vr_min3 = np.nanmin(np.where(tr["ok"], tr["v"] / ctx.vf, np.nan)[:, -3:], 1)
        for name, f in [("vr_min3", vr_min3)] + [(n, It[..., k]) for k, n in enumerate(I_NAMES)]:
            diag.setdefault(("train", "all", name), []).append(win_auc(f, tr["cells"], valid))
            diag.setdefault(("train", "cand", name), []).append(win_auc(f, tr["cells"], valid & ctx.cand[None]))
        # ---- paired training rows (earliest-onset windows, as in production)
        d6, t6 = ctx.eo
        if len(d6) > MAX_WIN:
            s = np.sort(rng.choice(len(d6), MAX_WIN, replace=False))
            d6, t6 = d6[s], t6[s]
        h6 = hist_from_truth(ctx, d6, t6)
        Is.append(inst_feats(ctx, *h6).reshape(-1, len(I_NAMES)))
        ys.append(ctx.q[d6, t6 + F].reshape(-1))
        for sd in SEEDS:
            t2_prior_robust.rng = np.random.default_rng(1000 * (sd + 1) + i)
            for b in range(N_BAGS):
                pri = t2_prior_robust.dropout_prior(ctx)
                names_on, X = feats_on(ctx, False, *h6, t6, ctx.g[d6], pri, d6, z["weekday"][d6])
                Xb[(sd, b)].append(X.reshape(-1, X.shape[-1]))
        ctx.z = {k: z[k] for k in ("link_ids", "weekday", "dates")}
        ctx.q = ctx.el = None
        ctxs.append(ctx)
        print(f"[{p}] train events {len(tr['d'])}, training windows {len(d6)} ({time.time() - t_start:.0f}s)", flush=True)
    I_all, y_all = np.concatenate(Is), np.concatenate(ys)
    models = {}
    for sd in SEEDS:
        for var, idx in VARS.items():
            ms = []
            for b in range(N_BAGS):
                X = np.concatenate(Xb[(sd, b)])
                if idx:
                    X = np.concatenate([X, I_all[:, idx]], 1)
                nm = names_on + [I_NAMES[k] for k in idx]
                ms.append(lgb.train(dict(PARAMS, num_leaves=63, seed=b + 100 * sd), lgb.Dataset(X, y_all, feature_name=nm), 600))
            models[(sd, var)] = ms
            if idx:
                imp = ms[0].feature_importance("gain")
                print(f"trained seed {sd} {var} ({time.time() - t_start:.0f}s); new share of gain {imp[-len(idx):].sum() / imp.sum():.3f}", flush=True)
    del Xb
    res: dict = {}
    for ctx in ctxs:
        for split in ("validation", "private"):
            c = np.load(CACHE / f"t2probs_{split}_{ctx.panel}.npz")
            if not len(c["on_d"]):
                continue
            z = load(ctx.panel, split)
            blank = official_blank(ctx.panel, split, z)
            on = first_onsets(ctx.vf, z, blank=blank)
            assert (on["d"] == c["on_d"]).all(), "window sets differ from the cache"
            cn = np.mean([np.load(CACHE / f"t2cnn_{t}_{split}_{ctx.panel}.npz")["on_p"] for t in TAGS], 0)
            truth, valid = c["on_fut"][:, F - 1], c["on_efut"][:, F - 1]
            I = inst_feats(ctx, on["v"], on["qf"], on["ok"])
            vr_min3 = np.nanmin(np.where(on["ok"], on["v"] / ctx.vf, np.nan)[:, -3:], 1)
            m_prob = 0.7 * c["on_p"] + 0.3 * cn
            for name, f in [("M prob", m_prob), ("vr_min3", vr_min3)] + [(n, I[..., k]) for k, n in enumerate(I_NAMES)]:
                diag.setdefault((split, "all", name), []).append(win_auc(f, truth, valid))
                diag.setdefault((split, "cand", name), []).append(win_auc(f, truth, valid & ctx.cand[None]))
            grp = (z["weekday"][on["d"]] >= 5).astype(int)
            key = np.stack([on["d"], grp], 1)
            _, X = feats_on(ctx, False, on["v"], on["qf"], on["ok"], on["t0"], grp, table_prior(ctx), key,
                            z["weekday"][on["d"]].astype(int))
            for (sd, var), ms in models.items():
                idx = VARS[var]
                Xo = np.concatenate([X, I[..., idx]], -1) if idx else X
                p = np.mean([m.predict(Xo.reshape(-1, Xo.shape[-1])) for m in ms], 0).reshape(Xo.shape[:2])
                for mode, pp in [("alone", p)] + [(f"blend{w}", w * p + (1 - w) * cn) for w in W_BLEND]:
                    pr = np.zeros(on["fut"].shape, bool)
                    pr[:, F - 1] = np.stack([select(q, "ratio", 1.0) for q in pp])
                    s = iou(pr, on["fut"], on["efut"])
                    for ds, dmax in DAY_SETS.items():
                        m = on["d"] < dmax
                        res.setdefault((mode, split, ds, sd), {}).setdefault(var, {})[ctx.panel] = s[m].mean() if m.any() else np.nan
        print(f"  scored {ctx.panel} ({time.time() - t_start:.0f}s)", flush=True)
    print("\n== per-window AUC for 'link in the T+30 set' (0.5 = no signal; <0.5 = lower value -> onset), mean over panels ==")
    names = ["M prob", "vr_min3"] + I_NAMES
    for split in ("train", "validation", "private"):
        for scope in ("all", "cand"):
            vals = [f"{n} {np.nanmean(diag[(split, scope, n)]):.3f}" for n in names if (split, scope, n) in diag]
            print(f"{split:<10} {scope:<4}: " + "  ".join(vals))
    print("\n== paired onset IoU vs base (family mean; per seed, then mean over 3 seeds) ==")
    ref = np.mean([fmean(res[("blend0.7", sp, ds, sd)]["base"]) for sp in ("validation", "private") for ds in DAY_SETS for sd in SEEDS])
    print(f"   (reference: base in blend 0.7, mean of the 4 conditions = {ref:.3f})")
    for var in [v for v in VARS if v != "base"]:
        for mode in ["alone"] + [f"blend{w}" for w in W_BLEND]:
            cells = []
            for split in ("validation", "private"):
                for ds in DAY_SETS:
                    deltas = []
                    for sd in SEEDS:
                        r = res[(mode, split, ds, sd)]
                        # blends are compared with base in the SAME blend weight 0.7 (M's rule), alone with base alone
                        b0 = res[("blend0.7" if mode.startswith("blend") else mode, split, ds, sd)]["base"]
                        deltas.append(fmean(r[var]) - fmean(b0))
                    cells.append(f"{split[:4]} {ds}: {np.mean(deltas):+.4f} ({sum(x > 0 for x in deltas)}/3)")
            print(f"  {var:<4} {mode:<9} | " + " | ".join(cells))
        print()
    for mode in [f"blend{w}" for w in W_BLEND]:
        cells = []
        for split in ("validation", "private"):
            for ds in DAY_SETS:
                deltas = [fmean(res[(mode, split, ds, sd)]["base"]) - fmean(res[("blend0.7", split, ds, sd)]["base"]) for sd in SEEDS]
                cells.append(f"{split[:4]} {ds}: {np.mean(deltas):+.4f}")
        print(f"  base {mode:<9} vs base blend0.7 | " + " | ".join(cells))
    print(f"total {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
