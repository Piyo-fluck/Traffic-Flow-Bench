"""Task 2 onset as a first-breakdown problem (the user's competing-risk proposal; note 44), paired LightGBM test.

Two untried parts of the proposal, each as a paired variant of H's onset configuration (prior dropout; expected-IoU
ratio decision, c 1.0):
  lead   hazard-style training: besides the usual windows whose first queue is at T+30 (lead 6 slots), add windows
         whose first queue is at T+20, T+25, T+35, T+40 (lead 4, 5, 7, 8; P_EXTRA of them). The label is the set of
         cells queued at the first-breakdown slot, and the lead is a feature. Scoring uses lead 6.
  rank   relative state inside the window (percentile over the panel's links): last-3-slot q/cap, its growth, the v/vf
         drop, the q/cap step to each neighbour, and the margin to an EMPIRICAL breakdown flow (the median last-3-slot
         q/cap before train breakdowns at that link), plus the margin itself.
Variants: base, rank, lead, rank+lead (identical prior-dropout draws per bag). Scored on the blank-aware official-like
onset windows, alone and in M's blend (0.7 LightGBM + 0.3 CNN x3 from the caches), with the share of windows missed
entirely (IoU = 0), which carries most of M's onset loss (note 44).

    .venv/Scripts/python.exe analysis/t2_onset_hazard.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "scripts"))
from t2_eda02 import T2_PANELS  # noqa: E402
from t2_eda03 import F, H  # noqa: E402
from t2_expected_iou import select  # noqa: E402
from t2_first_onset_eval import first_onsets, official_blank  # noqa: E402
from t2_m1_models import Ctx, hist_from_truth  # noqa: E402
from t2_m1_predict import feats_on, table_prior  # noqa: E402
from t2_prior_robust import dropout_prior  # noqa: E402
from t2_pseudo_eval import iou  # noqa: E402
from t2_shift_remedies import PARAMS  # noqa: E402
from tfb.data import CACHE, load  # noqa: E402
from tfb.t1eval import family_mean  # noqa: E402

import os  # noqa: E402

import t2_prior_robust  # noqa: E402

# Run 1 used HZ_BAGS=2 HZ_LEADS=4,5,7,8 HZ_PEXTRA=0.25 HZ_SEED=0 (the prior-dropout draw alone moves onset by ~0.08).
N_BAGS = int(os.environ.get("HZ_BAGS", 5))
LEADS_EXTRA = tuple(int(x) for x in os.environ.get("HZ_LEADS", "2,3,4,5,7,8,9,10").split(","))
P_EXTRA = float(os.environ.get("HZ_PEXTRA", 1.0))
SEED = int(os.environ.get("HZ_SEED", 0))
MAX_WIN = 2000
VARIANTS = ("base", "rank", "lead", "rank+lead")
R_NAMES = ["r_qc3", "r_qc_growth", "r_vr_drop", "r_qc_step_a", "r_qc_step_b", "r_margin", "margin_emp"]
TAGS = ["t2_cnn_lov", "t2_cnn_lov_s1", "t2_cnn_lov_s2"]
DAY_SETS = {"d1-14": 14, "all": 31}
rng = np.random.default_rng(23 + SEED)
t2_prior_robust.rng = np.random.default_rng(1 + SEED)  # the prior-dropout draws differ per repeat


def fmean(d):
    d = {k: v for k, v in d.items() if np.isfinite(v)}
    return family_mean(d) if d else float("nan")


def lead_windows(ctx, k: int):
    """Origins whose history holds no ongoing queue and whose FIRST queue is at T + 5k (k slots ahead)."""
    q = ctx.q
    ds, ts = [], []
    for t0 in range(H, q.shape[1] - k):
        hist = q[:, t0 - H:t0]
        ongoing = (hist.sum((1, 2)) >= 2) & (hist.sum(1).max(1) >= 2)
        ok = ~ongoing & ~q[:, t0 + 1:t0 + k].any((1, 2)) & q[:, t0 + k].any(1)
        dd = np.flatnonzero(ok)
        ds.append(dd); ts.append(np.full(len(dd), t0))
    return np.concatenate(ds), np.concatenate(ts)


def last_first(ctx, v, qf, ok):
    qc = np.where(ok, qf / ctx.cap, np.nan)
    vr = np.where(ok, v / ctx.vf, np.nan)
    with np.errstate(all="ignore"):
        return (np.nanmean(qc[:, -3:], 1), np.nanmean(qc[:, :3], 1), np.nanmean(vr[:, -3:], 1), np.nanmean(vr[:, :3], 1))


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


def empirical_ref(ctx, v, qf, ok, y) -> np.ndarray:
    """Per link: median last-3-slot q/cap before train breakdowns there (global median where too few)."""
    qc3 = last_first(ctx, v, qf, ok)[0]
    hit = y & np.isfinite(qc3)
    glob = float(np.median(qc3[hit])) if hit.any() else 1.0
    ref = np.array([np.median(qc3[hit[:, i], i]) if hit[:, i].sum() >= 3 else glob for i in range(ctx.S)])
    return np.clip(ref, 0.05, None).astype(np.float32)


def rank_feats(ctx, v, qf, ok) -> np.ndarray:
    qc3, qc0, vr3, vr0 = last_first(ctx, v, qf, ok)
    margin = qc3 / ctx.cemp
    cols = [wrank(qc3), wrank(qc3 - qc0), wrank(vr0 - vr3), wrank(qc3 - nb(qc3, 1)), wrank(qc3 - nb(qc3, -1)),
            wrank(margin), np.nan_to_num(margin, nan=-1.0)]
    return np.stack(cols, -1).astype(np.float32)                                  # (N, S, 7)


def main() -> None:
    t_start = time.time()
    store = {(b, k): [] for b in range(N_BAGS) for k in (F,) + LEADS_EXTRA}
    ranks = {k: [] for k in (F,) + LEADS_EXTRA}
    ys = {k: [] for k in (F,) + LEADS_EXTRA}
    names_on, ctxs = None, []
    for i, p in enumerate(T2_PANELS):
        ctx = Ctx(p, i)
        ctx.fitd = np.ones_like(ctx.fitd)
        z = ctx.z
        d6, t6 = ctx.eo
        if len(d6) > MAX_WIN:
            s = np.sort(rng.choice(len(d6), MAX_WIN, replace=False))
            d6, t6 = d6[s], t6[s]
        h6 = hist_from_truth(ctx, d6, t6)
        y6 = ctx.q[d6, t6 + F]
        ctx.cemp = empirical_ref(ctx, *h6, y6)
        groups = [(F, d6, t6, h6, y6)]
        for k in LEADS_EXTRA:
            dk, tk = lead_windows(ctx, k)
            sel = rng.random(len(dk)) < P_EXTRA * min(1.0, MAX_WIN / max(len(dk), 1))
            dk, tk = dk[sel], tk[sel]
            groups.append((k, dk, tk, hist_from_truth(ctx, dk, tk), ctx.q[dk, tk + k]))
        for k, d, t, h, y in groups:
            ranks[k].append(rank_feats(ctx, *h).reshape(-1, len(R_NAMES)))
            ys[k].append(y.reshape(-1))
        for b in range(N_BAGS):
            pri = dropout_prior(ctx)
            for k, d, t, h, y in groups:
                pk = (lambda name, key, tt, _k=k, _p=pri: _p(name, key, tt + (_k - F)))  # priors at the breakdown slot
                names_on, X = feats_on(ctx, False, *h, t, ctx.g[d], pk, d, z["weekday"][d])
                store[(b, k)].append(X.reshape(-1, X.shape[-1]))
        print(f"[{p}] windows " + ", ".join(f"lead{k} {len(d)}" for k, d, *_ in groups) + f" ({time.time() - t_start:.0f}s)", flush=True)
        ctx.z = {k: z[k] for k in ("link_ids", "weekday", "dates")}
        ctx.q = ctx.el = None
        ctxs.append(ctx)
    for key in store:
        store[key] = np.concatenate(store[key])
    for k in ranks:
        ranks[k], ys[k] = np.concatenate(ranks[k]), np.concatenate(ys[k])
    models = {}
    for var in VARIANTS:
        leads = (F,) + LEADS_EXTRA if "lead" in var else (F,)
        nm = names_on + (R_NAMES if "rank" in var else []) + (["lead"] if "lead" in var else [])
        models[var] = []
        for b in range(N_BAGS):
            parts = []
            for k in leads:
                cols = [store[(b, k)]]
                if "rank" in var:
                    cols.append(ranks[k])
                if "lead" in var:
                    cols.append(np.full((len(store[(b, k)]), 1), k, np.float32))
                parts.append(np.concatenate(cols, 1) if len(cols) > 1 else cols[0])
            A = np.concatenate(parts) if len(parts) > 1 else parts[0]
            yy = np.concatenate([ys[k] for k in leads])
            models[var].append(lgb.train(dict(PARAMS, num_leaves=63, seed=b + 100 * SEED), lgb.Dataset(A, yy, feature_name=nm), 600))
            del A
        imp = models[var][0].feature_importance("gain")
        n_new = (len(R_NAMES) if "rank" in var else 0) + (1 if "lead" in var else 0)
        share = f"; new features' share of gain {imp[-n_new:].sum() / imp.sum():.3f}" if n_new else ""
        print(f"trained {var} ({time.time() - t_start:.0f}s){share}", flush=True)
    del store
    res: dict = {}
    zero: dict = {}
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
            grp = (z["weekday"][on["d"]] >= 5).astype(int)
            key = np.stack([on["d"], grp], 1)
            _, X = feats_on(ctx, False, on["v"], on["qf"], on["ok"], on["t0"], grp, table_prior(ctx), key,
                            z["weekday"][on["d"]].astype(int))
            R = rank_feats(ctx, on["v"], on["qf"], on["ok"])
            for var in VARIANTS:
                cols = [X]
                if "rank" in var:
                    cols.append(R)
                if "lead" in var:
                    cols.append(np.full(X.shape[:2] + (1,), F, np.float32))
                Xo = np.concatenate(cols, -1)
                p = np.mean([m.predict(Xo.reshape(-1, Xo.shape[-1])) for m in models[var]], 0).reshape(Xo.shape[:2])
                for mode, pp in (("alone", p), ("blend", 0.7 * p + 0.3 * cn)):
                    pr = np.zeros(on["fut"].shape, bool)
                    pr[:, F - 1] = np.stack([select(q, "ratio", 1.0) for q in pp])
                    s = iou(pr, on["fut"], on["efut"])
                    zero.setdefault((mode, var, split), []).append(s == 0)
                    for ds, dmax in DAY_SETS.items():
                        m = on["d"] < dmax
                        res.setdefault((mode, split, ds), {}).setdefault(var, {})[ctx.panel] = s[m].mean() if m.any() else np.nan
        print(f"  scored {ctx.panel} ({time.time() - t_start:.0f}s)", flush=True)
    for (mode, split, ds), r in res.items():
        base = r["base"]
        line = []
        for var in VARIANTS:
            d = r[var]
            diffs = [d[p] - base[p] for p in d if np.isfinite(d[p]) and np.isfinite(base.get(p, np.nan))]
            line.append(f"{var} {fmean(d):.3f} ({fmean(d) - fmean(base):+.3f}; {sum(x > 1e-9 for x in diffs)}/{sum(x < -1e-9 for x in diffs)})")
        print(f"== onset {mode} {split} {ds}: " + "  ".join(line))
    for (mode, var, split), z0 in sorted(zero.items()):
        print(f"   IoU=0 share {mode} {var:<10} {split}: {np.concatenate(z0).mean():.3f}")
    print(f"total {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
