"""Task 2 onset: re-test onset-model choices on the official-like windows (first morning onset).

Notes 22-26 showed the generic pseudo windows mis-rank onset methods, so choices rejected on them are re-tested here.
All variants share the same prior-dropout draws (one draw per bag, reused across variants) and 5 bags:
    hist     el   : only eligible history cells (current)
             fin  : every valued history cell. The official window history carries about 7% ineligible valued
                    cells (pct_observed 20-67), and they are as clean as eligible ones (note 01)
    cols     base : production features;  +struct, +mg (capacity margins), +struct+mg (t2_struct_noid)
    windows  all  : every earliest-onset training window;  morning : queue at 04-14 h only (like the official)
    alpha    k = round(alpha * sum p) with alpha 1.0 / 1.2 / 1.5
The score is onset IoU on first-morning-onset events (t2_first_onset_eval) with days 1-7 and 1-14, validation and
private, family mean. The reference is el/base/all/alpha 1.2 (= H's onset).

    py analysis/t2_onset_variants.py
"""
from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "scripts"))
from t2_demand_stress import EXTRA_ON, load_tq  # noqa: E402
from t2_eda02 import T2_PANELS  # noqa: E402
from t2_eda03 import F, H  # noqa: E402
from t2_first_onset_eval import HI, LO, first_onsets  # noqa: E402
from t2_m1_models import Ctx, onset_decide  # noqa: E402
from t2_m1_predict import table_prior  # noqa: E402
from t2_prior_robust import dropout_prior  # noqa: E402
from t2_pseudo_eval import iou  # noqa: E402
from t2_shift_remedies import PARAMS  # noqa: E402
from t2_struct_noid import effective_capacity, full_onset, struct_vecs  # noqa: E402
from tfb.data import load  # noqa: E402
from tfb.t1eval import family_mean  # noqa: E402

N_BAGS = 5
ALPHAS = (1.0, 1.2, 1.5)
DAY_SETS = {"d1-7": 7, "d1-14": 14}


def hist_arrays(v_all, q_all, ok_el, d, t0):
    hi = t0[:, None] + np.arange(-H, 0)
    v, qf = v_all[d[:, None], hi], q_all[d[:, None], hi]
    return v, qf, ok_el[d[:, None], hi], np.isfinite(v) & np.isfinite(qf)


def col_sets(names):
    st = [n for n in names if n.startswith("st_") and n not in EXTRA_ON]
    mg = [n for n in names if n.startswith("mg_")]
    base = [n for n in names if not n.startswith(("st_", "mg_"))]
    idx = {n: i for i, n in enumerate(names)}
    sets = {"base": base, "+struct": base + st, "+mg": base + mg, "+struct+mg": base + st + mg}
    return {k: np.array([idx[n] for n in v]) for k, v in sets.items()}


def fmean(d):
    d = {k: v for k, v in d.items() if np.isfinite(v)}
    return family_mean(d) if d else float("nan")


def main() -> None:
    t_start = time.time()
    ctxs, extra = [], {}
    X = {(h, b): [] for h in ("el", "fin") for b in range(N_BAGS)}
    ys, morning, names = [], [], None
    for i, p in enumerate(T2_PANELS):
        ctx = Ctx(p, i)
        ctx.fitd = np.ones_like(ctx.fitd)
        ceff = effective_capacity(ctx)
        extra[p] = (struct_vecs(ctx, ceff), ceff, load_tq(p))
        d, t0 = ctx.eo
        v, qf, ok_el, ok_fin = hist_arrays(ctx.z["true_speed"], ctx.z["true_flow"], ctx.el, d, t0)
        wd = ctx.z["weekday"][d]
        for b in range(N_BAGS):
            pri = dropout_prior(ctx)  # one draw per bag, shared by el and fin (same key object -> same mask)
            for h, ok in (("el", ok_el), ("fin", ok_fin)):
                names, Xb = full_onset(ctx, *extra[p], v, qf, ok, t0, ctx.g[d], pri, d, wd)
                X[(h, b)].append(Xb.reshape(-1, Xb.shape[-1]))
        ys.append(ctx.q[d, t0 + F].reshape(-1))
        morning.append(np.repeat((t0 + F >= LO) & (t0 + F < HI), ctx.S))
        ctx.z = {k: ctx.z[k] for k in ("link_ids", "weekday")}
        ctx.q = ctx.el = None
        ctxs.append(ctx)
        print(f"[{p}] onset rows ({time.time() - t_start:.0f}s)", flush=True)
    y, morning = np.concatenate(ys), np.concatenate(morning)
    cols = col_sets(names)
    configs = list(itertools.product(("el", "fin"), cols, ("all", "morning")))
    models = {}
    for h, c, win in configs:
        sel = morning if win == "morning" else np.ones_like(morning)
        ms = []
        for b in range(N_BAGS):
            Xb = np.concatenate(X[(h, b)])[sel][:, cols[c]]
            ms.append(lgb.train(dict(PARAMS, num_leaves=63, seed=b),
                                lgb.Dataset(Xb, y[sel], feature_name=[names[j] for j in cols[c]]), 600))
        models[(h, c, win)] = ms
        print(f"trained {h}/{c}/{win} ({time.time() - t_start:.0f}s)", flush=True)
    del X
    res: dict = {}
    for ctx in ctxs:
        for split in ("validation", "private"):
            z = load(ctx.panel, split)
            ev = first_onsets(ctx.vf, z)
            if not len(ev["d"]):
                continue
            ok_el = z["elig"] & np.isfinite(z["obs_speed"]) & np.isfinite(z["obs_flow"])
            v, qf, oke, okf = hist_arrays(z["obs_speed"], z["obs_flow"], ok_el, ev["d"], ev["t0"])
            grp = (z["weekday"][ev["d"]] >= 5).astype(int)
            key = np.stack([ev["d"], grp], 1)
            wd = z["weekday"][ev["d"]].astype(int)
            feats = {h: full_onset(ctx, *extra[ctx.panel], v, qf, ok, ev["t0"], grp, table_prior(ctx), key, wd)[1]
                     for h, ok in (("el", oke), ("fin", okf))}
            for (h, c, win), ms in models.items():
                Xs = feats[h][..., cols[c]]
                p = np.mean([m.predict(Xs.reshape(-1, Xs.shape[-1])) for m in ms], 0).reshape(Xs.shape[:2])
                for a in ALPHAS:
                    pr = np.zeros((len(ev["d"]), F, ctx.S), bool)
                    pr[:, F - 1] = onset_decide(p, ("sum", a))
                    s = iou(pr, ev["fut"], ev["efut"])
                    for ds, dmax in DAY_SETS.items():
                        m = ev["d"] < dmax
                        res.setdefault((split, ds), {}).setdefault((h, c, win, a), {})[ctx.panel] = s[m].mean() if m.any() else np.nan
        print(f"  scored {ctx.panel} ({time.time() - t_start:.0f}s)", flush=True)
    ref = ("el", "base", "all", 1.2)
    for (split, ds), r in res.items():
        print(f"\n== {split} {ds}: onset IoU on first-morning-onset windows, family mean (ref el/base/all/a1.2 = {fmean(r[ref]):.3f}) ==")
        rows = sorted(r.items(), key=lambda kv: -fmean(kv[1]))
        for cfg, d in rows[:12]:
            diffs = [d[p] - r[ref][p] for p in d if np.isfinite(d[p]) and np.isfinite(r[ref][p])]
            print(f"  {'/'.join(map(str, cfg)):<28} {fmean(d):.3f}  (vs ref {fmean(d) - fmean(r[ref]):+.3f}; "
                  f"panels better/worse {sum(x > 1e-9 for x in diffs)}/{sum(x < -1e-9 for x in diffs)})")
    print(f"total {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
