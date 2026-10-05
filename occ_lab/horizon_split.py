"""Step 3 of the user's plan after note 86: separate correctors by horizon.

AI's corrector is one LightGBM with `horizon` as a feature. Ongoing errors concentrate at T+25 / T+30 (note 52),
so the far horizons may want a different correction than the near ones. Everything else is AI's recipe: the 15
features, the tail's masked-layer windows, L2 on r = y - p0, 400 trees, p1 = clip(p0 + eta * r_hat).

    one      AI: one corrector over all six horizons
    near/far two correctors, T+5..T+20 and T+25..T+30
    six      one corrector per horizon (for reference - the user's first choice is two)

Tests as in notes 87 / 88: January -> February inside the tail (obs and full coverage) and the whole tail ->
March / April pseudo windows. Reported at AI's setting (eta 0.75, gate 0.10) and at each variant's best, with the
gain split into near and far horizons.

    python occ_lab/horizon_split.py
"""
from __future__ import annotations

import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta"))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

import common as C  # noqa: E402
from logit_update import IDX  # noqa: E402
from residual_cov import CACHE, PARAMS, load_env  # noqa: E402
from residual_diag import NAMES, TAU  # noqa: E402
from residual_oof import panel_data as scored_panel  # noqa: E402

SPLITS_H = {"one": [list(range(6))], "near/far": [[0, 1, 2, 3], [4, 5]], "six": [[h] for h in range(6)]}
SETTINGS = [(e, g) for e in (0.50, 0.75, 1.00) for g in (0.0, 0.05, 0.10)]
NEAR, FAR = [0, 1, 2, 3], [4, 5]


def fit_groups(train, groups):
    """train: list of (f15 (N,F,S,15), y, e, base). One model per horizon group."""
    models = []
    for hs in groups:
        Xs, ys, ps = [], [], []
        for f, y, e, base in train:
            m = e.copy()
            m[:, [h for h in range(6) if h not in hs]] = False
            Xs.append(f[m]); ys.append(y[m].astype(np.float32)); ps.append(base[m])
        X, y, p0 = np.concatenate(Xs), np.concatenate(ys), np.concatenate(ps)
        models.append((hs, lgb.train(PARAMS, lgb.Dataset(X, label=y - p0, feature_name=NAMES, free_raw_data=False),
                                     num_boost_round=400)))
    return models


def evaluate(models, panels):
    fam_b, fam, part = {}, {}, {}
    for panel, f, y, e, base in panels:
        rh = np.zeros(base.shape, np.float32)
        for hs, gbm in models:
            m = e.copy()
            m[:, [h for h in range(6) if h not in hs]] = False
            rh[m] = gbm.predict(f[m]).astype(np.float32)
        fam_b[panel] = C.iou(base >= TAU, y, e).mean()
        for eta, g in SETTINGS:
            p = np.clip(base + eta * np.where(np.abs(rh) >= g, rh, 0.0), 0, 1)
            fam.setdefault((eta, g), {})[panel] = C.iou(p >= TAU, y, e).mean()
            for name, hs in (("near", NEAR), ("far", FAR)):
                q = base >= TAU                              # corrected only on this horizon group
                q[:, hs] = p[:, hs] >= TAU
                part.setdefault((eta, g, name), {})[panel] = C.iou(q, y, e).mean()
    b = C.family_mean(fam_b)
    return {k: (C.family_mean(d) - b, C.family_mean(part[k + ("near",)]) - b, C.family_mean(part[k + ("far",)]) - b)
            for k, d in fam.items()}


def show(title, res):
    print(f"  {title}")
    for name, r in res.items():
        ai = r[(0.75, 0.10)]
        (be, bg), best = max(r.items(), key=lambda kv: kv[1][0])
        print(f"    {name:<9} AI setting {ai[0]:+.4f} (near {ai[1]:+.4f}, far {ai[2]:+.4f})   "
              f"best {best[0]:+.4f} at {be:.2f}/{bg:.2f} (near {best[1]:+.4f}, far {best[2]:+.4f})", flush=True)


def main() -> None:
    lab = {p: np.load(CACHE / f"cov_labels_{p}.npz") for p in C.T2_PANELS}
    env = {(v, p): load_env(v, p) for v in ("obs", "full") for p in C.T2_PANELS}
    jan = {p: np.char.startswith(lab[p]["date"].astype(str), "2031-01") for p in C.T2_PANELS}

    def tail(v, sel):
        out = []
        for p in C.T2_PANELS:
            f, base, _ = env[(v, p)]
            w = sel(p)
            out.append((p, f[w][..., IDX], lab[p]["y"][w], lab[p]["e"][w], base[w]))
        return out

    print("=== 1. fit on 2031-01 (masked-layer windows), score 2031-02 ===", flush=True)
    train = [t[1:] for t in tail("obs", lambda p: jan[p])]
    res = {v: {} for v in ("obs", "full")}
    for name, groups in SPLITS_H.items():
        models = fit_groups(train, groups)
        for v in ("obs", "full"):
            res[v][name] = evaluate(models, tail(v, lambda p: ~jan[p]))
    for v in ("obs", "full"):
        show(f"February, {v} coverage", res[v])

    print("\n=== 2. fit on the whole tail, score March / April pseudo windows ===", flush=True)
    train = [t[1:] for t in tail("obs", lambda p: np.ones(len(jan[p]), bool))]
    scored = {s: [(p,) + scored_panel(p, s) for p in C.T2_PANELS if scored_panel(p, s) is not None]
              for s in C.SPLITS}
    res = {s: {} for s in C.SPLITS}
    for name, groups in SPLITS_H.items():
        models = fit_groups(train, groups)
        for s in C.SPLITS:
            res[s][name] = evaluate(models, scored[s])
    for s in C.SPLITS:
        show(s, res[s])


if __name__ == "__main__":
    main()
