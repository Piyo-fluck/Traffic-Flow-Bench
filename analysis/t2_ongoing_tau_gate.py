"""Task 2 ONGOING: how much of note 52's "+0.080 from an oracle per-window threshold" is actually learnable?

Note 52 measured the ceiling; this measures what a rule can reach. For every official-like ongoing window we store
the whole IoU-vs-tau curve (15 thresholds) plus cheap window features, so any tau-choosing rule can be evaluated by
indexing the curve. Rules are FITTED ON ONE SPLIT AND APPLIED TO THE OTHER (both directions), so the reported gain
is out-of-sample. This is a derivation/evaluation device: a production gate would be fitted on the train run, which
is only worth building if the numbers here are positive.

  global            one tau for everything (the note-52 result, as an out-of-sample reference)
  panel             one tau per panel
  qsize / pmass /   one tau per quintile of the queue size at T, of the mean probability, and per hour bucket
  hour
  lgbm              a small LightGBM regression, window features -> the window's best tau
  lgbm+iou          the same features, but the target is chosen to maximise IoU directly: for each window the
                    model scores every tau (pairwise-free listwise surrogate: regression on the IoU curve)
  ORACLE            the per-window best tau (note 52's ceiling)

Window features (history only, no horizon information): queue size and its trend at T, number of queue runs, mean /
max / mass of the predicted probability, probability entropy, the share of cells above several thresholds, history
coverage, hour, weekday flag, panel id, and the corridor length.

    .venv/Scripts/python.exe analysis/t2_ongoing_tau_gate.py
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
from t2_first_onset_eval import official_blank  # noqa: E402
from t2_m1_models import Ctx  # noqa: E402
from t2_official_like_eval import ongoing_official  # noqa: E402
from t2_pseudo_eval import iou  # noqa: E402
from tfb.data import CACHE, load  # noqa: E402

TAGS = ["t2_cnn_lov", "t2_cnn_lov_s1", "t2_cnn_lov_s2"]
TAU, W_L = 0.60, 0.5
TAUS = np.round(np.arange(0.20, 0.91, 0.05), 2)
FEATS = ["nq", "fq", "runs", "trend", "cover", "hour", "wknd", "panel", "S", "p_mean", "p_max", "p_mass",
         "p_ent", "s40", "s50", "s60", "s70", "ratio", "spread"]
PARAMS = dict(objective="l2", learning_rate=0.05, num_leaves=31, min_data_in_leaf=50, feature_fraction=0.9,
              bagging_fraction=0.8, bagging_freq=1, verbose=-1)
boot = np.random.default_rng(23)


def ci(delta: np.ndarray, n_boot: int = 2000) -> str:
    if not len(delta):
        return "n/a"
    idx = boot.integers(0, len(delta), (n_boot, len(delta)))
    m = delta[idx].mean(1)
    return f"{delta.mean():+.4f} [{np.percentile(m, 2.5):+.4f}, {np.percentile(m, 97.5):+.4f}]"


def last_state(vis_q: np.ndarray, ok: np.ndarray) -> np.ndarray:
    has = ok.any(1)
    idx = ok.shape[1] - 1 - np.argmax(ok[:, ::-1], axis=1)
    return np.take_along_axis(vis_q, idx[:, None], axis=1)[:, 0] & has


def pick(curve: np.ndarray, k: np.ndarray) -> np.ndarray:
    """IoU of each window when threshold index k[j] is used."""
    return curve[np.arange(len(curve)), k]


def by_group(curve_fit: np.ndarray, grp_fit: np.ndarray, grp_app: np.ndarray, n_grp: int) -> np.ndarray:
    """Best tau index per group on the fitting split, applied to the other split's groups."""
    best = np.full(n_grp, int(np.argmax(curve_fit.mean(0))), int)
    for g in range(n_grp):
        m = grp_fit == g
        if m.sum() >= 30:
            best[g] = int(np.argmax(curve_fit[m].mean(0)))
    return best[grp_app]


def quintile(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(edges, x), 0, len(edges)).astype(int)


def main() -> None:
    t0_all = time.time()
    D: dict = {}
    for i, panel in enumerate(T2_PANELS):
        ctx = Ctx(panel, i)
        ctx.fitd = np.ones_like(ctx.fitd)
        for split in ("validation", "private"):
            z = load(panel, split)
            og, _ = ongoing_official(ctx, z, official_blank(panel, split, z))
            n = len(og["d"])
            if not n:
                continue
            c = np.load(CACHE / f"t2probs_{split}_{panel}.npz")
            assert (c["og_d"] == og["d"]).all()
            pC = np.mean([np.load(CACHE / f"t2cnn_{t}_{split}_{panel}.npz")["og_p"] for t in TAGS], 0)
            pM = W_L * c["og_p"] + (1 - W_L) * pC
            truth, elig = og["fut"], og["efut"]
            curve = np.stack([iou(pM >= t, truth, elig) for t in TAUS], 1)          # (N, len(TAUS))
            vis_q = og["ok"] & (np.nan_to_num(og["v"], nan=1e9) <= 0.6 * ctx.vf)
            base = last_state(vis_q, og["ok"])
            early = vis_q[:, :3].mean(1) > 0.5
            runs = (base[:, 1:] & ~base[:, :-1]).sum(1) + base[:, :1].sum(1)
            with np.errstate(all="ignore"):
                ent = -(pM * np.log(np.maximum(pM, 1e-6)) + (1 - pM) * np.log(np.maximum(1 - pM, 1e-6)))
            f = np.stack([base.sum(1), base.mean(1), runs, base.sum(1) - early.sum(1), og["ok"].mean((1, 2)),
                          og["t0"] / 12.0, (z["weekday"][og["d"]] >= 5).astype(float), np.full(n, i), np.full(n, ctx.S),
                          pM.mean((1, 2)), pM.max((1, 2)), pM.sum((1, 2)) / ctx.S, ent.mean((1, 2)),
                          (pM >= 0.4).mean((1, 2)), (pM >= 0.5).mean((1, 2)), (pM >= 0.6).mean((1, 2)),
                          (pM >= 0.7).mean((1, 2)),
                          pM.sum((1, 2)) / np.maximum(base.sum(1) * pM.shape[1], 1),
                          (pM >= 0.4).mean((1, 2)) - (pM >= 0.7).mean((1, 2))], 1).astype(np.float32)
            d = D.setdefault(split, {"curve": [], "X": []})
            d["curve"].append(curve); d["X"].append(f)
        print(f"[{panel}] collected ({time.time() - t0_all:.0f}s)", flush=True)
    for k in D:
        D[k] = {a: np.concatenate(b) for a, b in D[k].items()}
    prod = np.argmin(np.abs(TAUS - TAU))
    res: dict = {}
    for fit, app in (("validation", "private"), ("private", "validation")):
        cf, ca = D[fit]["curve"], D[app]["curve"]
        Xf, Xa = D[fit]["X"], D[app]["X"]
        base_iou = pick(ca, np.full(len(ca), prod))
        out = {"production tau 0.60": base_iou, "ORACLE per-window tau": ca.max(1)}
        out["global tau (fitted)"] = pick(ca, np.full(len(ca), int(np.argmax(cf.mean(0)))))
        for name, col in (("panel", FEATS.index("panel")),):
            out[f"tau per {name}"] = pick(ca, by_group(cf, Xf[:, col].astype(int), Xa[:, col].astype(int), len(T2_PANELS)))
        for name in ("nq", "p_mean", "hour", "ratio"):
            col = FEATS.index(name)
            edges = np.percentile(Xf[:, col], [20, 40, 60, 80])
            out[f"tau per {name} quintile"] = pick(ca, by_group(cf, quintile(Xf[:, col], edges), quintile(Xa[:, col], edges), 5))
        y = TAUS[cf.argmax(1)]
        m = lgb.train(PARAMS, lgb.Dataset(Xf, y, feature_name=FEATS), 400)
        k = np.abs(np.clip(m.predict(Xa), TAUS[0], TAUS[-1])[:, None] - TAUS[None]).argmin(1)
        out["lgbm -> best tau"] = pick(ca, k)
        gain = cf - cf[:, prod:prod + 1]                                            # per-window gain of each tau
        ms = [lgb.train(PARAMS, lgb.Dataset(Xf, gain[:, t], feature_name=FEATS), 300) for t in range(len(TAUS))]
        pred_gain = np.stack([mm.predict(Xa) for mm in ms], 1)
        out["lgbm -> argmax predicted gain"] = pick(ca, pred_gain.argmax(1))
        res[(fit, app)] = out
        print(f"fitted on {fit} ({time.time() - t0_all:.0f}s); lgbm tau range "
              f"{TAUS[k].min():.2f}-{TAUS[k].max():.2f}, median {np.median(TAUS[k]):.2f}", flush=True)
    for (fit, app), out in res.items():
        b = out["production tau 0.60"]
        orc = out["ORACLE per-window tau"].mean() - b.mean()
        print(f"\n=== fitted on {fit}, applied to {app} ({len(b)} windows) ===")
        for name, v in out.items():
            extra = "" if name.startswith(("production", "ORACLE")) else f"  captures {(v.mean() - b.mean()) / orc:.2f} of the oracle"
            print(f"  {name:<30} IoU {v.mean():.4f}  delta {ci(v - b)}{extra}")
    print(f"total {time.time() - t0_all:.0f}s")


if __name__ == "__main__":
    main()
