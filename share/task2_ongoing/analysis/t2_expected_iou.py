"""Task 2: pick each window's prediction set by maximising the EXPECTED IoU, instead of a fixed alpha / tau.

IoU is a set metric, and its Bayes-optimal decision depends on the whole probability vector of a window (F-measure
/ Jaccard maximisers, e.g. arXiv:1310.4849). With cells sorted by p, the optimum is a top-k set whose k maximises
E[IoU_k]. Production uses one rule for every window (onset: k = round(1.2 sum p); ongoing: p >= 0.65). Here, with
the saved models (onset = H's 5 bags, ongoing = I's stress model), on the blank-aware official-like windows:
    current    production rules
    ratio_c    k* = argmax_k S_k / (k + P - S_k), with S_k = the top-k sum and P = the sum of all p. This is the
               ratio-of-expectations approximation, applied to p' = min(1, c p) for c in 1.0 / 1.2 / 1.5
    mc_c       exact E[IoU_k] under independent Bernoulli(p') labels, by Monte Carlo (1,000 draws), conditioned on
               a non-empty truth (official windows always hold a queue)
Onset uses only the T+30 links. Ongoing uses every horizon cell with p' > 0.01. Days 1-14 and all days are scored,
validation and private, family mean.

    py analysis/t2_expected_iou.py
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
from t2_eda03 import F  # noqa: E402
from t2_first_onset_eval import first_onsets, official_blank  # noqa: E402
from t2_m1_models import Ctx, onset_decide  # noqa: E402
from t2_m1_predict import feats_og, feats_on, table_prior, zero_prior  # noqa: E402
from t2_official_like_eval import ongoing_official  # noqa: E402
from t2_pseudo_eval import iou  # noqa: E402
from tfb.data import load  # noqa: E402
from tfb.t1eval import family_mean  # noqa: E402

CS = (1.0, 1.2, 1.5)
DAY_SETS = {"d1-14": 14, "all": 31}
N_MC = 1000
rng = np.random.default_rng(11)


def fmean(d):
    d = {k: v for k, v in d.items() if np.isfinite(v)}
    return family_mean(d) if d else float("nan")


def best_k_ratio(ps: np.ndarray) -> int:
    """ps sorted descending. argmax_k S_k / (k + P - S_k)."""
    s = np.cumsum(ps)
    k = np.arange(1, len(ps) + 1)
    return int(np.argmax(s / (k + ps.sum() - s))) + 1


def best_k_mc(ps: np.ndarray) -> int:
    """ps sorted descending. Exact E[IoU_k] by Monte Carlo, truth drawn Bernoulli(ps), conditioned on non-empty."""
    y = rng.random((N_MC, len(ps))) < ps[None]
    y = y[y.any(1)]
    if len(y) == 0:
        return 1
    inter = np.cumsum(y, 1)                         # |top-k ∩ y| for k = 1..n
    k = np.arange(1, len(ps) + 1)
    union = k[None] + y.sum(1, keepdims=True) - inter
    return int(np.argmax((inter / union).mean(0))) + 1


def select(p_flat: np.ndarray, rule: str, c: float, cand_min: float = 0.0) -> np.ndarray:
    """p_flat (n,) -> bool (n,), the chosen top-k set."""
    pc = np.minimum(1.0, c * p_flat)
    idx = np.flatnonzero(pc > cand_min) if cand_min > 0 else np.arange(len(pc))
    out = np.zeros(len(pc), bool)
    if len(idx) == 0:
        out[np.argmax(pc)] = True
        return out
    order = idx[np.argsort(-pc[idx])]
    ps = pc[order]
    k = best_k_ratio(ps) if rule == "ratio" else best_k_mc(ps)
    out[order[:k]] = True
    return out


def main() -> None:
    t_start = time.time()
    m_on = [lgb.Booster(model_file=str(f)) for f in sorted((ROOT / "work" / "sub_t2robust_bag5").glob("t2_onset*.txt"))]
    m_og = lgb.Booster(model_file=str(ROOT / "work" / "sub_t2robust_bag5_stress" / "t2_ongoing.txt"))
    rules = ["current"] + [f"{r}_c{c}" for r in ("ratio", "mc") for c in CS]
    res: dict = {}
    for i, panel in enumerate(T2_PANELS):
        ctx = Ctx(panel, i)
        ctx.fitd = np.ones_like(ctx.fitd)
        for split in ("validation", "private"):
            z = load(panel, split)
            blank = official_blank(panel, split, z)
            # onset
            on = first_onsets(ctx.vf, z, blank=blank)
            s_on = {r: np.zeros(0) for r in rules}
            if len(on["d"]):
                grp = (z["weekday"][on["d"]] >= 5).astype(int)
                key = np.stack([on["d"], grp], 1)
                _, X = feats_on(ctx, False, on["v"], on["qf"], on["ok"], on["t0"], grp, table_prior(ctx), key,
                                z["weekday"][on["d"]].astype(int))
                p = np.mean([m.predict(X.reshape(-1, X.shape[-1])) for m in m_on], 0).reshape(X.shape[:2])
                for r in rules:
                    pr = np.zeros((len(on["d"]), F, ctx.S), bool)
                    if r == "current":
                        pr[:, F - 1] = onset_decide(p, ("sum", 1.2))
                    else:
                        rule, c = r.split("_c")
                        pr[:, F - 1] = np.stack([select(p[j], rule, float(c)) for j in range(len(p))])
                    s_on[r] = iou(pr, on["fut"], on["efut"])
            # ongoing
            og, _ = ongoing_official(ctx, z, blank)
            s_og = {r: [] for r in rules}
            grp = (z["weekday"][og["d"]] >= 5).astype(int)
            key = np.stack([og["d"], grp], 1)
            wdo = z["weekday"][og["d"]].astype(int)
            for j in range(0, len(og["d"]), 200):
                b = slice(j, j + 200)
                _, Xh, _ = feats_og(ctx, True, og["v"][b], og["qf"][b], og["ok"][b], og["t0"][b], grp[b], zero_prior(ctx),
                                    key[b], wdo[b])
                p = m_og.predict(Xh.reshape(-1, Xh.shape[-1])).reshape(Xh.shape[:3])
                fut, efut = og["fut"][b], og["efut"][b]
                for r in rules:
                    if r == "current":
                        pr = p >= 0.65
                    else:
                        rule, c = r.split("_c")
                        pr = np.stack([select(p[w].reshape(-1), rule, float(c), cand_min=0.01).reshape(p.shape[1:])
                                       for w in range(len(p))])
                    s_og[r].append(iou(pr, fut, efut))
            s_og = {r: (np.concatenate(v) if v else np.zeros(0)) for r, v in s_og.items()}
            for r in rules:
                for ds, dmax in DAY_SETS.items():
                    a = s_on[r][on["d"] < dmax] if len(s_on[r]) else s_on[r]
                    b2 = s_og[r][og["d"] < dmax] if len(s_og[r]) else s_og[r]
                    rr = res.setdefault((split, ds), {}).setdefault(r, {"onset": {}, "ongoing": {}})
                    rr["onset"][panel] = a.mean() if len(a) else np.nan
                    rr["ongoing"][panel] = b2.mean() if len(b2) else np.nan
            cur = res[(split, "all")]
            print(f"[{panel} {split}] onset events {len(on['d'])}, ongoing windows {len(og['d'])} | "
                  + "  ".join(f"{r}: {cur[r]['onset'][panel]:.3f}/{cur[r]['ongoing'][panel]:.3f}" for r in rules)
                  + f" ({time.time() - t_start:.0f}s)", flush=True)
    for (split, ds), rr in res.items():
        print(f"\n== {split} {ds}: onset / ongoing IoU, family mean (blank-aware official-like windows) ==")
        base = rr["current"]
        for r, d in rr.items():
            a, b2 = fmean(d["onset"]), fmean(d["ongoing"])
            da = [d["onset"][p] - base["onset"][p] for p in d["onset"] if np.isfinite(d["onset"][p]) and np.isfinite(base["onset"][p])]
            db = [d["ongoing"][p] - base["ongoing"][p] for p in d["ongoing"] if np.isfinite(d["ongoing"][p]) and np.isfinite(base["ongoing"][p])]
            print(f"  {r:<12} onset {a:.3f} ({a - fmean(base['onset']):+.3f}, better/worse {sum(x > 1e-9 for x in da)}/{sum(x < -1e-9 for x in da)})"
                  f"  ongoing {b2:.3f} ({b2 - fmean(base['ongoing']):+.3f}, {sum(x > 1e-9 for x in db)}/{sum(x < -1e-9 for x in db)})"
                  f"  -> S_queue {((a + b2) / 2) - (fmean(base['onset']) + fmean(base['ongoing'])) / 2:+.4f} "
                  f"(total {0.3 * (((a + b2) / 2) - (fmean(base['onset']) + fmean(base['ongoing'])) / 2):+.4f})")
    print(f"total {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
