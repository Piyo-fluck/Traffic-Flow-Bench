"""Task 2: cheap decision / post-processing variants on cached probabilities (no retraining).

The first run computes and caches the probabilities of the saved models on the blank-aware official-like windows
(notes 26, 28): onset = H's 5 bags at T+30, ongoing = I's stress model over the horizon. The cache is
work/cache/t2probs_<split>_<panel>.npz. After that, every variant scores in seconds.

onset (baseline = K's rule: expected-IoU ratio, c 1.0; note 31). Onset queues cover 1.5-4 ADJACENT links, and
the ratio rule treats links as independent:
    block      the best single contiguous block (length 1..8) by the same expected-IoU ratio
    smooth_l   ratio rule on p + l * mean(neighbours), l in 0.25 / 0.5
ongoing (baseline = p >= 0.65; note 29). Queues are contiguous in space and time:
    sm_s_w     spatial smoothing [w, 1, w] (normalised) then tau;  sm_st  3x3 time x space mean then tau
    close      fill 1-link gaps between predicted cells of the same step;  open  drop cells without any
               space/time neighbour;  close+open
Days 1-14 and all days, validation and private, family mean, panels better/worse vs the baseline.

    py analysis/t2_decision_variants.py
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
from t2_expected_iou import select  # noqa: E402
from t2_first_onset_eval import first_onsets, official_blank  # noqa: E402
from t2_m1_models import Ctx  # noqa: E402
from t2_m1_predict import feats_og, feats_on, table_prior, zero_prior  # noqa: E402
from t2_official_like_eval import ongoing_official  # noqa: E402
from t2_pseudo_eval import iou  # noqa: E402
from tfb.data import CACHE, load  # noqa: E402
from tfb.t1eval import family_mean  # noqa: E402

DAY_SETS = {"d1-14": 14, "all": 31}


def fmean(d):
    d = {k: v for k, v in d.items() if np.isfinite(v)}
    return family_mean(d) if d else float("nan")


def cache_probs(panel: str, i: int, split: str, ctx, m_on, m_og) -> Path:
    path = CACHE / f"t2probs_{split}_{panel}.npz"
    if path.exists():
        return path
    z = load(panel, split)
    blank = official_blank(panel, split, z)
    on = first_onsets(ctx.vf, z, blank=blank)
    out = dict(on_d=on["d"], on_fut=on["fut"], on_efut=on["efut"])
    if len(on["d"]):
        grp = (z["weekday"][on["d"]] >= 5).astype(int)
        key = np.stack([on["d"], grp], 1)
        _, X = feats_on(ctx, False, on["v"], on["qf"], on["ok"], on["t0"], grp, table_prior(ctx), key,
                        z["weekday"][on["d"]].astype(int))
        out["on_p"] = np.mean([m.predict(X.reshape(-1, X.shape[-1])) for m in m_on], 0).reshape(X.shape[:2]).astype(np.float32)
    else:
        out["on_p"] = np.zeros((0, ctx.S), np.float32)
    og, _ = ongoing_official(ctx, z, blank)
    grp = (z["weekday"][og["d"]] >= 5).astype(int)
    key = np.stack([og["d"], grp], 1)
    wdo = z["weekday"][og["d"]].astype(int)
    ps = []
    for j in range(0, len(og["d"]), 200):
        b = slice(j, j + 200)
        _, Xh, _ = feats_og(ctx, True, og["v"][b], og["qf"][b], og["ok"][b], og["t0"][b], grp[b], zero_prior(ctx), key[b], wdo[b])
        ps.append(m_og.predict(Xh.reshape(-1, Xh.shape[-1])).reshape(Xh.shape[:3]).astype(np.float32))
    out.update(og_d=og["d"], og_fut=og["fut"], og_efut=og["efut"],
               og_p=np.concatenate(ps) if ps else np.zeros((0, F, ctx.S), np.float32))
    np.savez_compressed(path, **out)
    return path


# ---------------------------------------------------------------- onset rules (p: (S,) -> bool (S,))
def block_rule(p: np.ndarray, max_len: int = 8) -> np.ndarray:
    S, P = len(p), p.sum()
    cs = np.concatenate([[0.0], np.cumsum(p)])
    best, arg = -1.0, (int(np.argmax(p)), 1)
    for L in range(1, min(max_len, S) + 1):
        sb = cs[L:] - cs[:-L]
        sc = sb / (L + P - sb)
        a = int(np.argmax(sc))
        if sc[a] > best:
            best, arg = float(sc[a]), (a, L)
    out = np.zeros(S, bool)
    out[arg[0]:arg[0] + arg[1]] = True
    return out


def smooth_rule(p: np.ndarray, lam: float) -> np.ndarray:
    pp = np.pad(p, 1, mode="edge")
    return select(np.clip(p + lam * (pp[:-2] + pp[2:]) / 2, 0, 1), "ratio", 1.0)


ONSET_RULES = {"ratio (K)": lambda p: select(p, "ratio", 1.0), "block": block_rule,
               "smooth_0.25": lambda p: smooth_rule(p, 0.25), "smooth_0.5": lambda p: smooth_rule(p, 0.5)}


# ---------------------------------------------------------------- ongoing rules (p: (N, F, S) -> bool)
def sm_space(p, w):
    pp = np.pad(p, ((0, 0), (0, 0), (1, 1)), mode="edge")
    return (w * pp[..., :-2] + p + w * pp[..., 2:]) / (1 + 2 * w)


def sm_st(p):
    pp = np.pad(p, ((0, 0), (1, 1), (1, 1)), mode="edge")
    return sum(pp[:, 1 + a:1 + a + p.shape[1], 1 + b:1 + b + p.shape[2]] for a in (-1, 0, 1) for b in (-1, 0, 1)) / 9


def close_space(pred):
    pp = np.pad(pred, ((0, 0), (0, 0), (1, 1)))
    return pred | (pp[..., :-2] & pp[..., 2:])


def open_st(pred):
    pp = np.pad(pred, ((0, 0), (1, 1), (1, 1)))
    nb = pp[:, :-2, 1:-1] | pp[:, 2:, 1:-1] | pp[:, 1:-1, :-2] | pp[:, 1:-1, 2:]
    return pred & nb


OG_RULES = {"tau0.65": lambda p: p >= 0.65, "tau0.60": lambda p: p >= 0.60,
            "sm_s_0.25": lambda p: sm_space(p, 0.25) >= 0.65, "sm_s_0.5": lambda p: sm_space(p, 0.5) >= 0.65,
            "sm_s_0.5/0.6": lambda p: sm_space(p, 0.5) >= 0.60, "sm_st": lambda p: sm_st(p) >= 0.65,
            "sm_st/0.55": lambda p: sm_st(p) >= 0.55, "close": lambda p: close_space(p >= 0.65),
            "open": lambda p: open_st(p >= 0.65), "close+open": lambda p: open_st(close_space(p >= 0.65))}


def main() -> None:
    t_start = time.time()
    m_on = [lgb.Booster(model_file=str(f)) for f in sorted((ROOT / "work" / "sub_t2robust_bag5").glob("t2_onset*.txt"))]
    m_og = lgb.Booster(model_file=str(ROOT / "work" / "sub_t2robust_bag5_stress" / "t2_ongoing.txt"))
    res: dict = {}
    for i, panel in enumerate(T2_PANELS):
        ctx = None
        for split in ("validation", "private"):
            path = CACHE / f"t2probs_{split}_{panel}.npz"
            if not path.exists():
                if ctx is None:
                    ctx = Ctx(panel, i)
                    ctx.fitd = np.ones_like(ctx.fitd)
                cache_probs(panel, i, split, ctx, m_on, m_og)
            c = np.load(path)
            for name, rule in ONSET_RULES.items():
                if len(c["on_d"]):
                    pr = np.zeros(c["on_fut"].shape, bool)
                    pr[:, F - 1] = np.stack([rule(p) for p in c["on_p"]])
                    s = iou(pr, c["on_fut"], c["on_efut"])
                else:
                    s = np.zeros(0)
                for ds, dmax in DAY_SETS.items():
                    m = c["on_d"] < dmax
                    res.setdefault(("onset", split, ds), {}).setdefault(name, {})[panel] = s[m].mean() if m.any() else np.nan
            for name, rule in OG_RULES.items():
                s = iou(rule(c["og_p"]), c["og_fut"], c["og_efut"]) if len(c["og_d"]) else np.zeros(0)
                for ds, dmax in DAY_SETS.items():
                    m = c["og_d"] < dmax
                    res.setdefault(("ongoing", split, ds), {}).setdefault(name, {})[panel] = s[m].mean() if m.any() else np.nan
        print(f"[{panel}] done ({time.time() - t_start:.0f}s)", flush=True)
    for (kind, split, ds), r in res.items():
        base = r["ratio (K)" if kind == "onset" else "tau0.65"]
        print(f"\n== {kind} {split} {ds}: IoU family mean (baseline {fmean(base):.3f}) ==")
        for name, d in r.items():
            diffs = [d[p] - base[p] for p in d if np.isfinite(d[p]) and np.isfinite(base[p])]
            print(f"  {name:<14} {fmean(d):.3f} ({fmean(d) - fmean(base):+.3f}; better/worse "
                  f"{sum(x > 1e-9 for x in diffs)}/{sum(x < -1e-9 for x in diffs)})")
    print(f"total {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
