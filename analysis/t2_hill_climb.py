"""Task 2: hill-climbing the blend over every cached model family (the ensemble step, now authorised by the user).

M's weights were set by hand: onset 0.7*LightGBM + 0.3*CNN x3, ongoing 0.5*LightGBM + 0.5*CNN x3 at tau 0.60.
Note 55 put the realistic route to +0.01 total in stacking, and the blend weights are one of its untried items.
This runs the standard greedy ensemble selection (Caruana: repeatedly add the member that improves the score most,
with replacement, so members get integer weights) over every probability source cached on the official-like
windows, choosing tau jointly for ongoing and using the expected-IoU rule for onset.

Everything is FITTED ON ONE SPLIT AND APPLIED TO THE OTHER, both directions, so the reported gain is out-of-sample
(the same protocol as note 53; a production ensemble would keep the weights found here, which is legitimate because
they are selected on a scored month only as a derivation device, then applied unchanged).

References printed alongside: M's production recipe, M at tau 0.50 (notes 52/53), and candidate O's ongoing recipe
(0.5*gs1 + 0.5*TFM x3 at tau 0.55, note 45).

    .venv/Scripts/python.exe analysis/t2_hill_climb.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

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
from t2_official_like_eval import ongoing_official  # noqa: E402
from t2_pseudo_eval import iou  # noqa: E402
from tfb.data import CACHE, load  # noqa: E402

CNN3 = ["t2_cnn_lov", "t2_cnn_lov_s1", "t2_cnn_lov_s2"]
TFM3 = ["t2_cnn_lov_tfm", "t2_cnn_lov_tfm_s1", "t2_cnn_lov_tfm_s2"]
TFM_MORE = ["t2_cnn_lov_tfm_s3", "t2_cnn_lov_tfm_s4", "t2_cnn_lov_tfm_s5", "t2_cnn_lov_tfm_s6"]
TFM8 = TFM_MORE + ["t2_cnn_lov_tfm_s7", "t2_cnn_lov_tfm_s8", "t2_cnn_lov_tfm_s9", "t2_cnn_lov_tfm_s10"]
GROUPS = {"cnn3": CNN3, "tfm3": TFM3, "tfm_more": TFM_MORE, "tfm8": TFM8, "cnn_glob": ["t2_cnn_lov_glob"],
          "cnn_aux": ["t2_cnn_lov_aux_ema"], "cnn_aux8": ["t2_cnn_lov_aux_emb8_ema"],
          "tfm_w128": ["t2_cnn_lov_tfm_w128"], "tfm_pri": ["t2_cnn_pri_lov_tfm_on0.5"]}
TAUS = np.round(np.arange(0.30, 0.81, 0.05), 2)
N_ITER = 25
boot = np.random.default_rng(53)


def ci(delta: np.ndarray, n_boot: int = 2000) -> str:
    idx = boot.integers(0, len(delta), (n_boot, len(delta)))
    m = delta[idx].mean(1)
    return f"{delta.mean():+.4f} [{np.percentile(m, 2.5):+.4f}, {np.percentile(m, 97.5):+.4f}]"


def load_group(tags, split, panel, key):
    """Mean probability over the tags of a group, or None when any file is missing."""
    out = []
    for t in tags:
        f = CACHE / f"t2cnn_{t}_{split}_{panel}.npz"
        if not f.exists():
            return None
        z = np.load(f)
        if key not in z:                      # sources cached for one condition only (e.g. ongoing-only models)
            return None
        out.append(z[key])
    return np.mean(out, 0).astype(np.float32)


def collect(kind: str) -> tuple[dict, dict]:
    """kind 'og' or 'on' -> {split: {source: [per-panel arrays]}}, {split: (truth, elig) per panel}."""
    src: dict = {}
    tgt: dict = {}
    for i, panel in enumerate(T2_PANELS):
        ctx = Ctx(panel, i)
        ctx.fitd = np.ones_like(ctx.fitd)
        for split in ("validation", "private"):
            z = load(panel, split)
            blank = official_blank(panel, split, z)
            if kind == "og":
                w, _ = ongoing_official(ctx, z, blank)
                truth, elig = w["fut"], w["efut"]
            else:
                w = first_onsets(ctx.vf, z, blank=blank)
                truth, elig = w["fut"], w["efut"]
            if not len(w["d"]):
                continue
            c = np.load(CACHE / f"t2probs_{split}_{panel}.npz")
            assert (c[f"{kind}_d"] == w["d"]).all()
            here = {"lgbm": c[f"{kind}_p"].astype(np.float32)}
            g = np.load(CACHE / f"t2gs_{split}_{panel}.npz")
            for k in ("0", "1"):
                if f"{kind}_p{k}" in g:
                    here[f"gs{k}"] = g[f"{kind}_p{k}"].astype(np.float32)
            for name, tags in GROUPS.items():
                p = load_group(tags, split, panel, f"{kind}_p")
                if p is not None:
                    here[name] = p
            for name, p in here.items():
                src.setdefault(split, {}).setdefault(name, []).append(p)
            tgt.setdefault(split, []).append((truth, elig))
        print(f"  [{panel}] {kind} collected", flush=True)
    for split in list(src):
        keep = [n for n, v in src[split].items() if len(v) == len(tgt[split])]
        src[split] = {n: src[split][n] for n in keep}
    return src, tgt


def og_iou(src, tgt, w: dict, tau: float) -> np.ndarray:
    out = []
    for j, (truth, elig) in enumerate(tgt):
        tot = sum(w.values())
        p = sum(c * src[n][j] for n, c in w.items()) / tot
        out.append(iou(p >= tau, truth, elig))
    return np.concatenate(out)


def on_iou(src, tgt, w: dict) -> np.ndarray:
    out = []
    for j, (truth, elig) in enumerate(tgt):
        tot = sum(w.values())
        p = sum(c * src[n][j] for n, c in w.items()) / tot
        pr = np.zeros(truth.shape, bool)
        pr[:, F - 1] = np.stack([select(q, "ratio", 1.0) for q in p])
        out.append(iou(pr, truth, elig))
    return np.concatenate(out)


def climb_og(src, tgt, names):
    w: dict = {}
    best_tau, best = 0.5, -1.0
    for it in range(N_ITER):
        cand_best = None
        for n in names:
            w2 = dict(w)
            w2[n] = w2.get(n, 0) + 1
            for tau in TAUS:
                v = og_iou(src, tgt, w2, tau).mean()
                if cand_best is None or v > cand_best[0]:
                    cand_best = (v, n, tau)
        if cand_best[0] <= best + 1e-6 and it > 0:
            break
        best, n, best_tau = cand_best
        w[n] = w.get(n, 0) + 1
    return w, best_tau, best


def climb_on(src, tgt, names):
    w: dict = {}
    best = -1.0
    for it in range(N_ITER):
        cand_best = None
        for n in names:
            w2 = dict(w)
            w2[n] = w2.get(n, 0) + 1
            v = on_iou(src, tgt, w2).mean()
            if cand_best is None or v > cand_best[0]:
                cand_best = (v, n)
        if cand_best[0] <= best + 1e-6 and it > 0:
            break
        best, n = cand_best
        w[n] = w.get(n, 0) + 1
    return w, best


def show(w: dict) -> str:
    tot = sum(w.values())
    return " + ".join(f"{c / tot:.2f}*{n}" for n, c in sorted(w.items(), key=lambda kv: -kv[1]))


def main() -> None:
    t0 = time.time()
    print("collecting ongoing probabilities", flush=True)
    src_og, tgt_og = collect("og")
    print(f"sources: {sorted(src_og['validation'])} ({time.time() - t0:.0f}s)", flush=True)
    for fit, app in (("validation", "private"), ("private", "validation")):
        names = sorted(set(src_og[fit]) & set(src_og[app]))
        prod = og_iou(src_og[app], tgt_og[app], {"lgbm": 1, "cnn3": 1}, 0.60)
        refs = {"M production (0.5 lgbm + 0.5 cnn3, tau 0.60)": ({"lgbm": 1, "cnn3": 1}, 0.60),
                "M at tau 0.50": ({"lgbm": 1, "cnn3": 1}, 0.50),
                "candidate O (0.5 gs1 + 0.5 tfm3, tau 0.55)": ({"gs1": 1, "tfm3": 1}, 0.55)}
        print(f"\n=== ONGOING: fitted on {fit}, applied to {app} ===")
        for name, (w, tau) in refs.items():
            if all(n in src_og[app] for n in w):
                v = og_iou(src_og[app], tgt_og[app], w, tau)
                print(f"  {name:<44} IoU {v.mean():.4f}  delta {ci(v - prod)}")
        w, tau, fit_v = climb_og(src_og[fit], tgt_og[fit], names)
        v = og_iou(src_og[app], tgt_og[app], w, tau)
        print(f"  hill climb (fit IoU {fit_v:.4f}) tau {tau:.2f}: {show(w)}")
        print(f"  {'-> applied':<44} IoU {v.mean():.4f}  delta {ci(v - prod)}  -> total {0.15 * (v.mean() - prod.mean()):+.4f}",
              flush=True)
    print("\ncollecting onset probabilities", flush=True)
    src_on, tgt_on = collect("on")
    for fit, app in (("validation", "private"), ("private", "validation")):
        names = sorted(set(src_on[fit]) & set(src_on[app]))
        prod = on_iou(src_on[app], tgt_on[app], {"lgbm": 7, "cnn3": 3})
        print(f"\n=== ONSET: fitted on {fit}, applied to {app} ===")
        print(f"  M production (0.7 lgbm + 0.3 cnn3)          IoU {prod.mean():.4f}")
        w, fit_v = climb_on(src_on[fit], tgt_on[fit], names)
        v = on_iou(src_on[app], tgt_on[app], w)
        print(f"  hill climb (fit IoU {fit_v:.4f}): {show(w)}")
        print(f"  {'-> applied':<44} IoU {v.mean():.4f}  delta {ci(v - prod)}  -> total {0.15 * (v.mean() - prod.mean()):+.4f}",
              flush=True)
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
