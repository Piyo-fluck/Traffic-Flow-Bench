"""Task 2 ONGOING: fix a production recipe from the note-56 hill climb and check it on both months.

Note 56 ran greedy ensemble selection in both directions. The two fits agree on the core members (transformer
seeds, CNN x3, global-state LightGBM) but differ in the tail, and a 6-member recipe with 1/7-style weights is the
kind of thing that fits window noise. This script evaluates a small number of ROUND-NUMBER recipes over the agreed
members on BOTH splits, so the production choice is a simple one that behaves the same in both months rather than
whichever exact weight vector the greedy search stopped at.

Caveat kept explicit: the member set comes from fits on these same two months, so these numbers are not fully
out-of-sample. The honest headline stays the cross-split +0.0294 / +0.0329 of note 56; this run only decides which
simple recipe to ship and checks it is not carried by one panel.

Optional extra source: if `T2_EXTRA_TAG` is set, `work/cache/t2cnn_<tag>_<split>_<panel>.npz` is added to the pool
(used to bring the shockwave LightGBM of note 54 in once its probabilities are cached).

    .venv/Scripts/python.exe analysis/t2_ongoing_consensus.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "scripts"))
from t2_eda02 import T2_PANELS  # noqa: E402
from t2_first_onset_eval import official_blank  # noqa: E402
from t2_hill_climb import GROUPS, load_group  # noqa: E402
from t2_m1_models import Ctx  # noqa: E402
from t2_official_like_eval import ongoing_official  # noqa: E402
from t2_pseudo_eval import iou  # noqa: E402
from tfb.data import CACHE, load  # noqa: E402
from tfb.t1eval import family_mean  # noqa: E402

TAUS = (0.40, 0.45, 0.50, 0.55, 0.60)
EXTRA = os.environ.get("T2_EXTRA_TAG", "")
RECIPES = {
    "M production": ({"lgbm": 1, "cnn3": 1}, 0.60),
    "M tau 0.50": ({"lgbm": 1, "cnn3": 1}, 0.50),
    "candidate O": ({"gs1": 1, "tfm3": 1}, 0.55),
    "tfm_more + cnn3 + gs1 (2:1:1)": ({"tfm_more": 2, "cnn3": 1, "gs1": 1}, None),
    "tfm_more + cnn3 + gs1 (1:1:1)": ({"tfm_more": 1, "cnn3": 1, "gs1": 1}, None),
    "tfm_all + cnn3 + gs1 (2:1:1)": ({"tfm_more": 1, "tfm3": 1, "cnn3": 1, "gs1": 1}, None),
    "+ lgbm kept (2:1:1:1)": ({"tfm_more": 2, "cnn3": 1, "gs1": 1, "lgbm": 1}, None),
    "+ gs0 (2:1:1:1)": ({"tfm_more": 2, "cnn3": 1, "gs1": 1, "gs0": 1}, None),
    "R with tfm8 (2:1:1)": ({"tfm8": 2, "cnn3": 1, "gs1": 1}, None),
    "tfm8 3:1:1": ({"tfm8": 3, "cnn3": 1, "gs1": 1}, None),
    "rec replaces cnn3 (3:1:1)": ({"tfm8": 3, "extra": 1, "gs1": 1}, None),
    "rec beside cnn3 (3:1:1:1)": ({"tfm8": 3, "cnn3": 1, "extra": 1, "gs1": 1}, None),
    "tfm8 + tfm3 (2:1:1:1)": ({"tfm8": 2, "tfm3": 1, "cnn3": 1, "gs1": 1}, None),
}
boot = np.random.default_rng(67)


def ci(delta: np.ndarray, n_boot: int = 2000) -> str:
    idx = boot.integers(0, len(delta), (n_boot, len(delta)))
    m = delta[idx].mean(1)
    return f"{delta.mean():+.4f} [{np.percentile(m, 2.5):+.4f}, {np.percentile(m, 97.5):+.4f}]"


def main() -> None:
    t0 = time.time()
    src: dict = {}
    tgt: dict = {}
    panels: dict = {}
    for i, panel in enumerate(T2_PANELS):
        ctx = Ctx(panel, i)
        ctx.fitd = np.ones_like(ctx.fitd)
        for split in ("validation", "private"):
            z = load(panel, split)
            w, _ = ongoing_official(ctx, z, official_blank(panel, split, z))
            if not len(w["d"]):
                continue
            c = np.load(CACHE / f"t2probs_{split}_{panel}.npz")
            assert (c["og_d"] == w["d"]).all()
            here = {"lgbm": c["og_p"].astype(np.float32)}
            g = np.load(CACHE / f"t2gs_{split}_{panel}.npz")
            for k in ("0", "1"):
                if f"og_p{k}" in g:
                    here[f"gs{k}"] = g[f"og_p{k}"].astype(np.float32)
            for name, tags in GROUPS.items():
                p = load_group(tags, split, panel, "og_p")
                if p is not None:
                    here[name] = p
            if EXTRA:
                p = load_group([EXTRA], split, panel, "og_p")
                if p is not None:
                    here["extra"] = p
            for name, p in here.items():
                src.setdefault(split, {}).setdefault(name, []).append(p)
            tgt.setdefault(split, []).append((w["fut"], w["efut"]))
            panels.setdefault(split, []).append(panel)
        print(f"  [{panel}] collected ({time.time() - t0:.0f}s)", flush=True)
    recipes = dict(RECIPES)
    if EXTRA:
        recipes[f"+ {EXTRA} (2:1:1:1)"] = ({"tfm_more": 2, "cnn3": 1, "gs1": 1, "extra": 1}, None)
    for split in ("validation", "private"):
        s, t = src[split], tgt[split]
        ref = None
        print(f"\n=== {split}: ongoing IoU on official-like windows ({sum(len(x[0]) for x in t)} windows) ===")
        for name, (w, tau) in recipes.items():
            if not all(n in s for n in w):
                continue
            def sc(tau):
                out, per = [], {}
                for j, (truth, elig) in enumerate(t):
                    tot = sum(w.values())
                    p = sum(c * s[n][j] for n, c in w.items()) / tot
                    v = iou(p >= tau, truth, elig)
                    out.append(v); per[panels[split][j]] = v.mean()
                return np.concatenate(out), per
            if tau is None:
                cand = {tt: sc(tt) for tt in TAUS}
                tau = max(cand, key=lambda k: cand[k][0].mean())
                v, per = cand[tau]
            else:
                v, per = sc(tau)
            if ref is None:
                ref = v
            worst = min(per.items(), key=lambda kv: kv[1])
            print(f"  {name:<30} tau {tau:.2f}  IoU {v.mean():.4f}  family {family_mean(per):.4f}  "
                  f"delta {ci(v - ref)}  worst panel {worst[0]} {worst[1]:.3f}")
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
