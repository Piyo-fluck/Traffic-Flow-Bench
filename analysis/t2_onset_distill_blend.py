"""Does the distilled CNN of idea 4 help M production onset blend? (0.7 * LightGBM + 0.3 * CNN, ratio rule)"""
from __future__ import annotations
import os, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "analysis"), str(ROOT / "scripts")]
from t2_eda02 import T2_PANELS
from t2_eda03 import F
from t2_expected_iou import select
from t2_first_onset_eval import first_onsets, official_blank
from t2_m1_models import Ctx
from t2_pseudo_eval import iou
from tfb.data import CACHE, load
from tfb.t1eval import family_mean

CNN3 = ["t2_cnn_lov", "t2_cnn_lov_s1", "t2_cnn_lov_s2"]
DIS = [os.environ.get("T2_CNN_TAG", "t2_cnn_lov_dis_s20")]
W = 0.7
boot = np.random.default_rng(41)

def ci(d):
    idx = boot.integers(0, len(d), (2000, len(d)))
    m = d[idx].mean(1)
    return f"{d.mean():+.4f} [{np.percentile(m, 2.5):+.4f}, {np.percentile(m, 97.5):+.4f}]"

def grp(tags, split, panel):
    out = []
    for t in tags:
        f = CACHE / f"t2cnn_{t}_{split}_{panel}.npz"
        if not f.exists():
            return None
        out.append(np.load(f)["on_p"])
    return np.mean(out, 0)

def main():
    t0 = time.time()
    res, fam = {}, {}
    for i, panel in enumerate(T2_PANELS):
        ctx = Ctx(panel, i); ctx.fitd = np.ones_like(ctx.fitd)
        for split in ("validation", "private"):
            z = load(panel, split)
            on = first_onsets(ctx.vf, z, blank=official_blank(panel, split, z))
            if not len(on["d"]):
                continue
            c = np.load(CACHE / f"t2probs_{split}_{panel}.npz")
            assert (c["on_d"] == on["d"]).all()
            cnn3, dis = grp(CNN3, split, panel), grp(DIS, split, panel)
            if dis is None:
                print("missing distilled cache for", panel, split); return
            variants = {"M production (cnn3)": cnn3, "variant only": dis, "cnn3 + variant": 0.5 * (cnn3 + dis)}
            for name, cn in variants.items():
                p = W * c["on_p"] + (1 - W) * cn
                pr = np.zeros(on["fut"].shape, bool)
                pr[:, F - 1] = np.stack([select(q, "ratio", 1.0) for q in p])
                s = iou(pr, on["fut"], on["efut"])
                res.setdefault((split, name), []).append(s)
                fam.setdefault((split, name), {})[panel] = s.mean()
        print(f"[{panel}] done ({time.time() - t0:.0f}s)", flush=True)
    for split in ("validation", "private"):
        base = np.concatenate(res[(split, "M production (cnn3)")])
        print(f"\n=== {split}: onset, {len(base)} official-like windows ===")
        for name in ("M production (cnn3)", "variant only", "cnn3 + variant"):
            v = np.concatenate(res[(split, name)])
            print(f"  {name:<22} IoU {v.mean():.4f}  family {family_mean(fam[(split, name)]):.4f}  delta {ci(v - base)}")
    print(f"total {time.time() - t0:.0f}s")

if __name__ == "__main__":
    main()
