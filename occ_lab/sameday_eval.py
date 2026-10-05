"""Score the same-day adaptation (occ_lab/pseudo_adapt_sameday.py) in AT's ongoing stack.

Averages the seed caches pseudo_sd_s<seed>_<mode>_* into pseudo_ssl_sd4_<mode>_*, then rebuilds AT's ongoing p
(0.40 old + 0.60 p_PS, gated residual corrector, tau 0.50) with each mode's p_PS. Reported on all official-like
windows and on the early days only (day index <= 4): official ongoing windows have 1-4 earlier days of their own
month, the lab pool mostly 10-20, so the early subset is the regime the official windows are in.

    .venv/Scripts/python.exe occ_lab/sameday_eval.py
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
from online_adapter import ETA, GATE, global_features, load_split  # noqa: E402

CACHE = LAB / "cache"
SEEDS = (20, 21, 22, 23)
MODES = ("base", "sameday")
EARLY = 4


def average():
    for f in sorted(CACHE.glob(f"pseudo_sd_s{SEEDS[0]}_base_*.npz")):
        suf = f.name.split("_base_")[1]
        for mode in MODES:
            z = [np.load(CACHE / f"pseudo_sd_s{s}_{mode}_{suf}") for s in SEEDS]
            np.savez(CACHE / f"pseudo_ssl_sd4_{mode}_{suf}", on=np.mean([a["on"] for a in z], 0).astype(np.float32),
                     og=np.mean([a["og"].astype(np.float32) for a in z], 0).astype(np.float16))


def main() -> None:
    average()
    gbm = lgb.Booster(model_file=str(LAB / "models" / "residual_lgb.txt"))
    fam = {}
    for i, panel in enumerate(C.T2_PANELS):
        for mode in MODES:
            for s, D in load_split(i, panel, f"ssl_sd4_{mode}").items():
                blend = D["base"]
                F = global_features(D["P"], blend)
                rh = gbm.predict(F.reshape(-1, F.shape[-1])).reshape(blend.shape).astype(np.float32)
                p = np.clip(blend + ETA * np.where(np.abs(rh) >= GATE, rh, 0.0), 0, 1)
                y, e, d = D["y"], D["e"], D["d"]
                for nm, q in (("p_PS alone", D["ps"]), ("AT stack", p)):
                    s_ = C.iou(q >= 0.5, y, e)
                    fam.setdefault((s, nm, mode, "all"), {})[panel] = s_.mean()
                    if (d <= EARLY).any():
                        fam.setdefault((s, nm, mode, "early"), {})[panel] = s_[d <= EARLY].mean()
        print(f"  {panel}", flush=True)
    for sub in ("all", "early"):
        print(f"\n=== ongoing family IoU, {sub} windows ===")
        for nm in ("p_PS alone", "AT stack"):
            for mode in MODES:
                v = [C.family_mean(fam[(s, nm, mode, sub)]) for s in C.SPLITS]
                print(f"  {nm:<11} {mode:<8} " + "  ".join(f"{s} {x:.4f}" for s, x in zip(C.SPLITS, v))
                      + f"  mean {np.mean(v):.4f}")


if __name__ == "__main__":
    main()
