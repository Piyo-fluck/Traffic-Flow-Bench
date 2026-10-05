"""AT's ongoing stack with the transformer family adapted to the scored run (occ_lab/pseudo_adapt_tfm.py).

AT: p0 = 0.40 old + 0.60 p_PS, old = 0.20 gs1 + 0.60 tfm8 + 0.20 cnn3, then the gated residual corrector, tau 0.50.
Here tfm8 (frozen) is replaced by the mean of the eight adapted transformers, same weights. The corrector was fitted
with the frozen tfm8 as a feature column, so it is fed either the frozen column (its own training distribution) or
the adapted one. Two re-weighted variants are listed for reference only (weight changes alone have not moved the LB).

    .venv/Scripts/python.exe occ_lab/tfmps_stack_eval.py
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
TAG = "ssl_w192ssl_avg4"
SEEDS = range(3, 11)
MODE = "sup0.0001"


def main() -> None:
    gbm = lgb.Booster(model_file=str(LAB / "models" / "residual_lgb.txt"))
    corr = lambda base, P: np.clip(base + ETA * np.where(np.abs(  # noqa: E731
        r := gbm.predict(global_features(P, base).reshape(-1, 15)).reshape(base.shape).astype(np.float32)) >= GATE, r, 0.0), 0, 1)
    fam = {}
    for i, panel in enumerate(C.T2_PANELS):
        for s, D in load_split(i, panel, TAG).items():
            files = [CACHE / f"tfmps_s{k}_{MODE}_{s}_{panel}.npz" for k in SEEDS]
            if not all(f.exists() for f in files):
                continue
            tad = np.mean([np.load(f)["og"].astype(np.float32) for f in files], 0)
            P, ps = D["P"], D["ps"]
            Pad = dict(P, tfm8=tad)
            old_ad = 0.2 * P["gs1"] + 0.6 * tad + 0.2 * P["cnn3"]
            y, e, d = D["y"], D["e"], D["d"]
            sfiles = [CACHE / f"stidps_s{k}_sup_{s}_{panel}.npz" for k in range(4)]
            sad = np.mean([np.load(f)["og"].astype(np.float32) for f in sfiles], 0) if all(f.exists() for f in sfiles) else None
            variants = {
                "AT": corr(D["base"], P),
                "tfm8 -> adapted (corrector sees frozen tfm8)": corr(0.4 * old_ad + 0.6 * ps, P),
                "tfm8 -> adapted (corrector sees adapted tfm8)": corr(0.4 * old_ad + 0.6 * ps, Pad),
                "adapted tfm8 alone": tad,
                **({"tfm8 + stid -> adapted (corrector sees adapted tfm8 and stid)":
                    corr(0.4 * old_ad + 0.6 * ps, dict(Pad, stid=sad)),
                    "tfm8 -> adapted, stid -> adapted (corrector sees frozen tfm8, adapted stid)":
                    corr(0.4 * old_ad + 0.6 * ps, dict(P, stid=sad)),
                    "adapted stid alone": sad} if sad is not None else {}),
                "ref: 0.5 p_PS + 0.5 adapted tfm8, corrector": corr(0.5 * ps + 0.5 * tad, Pad),
                "ref: 0.4 p_PS + 0.4 adapted tfm8 + 0.2 old, corrector": corr(0.4 * ps + 0.4 * tad + 0.2 * D["old"], Pad),
            }
            for nm, q in variants.items():
                sc = C.iou(q >= 0.5, y, e)
                fam.setdefault((s, nm, "all"), {})[panel] = sc.mean()
                if (d <= 4).any():
                    fam.setdefault((s, nm, "early"), {})[panel] = sc[d <= 4].mean()
        print(f"  {panel}", flush=True)
    names = sorted({k[1] for k in fam}, key=lambda n: (n != "AT", n))
    for sub in ("all", "early"):
        print(f"\n=== ongoing family IoU, {sub} windows ===")
        ref = np.mean([C.family_mean(fam[(s, "AT", sub)]) for s in C.SPLITS])
        for nm in names:
            if all((s, nm, sub) in fam for s in C.SPLITS):
                v = [C.family_mean(fam[(s, nm, sub)]) for s in C.SPLITS]
                print(f"  {nm:<52} " + "  ".join(f"{s[:3]} {x:.4f}" for s, x in zip(C.SPLITS, v))
                      + f"  mean {np.mean(v):.4f} ({np.mean(v) - ref:+.4f})")


if __name__ == "__main__":
    main()
