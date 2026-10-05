"""How far should the pseudo-supervised model's ongoing weight go past AP's 0.30?

AP put it at 0.30 inside the frozen production blend and gained +0.00158 on the leaderboard, with the ongoing
family moving +0.0125 locally - a realisation of about 0.8, the only local surface that has ever transferred
(note 106). This sweeps the weight on that same surface: the scored months, the right RUN (note 108 showed the
run is what matters), the AI corrector applied on top exactly as production does.

    p0(a) = (1 - a) * (0.20 gs1 + 0.60 tfm8 + 0.20 cnn3) + a * pseudo
    p     = clip(p0 + 0.75 * r_hat where |r_hat| >= 0.10),  decision p >= 0.50

    python occ_lab/ongoing_pseudo_sweep.py --tag s20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta"))
sys.path.insert(0, str(LAB.parent / "analysis"))
sys.path.insert(0, str(LAB.parent / "scripts"))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

import common as C  # noqa: E402
from residual_diag import NAMES  # noqa: E402
from t2_residual_predict import features  # noqa: E402

CACHE = LAB / "cache"
TAU, ETA, GATE = 0.50, 0.75, 0.10


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="s20")
    ap.add_argument("--weights", type=float, nargs="+",
                    default=[0.0, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.85, 1.0])
    args = ap.parse_args()
    gbm = lgb.Booster(model_file=str(LAB / "models" / "residual_lgb.txt"))
    assert gbm.feature_name() == NAMES, "the corrector was fitted on a different feature order"
    raw, cor = {}, {}
    for i, panel in enumerate(C.T2_PANELS):
        f = CACHE / f"pseudo_{args.tag}_{C.SPLITS[0]}_{panel}.npz"
        if not f.exists():
            continue
        ctx, stat, W = C.windows(i, panel)
        for s in C.SPLITS:
            og, ref = W[s]["og"], W[s]["ref"]
            if not len(og["d"]):
                continue
            nw = np.load(CACHE / f"pseudo_{args.tag}_{s}_{panel}.npz")["og"].astype(np.float32)
            mem = np.load(CACHE / f"ogmem_{s}_{panel}.npz")           # the real corrector features
            P = {"gs1": ref["og_gs1"], "tfm8": ref["og_tfm8"], "cnn3": ref["og_cnn3"],
                 "occ_cnn": mem["occ_cnn"], "occ_tfm": mem["occ_tfm"], "stid": mem["stid"]}
            base = 0.2 * P["gs1"] + 0.6 * P["tfm8"] + 0.2 * P["cnn3"]
            y, e = og["fut"], og["efut"]
            for a in args.weights:
                p0 = (1 - a) * base + a * nw
                raw.setdefault((s, a), {})[panel] = C.iou(p0 >= TAU, y, e).mean()
                feat = features(P, p0)
                rh = gbm.predict(feat.reshape(-1, feat.shape[-1])).reshape(p0.shape).astype(np.float32)
                p = np.clip(p0 + ETA * np.where(np.abs(rh) >= GATE, rh, 0.0), 0, 1)
                cor.setdefault((s, a), {})[panel] = C.iou(p >= TAU, y, e).mean()
        ctx.z = None
        print(f"  {panel}", flush=True)

    for s in C.SPLITS:
        b = C.family_mean(raw[(s, 0.30)])
        print(f"\n=== ongoing {s} (AP's 0.30 as the reference) ===")
        print(f"{'weight':>7} {'blend only':>12} {'+ corrector':>13} {'vs AP':>9}")
        for a in args.weights:
            if (s, a) not in raw:
                continue
            r, c = C.family_mean(raw[(s, a)]), C.family_mean(cor[(s, a)])
            print(f"{a:>7.2f} {r:>12.4f} {c:>13.4f} {c - C.family_mean(cor[(s, 0.30)]):>+9.4f}")
        _ = b


if __name__ == "__main__":
    main()
