"""What each ongoing subsystem is actually worth, on the scored months' official-like pool.

The ongoing half carries 24 trained models and a residual corrector. Twelve of them - occ_cnn x4, occ_tfm x4 and
the ongoing STID x4 - never enter the blend at all; `scripts/t2_residual_predict.py` uses them only as corrector
features. This strips the stack down one layer at a time and prices each layer.

    old      0.20 gs1 + 0.60 tfm8 + 0.20 cnn3                     (12 models: 1 + 8 + 3)
    blend    0.40 old + 0.60 p_PS                                 (+ the pseudo-supervised adaptation)
    + corr   clip(blend + 0.75 r_hat 1[|r_hat| >= 0.10])          (+ 12 more models, as features only)

and, to price the corrector's inputs rather than the corrector itself, a refit of the same LightGBM on the same
OOF cache with the twelve feature columns that come from those models held out.

    python occ_lab/ongoing_ablation.py --tag s20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta"))
sys.path.insert(0, str(LAB.parent / "analysis"))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

import common as C  # noqa: E402
from online_adapter import ETA, GATE, global_features, load_split, mean_of  # noqa: E402
from residual_diag import NAMES, TAU  # noqa: E402

CACHE = LAB / "cache"
W_PS = 0.60
# the corrector columns that exist only because occ_cnn, occ_tfm and the ongoing STID were trained
EXTRA = ["occ_cnn", "occ_tfm", "stid", "d_logit_stid", "d_logit_occ_cnn", "member_std", "member_range"]


def oof_rows(panel):
    f, fs, fp = (CACHE / f"og_oof_{panel}.npz", CACHE / f"stid_oof_{panel}.npz",
                 CACHE / f"pseudo_oof_{panel}.npz")
    if not (f.exists() and fs.exists() and fp.exists()):
        return None
    z, zp = np.load(f), np.load(fp)
    P = {"gs1": z["gs1"].astype(np.float32), "tfm8": z["tfm8"].astype(np.float32),
         "cnn3": z["cnn3"].astype(np.float32), "occ_cnn": mean_of(z, "occ_cnn"),
         "occ_tfm": mean_of(z, "occ_tfm"), "stid": mean_of(np.load(fs), "stid")}
    old = 0.2 * P["gs1"] + 0.6 * P["tfm8"] + 0.2 * P["cnn3"]
    base = ((1 - W_PS) * old + W_PS * zp["pseudo"].astype(np.float32)).astype(np.float32)
    return global_features(P, base), z["y"], z["e"], base


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="s20")
    ap.add_argument("--rounds", type=int, default=400)
    args = ap.parse_args()
    gbm = lgb.Booster(model_file=str(LAB / "models" / "residual_lgb.txt"))

    keep = [i for i, n in enumerate(NAMES) if n not in EXTRA]
    X, Y, P0 = [], [], []
    for panel in C.T2_PANELS:
        r = oof_rows(panel)
        if r is None:
            continue
        F, y, e, base = r
        X.append(F[e][:, keep])
        Y.append(y[e].astype(np.float32))
        P0.append(base[e])
    X, Y, P0 = np.concatenate(X), np.concatenate(Y), np.concatenate(P0)
    print(f"refit without {len(EXTRA)} columns: {len(Y):,} OOF cells, {X.shape[1]} features", flush=True)
    par = {"objective": "l2", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 200,
           "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1,
           "num_threads": 8}
    lean = lgb.train(par, lgb.Dataset(X, label=Y - P0, feature_name=[NAMES[i] for i in keep]),
                     num_boost_round=args.rounds)

    fam = {}
    for i, panel in enumerate(C.T2_PANELS):
        sp = load_split(i, panel, args.tag)
        for s, D in sp.items():
            y, e = D["y"], D["e"]
            old, ps = D["old"], D["ps"]
            blend = D["base"]
            F = global_features(D["P"], blend)
            flat = F.reshape(-1, F.shape[-1])
            rh = gbm.predict(flat).reshape(blend.shape).astype(np.float32)
            rl = lean.predict(flat[:, keep]).reshape(blend.shape).astype(np.float32)
            step = lambda r: np.clip(blend + ETA * np.where(np.abs(r) >= GATE, r, 0.0), 0, 1)  # noqa: E731
            Fo = global_features(D["P"], old)
            rho = gbm.predict(Fo.reshape(-1, Fo.shape[-1])).reshape(old.shape).astype(np.float32)
            for nm, p in (("old (12 models)", old),
                          ("old + corrector", np.clip(old + ETA * np.where(np.abs(rho) >= GATE, rho, 0.0), 0, 1)),
                          ("blend (+ pseudo)", blend),
                          ("blend + corrector = AS", step(rh)),
                          ("blend + lean corrector", step(rl)),
                          ("pseudo alone", ps)):
                fam.setdefault((s, nm), {})[panel] = C.iou(p >= TAU, y, e).mean()
        print(f"  {panel}", flush=True)

    names = ["old (12 models)", "old + corrector", "pseudo alone", "blend (+ pseudo)",
             "blend + corrector = AS", "blend + lean corrector"]
    print(f"\nongoing family IoU after tau 0.50, official-like pool of the scored months\n")
    print(f"{'stack':<26}" + "".join(f"{s:>13}" for s in C.SPLITS) + f"{'mean':>10}{'vs AS':>9}")
    ref = float(np.mean([C.family_mean(fam[(s, "blend + corrector = AS")]) for s in C.SPLITS]))
    for nm in names:
        v = [C.family_mean(fam[(s, nm)]) for s in C.SPLITS]
        m = float(np.mean(v))
        print(f"{nm:<26}" + "".join(f"{x:>13.4f}" for x in v) + f"{m:>10.4f}{m - ref:>+9.4f}")


if __name__ == "__main__":
    main()
