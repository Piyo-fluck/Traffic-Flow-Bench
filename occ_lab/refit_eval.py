"""Refit the residual corrector on ADAPTED members, and score the adapted stacks (notes 125-126).

The production corrector (occ_lab/models/residual_lgb.txt) was fitted on the OOF cache with p0 = old (no p_PS) and
frozen members as features; AU feeds it p0 = 0.40 old(adapted tfm8) + 0.60 p_PS and the frozen feature columns.
Here the OOF cache is rebuilt along the same causal trajectory for every adapted family (occ_lab/adapt_family.py
--mode oof, occ_lab/cache_pseudo_oof.py --tag _ssl_s*), so the corrector can be refitted on exactly the p0 and the
feature columns it is applied to.

    AU   old corrector, p0 with adapted tfm8, frozen feature columns          (submitted, LB 0.88635)
    AUc  AU with cnn3 adapted in p0 too (old corrector, frozen feature columns)
    R0   refitted corrector, same p0, frozen feature columns
    R1   refitted, same p0, adapted feature columns (tfm8, stid, occ_cnn, occ_tfm)
    R2   R1 with cnn3 adapted too (in p0 and in the features)

    .venv/Scripts/python.exe occ_lab/refit_eval.py
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
from online_adapter import ETA, GATE, global_features, load_split, mean_of  # noqa: E402

CACHE = LAB / "cache"
SEEDS = {"tfm": range(3, 11), "stid": range(4), "cnn3": range(3), "occ_cnn": range(4), "occ_tfm": range(4)}
W_PS = 0.60
PAR = {"objective": "l2", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 200, "feature_fraction": 0.8,
       "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "num_threads": 8}


def fam_oof(fam, panel):
    fs = [CACHE / f"fam_{fam}_s{k}_oof_{panel}.npz" for k in SEEDS[fam]]
    return np.mean([np.load(f)["og"].astype(np.float32) for f in fs], 0) if all(f.exists() for f in fs) else None


def fam_lab(fam, split, panel):
    if fam == "tfm":
        fs = [CACHE / f"tfmps_s{k}_sup0.0001_{split}_{panel}.npz" for k in SEEDS[fam]]
    elif fam == "stid":
        fs = [CACHE / f"stidps_s{k}_sup_{split}_{panel}.npz" for k in SEEDS[fam]]
    else:
        fs = [CACHE / f"fam_{fam}_s{k}_sup_{split}_{panel}.npz" for k in SEEDS[fam]]
    return np.mean([np.load(f)["og"].astype(np.float32) for f in fs], 0) if all(f.exists() for f in fs) else None


def compose(P, ps, ad, variant):
    """p0 and the feature dict for a variant; P frozen members, ad adapted ones (dict, may lack keys)."""
    cnn3 = ad["cnn3"] if variant in ("R2", "AUc") else P["cnn3"]
    old = 0.2 * P["gs1"] + 0.6 * ad["tfm"] + 0.2 * cnn3
    base = (1 - W_PS) * old + W_PS * ps
    if variant in ("AU", "R0", "AUc"):
        feats = P
    else:
        feats = dict(P, tfm8=ad["tfm"], stid=ad["stid"], occ_cnn=ad["occ_cnn"], occ_tfm=ad["occ_tfm"],
                     cnn3=cnn3)
    return base, feats


def main() -> None:
    # ---- OOF (train run 2031-01/02), adapted along the causal trajectory
    rows = {v: ([], [], []) for v in ("R0", "R1", "R2")}
    for panel in C.T2_PANELS:
        z = np.load(CACHE / f"og_oof_{panel}.npz")
        P = {"gs1": z["gs1"].astype(np.float32), "tfm8": z["tfm8"].astype(np.float32),
             "cnn3": z["cnn3"].astype(np.float32), "occ_cnn": mean_of(z, "occ_cnn"),
             "occ_tfm": mean_of(z, "occ_tfm"), "stid": mean_of(np.load(CACHE / f"stid_oof_{panel}.npz"), "stid")}
        pss = [CACHE / f"pseudo_oof_ssl_s{k}_{panel}.npz" for k in (20, 21, 22, 23)]
        ad = {f: fam_oof(f, panel) for f in SEEDS}
        if not all(p.exists() for p in pss) or any(v is None for v in ad.values()):
            print(f"  OOF {panel}: missing caches, skipped", flush=True)
            continue
        ps = np.mean([np.load(p)["pseudo"].astype(np.float32) for p in pss], 0)
        assert ps.shape == P["gs1"].shape and ad["tfm"].shape == ps.shape, "OOF windows do not align"
        y, e = z["y"], z["e"]
        for v in rows:
            base, feats = compose(P, ps, ad, v)
            Fe = global_features(feats, base)
            rows[v][0].append(Fe[e]); rows[v][1].append((y[e].astype(np.float32) - base[e])); rows[v][2].append(base[e])
    gbms = {}
    for v, (X, r, _) in rows.items():
        if not X:
            continue
        gbms[v] = lgb.train(PAR, lgb.Dataset(np.concatenate(X), label=np.concatenate(r)), num_boost_round=400)
        gbms[v].save_model(str(LAB / "models" / f"residual_lgb_{v}.txt"))
        print(f"refit {v}: {sum(len(a) for a in r):,} OOF cells", flush=True)
    old_gbm = lgb.Booster(model_file=str(LAB / "models" / "residual_lgb.txt"))

    # ---- scored months, official-like pool
    fam = {}
    for i, panel in enumerate(C.T2_PANELS):
        for s, D in load_split(i, panel, "ssl_w192ssl_avg4").items():
            ad = {f: fam_lab(f, s, panel) for f in SEEDS}
            P, ps, y, e, d = D["P"], D["ps"], D["y"], D["e"], D["d"]
            out = {"AT": (D["base"], P, old_gbm)}
            if ad["tfm"] is not None:
                out["AU"] = (*compose(P, ps, ad, "AU"), old_gbm)
                if ad["cnn3"] is not None:
                    out["AUc"] = (*compose(P, ps, ad, "AUc"), old_gbm)
                for v in gbms:
                    if all(ad[f] is not None for f in (["tfm"] if v == "R0" else list(SEEDS) if v == "R2"
                                                       else ["tfm", "stid", "occ_cnn", "occ_tfm"])):
                        out[v] = (*compose(P, ps, ad, v), gbms[v])
            for nm, (base, feats, g) in out.items():
                Fe = global_features(feats, base)
                rh = g.predict(Fe.reshape(-1, Fe.shape[-1])).reshape(base.shape).astype(np.float32)
                p = np.clip(base + ETA * np.where(np.abs(rh) >= GATE, rh, 0.0), 0, 1)
                sc = C.iou(p >= 0.5, y, e)
                fam.setdefault((s, nm, "all"), {})[panel] = sc.mean()
                if (d <= 4).any():
                    fam.setdefault((s, nm, "early"), {})[panel] = sc[d <= 4].mean()
        print(f"  {panel}", flush=True)
    for sub in ("all", "early"):
        print(f"\n=== ongoing family IoU, {sub} windows ===")
        ref = np.mean([C.family_mean(fam[(s, "AT", sub)]) for s in C.SPLITS])
        for nm in ("AT", "AU", "AUc", "R0", "R1", "R2"):
            if all((s, nm, sub) in fam and len(fam[(s, nm, sub)]) == len(fam[(s, "AT", sub)]) for s in C.SPLITS):
                v = [C.family_mean(fam[(s, nm, sub)]) for s in C.SPLITS]
                print(f"  {nm:<4} val {v[0]:.4f}  priv {v[1]:.4f}  mean {np.mean(v):.4f} ({np.mean(v) - ref:+.4f})")


if __name__ == "__main__":
    main()
