"""Refit the residual corrector on the p0 candidate AR actually uses, along the adaptation's causal trajectory.

The shipped corrector (`occ_lab/models/residual_lgb.txt`) was fitted with

    p0 = 0.20 gs1 + 0.60 tfm8 + 0.20 cnn3

and AR applies it to `0.40 * that + 0.60 * p_PS`. Five of the fifteen features are functions of p0, so the
corrector runs out of distribution. And p_PS is adapted per day, so its own distribution moves: the refit has to
use the trajectory, which `cache_pseudo_oof.py` reproduces on the held-out tail.

Four configurations are compared on both scored months, after the decision threshold, which is the only
comparison that matters:

    none        AR's blend with no corrector
    shipped     AR's blend with the corrector fitted on the OLD p0 (what AR submitted)
    refit       the same feature set, fitted on AR's p0 along the trajectory
    refit16     refit plus p_PS itself as a sixteenth feature

    python occ_lab/fit_corrector_ar.py --tag s20
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
from residual_diag import NAMES, TAU, lg  # noqa: E402

CACHE = LAB / "cache"
W_PS = 0.60
ETA, GATE = 0.75, 0.10


def mean_of(z, prefix):
    ks = [k for k in z.files if k.startswith(prefix)]
    return np.mean([z[k].astype(np.float32) for k in ks], 0) if ks else None


def features(P, base, extra=None):
    N, Fh, S = base.shape
    hor = np.broadcast_to(np.arange(Fh, dtype=np.float32)[None, :, None], base.shape)
    pos = np.broadcast_to((np.arange(S, dtype=np.float32) / max(S - 1, 1))[None, None, :], base.shape)
    members = np.stack([P[k] for k in ("gs1", "tfm8", "cnn3", "occ_cnn", "occ_tfm", "stid")])
    cols = [base, lg(base), np.abs(base - 0.5), members.std(0), members.max(0) - members.min(0),
            P["gs1"], P["tfm8"], P["cnn3"], P["occ_cnn"], P["occ_tfm"], P["stid"],
            lg(P["stid"]) - lg(base), lg(P["occ_cnn"]) - lg(base), hor, pos]
    if extra is not None:
        cols.append(extra)
    return np.stack(cols, -1)


def oof_panel(panel):
    f, fs, fp = (CACHE / f"og_oof_{panel}.npz", CACHE / f"stid_oof_{panel}.npz",
                 CACHE / f"pseudo_oof_{panel}.npz")
    if not (f.exists() and fs.exists() and fp.exists()):
        return None
    z, zp = np.load(f), np.load(fp)
    assert z["y"].shape == zp["y"].shape and (z["y"] == zp["y"]).all(), f"{panel}: OOF windows differ"
    P = {"gs1": z["gs1"].astype(np.float32), "tfm8": z["tfm8"].astype(np.float32),
         "cnn3": z["cnn3"].astype(np.float32), "occ_cnn": mean_of(z, "occ_cnn"),
         "occ_tfm": mean_of(z, "occ_tfm"), "stid": mean_of(np.load(fs), "stid")}
    ps = zp["pseudo"].astype(np.float32)
    old = 0.2 * P["gs1"] + 0.6 * P["tfm8"] + 0.2 * P["cnn3"]
    base = (1 - W_PS) * old + W_PS * ps
    return P, base, ps, z["y"], z["e"]


def scored_panel(i, panel, tag):
    ctx, stat, Wd = C.windows(i, panel)
    out = {}
    for s in C.SPLITS:
        og, ref = Wd[s]["og"], Wd[s]["ref"]
        fm, fp = CACHE / f"ogmem_{s}_{panel}.npz", CACHE / f"pseudo_{tag}_{s}_{panel}.npz"
        if not len(og["d"]) or not (fm.exists() and fp.exists()):
            continue
        mem = np.load(fm)
        P = {"gs1": ref["og_gs1"], "tfm8": ref["og_tfm8"], "cnn3": ref["og_cnn3"],
             "occ_cnn": mem["occ_cnn"], "occ_tfm": mem["occ_tfm"], "stid": mem["stid"]}
        ps = np.load(fp)["og"].astype(np.float32)
        old = 0.2 * P["gs1"] + 0.6 * P["tfm8"] + 0.2 * P["cnn3"]
        out[s] = (P, (1 - W_PS) * old + W_PS * ps, ps, og["fut"], og["efut"])
    ctx.z = None
    return out


def apply_corr(gbm, P, base, ps, extra):
    feat = features(P, base, ps if extra else None)
    rh = gbm.predict(feat.reshape(-1, feat.shape[-1])).reshape(base.shape).astype(np.float32)
    return np.clip(base + ETA * np.where(np.abs(rh) >= GATE, rh, 0.0), 0, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="s20")
    ap.add_argument("--rounds", type=int, default=400)
    args = ap.parse_args()
    oof = {p: oof_panel(p) for p in C.T2_PANELS}
    oof = {p: v for p, v in oof.items() if v is not None}
    print(f"OOF panels with a full trajectory: {len(oof)}", flush=True)
    if not oof:
        raise SystemExit("no pseudo_oof caches yet")
    par = {"objective": "l2", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 200,
           "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "num_threads": 8}
    fits = {}
    for extra in (False, True):
        X = np.concatenate([features(P, b, ps if extra else None)[e] for P, b, ps, y, e in oof.values()])
        y = np.concatenate([yy[e].astype(np.float32) for P, b, ps, yy, e in oof.values()])
        p0 = np.concatenate([b[e] for P, b, ps, yy, e in oof.values()])
        nm = NAMES + (["pseudo"] if extra else [])
        print(f"  fit {'refit16' if extra else 'refit':<8} {len(y):,} cells, blend wrong on "
              f"{(((p0 >= TAU) != (y >= 0.5)).mean() * 100):.2f}%", flush=True)
        g = lgb.train(par, lgb.Dataset(X, label=y - p0, feature_name=nm, free_raw_data=False),
                      num_boost_round=args.rounds)
        out = LAB / "models" / (f"residual_lgb_ar{'16' if extra else ''}.txt")
        g.save_model(str(out))
        fits["refit16" if extra else "refit"] = (g, extra)
        print(f"  saved {out.name}", flush=True)
    shipped = lgb.Booster(model_file=str(LAB / "models" / "residual_lgb.txt"))

    fam = {}
    for i, panel in enumerate(C.T2_PANELS):
        d = scored_panel(i, panel, args.tag)
        for s, (P, base, ps, y, e) in d.items():
            fam.setdefault((s, "none"), {})[panel] = C.iou(base >= TAU, y, e).mean()
            fam.setdefault((s, "shipped"), {})[panel] = C.iou(
                apply_corr(shipped, P, base, ps, False) >= TAU, y, e).mean()
            for nm, (g, extra) in fits.items():
                fam.setdefault((s, nm), {})[panel] = C.iou(
                    apply_corr(g, P, base, ps, extra) >= TAU, y, e).mean()
        print(f"  {panel}", flush=True)

    print(f"\nongoing family IoU after the threshold (AR's blend, w_PS {W_PS}, eta {ETA}, gate {GATE})")
    print(f"{'corrector':<12}" + "".join(f"{s:>13}" for s in C.SPLITS) + f"{'mean':>10}")
    base_v = None
    for nm in ("none", "shipped", "refit", "refit16"):
        if (C.SPLITS[0], nm) not in fam:
            continue
        v = [C.family_mean(fam[(s, nm)]) for s in C.SPLITS]
        m = float(np.mean(v))
        if base_v is None:
            base_v = m
        print(f"{nm:<12}" + "".join(f"{x:>13.4f}" for x in v) + f"{m:>10.4f}   ({m - base_v:+.4f})")


if __name__ == "__main__":
    main()
