"""Cheap diagnostic for the residual-corrector branch (the user's step 4).

Instead of asking a new member to predict queues, ask a small model to predict what the existing ensemble got
WRONG, and add that back:

    r = y - p0        with p0 the frozen production blend (0.20 gs1 + 0.60 tfm8 + 0.20 cnn3)
    r_hat = f(member probabilities, disagreement, horizon, position)
    p_final = clip(p0 + eta * r_hat, 0, 1)

If this cannot beat the blend, the residual-STID is not worth building; if it can, the same target trained with a
STID body and the harm-aware loss is the next step.

Out-of-fold by construction: the corrector is fitted on ONE scored month and applied to the OTHER, and the base
members saw neither (they are fitted on the train period up to 2031-01-01). Only the March -> April direction is
admissible for a submission - April is after every March origin, so an April-fitted corrector could not be used on
March. The reverse direction is reported as a stability check only; a production corrector for March would have to
be fitted on out-of-fold train-period windows.

    python occ_lab/residual_diag.py
"""
from __future__ import annotations

import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta"))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

import common as C  # noqa: E402

CACHE = LAB / "cache"
TAU = 0.50
ETAS = (0.25, 0.50, 0.75, 1.00)
GATES = (0.0, 0.05, 0.10, 0.20)          # apply the correction only where |r_hat| exceeds this


def mean_of(z, prefix):
    ks = [k for k in z.files if k.startswith(prefix)]
    return np.mean([z[k].astype(np.float32) for k in ks], 0) if ks else None


def lg(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p)).astype(np.float32)


def panel_data(panel, split):
    f, fs = CACHE / f"og_{split}_{panel}.npz", CACHE / f"stid_{split}_{panel}.npz"
    if not (f.exists() and fs.exists()):
        return None
    z = np.load(f)
    y, e = z["y"], z["e"]
    P = {"gs1": z["gs1"].astype(np.float32), "tfm8": z["tfm8"].astype(np.float32),
         "cnn3": z["cnn3"].astype(np.float32), "occ_cnn": mean_of(z, "occ_cnn"),
         "occ_tfm": mean_of(z, "occ_tfm"), "stid": mean_of(np.load(fs), "stid")}
    base = 0.2 * P["gs1"] + 0.6 * P["tfm8"] + 0.2 * P["cnn3"]
    N, Fh, S = base.shape
    hor = np.broadcast_to(np.arange(Fh, dtype=np.float32)[None, :, None], base.shape)
    pos = np.broadcast_to((np.arange(S, dtype=np.float32) / max(S - 1, 1))[None, None, :], base.shape)
    members = np.stack([P[k] for k in ("gs1", "tfm8", "cnn3", "occ_cnn", "occ_tfm", "stid")])
    feat = np.stack([base, lg(base), np.abs(base - 0.5), members.std(0), members.max(0) - members.min(0),
                     P["gs1"], P["tfm8"], P["cnn3"], P["occ_cnn"], P["occ_tfm"], P["stid"],
                     lg(P["stid"]) - lg(base), lg(P["occ_cnn"]) - lg(base), hor, pos], -1)
    return feat[e], y[e].astype(np.float32), base[e], (base, y, e, feat)


NAMES = ["p_blend", "logit_blend", "margin", "member_std", "member_range", "gs1", "tfm8", "cnn3", "occ_cnn",
         "occ_tfm", "stid", "d_logit_stid", "d_logit_occ_cnn", "horizon", "position"]


def fit(X, y, p0, wrong_only):
    r = y - p0
    m = ((p0 >= TAU) != (y >= 0.5)) if wrong_only else np.ones(len(y), bool)
    d = lgb.Dataset(X[m], label=r[m], feature_name=NAMES, free_raw_data=False)
    return lgb.train({"objective": "l2", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 200,
                      "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1,
                      "num_threads": 8}, d, num_boost_round=400)


def main() -> None:
    data = {s: {p: panel_data(p, s) for p in C.T2_PANELS} for s in C.SPLITS}
    data = {s: {p: v for p, v in d.items() if v is not None} for s, d in data.items()}
    for src, dst in (("validation", "private"), ("private", "validation")):
        X = np.concatenate([data[src][p][0] for p in data[src]])
        y = np.concatenate([data[src][p][1] for p in data[src]])
        p0 = np.concatenate([data[src][p][2] for p in data[src]])
        print(f"\n=== corrector fitted on {src} ({len(y):,} eligible cells), applied to {dst} ===", flush=True)
        for wrong_only in (False, True):
            gbm = fit(X, y, p0, wrong_only)
            tag = "base-wrong cells only" if wrong_only else "all cells"
            imp = sorted(zip(gbm.feature_importance("gain"), NAMES), reverse=True)[:5]
            print(f"  [{tag}] top gain: " + ", ".join(f"{n} {g / sum(i[0] for i in imp) * 100:.0f}%" for g, n in imp),
                  flush=True)
            fam = {}
            rescue = {}
            for panel, (_, _, _, (base, yy, ee, feat)) in data[dst].items():
                rh = np.zeros(base.shape, np.float32)
                rh[ee] = gbm.predict(feat[ee]).astype(np.float32)
                fam.setdefault("blend (T)", {})[panel] = C.iou(base >= TAU, yy, ee).mean()
                for eta in ETAS:
                    for g in GATES:
                        use = np.abs(rh) >= g
                        p = np.clip(base + eta * np.where(use, rh, 0.0), 0, 1)
                        fam.setdefault(f"eta {eta:.2f} gate {g:.2f}", {})[panel] = C.iou(p >= TAU, yy, ee).mean()
                        if eta == 1.0:
                            bw = ((base >= TAU) != yy)[ee]
                            w = ((p >= TAU) != yy)[ee]
                            a = rescue.setdefault(g, [0, 0, 0, 0])
                            a[0] += int((bw & ~w).sum()); a[1] += int(bw.sum())
                            a[2] += int((~bw & w).sum()); a[3] += int((~bw).sum())
            b = C.family_mean(fam["blend (T)"])
            best = sorted(((C.family_mean(d), n) for n, d in fam.items() if n != "blend (T)"), reverse=True)
            print(f"     blend {b:.4f}; best 6:")
            for v, n in best[:6]:
                print(f"       {n:<24} {v:.4f} ({v - b:+.4f})")
            print("       at eta 1.00: " + "  ".join(
                f"gate {g:.2f} rescue {a[0] / max(a[1], 1):.3f} spoil {a[2] / max(a[3], 1):.4f}"
                for g, a in sorted(rescue.items())), flush=True)


if __name__ == "__main__":
    main()
