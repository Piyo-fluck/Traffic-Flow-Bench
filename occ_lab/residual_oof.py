"""The admissible version of the residual corrector: fitted on the held-out train tail, applied to both months.

`occ_lab/residual_diag.py` showed the idea works across months (+0.0079 / +0.0117 family IoU) but a corrector
fitted on April cannot be used on March. This fits it on the 2031-01/02 windows from `cache_oof.py` - days no
member was fitted on, and earlier than every scored origin - so ONE corrector is admissible for both months.

Reports, for each (weighting, eta, gate): family IoU on validation and private, and the rescue / spoil pair.

    python occ_lab/residual_oof.py
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
from residual_diag import NAMES, TAU, lg, mean_of  # noqa: E402

CACHE = LAB / "cache"
ETAS = (0.25, 0.50, 0.75, 1.00)
GATES = (0.0, 0.05, 0.10, 0.20)
ALPHAS = (0.0, 3.0, 9.0)        # sample weight 1 + alpha on the cells the frozen blend gets wrong


def panel_data(panel, tag):
    f = CACHE / (f"og_{tag}_{panel}.npz" if tag != "oof" else f"og_oof_{panel}.npz")
    fs = CACHE / (f"stid_{tag}_{panel}.npz" if tag != "oof" else f"stid_oof_{panel}.npz")
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
    return feat, y, e, base


def main() -> None:
    oof = {p: panel_data(p, "oof") for p in C.T2_PANELS}
    oof = {p: v for p, v in oof.items() if v is not None}
    scored = {s: {p: panel_data(p, s) for p in C.T2_PANELS} for s in C.SPLITS}
    scored = {s: {p: v for p, v in d.items() if v is not None} for s, d in scored.items()}
    X = np.concatenate([oof[p][0][oof[p][2]] for p in oof])
    y = np.concatenate([oof[p][1][oof[p][2]].astype(np.float32) for p in oof])
    p0 = np.concatenate([oof[p][3][oof[p][2]] for p in oof])
    wrong = (p0 >= TAU) != (y >= 0.5)
    print(f"out-of-fold training set: {len(y):,} cells from {len(oof)} panels, "
          f"{wrong.mean() * 100:.2f}% of them the frozen blend gets wrong", flush=True)
    for alpha in ALPHAS:
        w = 1.0 + alpha * wrong
        d = lgb.Dataset(X, label=y - p0, weight=w, feature_name=NAMES, free_raw_data=False)
        gbm = lgb.train({"objective": "l2", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 200,
                         "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1,
                         "num_threads": 8}, d, num_boost_round=400)
        imp = sorted(zip(gbm.feature_importance("gain"), NAMES), reverse=True)[:5]
        tot = sum(g for g, _ in imp)
        print(f"\n=== alpha {alpha:.0f} (weight on blend-wrong cells) ===")
        print("   top gain: " + ", ".join(f"{n} {g / tot * 100:.0f}%" for g, n in imp), flush=True)
        for s, data in scored.items():
            fam, res = {}, {}
            for panel, (feat, yy, ee, base) in data.items():
                rh = np.zeros(base.shape, np.float32)
                rh[ee] = gbm.predict(feat[ee]).astype(np.float32)
                fam.setdefault("blend", {})[panel] = C.iou(base >= TAU, yy, ee).mean()
                for eta in ETAS:
                    for g in GATES:
                        p = np.clip(base + eta * np.where(np.abs(rh) >= g, rh, 0.0), 0, 1)
                        fam.setdefault((eta, g), {})[panel] = C.iou(p >= TAU, yy, ee).mean()
                        bw = ((base >= TAU) != yy)[ee]
                        ww = ((p >= TAU) != yy)[ee]
                        a = res.setdefault((eta, g), [0, 0, 0, 0])
                        a[0] += int((bw & ~ww).sum()); a[1] += int(bw.sum())
                        a[2] += int((~bw & ww).sum()); a[3] += int((~bw).sum())
            b = C.family_mean(fam["blend"])
            best = sorted(((C.family_mean(v), k) for k, v in fam.items() if k != "blend"), reverse=True)
            print(f"   {s}: blend {b:.4f}")
            for v, k in best[:4]:
                a = res[k]
                print(f"      eta {k[0]:.2f} gate {k[1]:.2f}  {v:.4f} ({v - b:+.4f})  "
                      f"rescue {a[0] / max(a[1], 1):.3f}  spoil {a[2] / max(a[3], 1):.4f}")
            v, k = best[-1]
            print(f"      worst: eta {k[0]:.2f} gate {k[1]:.2f}  {v:.4f} ({v - b:+.4f})", flush=True)


if __name__ == "__main__":
    main()
