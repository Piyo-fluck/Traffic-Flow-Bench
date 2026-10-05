"""Idea A in AU's ongoing stack: oriented members replace their seeds, against identically fine-tuned controls.

AU: p0 = 0.40 (0.20 gs1 + 0.60 tfm8 + 0.20 cnn3) + 0.60 p_PS, tfm8 = the eight adapted transformers (s3..s10),
p_PS = the four SSL w192 CNNs (s20..s23), then the production corrector, tau 0.50.
    tfm oriented / control      s3..s6 of tfm8 replaced by occ_lab/orient_tfm.py's adapted models
    cnn oriented / control      s20, s21 of p_PS replaced by occ_lab/leap_t2.py's (--orient / ctrl) adapted models
    both                        the two replacements together
All windows and days 1-5 (where the official windows are).

    .venv/Scripts/python.exe occ_lab/orient_stack_eval.py
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
TFM = range(3, 11)
PS = (20, 21, 22, 23)


def ld(name, s, panel):
    return np.load(CACHE / f"{name}_{s}_{panel}.npz")["og"].astype(np.float32)


def main() -> None:
    gbm = lgb.Booster(model_file=str(LAB / "models" / "residual_lgb.txt"))

    def corr(base, P):
        Fe = global_features(P, base)
        r = gbm.predict(Fe.reshape(-1, 15)).reshape(base.shape).astype(np.float32)
        return np.clip(base + ETA * np.where(np.abs(r) >= GATE, r, 0.0), 0, 1)

    fam, single = {}, {}
    for i, panel in enumerate(C.T2_PANELS):
        for s, D in load_split(i, panel, "ssl_w192ssl_avg4").items():
            P, y, e, d = D["P"], D["y"], D["e"], D["d"]
            tf = {k: ld(f"tfmps_s{k}_sup0.0001", s, panel) for k in TFM}
            ps = {k: ld(f"pseudo_ssl_t2_cnn_lov_w192_rec_ssl30k_s{k}", s, panel) for k in PS}
            alt_t = {kind: {k: ld(f"fam_otfm_{kind}_s{k}_sup", s, panel) for k in range(3, 7)} for kind in ("o", "c")}
            alt_c = {kind: {k: ld(f"fam_leap_{tag}_s{k}_sup+rec", s, panel) for k in (20, 21)}
                     for kind, tag in (("o", "orient"), ("c", "ctrl"))}

            def stack(tf_rep=None, ps_rep=None):
                t = np.mean([(tf_rep or {}).get(k, tf[k]) for k in TFM], 0)
                p = np.mean([(ps_rep or {}).get(k, ps[k]) for k in PS], 0)
                return corr(0.4 * (0.2 * P["gs1"] + 0.6 * t + 0.2 * P["cnn3"]) + 0.6 * p, P)

            V = {"AU": stack(),
                 "tfm s3-6 control": stack(tf_rep=alt_t["c"]), "tfm s3-6 ORIENTED": stack(tf_rep=alt_t["o"]),
                 "cnn s20-21 control": stack(ps_rep=alt_c["c"]), "cnn s20-21 ORIENTED": stack(ps_rep=alt_c["o"]),
                 "both control": stack(alt_t["c"], alt_c["c"]), "both ORIENTED": stack(alt_t["o"], alt_c["o"])}
            for nm, q in V.items():
                sc = C.iou(q >= 0.5, y, e)
                fam.setdefault((s, nm, "all"), {})[panel] = sc.mean()
                if (d <= 4).any():
                    fam.setdefault((s, nm, "days 1-5"), {})[panel] = sc[d <= 4].mean()
            for kind in ("o", "c"):
                for k, v in alt_t[kind].items():
                    single.setdefault((s, f"tfm {kind} s{k}"), {})[panel] = C.iou(v >= 0.5, y, e).mean()
                for k, v in alt_c[kind].items():
                    single.setdefault((s, f"cnn {kind} s{k}"), {})[panel] = C.iou(v >= 0.5, y, e).mean()
        print(f"  {panel}", flush=True)
    print("\n=== single adapted members, ongoing (o = oriented, c = control) ===")
    for nm in sorted({k[1] for k in single}):
        v = [C.family_mean(single[(s, nm)]) for s in C.SPLITS]
        print(f"  {nm:<12} val {v[0]:.4f}  priv {v[1]:.4f}")
    for sub in ("all", "days 1-5"):
        print(f"\n=== ongoing family IoU in AU's stack, {sub} ===")
        ref = {s: C.family_mean(fam[(s, "AU", sub)]) for s in C.SPLITS}
        for nm in ("AU", "tfm s3-6 control", "tfm s3-6 ORIENTED", "cnn s20-21 control", "cnn s20-21 ORIENTED",
                   "both control", "both ORIENTED"):
            v = [C.family_mean(fam[(s, nm, sub)]) for s in C.SPLITS]
            print(f"  {nm:<22} val {v[0]:.4f} ({v[0] - ref['validation']:+.4f})  priv {v[1]:.4f} "
                  f"({v[1] - ref['private']:+.4f})")


if __name__ == "__main__":
    main()
