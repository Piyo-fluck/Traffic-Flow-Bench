"""Note 163 in production: AU's ongoing stack with p_PS rebuilt on the widened history mask (T2_OKALL=1).

p_PS (0.60 of AU's p0) = the 4-seed mean of the w192 SSL CNN after the pseudo-supervised adaptation (sup+rec);
the OKALL version is the same recipe with the history mask = 'has a value' in training, adaptation and scoring
(Ada: work/logs_ssl/run_ssl_okall.sh -> occ_lab/cache/pseudo_ssl_t2_cnn_lov_w192_rec_ssl30k_s<k>_okall_*).
The other members and the corrector are AU's. Single-seed comparisons use the matching production seed.

    .venv/Scripts/python.exe occ_lab/okall_stack_eval.py
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
import select_bank as SB  # noqa: E402
from online_adapter import ETA, GATE, global_features  # noqa: E402

CACHE = LAB / "cache"
SEEDS = (20, 21, 22, 23)


def ps_of(tag, s, panel):
    f = CACHE / f"pseudo_{tag}_{s}_{panel}.npz"
    return np.load(f)["og"].astype(np.float32) if f.exists() else None


def main() -> None:
    gbm = lgb.Booster(model_file=str(LAB / "models" / "residual_lgb.txt"))

    def corr(base, P):
        r = gbm.predict(global_features(P, base).reshape(-1, 15)).reshape(base.shape).astype(np.float32)
        return np.clip(base + ETA * np.where(np.abs(r) >= GATE, r, 0.0), 0, 1)

    fam, single = {}, {}
    for i, panel in enumerate(C.T2_PANELS):
        for s in C.SPLITS:
            B = SB.bank_for(i, panel, s, "solid3", "")
            yb, eb, d = B["y"] > 0.5, B["e"] > 0.5, B["d"]
            M0 = B["M"].copy()
            M0[3] = B["ps_ship"]
            prod = [ps_of(f"ssl_t2_cnn_lov_w192_rec_ssl30k_s{k}", s, panel) for k in SEEDS]
            okal = [ps_of(f"ssl_t2_cnn_lov_w192_rec_ssl30k_s{k}_okall", s, panel) for k in SEEDS]
            V = {"AU": M0}
            if all(x is not None for x in okal):
                M1 = M0.copy(); M1[3] = np.mean(okal, 0); V["AU, p_PS -> okall (4 seeds)"] = M1
            tf = [CACHE / f"tfmps_s{k}okall_sup0.0001_{s}_{panel}.npz" for k in range(3, 11)]
            if all(f.exists() for f in tf):
                t8 = np.mean([np.load(f)["og"].astype(np.float32) for f in tf], 0)
                M2 = M0.copy(); M2[1] = t8; V["AU, tfm8 -> okall"] = M2
                if "AU, p_PS -> okall (4 seeds)" in V:
                    M3 = M1.copy(); M3[1] = t8; V["AU, p_PS + tfm8 -> okall"] = M3
                cf = [C.CACHE / f"t2cnn_{t}_okall_{s}_{panel}.npz" for t in C.CNN3]
                if all(f.exists() for f in cf) and "AU, p_PS + tfm8 -> okall" in V:
                    c3 = np.mean([np.load(f)["og_p"].astype(np.float32) for f in cf], 0)
                    M4 = M3.copy(); M4[2] = c3; V["AU, p_PS + tfm8 + cnn3 -> okall"] = M4
                    for nm, x in (("cnn3 prod", B["P"]["cnn3"]), ("cnn3 okall", c3)):
                        sc = C.iou(x >= 0.5, yb, eb)
                        single.setdefault((s, nm, "all"), {})[panel] = sc.mean()
                        single.setdefault((s, nm, "d15"), {})[panel] = sc[d <= 4].mean()
                for nm, x in (("tfm8 prod", B["tfm_ship"]), ("tfm8 okall", t8)):
                    sc = C.iou(x >= 0.5, yb, eb)
                    single.setdefault((s, nm, "all"), {})[panel] = sc.mean()
                    single.setdefault((s, nm, "d15"), {})[panel] = sc[d <= 4].mean()
            for nm, M in V.items():
                q = corr(np.tensordot(SB.WAU, M, 1), B["P"])
                sc = C.iou(q >= 0.5, yb, eb)
                fam.setdefault((s, nm, "all"), {})[panel] = sc.mean()
                fam.setdefault((s, nm, "d15"), {})[panel] = sc[d <= 4].mean()
            for k, a, b in zip(SEEDS, prod, okal):
                for nm, x in ((f"prod s{k}", a), (f"okall s{k}", b)):
                    if x is None:
                        continue
                    sc = C.iou(x >= 0.5, yb, eb)
                    single.setdefault((s, nm, "all"), {})[panel] = sc.mean()
                    single.setdefault((s, nm, "d15"), {})[panel] = sc[d <= 4].mean()
        print(f"  {panel}", flush=True)
    for sub in ("all", "d15"):
        print(f"\n=== ongoing family IoU, {'all days' if sub == 'all' else 'days 1-5'} ===")
        for k in list(SEEDS) + ["tfm8", "cnn3"]:
            for nm in ((f"prod s{k}", f"okall s{k}") if k not in ("tfm8", "cnn3") else (f"{k} prod", f"{k} okall")):
                if all((s, nm, sub) in single for s in C.SPLITS):
                    print(f"  single {nm:<12}" + "".join(f"  {s[:4]} {C.family_mean(single[(s, nm, sub)]):.4f}" for s in C.SPLITS))
        ref = {s: C.family_mean(fam[(s, "AU", sub)]) for s in C.SPLITS}
        for nm in ("AU", "AU, p_PS -> okall (4 seeds)", "AU, tfm8 -> okall", "AU, p_PS + tfm8 -> okall",
                   "AU, p_PS + tfm8 + cnn3 -> okall"):
            if all((s, nm, sub) in fam for s in C.SPLITS):
                print(f"  stack {nm:<30}" + "".join(f"  {s[:4]} {C.family_mean(fam[(s, nm, sub)]):.4f} "
                                                    f"({C.family_mean(fam[(s, nm, sub)]) - ref[s]:+.4f})" for s in C.SPLITS))


if __name__ == "__main__":
    main()
