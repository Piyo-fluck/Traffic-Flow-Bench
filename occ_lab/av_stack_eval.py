"""The adaptation variants of occ_lab/adapt_variants.py in AU's ongoing stack (p_PS seeds 20 and 21 replaced).

The in-script baseline (variant 'base', note 106's recipe re-run with the same batches) replaces the same seeds, so
each row reads against it as well as against AU. All windows and days 1-5.

    .venv/Scripts/python.exe occ_lab/av_stack_eval.py base wise_0.25 wise_0.5 wise_0.75 ...
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
from refit_eval import fam_lab  # noqa: E402

CACHE = LAB / "cache"


def main() -> None:
    variants = sys.argv[1:]
    gbm = lgb.Booster(model_file=str(LAB / "models" / "residual_lgb.txt"))

    def corr(base, P):
        Fe = global_features(P, base)
        r = gbm.predict(Fe.reshape(-1, 15)).reshape(base.shape).astype(np.float32)
        return np.clip(base + ETA * np.where(np.abs(r) >= GATE, r, 0.0), 0, 1)

    fam, single = {}, {}
    for i, panel in enumerate(C.T2_PANELS):
        for s, D in load_split(i, panel, "ssl_w192ssl_avg4").items():
            P, y, e, d = D["P"], D["y"], D["e"], D["d"]
            tad = fam_lab("tfm", s, panel)
            ps = {k: np.load(CACHE / f"pseudo_ssl_t2_cnn_lov_w192_rec_ssl30k_s{k}_{s}_{panel}.npz")["og"].astype(np.float32)
                  for k in (20, 21, 22, 23)}
            old = 0.2 * P["gs1"] + 0.6 * tad + 0.2 * P["cnn3"]
            V = {"AU": corr(0.4 * old + 0.6 * np.mean(list(ps.values()), 0), P)}
            for v in variants:
                X = {k: np.load(CACHE / f"fam_av_s{k}_{v}_{s}_{panel}.npz")["og"].astype(np.float32) for k in (20, 21)}
                for k in (20, 21):
                    for sub, msk in (("all", slice(None)), ("days 1-5", d <= 4)):
                        single.setdefault((s, v, sub), {})[f"{panel}{k}"] = C.iou(X[k] >= 0.5, y, e)[msk].mean()
                V[v] = corr(0.4 * old + 0.6 * np.mean([X[20], X[21], ps[22], ps[23]], 0), P)
            for nm, q in V.items():
                sc = C.iou(q >= 0.5, y, e)
                fam.setdefault((s, nm, "all"), {})[panel] = sc.mean()
                if (d <= 4).any():
                    fam.setdefault((s, nm, "days 1-5"), {})[panel] = sc[d <= 4].mean()
        print(f"  {panel}", flush=True)
    for sub in ("all", "days 1-5"):
        print(f"\n=== {sub}: single member (mean of s20, s21) | in AU's stack (vs AU, vs base) ===")
        ref = np.mean([C.family_mean(fam[(s, "AU", sub)]) for s in C.SPLITS])
        base = np.mean([C.family_mean(fam[(s, "base", sub)]) for s in C.SPLITS]) if "base" in variants else ref
        for v in variants:
            sv = [np.mean(list(single[(s, v, sub)].values())) for s in C.SPLITS]
            st = [C.family_mean(fam[(s, v, sub)]) for s in C.SPLITS]
            print(f"  {v:<16} single {sv[0]:.4f} / {sv[1]:.4f}   stack {st[0]:.4f} / {st[1]:.4f}  "
                  f"({np.mean(st) - ref:+.4f} vs AU, {np.mean(st) - base:+.4f} vs base)")


if __name__ == "__main__":
    main()
