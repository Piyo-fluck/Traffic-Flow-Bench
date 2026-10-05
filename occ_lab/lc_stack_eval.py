"""The long-context CNN (occ_lab/longctx.py) in AU's ongoing stack, against its no-long control.

AU: p0 = 0.40 old + 0.60 p_PS, old = 0.20 gs1 + 0.60 tfm8(adapted) + 0.20 cnn3, p_PS = the four SSL w192 seeds
(s20..s23) pseudo-supervised adapted, then the production residual corrector, tau 0.50. The long-context model is
the same backbone with a long branch, so it enters p_PS seed-matched: its seeds replace the same seeds of p_PS.
The control (--no-long, the same fine-tuning without the branch) enters the same way, so the difference between
the two rows is the long context alone.

    .venv/Scripts/python.exe occ_lab/lc_stack_eval.py [lc | proceed]      ("long" rows = the new model)
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
PS_SEEDS = (20, 21, 22, 23)


def load(name, s, panel):
    f = CACHE / f"{name}_{s}_{panel}.npz"
    return np.load(f)["og"].astype(np.float32) if f.exists() else None


def ps_seed(k, s, panel):
    return load(f"pseudo_ssl_t2_cnn_lov_w192_rec_ssl30k_s{k}", s, panel)


FAMILY = "lc"


def lc(tag, s, panel, mode="sup+rec"):
    return load(f"fam_{FAMILY}_{tag}_{mode}", s, panel)


KINDS = {"lc": (("long", {20: "s20", 21: "s21", 22: "s22"}), ("ctrl", {20: "ctrl_s20", 21: "ctrl_s21"}),
                ("nopre", {20: "nopre_s20"})),
         "proceed": (("long", {20: "s20", 21: "s21"}), ("ctrl", {20: "ctrl_s20", 21: "ctrl_s21"})),
         "leap": tuple((k, {20: f"{k}_s20", 21: f"{k}_s21"}) for k in ("ctrl", "conf", "mrep", "both"))}


def main() -> None:
    global FAMILY
    if len(sys.argv) > 1:
        FAMILY = sys.argv[1]
    gbm = lgb.Booster(model_file=str(LAB / "models" / "residual_lgb.txt"))

    def corr(base, P):
        Fe = global_features(P, base)
        r = gbm.predict(Fe.reshape(-1, 15)).reshape(base.shape).astype(np.float32)
        return np.clip(base + ETA * np.where(np.abs(r) >= GATE, r, 0.0), 0, 1)

    fam, single = {}, {}
    for i, panel in enumerate(C.T2_PANELS):
        for s, D in load_split(i, panel, "ssl_w192ssl_avg4").items():
            tad = fam_lab("tfm", s, panel)
            P, ps, y, e, d = D["P"], D["ps"], D["y"], D["e"], D["d"]
            old = 0.2 * P["gs1"] + 0.6 * tad + 0.2 * P["cnn3"]
            V = {"AU": corr(0.4 * old + 0.6 * ps, P)}
            seeds = {k: ps_seed(k, s, panel) for k in PS_SEEDS}
            for kind, tags in KINDS[FAMILY]:
                got = {k: lc(t, s, panel) for k, t in tags.items()}
                got = {k: v for k, v in got.items() if v is not None}
                if not got:
                    continue
                for k, v in got.items():
                    single.setdefault((s, f"{kind} s{k}"), {})[panel] = C.iou(v >= 0.5, y, e).mean()
                    single.setdefault((s, f"PS s{k}"), {})[panel] = C.iou(seeds[k] >= 0.5, y, e).mean()
                ps2 = np.mean([got.get(k, seeds[k]) for k in PS_SEEDS], 0)
                V[f"{kind} replaces PS seeds {sorted(got)}"] = corr(0.4 * old + 0.6 * ps2, P)
                if kind in ("long", "both"):
                    g = np.mean(list(got.values()), 0)
                    V["long 0.10 of p0"] = corr(0.4 * old + 0.5 * ps + 0.1 * g, P)
                    V["long 0.20 of p0"] = corr(0.4 * old + 0.4 * ps + 0.2 * g, P)
            for nm, q in V.items():
                sc = C.iou(q >= 0.5, y, e)
                fam.setdefault((s, nm, "all"), {})[panel] = sc.mean()
                if (d <= 4).any():
                    fam.setdefault((s, nm, "early"), {})[panel] = sc[d <= 4].mean()
        print(f"  {panel}", flush=True)
    print("\n=== single models, sup+rec adapted, ongoing family IoU ===")
    for nm in sorted({k[1] for k in single}):
        if all((s, nm) in single for s in C.SPLITS):
            v = [C.family_mean(single[(s, nm)]) for s in C.SPLITS]
            print(f"  {nm:<14} val {v[0]:.4f}  priv {v[1]:.4f}  mean {np.mean(v):.4f}")
    names = sorted({k[1] for k in fam}, key=lambda n: (n != "AU", n))
    for sub in ("all", "early"):
        print(f"\n=== ongoing family IoU in AU's stack, {sub} windows ===")
        ref = np.mean([C.family_mean(fam[(s, "AU", sub)]) for s in C.SPLITS])
        for nm in names:
            if all((s, nm, sub) in fam and len(fam[(s, nm, sub)]) == len(fam[(s, "AU", sub)]) for s in C.SPLITS):
                v = [C.family_mean(fam[(s, nm, sub)]) for s in C.SPLITS]
                print(f"  {nm:<40} val {v[0]:.4f}  priv {v[1]:.4f}  mean {np.mean(v):.4f} ({np.mean(v) - ref:+.4f})")


if __name__ == "__main__":
    main()
