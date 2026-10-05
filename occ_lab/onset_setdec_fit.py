"""Fit the onset set decoder (notes 152-153) for production: stage 1 IoU-LambdaRank + swap, stage 2 marginal-utility
add. Seed-averaged (6 LightGBM seeds per stage) to remove the seed dependence seen in note 153.

1. check: the seed-averaged decoder, cross-month (fit one scored month, apply the other), vs note 145's swap
2. fit on both scored months: stage-1 rankers; stage-2 rows from each month's CROSS-MONTH stage-1 output (as in the
   check); stage-2 regressors on both months; the threshold from 5-fold out-of-fold predictions on both months
   (largest threshold within 0.001 of the best summed gain) - never tuned on the official windows
Saves occ_lab/models/onset_rank_s<k>.txt, onset_du_s<k>.txt, onset_du_tau.txt.

    .venv/Scripts/python.exe occ_lab/onset_setdec_fit.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta"))
sys.path.insert(0, str(LAB.parent))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

import bottleneck_cls as BC  # noqa: E402
import common as C  # noqa: E402
import onset_rerank as RR  # noqa: E402
import onset_stage2_du as S2  # noqa: E402

SEEDS = range(6)
MODELS = LAB / "models"
KEY = "Xn" if "--plus" in sys.argv else "Xp"       # note 155 (C): the ranker with note 152's new candidate evidence
TAG = ("_plus" if KEY == "Xn" else "") + __import__("os").environ.get("SETDEC_TAG", "")   # note 169: never overwrite production


def fit_rank(Ds, sd):
    lab = np.minimum((np.concatenate([D["U"] for D in Ds]) * 10).round().astype(int), 10)
    return lgb.train({**S2.PAR, "objective": "lambdarank", "seed": sd, "label_gain": list(range(11)),
                      "min_data_in_leaf": 20},
                     lgb.Dataset(np.concatenate([D[KEY] for D in Ds]), lab,
                                 group=np.concatenate([D["g"] for D in Ds])), 200)


def apply_rank(models, D):
    sc = np.mean([m.predict(D[KEY]) for m in models], 0)
    P1 = D["base"].copy()
    off = 0
    for (n, cand, reg, mass), g in zip(D["meta"], D["g"]):
        s_ = sc[off:off + g]
        P1[n] = BC.swap(D["base"][n], cand, reg, mass, s_ - s_.min() + 1e-3); off += g
    D["s1"], D["P1"] = sc, P1


def rows_of(Ts):
    X = np.array([f for T in Ts for _, rows in T.values() for f, _, _ in rows], np.float32)
    y = np.array([dU for T in Ts for _, rows in T.values() for _, dU, _ in rows], np.float32)
    win = np.array([k for k, (T, n) in enumerate((T, n) for T in Ts for n in T) for _ in T[n][1]])
    return X, y, win


def choose_tau(X, y, win):
    rng = np.random.default_rng(0)
    fold = rng.integers(0, 5, win.max() + 1)[win]
    oof = np.zeros(len(y))
    for k in range(5):
        oof[fold == k] = np.mean([lgb.train({**S2.PAR, "objective": "regression", "seed": sd},
                                            lgb.Dataset(X[fold != k], y[fold != k]), 150).predict(X[fold == k])
                                  for sd in SEEDS], 0)
    gain = {}
    for tau in S2.TAUS:
        g = 0.0
        for w in np.unique(win):
            j = np.flatnonzero(win == w)
            b = j[np.argmax(oof[j])]
            if oof[b] > tau:
                g += y[b]
        gain[tau] = g
    best = max(gain.values())
    return (max(t for t in S2.TAUS if gain[t] >= best - 1e-3) if best > 0 else np.inf), gain


def main() -> None:
    warnings.simplefilter("ignore")
    data = RR.prepare()
    WW = {}
    # ---- 1. cross-month check of the seed-averaged decoder
    for s_fit, s_app in (("validation", "private"), ("private", "validation")):
        rk = [fit_rank([d[s_fit] for d in data.values() if s_fit in d], sd) for sd in SEEDS]
        for d in data.values():
            if s_app in d:
                apply_rank(rk, d[s_app])
    T = {p_: {s: S2.pairs(D) for s, D in d.items()} for p_, d in data.items()}
    for s_fit, s_app in (("validation", "private"), ("private", "validation")):
        X, y, win = rows_of([T[p_][s_fit] for p_ in T if s_fit in T[p_]])
        tau, _ = choose_tau(X, y, win)
        du = [lgb.train({**S2.PAR, "objective": "regression", "seed": sd}, lgb.Dataset(X, y), 150) for sd in SEEDS]
        f145, f1, f2 = {}, {}, {}
        for p_, d in data.items():
            if s_app not in d:
                continue
            D = d[s_app]
            on = D["on"]
            i1, i2 = [], []
            for n in range(len(on["d"])):
                if n in T[p_][s_app]:
                    ib, rows = T[p_][s_app][n]
                    new = ib
                    if rows:
                        pr = np.mean([m.predict(np.array([f for f, _, _ in rows], np.float32)) for m in du], 0)
                        j = int(np.argmax(pr))
                        if pr[j] > tau:
                            new = ib + rows[j][1]
                else:
                    ib = new = S2.iou_of(C.select(D["P1"][n], "ratio", 1.0), on["fut"][n, -1], on["efut"][n, -1])
                i1.append(ib); i2.append(new)
            f1[p_], f2[p_] = np.mean(i1), np.mean(i2)
            WW.setdefault(s_app, {})[p_] = (np.array(i1), np.array(i2), C.onset_iou(D["base"], on))
            # note 145's classifier + swap, for reference (seed 0)
        print(f"  applied to {s_app:<10} stage 1 {C.family_mean(f1):.4f}  + stage 2 {C.family_mean(f2):.4f} "
              f"(+{C.family_mean(f2) - C.family_mean(f1):.4f}, tau {tau})", flush=True)
    ref = {s: C.family_mean({p_: C.onset_iou(d[s]["base"], d[s]["on"]).mean() for p_, d in data.items() if s in d})
           for s in C.SPLITS}
    print(f"  production (no swap): " + "  ".join(f"{s} {ref[s]:.4f}" for s in C.SPLITS))
    for s_ in C.SPLITS:                                   # note 169: window-weighted too, and per panel
        a = WW[s_]
        print(f"  window-weighted {s_:<10} no swap {np.concatenate([v[2] for v in a.values()]).mean():.4f}  stage 1 "
              f"{np.concatenate([v[0] for v in a.values()]).mean():.4f}  stage 2 {np.concatenate([v[1] for v in a.values()]).mean():.4f} | "
              + " ".join(f"{k}:{v[2].mean():.3f}/{v[1].mean():.3f}" for k, v in a.items()))
    np.save(MODELS / f"setdec_check{TAG}.npy", np.array([WW], dtype=object), allow_pickle=True)
    if __import__("os").environ.get("CHECK_ONLY", "") == "1":
        return
    # ---- 2. production fit on both months
    MODELS.mkdir(exist_ok=True)
    allD = [D for d in data.values() for D in d.values()]
    for sd in SEEDS:
        fit_rank(allD, sd).save_model(str(MODELS / f"onset_rank{TAG}_s{sd}.txt"))
    Ts = [T[p_][s] for p_ in T for s in T[p_]]          # stage-2 rows from the cross-month stage 1 (as checked)
    X, y, win = rows_of(Ts)
    tau, gain = choose_tau(X, y, win)
    for sd in SEEDS:
        lgb.train({**S2.PAR, "objective": "regression", "seed": sd}, lgb.Dataset(X, y), 150).save_model(
            str(MODELS / f"onset_du{TAG}_s{sd}.txt"))
    (MODELS / f"onset_du{TAG}_tau.txt").write_text(f"{tau}\n")
    print(f"saved: rankers and dU regressors x{len(SEEDS)}; stage-2 rows {len(y)}; threshold {tau} "
          f"(OOF gain by threshold: {', '.join(f'{k}: {v:+.2f}' for k, v in gain.items())})")


if __name__ == "__main__":
    main()
