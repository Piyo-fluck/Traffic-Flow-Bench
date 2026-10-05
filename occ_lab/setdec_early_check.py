"""note 169: does the onset set decoder still help under the official windows' condition (0-4 days of adaptation)?
For K in 0..4 (occ_lab/pseudo_adapt.py --hist K caches) and both month directions:
  base      the BB blend, ratio rule, no set decoder (early-K members)
  P         stage 1 + 2 fitted on the FULL-adaptation month A (as production is), applied to early-K month B
  E         fitted on early-K month A, applied to early-K month B
Window-weighted onset IoU on month B, averaged over K; per-K lines too."""
import os
import sys
import warnings
from pathlib import Path
LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB)); sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta")); sys.path.insert(0, str(LAB.parent))
import lightgbm as lgb
import numpy as np
import common as C
import onset_rerank as RR
import onset_stage2_du as S2
from onset_setdec_fit import SEEDS, fit_rank, apply_rank, rows_of, choose_tau
warnings.simplefilter("ignore")


def prep(k):
    if k is None:
        os.environ.pop("ONSET_EARLY_K", None)
    else:
        os.environ["ONSET_EARLY_K"] = str(k)
    return RR.prepare()


def decode(data_fit, s_fit, data_app, s_app):
    """fit stage 1 + 2 on data_fit[s_fit], apply to data_app[s_app]; per-window IoU arrays (base, stage 1, stage 2)"""
    rk = [fit_rank([d[s_fit] for d in data_fit.values() if s_fit in d], sd) for sd in SEEDS]
    for d in data_fit.values():                       # stage-2 rows on the fit month need its cross-month stage 1
        if s_fit in d:
            other = [x for x in C.SPLITS if x != s_fit][0]
            rk_o = [fit_rank([dd[other] for dd in data_fit.values() if other in dd], sd) for sd in SEEDS]
            apply_rank(rk_o, d[s_fit])
    Tf = {p_: {s_fit: S2.pairs(d[s_fit])} for p_, d in data_fit.items() if s_fit in d}
    X, y, win = rows_of([Tf[p_][s_fit] for p_ in Tf])
    tau, _ = choose_tau(X, y, win)
    du = [lgb.train({**S2.PAR, "objective": "regression", "seed": sd}, lgb.Dataset(X, y), 150) for sd in SEEDS]
    for d in data_app.values():
        if s_app in d:
            apply_rank(rk, d[s_app])
    b, i1, i2 = [], [], []
    for p_, d in data_app.items():
        if s_app not in d:
            continue
        D = d[s_app]; on = D["on"]
        T = S2.pairs(D)
        b.append(C.onset_iou(D["base"], on))
        for n in range(len(on["d"])):
            if n in T:
                ib, rows = T[n]; new = ib
                if rows:
                    pr = np.mean([m.predict(np.array([f for f, _, _ in rows], np.float32)) for m in du], 0)
                    j = int(np.argmax(pr))
                    if pr[j] > tau:
                        new = ib + rows[j][1]
            else:
                ib = new = S2.iou_of(C.select(D["P1"][n], "ratio", 1.0), on["fut"][n, -1], on["efut"][n, -1])
            i1.append(ib); i2.append(new)
    return np.concatenate(b), np.array(i1), np.array(i2)


def main():
    full = prep(None)
    res = {}
    for k in [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "0,1,2,3,4").split(",")]:
        early = prep(k)
        for s_fit, s_app in (("validation", "private"), ("private", "validation")):
            b, p1, p2 = decode(full, s_fit, early, s_app)
            _, e1, e2 = decode(early, s_fit, early, s_app)
            res.setdefault(s_app, []).append((b, p1, p2, e1, e2))
            print(f"K {k} applied to {s_app:<10} base {b.mean():.4f} | P stage1 {p1.mean():.4f} stage2 {p2.mean():.4f} | "
                  f"E stage1 {e1.mean():.4f} stage2 {e2.mean():.4f}", flush=True)
    for s_app, L in res.items():
        a = [np.mean([x[j] for x in L], 0) for j in range(5)]
        d_p, d_e = a[2] - a[0], a[4] - a[0]
        print(f"MEAN over K, {s_app:<10} base {a[0].mean():.4f} | P (production-like) {a[2].mean():.4f} ({d_p.mean():+.4f}, SE {d_p.std(ddof=1) / np.sqrt(len(d_p)):.4f}) | "
              f"E (refit early) {a[4].mean():.4f} ({d_e.mean():+.4f}, SE {d_e.std(ddof=1) / np.sqrt(len(d_e)):.4f})", flush=True)


if __name__ == "__main__":
    main()
