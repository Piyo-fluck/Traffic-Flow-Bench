"""note 171: does the CBM second opinion still help ON TOP of BB's onset set decoder, under the official windows'
condition? As setdec_early_check.py's production-like variant P (set decoder fitted on full-adaptation month A,
applied to the early-K month B), with the run evidence restricted to the K earlier days (ONSET_SHARE_K=1):

    setdec        stage 1 (ranker swap) + stage 2 (second region)
    setdec+CBM    after stage 1: rank = region mass of P1 x CBM region max (8 seeds, lambda 1), swap where the top
                  changes; the stage-1 score is re-ordered so stage 2 sees the new top; then stage 2

    ONSET_SHARE_K=1 .venv/Scripts/python.exe occ_lab/cbm_setdec_check.py
"""
import os
import sys
import warnings
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB)); sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta")); sys.path.insert(0, str(LAB.parent))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

import bottleneck_cls as BC  # noqa: E402
import common as C  # noqa: E402
import onset_stage2_du as S2  # noqa: E402
from onset_setdec_fit import SEEDS, fit_rank, apply_rank, rows_of, choose_tau  # noqa: E402
from setdec_early_check import prep  # noqa: E402

warnings.simplefilter("ignore")
CA = LAB / "cache"
CBM = os.environ.get("CBM_TAGS", ",".join(f"cbm_cbm_rf_s{k}" for k in range(8))).split(",")
LAM = float(os.environ.get("CBM_LAM", "1.0"))


def cbm_prob(split, panel):
    return np.mean([np.load(CA / f"cbm_{t}_{split}_{panel}.npz")["prob"] for t in CBM], 0)


def cbm_swap(D, split, panel):
    """apply the CBM second opinion to D['P1'] / D['s1'] in place; returns the number of windows changed."""
    pr = cbm_prob(split, panel)
    off, nch = 0, 0
    for (n, cand, reg, mass), g in zip(D["meta"], D["g"]):
        s_ = D["s1"][off:off + g]
        p1 = D["P1"][n]
        m1 = np.bincount(reg, weights=p1, minlength=len(cand))
        cb = np.array([pr[n][reg == r].max() for r in range(len(cand))])
        rk = m1 / max(m1.sum(), 1e-9) * np.maximum(cb, 1e-6) ** LAM
        q = BC.swap(p1, cand, reg, m1, rk)
        if not np.array_equal(q, p1):
            nch += 1
            D["P1"][n] = q
            a, b = int(np.argmax(s_)), int(np.argmax(rk))
            s_ = s_.copy(); s_[a], s_[b] = s_[b], s_[a]
            D["s1"][off:off + g] = s_
        off += g
    return nch


def decode(data_fit, s_fit, data_app, s_app, use_cbm):
    rk = [fit_rank([d[s_fit] for d in data_fit.values() if s_fit in d], sd) for sd in SEEDS]
    for d in data_fit.values():
        if s_fit in d:
            other = [x for x in C.SPLITS if x != s_fit][0]
            rk_o = [fit_rank([dd[other] for dd in data_fit.values() if other in dd], sd) for sd in SEEDS]
            apply_rank(rk_o, d[s_fit])
    Tf = {p_: S2.pairs(d[s_fit]) for p_, d in data_fit.items() if s_fit in d}
    X, y, win = rows_of(list(Tf.values()))
    tau, _ = choose_tau(X, y, win)
    du = [lgb.train({**S2.PAR, "objective": "regression", "seed": sd}, lgb.Dataset(X, y), 150) for sd in SEEDS]
    b, i2, nch = [], [], 0
    for p_, d in data_app.items():
        if s_app not in d:
            continue
        D = d[s_app]; on = D["on"]
        apply_rank(rk, D)
        if use_cbm:
            nch += cbm_swap(D, s_app, p_)
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
                new = S2.iou_of(C.select(D["P1"][n], "ratio", 1.0), on["fut"][n, -1], on["efut"][n, -1])
            i2.append(new)
    return np.concatenate(b), np.array(i2), nch


def main():
    full = prep(None)
    res = {}
    for k in range(5):
        early = prep(k)
        for s_fit, s_app in (("validation", "private"), ("private", "validation")):
            b, sd, _ = decode(full, s_fit, early, s_app, False)
            _, sc, nch = decode(full, s_fit, early, s_app, True)
            res.setdefault(s_app, []).append((b, sd, sc))
            print(f"K {k} {s_app:<10} base {b.mean():.4f}  setdec {sd.mean():.4f}  setdec+CBM {sc.mean():.4f}"
                  f"  (CBM changed {nch} windows)", flush=True)
    for s_app, L in res.items():
        b, sd, sc = (np.mean([x[j] for x in L], 0) for j in range(3))
        n = len(b)
        for nm, a, r in (("setdec vs base", sd, b), ("setdec+CBM vs base", sc, b), ("setdec+CBM vs setdec", sc, sd)):
            dd = a - r
            print(f"MEAN over K {s_app:<10} {nm:<22} {dd.mean():+.4f} (SE {dd.std(ddof=1) / np.sqrt(n):.4f})")


if __name__ == "__main__":
    main()
