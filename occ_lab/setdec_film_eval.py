"""Note 165: BB's onset set decoder (notes 152-153) with the STID member replaced by the FiLM STID (recovered
supervision), re-fitted and evaluated cross-month exactly as onset_setdec_fit.py's check. ONSET_ST=film selects it.

    .venv/Scripts/python.exe occ_lab/setdec_film_eval.py            (plain STID, = BB)
    ONSET_ST=film .venv/Scripts/python.exe occ_lab/setdec_film_eval.py
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta"))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

import common as C  # noqa: E402
import onset_rerank as RR  # noqa: E402
import onset_setdec_fit as SF  # noqa: E402
import onset_stage2_du as S2  # noqa: E402


def main() -> None:
    warnings.simplefilter("ignore")
    data = RR.prepare()
    af = os.environ.get("ONSET_ANOMFEAT", "")
    if af:                                                    # note 169: upstream entry-demand anomaly (or shuffled)
        for p, d in data.items():
            for s_, D in d.items():
                A = np.load(LAB / "cache" / f"entanom_{s_}_{p}.npz")
                TD, RN = A["today"], A["run"]
                if af.startswith("placebo"):
                    perm = np.random.default_rng(int(af[7:] or 0)).permutation(len(TD))
                    TD, RN = TD[perm], RN[perm]
                rows = []
                for n, cand, reg, mass in D["meta"]:
                    t, r = TD[n][cand], RN[n][cand]
                    rows.append(np.stack([t, r, t - t.max(), r - r.max(), (t == t.max()).astype(float)], 1))
                D["Xp"] = np.concatenate([D["Xp"], np.concatenate(rows).astype(np.float32)], 1)
    cf = os.environ.get("ONSET_CTMFEAT", "")
    if cf:                                                    # note 169: forward-CTM race evidence (or a placebo)
        rngp = np.random.default_rng(0)
        for p, d in data.items():
            for s_, D in d.items():
                SC = np.load(LAB / "cache" / f"ctmfwd_{s_}_{p}.npz")["sc"]
                rows = []
                for n, cand, reg, mass in D["meta"]:
                    f = SC[n] if cf == "ctm" else rngp.random(SC.shape[1]).astype(np.float32)
                    at = f[cand]
                    nb = np.array([f[max(0, j - 1):j + 2].max() for j in cand])
                    rows.append(np.stack([at, nb, nb - nb.max(), (nb == nb.max()).astype(float)], 1))
                D["Xp"] = np.concatenate([D["Xp"], np.concatenate(rows).astype(np.float32)], 1)
    if os.environ.get("ONSET_FILMFEAT", "") == "1":     # the FiLM STID only as extra candidate features
        CACHE = LAB / "cache"
        for p, d in data.items():
            for s, D in d.items():
                P = np.load(CACHE / f"onset_film_recov{os.environ.get('FILM_TAG', '')}_{s}_{p}.npz")["prob"].astype(np.float32)
                rows = []
                for n, cand, reg, mass in D["meta"]:
                    f = P[n]
                    at = f[cand]
                    nb = np.array([f[max(0, j - 2):j + 3].max() for j in cand])
                    share = np.array([f[reg == r].sum() for r in range(len(cand))]) / max(f.sum(), 1e-9)
                    rows.append(np.stack([at, nb, share, nb - nb.max(), (np.argsort(np.argsort(-nb)) == 0).astype(float)], 1))
                D["Xp"] = np.concatenate([D["Xp"], np.concatenate(rows).astype(np.float32)], 1)
    ref = {s: C.family_mean({p: C.onset_iou(d[s]["base"], d[s]["on"]).mean() for p, d in data.items() if s in d})
           for s in C.SPLITS}
    for s_fit, s_app in (("validation", "private"), ("private", "validation")):
        rk = [SF.fit_rank([d[s_fit] for d in data.values() if s_fit in d], sd) for sd in SF.SEEDS]
        for d in data.values():
            if s_app in d:
                SF.apply_rank(rk, d[s_app])
    T = {p: {s: S2.pairs(D) for s, D in d.items()} for p, d in data.items()}
    out = {}
    for s_fit, s_app in (("validation", "private"), ("private", "validation")):
        X, y, win = SF.rows_of([T[p][s_fit] for p in T if s_fit in T[p]])
        tau, _ = SF.choose_tau(X, y, win)
        du = [lgb.train({**S2.PAR, "objective": "regression", "seed": sd}, lgb.Dataset(X, y), 150) for sd in SF.SEEDS]
        f1, f2 = {}, {}
        for p, d in data.items():
            if s_app not in d:
                continue
            D, on = d[s_app], d[s_app]["on"]
            i1, i2 = [], []
            for n in range(len(on["d"])):
                if n in T[p][s_app]:
                    ib, rows = T[p][s_app][n]
                    new = ib
                    if rows:
                        pr = np.mean([m.predict(np.array([f for f, _, _ in rows], np.float32)) for m in du], 0)
                        j = int(np.argmax(pr))
                        if pr[j] > tau:
                            new = ib + rows[j][1]
                else:
                    ib = new = S2.iou_of(C.select(D["P1"][n], "ratio", 1.0), on["fut"][n, -1], on["efut"][n, -1])
                i1.append(ib); i2.append(new)
            f1[p], f2[p] = np.mean(i1), np.mean(i2)
        out[s_app] = (C.family_mean(f1), C.family_mean(f2), tau)
    tag = (os.environ.get("ONSET_ST", "") or "plain (AP STID)") + (" + FiLM features" if os.environ.get("ONSET_FILMFEAT") == "1" else "") + (f" + {os.environ.get('ONSET_CTMFEAT')} race features" if os.environ.get("ONSET_CTMFEAT") else "") + (f" + {os.environ.get('ONSET_ANOMFEAT')} anomaly features" if os.environ.get("ONSET_ANOMFEAT") else "")
    print(f"STID member: {tag}")
    for s in C.SPLITS:
        print(f"  {s:<10} blend ratio rule {ref[s]:.4f}   stage 1 {out[s][0]:.4f}   + stage 2 {out[s][1]:.4f} (tau {out[s][2]})")


if __name__ == "__main__":
    main()
