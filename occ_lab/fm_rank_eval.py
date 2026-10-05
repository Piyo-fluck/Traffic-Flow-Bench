"""Idea (4): do frozen foundation-model embeddings of the candidate's series help note 152's onset ranker?

note 152's IoU-LambdaRank (+ swap) on the note 145 features (Xp) against the same with K principal components of a
frozen model's embedding (occ_lab/fm_embed_onset.py) and their deviation from the window mean. The PCA is fitted on
all candidates of both scored months (unsupervised). Cross-month, LightGBM seeds 0-5.

    .venv/Scripts/python.exe occ_lab/fm_rank_eval.py chronos2 moment
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta"))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

import bottleneck_cls as BC  # noqa: E402
import common as C  # noqa: E402
import onset_rerank as RR  # noqa: E402

CACHE = LAB / "cache"
K = int(__import__("os").environ.get("FM_K", 4))
PAR = {"learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 20, "feature_fraction": 0.8,
       "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "num_threads": 8, "objective": "lambdarank",
       "label_gain": list(range(11))}


def main() -> None:
    warnings.simplefilter("ignore")
    models = sys.argv[1:] or ["chronos2"]
    data = RR.prepare()
    for mdl in models:
        E = {(p, s): np.nan_to_num(np.load(CACHE / f"fmemb_{mdl}_{s}_{p}.npz")["emb"].astype(np.float32))
             for p, d in data.items() for s in d}
        allE = np.concatenate(list(E.values()))
        mu = allE.mean(0)
        _, _, vt = np.linalg.svd(allE - mu, full_matrices=False)
        V = vt[:K].T
        for (p, s), e in E.items():
            D = data[p][s]
            z = (e - mu) @ V
            dev = np.empty_like(z)
            off = 0
            for g in D["g"]:
                dev[off:off + g] = z[off:off + g] - z[off:off + g].mean(0, keepdims=True); off += g
            D[f"X_{mdl}"] = np.concatenate([D["Xp"], z, dev], 1)
    keys = ["Xp"] + [f"X_{m}" for m in models]
    ref = {s: C.family_mean({p: C.onset_iou(d[s]["base"], d[s]["on"]).mean() for p, d in data.items() if s in d})
           for s in C.SPLITS}
    res = {}
    for sd in range(6):
        for key in keys:
            for s_fit, s_app in (("validation", "private"), ("private", "validation")):
                Ds = [d[s_fit] for d in data.values() if s_fit in d]
                lab = np.minimum((np.concatenate([D["U"] for D in Ds]) * 10).round().astype(int), 10)
                m = lgb.train({**PAR, "seed": sd}, lgb.Dataset(np.concatenate([D[key] for D in Ds]), lab,
                                                                 group=np.concatenate([D["g"] for D in Ds])), 200)
                fam = {}
                for p, d in data.items():
                    if s_app not in d:
                        continue
                    D = d[s_app]
                    sc = m.predict(D[key])
                    P2 = D["base"].copy()
                    off = 0
                    for (n, cand, reg, mass), g in zip(D["meta"], D["g"]):
                        s_ = sc[off:off + g]
                        P2[n] = BC.swap(D["base"][n], cand, reg, mass, s_ - s_.min() + 1e-3); off += g
                    fam[p] = C.onset_iou(P2, D["on"]).mean()
                res.setdefault((key, s_app), []).append(C.family_mean(fam) - ref[s_app])
    print(f"\nonset IoU vs production (IoU-LambdaRank + swap, 6 seeds, cross-month; K = {K} PCs + deviations)")
    base = {s: np.mean(res[("Xp", s)]) for s in C.SPLITS}
    for key in keys:
        v, pr = np.mean(res[(key, "validation")]), np.mean(res[(key, "private")])
        print(f"  {key:<12} val {v:+.4f}±{np.std(res[(key, 'validation')]):.4f}  priv {pr:+.4f}±"
              f"{np.std(res[(key, 'private')]):.4f}   vs Xp  {v - base['validation']:+.4f} / {pr - base['private']:+.4f}")


if __name__ == "__main__":
    main()
