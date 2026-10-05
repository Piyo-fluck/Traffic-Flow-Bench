"""Do the two onset changes of candidate S add up, or do they overlap?

Candidate S changes both halves of the onset blend at once:
    LightGBM  H + the scale-free Task 1 residual features (note 64, ideas 2/3)
    CNN       the past-day adapted model instead of CNN x3 (note 67, idea 9b)
Each was measured on its own; this measures the four combinations on the blank-aware official-like onset windows,
so the candidate expectation is not a sum of two numbers that might describe the same gain.

The LightGBM bags are retrained here exactly as scripts/t2_onset_t1res_predict.py trains them (H configuration,
prior dropout, 5 bags, 600 rounds, 63 leaves), and the adapted CNN probabilities come from the cache written by
the Ada run (`t2cnn_t2_cnn_lov_adapt_*`).

    .venv/Scripts/python.exe analysis/t2_onset_combo_eval.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "scripts"))
import t2_prior_robust  # noqa: E402
from t2_eda02 import T2_PANELS  # noqa: E402
from t2_eda03 import F  # noqa: E402
from t2_expected_iou import select  # noqa: E402
from t2_first_onset_eval import first_onsets, official_blank  # noqa: E402
from t2_m1_models import Ctx, hist_from_truth  # noqa: E402
from t2_m1_predict import feats_on, table_prior  # noqa: E402
from t2_onset_t1res_feats import NAMES, t1_feats  # noqa: E402
from t2_pseudo_eval import iou  # noqa: E402
from t2_shift_remedies import PARAMS  # noqa: E402
from tfb.data import CACHE, load  # noqa: E402
from tfb.t1eval import family_mean  # noqa: E402

CNN3 = ["t2_cnn_lov", "t2_cnn_lov_s1", "t2_cnn_lov_s2"]
ADAPT = "t2_cnn_lov_adapt"
COLS, N_BAGS, ROUNDS, LEAVES, MAX_WIN, W_ON = slice(8, 12), 5, 600, 63, 2000, 0.7
rng = np.random.default_rng(29)
boot = np.random.default_rng(7)


def ci(d):
    idx = boot.integers(0, len(d), (2000, len(d)))
    m = d[idx].mean(1)
    return f"{d.mean():+.4f} [{np.percentile(m, 2.5):+.4f}, {np.percentile(m, 97.5):+.4f}]"


def main() -> None:
    t0 = time.time()
    Xb, Eb, ys, ctxs, names_on = [[] for _ in range(N_BAGS)], [[] for _ in range(N_BAGS)], [], [], None
    for i, panel in enumerate(T2_PANELS):
        ctx = Ctx(panel, i)
        ctx.fitd = np.ones_like(ctx.fitd)
        z = ctx.z
        d, t0w = ctx.eo
        if len(d) > MAX_WIN:
            s = np.sort(rng.choice(len(d), MAX_WIN, replace=False))
            d, t0w = d[s], t0w[s]
        v, qf, ok = hist_from_truth(ctx, d, t0w)
        E, _ = t1_feats(panel, "train", d, t0w, v, qf, ok)
        t2_prior_robust.rng = np.random.default_rng(1000 + i)
        for b in range(N_BAGS):
            names_on, X = feats_on(ctx, False, v, qf, ok, t0w, ctx.g[d], t2_prior_robust.dropout_prior(ctx), d,
                                   z["weekday"][d])
            Xb[b].append(X.reshape(-1, X.shape[-1]))
            Eb[b].append(E[..., COLS].reshape(-1, 4))
        ys.append(ctx.q[d, t0w + F].reshape(-1))
        ctx.z = {k: z[k] for k in ("link_ids", "weekday", "dates")}
        ctx.q = ctx.el = None
        ctxs.append(ctx)
    y = np.concatenate(ys)
    nm = names_on + NAMES[COLS]
    base_models, t1_models = [], []
    for b in range(N_BAGS):
        Xp = np.concatenate(Xb[b])
        Ep = np.concatenate(Eb[b])
        base_models.append(lgb.train(dict(PARAMS, num_leaves=LEAVES, seed=b),
                                     lgb.Dataset(Xp, y, feature_name=names_on), ROUNDS))
        t1_models.append(lgb.train(dict(PARAMS, num_leaves=LEAVES, seed=b),
                                   lgb.Dataset(np.concatenate([Xp, Ep], 1), y, feature_name=nm), ROUNDS))
    del Xb, Eb
    print(f"trained {N_BAGS} bags ({time.time() - t0:.0f}s)", flush=True)
    res, fam = {}, {}
    for ctx in ctxs:
        for split in ("validation", "private"):
            z = load(ctx.panel, split)
            on = first_onsets(ctx.vf, z, blank=official_blank(ctx.panel, split, z))
            if not len(on["d"]):
                continue
            c = np.load(CACHE / f"t2probs_{split}_{ctx.panel}.npz")
            assert (c["on_d"] == on["d"]).all()
            cnn3 = np.mean([np.load(CACHE / f"t2cnn_{t}_{split}_{ctx.panel}.npz")["on_p"] for t in CNN3], 0)
            ad = np.load(CACHE / f"t2cnn_{ADAPT}_{split}_{ctx.panel}.npz")["on_p"]
            grp = (z["weekday"][on["d"]] >= 5).astype(int)
            key = np.stack([on["d"], grp], 1)
            _, X = feats_on(ctx, False, on["v"], on["qf"], on["ok"], on["t0"], grp, table_prior(ctx), key,
                            z["weekday"][on["d"]].astype(int))
            E, _ = t1_feats(ctx.panel, split, on["d"], on["t0"], on["v"], on["qf"], on["ok"])
            Xo = np.concatenate([X, E[..., COLS]], -1)
            p_new = np.mean([m.predict(Xo.reshape(-1, Xo.shape[-1])) for m in t1_models], 0).reshape(Xo.shape[:2])
            p_bas = np.mean([m.predict(X.reshape(-1, X.shape[-1])) for m in base_models], 0).reshape(X.shape[:2])
            for lname, pl in (("H cached", c["on_p"]), ("H retrained", p_bas), ("H+t1res", p_new)):
                for cname, pc in (("cnn3", cnn3), ("adapted", ad)):
                    p = W_ON * pl + (1 - W_ON) * pc
                    pr = np.zeros(on["fut"].shape, bool)
                    pr[:, F - 1] = np.stack([select(q, "ratio", 1.0) for q in p])
                    s = iou(pr, on["fut"], on["efut"])
                    res.setdefault((split, lname, cname), []).append(s)
                    fam.setdefault((split, lname, cname), {})[ctx.panel] = s.mean()
        print(f"  scored {ctx.panel} ({time.time() - t0:.0f}s)", flush=True)
    for split in ("validation", "private"):
        base = np.concatenate(res[(split, "H cached", "cnn3")])
        print(f"\n=== {split}: onset, {len(base)} official-like windows (M production = H + cnn3) ===")
        for lname in ("H cached", "H retrained", "H+t1res"):
            for cname in ("cnn3", "adapted"):
                v = np.concatenate(res[(split, lname, cname)])
                print(f"  {lname:<8} + {cname:<8} IoU {v.mean():.4f}  family {family_mean(fam[(split, lname, cname)]):.4f}"
                      f"  delta {ci(v - base)}")
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
