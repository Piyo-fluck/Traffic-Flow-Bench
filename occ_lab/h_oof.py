"""Out-of-fold H (the LightGBM onset member) on the TRAINING months' first-onset windows.

Production H (work/sub_t2robust_bag5): scripts/t2_m1_predict.py --onset-prior dropout --onset-bags 5, trained on the
earliest-onset origins of every train day, priors leave-one-day-out, scored with the train priors table. Here, per
calendar month m of the training data: the priors use only the other months (fitd = month != m), the model is
trained on the other months' earliest-onset origins (prior dropout, BAGS bags), and it scores month m's first-onset
windows (C.first_onsets on the RELEASED-style observations, as the lab pool of the scored months) with the prior
table of the other months - what an official window of an unseen month gets.

Saves occ_lab/cache/h_oof_<panel>.npz: d, t0, prob (N, S) at T+30, plus the window arrays needed downstream.

    .venv/Scripts/python.exe occ_lab/h_oof.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

LAB = Path(__file__).resolve().parent
ROOT = LAB.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "scripts"))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

from t2_eda02 import T2_PANELS  # noqa: E402
from t2_eda03 import F  # noqa: E402
from t2_first_onset_eval import first_onsets  # noqa: E402
from t2_m1_models import Ctx, hist_from_truth  # noqa: E402
from t2_m1_predict import feats_on, train_prior  # noqa: E402
from t2_prior_robust import dropout_prior  # noqa: E402

BAGS = 2
ROUNDS = 600
CACHE = LAB / "cache"


def main() -> None:
    t_start = time.time()
    ctxs = [Ctx(p, i) for i, p in enumerate(T2_PANELS)]
    months = {c.panel: np.array([str(x)[:7] for x in c.z["dates"]]) for c in ctxs}
    allm = sorted(set(np.concatenate(list(months.values()))))
    print(f"months: {allm}", flush=True)
    # scoring windows: first onsets of every train day from the observations
    ev = {}
    for c in ctxs:
        on = first_onsets(c.vf, c.z)
        ev[c.panel] = on
        print(f"  {c.panel}: first-onset windows {len(on['d'])}, earliest-onset training origins {len(c.eo[0])}",
              flush=True)
    prob = {c.panel: np.full((len(ev[c.panel]["d"]), c.S), np.nan, np.float32) for c in ctxs}
    params = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=50, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=8)
    for m in allm:
        for c in ctxs:
            c.fitd = months[c.panel] != m
        models = []
        for b in range(BAGS):
            Xs, ys = [], []
            for c in ctxs:
                d, t0 = c.eo
                keep = c.fitd[d]
                d, t0 = d[keep], t0[keep]
                if not len(d):
                    continue
                for a in range(0, len(d), 3000):
                    dd, tt = d[a:a + 3000], t0[a:a + 3000]
                    _, X = feats_on(c, False, *hist_from_truth(c, dd, tt), tt, c.g[dd], dropout_prior(c), dd,
                                    c.z["weekday"][dd])
                    Xs.append(X.reshape(-1, X.shape[-1]).astype(np.float32))
                    ys.append(c.q[dd, tt + F].reshape(-1))
            X, y = np.concatenate(Xs), np.concatenate(ys)
            del Xs
            models.append(lgb.train(dict(params, seed=b), lgb.Dataset(X, y), ROUNDS))
            del X, y
        for c in ctxs:
            on = ev[c.panel]
            sel = np.flatnonzero(months[c.panel][on["d"]] == m)
            if not len(sel):
                continue
            d = on["d"][sel]
            _, X = feats_on(c, False, on["v"][sel], on["qf"][sel], on["ok"][sel], on["t0"][sel], c.g[d],
                            train_prior(c), d, c.z["weekday"][d])
            prob[c.panel][sel] = np.mean([mm.predict(X.reshape(-1, X.shape[-1])) for mm in models], 0).reshape(
                len(sel), c.S)
        print(f"month {m} done ({time.time() - t_start:.0f}s)", flush=True)
    for c in ctxs:
        on = ev[c.panel]
        np.savez_compressed(CACHE / f"h_oof_{c.panel}.npz", prob=prob[c.panel], d=on["d"], t0=on["t0"],
                            v=on["v"], qf=on["qf"], ok=on["ok"], fut=on["fut"], efut=on["efut"],
                            month=months[c.panel][on["d"]])
    print(f"saved ({time.time() - t_start:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
