"""Ongoing cell stacker across months, with the away-from-queue evidence of note 160.

Note 156's LightGBM stacker (uncertain cells, AU's p in [0.05, 0.95]; 7 members, AU p0 / final, history, distances)
fitted on the OTHER scored month only (no same-month context), versus the same + the region evidence that separated
true from false away-queues in note 160 (AUC 0.65-0.70 -> 0.87-0.90 inside the decision band):
    distance to the T-queue (signed by the panel's upstream side), in-zone flag, observed share of the link in the
    last 15 min and at T, note 145 capacity headroom at the link (training-month prior + the month's <= 4 days),
    AU's max over the horizons at the link and its +-4 neighbourhood.
q = a p + (1 - a) p_AU on the modelled cells, tau 0.5; LightGBM seeds 0-5; all days and days 1-5.

    .venv/Scripts/python.exe occ_lab/og_cellstack.py
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

import bottleneck_capacity as BK  # noqa: E402
import common as C  # noqa: E402
import icl_ongoing as IO  # noqa: E402
import select_bank as SB  # noqa: E402
from online_adapter import ETA, GATE, global_features  # noqa: E402

F = C.F


def extra_feats(T, q_au, og, vf, A, Pr):
    """(cells, k) per uncertain cell of note 156's table T."""
    N, _, S = q_au.shape
    qT = og["ok"][:, -1] & (np.nan_to_num(og["v"][:, -1], nan=1e9) <= 0.6 * vf)
    pos = np.arange(S)
    sdist = np.full((N, S), 99.0, np.float32)
    for n in range(N):
        j = np.flatnonzero(qT[n])
        if len(j):
            dd = pos[:, None] - j[None, :]
            k = np.abs(dd).argmin(1)
            sdist[n] = np.clip(dd[pos, k], -99, 99)
    obs15 = og["ok"][:, -3:].mean(1)
    obsT = og["ok"][:, -1].astype(np.float32)
    pmax = q_au.max(1)
    pnb = IO.shiftmax(pmax, 4)
    capf = {}
    for n in np.unique(T["n"]):
        d, t0 = int(og["d"][n]), int(og["t0"][n])
        lo = max(0, d - 4)
        As = {k: w[lo:d + 1] for k, w in A.items()}
        cand = np.arange(S)
        cf = np.concatenate([BK.prior_rows(Pr, t0, og["qf"][n], og["ok"][n], cand),
                             BK.cap_rows(As, d - lo, t0, og["qf"][n], og["ok"][n], cand)], 1)
        capf[n] = cf[:, [0, 1, 2, 3, 8, 12, 13, 14, 15, 20]]          # absolute headrooms, step, month headrooms
    n_i, s_i = T["n"], T["s"]
    cfe = np.stack([capf[n][s] for n, s in zip(n_i, s_i)])
    return np.concatenate([np.stack([sdist[n_i, s_i], np.abs(sdist[n_i, s_i]) <= 2, obs15[n_i, s_i], obsT[n_i, s_i],
                                     pmax[n_i, s_i], pnb[n_i, s_i]], 1).astype(np.float32), cfe], 1)


def main() -> None:
    warnings.simplefilter("ignore")
    gbm = lgb.Booster(model_file=str(LAB / "models" / "residual_lgb.txt"))

    def corr(base, P):
        r = gbm.predict(global_features(P, base).reshape(-1, 15)).reshape(base.shape).astype(np.float32)
        return np.clip(base + ETA * np.where(np.abs(r) >= GATE, r, 0.0), 0, 1)

    built = {}
    for i, panel in enumerate(C.T2_PANELS):
        ctx, stat, W = C.windows(i, panel)
        vf = np.asarray(ctx.vf, np.float32)
        Pr = BK.prior_stats(panel, vf)
        for s in C.SPLITS:
            B = SB.bank_for(i, panel, s, "solid3", "")
            M0 = B["M"].copy(); M0[3] = B["ps_ship"]
            if "--okall" in sys.argv:            # note 163: p_PS and tfm8 rebuilt on the widened history mask
                M0[3] = np.mean([np.load(LAB / "cache" / f"pseudo_ssl_t2_cnn_lov_w192_rec_ssl30k_s{k}_okall_{s}_{panel}.npz")["og"]
                                 .astype(np.float32) for k in (20, 21, 22, 23)], 0)
                M0[1] = np.mean([np.load(LAB / "cache" / f"tfmps_s{k}okall_sup0.0001_{s}_{panel}.npz")["og"]
                                 .astype(np.float32) for k in range(3, 11)], 0)
            B["M"] = M0
            p0 = np.tensordot(SB.WAU, M0, 1)
            q_au = corr(p0, B["P"])
            T = IO.cell_table(B, q_au, p0)
            T["x2"] = extra_feats(T, q_au, W[s]["og"], vf, BK.day_arrays(W[s]["z"], vf), Pr)
            built[(panel, s)] = (B, q_au, T)
        print(f"  {panel}", flush=True)
    cat = {s: {k: np.concatenate([built[(p, s)][2][k] for p in C.T2_PANELS]) for k in ("stack", "x2", "y", "e")}
           for s in C.SPLITS}
    par = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 50,
           "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "num_threads": 8}
    res = {}
    for sd in range(6):
        for s_fit, s_app in (("validation", "private"), ("private", "validation")):
            c = cat[s_fit]
            ok = c["e"]
            for nm, X in (("stack (156)", c["stack"]), ("stack + away", np.concatenate([c["stack"], c["x2"]], 1))):
                m = lgb.train({**par, "seed": sd}, lgb.Dataset(X[ok], c["y"][ok]), 300)
                for p in C.T2_PANELS:
                    B, q_au, T = built[(p, s_app)]
                    Xa = T["stack"] if nm == "stack (156)" else np.concatenate([T["stack"], T["x2"]], 1)
                    pk = m.predict(Xa) if len(Xa) else np.zeros(0)
                    yb, eb, dd = B["y"] > 0.5, B["e"] > 0.5, B["d"]
                    for a in (0.5, 1.0):
                        q = q_au.copy()
                        q[T["n"], T["h"], T["s"]] = a * pk + (1 - a) * q_au[T["n"], T["h"], T["s"]]
                        sc = C.iou(q >= 0.5, yb, eb)
                        res.setdefault((s_app, nm, a, sd), {})[p] = (sc.mean(), sc[dd <= 4].mean())
                    sc = C.iou(q_au >= 0.5, yb, eb)
                    res.setdefault((s_app, "AU", 0, sd), {})[p] = (sc.mean(), sc[dd <= 4].mean())
        print(f"  seed {sd}", flush=True)
    print(f"\n{'variant':<22}{'all val':>9}{'all priv':>10}{'d1-5 val':>10}{'d1-5 priv':>11}")
    for nm in ("stack (156)", "stack + away"):
        for a in (0.5, 1.0):
            v = []
            for k in (0, 1):
                for s in C.SPLITS:
                    v.append(np.mean([C.family_mean({p: res[(s, nm, a, sd)][p][k] for p in C.T2_PANELS}) -
                                      C.family_mean({p: res[(s, "AU", 0, sd)][p][k] for p in C.T2_PANELS})
                                      for sd in range(6)]))
            print(f"{nm + f' a{a}':<22}{v[0]:>+9.4f}{v[1]:>+10.4f}{v[2]:>+10.4f}{v[3]:>+11.4f}")


if __name__ == "__main__":
    main()
