"""note 171: CBM-proposed candidates for the sites / shapes production's candidate pool cannot reach.

After BB's set decoder (stage 1 + 2, production-like: fitted on the full-adaptation other month) the decision A0 is
fixed. The CBM proposes sets Ac: its top-3 peaks (>= 4 links apart), region +-3 links, shape = the ratio rule on the
CBM ('cbm') or on sqrt(P1 * CBM) ('mix') inside the region. Two actions per proposal: replace A0 by Ac, or add Ac
to A0. A LightGBM regressor (6 seeds) predicts the IoU change of each action from run-invariant evidence; the best
action is taken when its prediction exceeds tau (5-fold out-of-fold on the fit month, as stage 2's threshold).

Fit on month A's early-K windows (K = 0..4 pooled), applied to month B's early-K windows, both directions.

    ONSET_SHARE_K=1 .venv/Scripts/python.exe occ_lab/cbm_override.py
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB)); sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta")); sys.path.insert(0, str(LAB.parent))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

import common as C  # noqa: E402
import onset_stage2_du as S2  # noqa: E402
from onset_setdec_fit import SEEDS, fit_rank, apply_rank, rows_of, choose_tau  # noqa: E402
from setdec_early_check import prep  # noqa: E402
from cbm_setdec_check import cbm_prob, CBM  # noqa: E402
import bottleneck_cls as BC  # noqa: E402
import cbm_onset as M  # noqa: E402
import torch  # noqa: E402

CA = LAB / "cache"
NOCON = os.environ.get("CBM_NOCON", "") == "1"


def cbm_concepts(split, panel, on):
    """(N, 5, S) concept values (q30, q15 probabilities; v30/vf, q30/q95, v15/vf), mean over the 8 CBM seeds."""
    f = CA / f"cbmc_rf8_{split}_{panel}.npz"
    if f.exists():
        return np.load(f)["c"]
    stc = M.panel_static(panel)
    z = M.load(panel, split)
    arrays = M.dyn(z, M.load_ramps(panel, split, z), stc["vf"], stc["q95"]) + (z,)

    class P:
        pass
    P.panel, P.vf, P.q95, P.S = panel, stc["vf"], stc["q95"], len(stc["vf"])
    out = []
    with torch.no_grad():
        for t in CBM:
            ck = torch.load(M.MODELS / f"{t}.pt", map_location=M.DEV)
            st = stc["st"][ck.get("st", list(range(8)))]
            x = torch.from_numpy(M.scored_inputs(P, split, on, st, arrays=arrays)).to(M.DEV)
            cn = M.ConceptNet(ck["cin"], dil=tuple(ck["dil"])).to(M.DEV); cn.load_state_dict(ck["cn"]); cn.eval()
            out.append(M.concept_feats(cn(x)).cpu().numpy())
    c = np.mean(out, 0).astype(np.float32)
    np.savez_compressed(f, c=c)
    return c

warnings.simplefilter("ignore")
R = 3
PAR = {"objective": "regression", "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 20,
       "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "num_threads": 8}


def peaks(p, k=3, sep=4):
    q = p.copy(); out = []
    for _ in range(k):
        j = int(np.argmax(q))
        if q[j] <= 1e-4:
            break
        out.append(j); q[max(0, j - sep + 1):j + sep] = -1
    return out


def proposals(P1, A0, cb, cand, s1, ctx=None):
    """feature rows and sets of every (proposal, action) of one window."""
    S = len(P1)
    a0j = np.flatnonzero(A0)
    c0 = a0j.mean() if len(a0j) else -100.0
    pk = peaks(cb)
    cbmax = max(cb.max(), 1e-6)
    s1n = s1 - s1.max() if len(s1) else s1
    out = []
    for r, j in enumerate(pk):
        m = np.zeros(S, bool); m[max(0, j - R):j + R + 1] = True
        # production's own candidate nearest to this CBM peak and its stage-1 score (relative to the chosen one)
        dc = np.abs(cand - j); ci = int(np.argmin(dc))
        for si, src in enumerate((cb, np.sqrt(np.maximum(P1, 1e-9) * cb))):
            Ac = C.select(src * m, "ratio", 1.0) & m
            if not Ac.any() or (Ac & ~A0).sum() == 0:
                continue
            ov = (Ac & A0).sum() / max((Ac | A0).sum(), 1)
            base = [r, si, cb[j], cb[j] / cbmax, cb[m].sum(), cb[A0].max() if len(a0j) else 0.0,
                    P1[Ac].max(), P1[Ac].sum(), P1.max(), P1[A0].max() if len(a0j) else 0.0, P1[A0].sum(),
                    Ac.sum(), A0.sum(), ov, abs(j - c0), dc[ci], s1n[ci],
                    cb[Ac].mean() - (cb[A0].mean() if len(a0j) else 0.0), float((P1 > P1[j]).sum())]
            if ctx is not None and not NOCON:
                cc, vl, ql, sh = ctx["c"], ctx["vlast"], ctx["qlast"], ctx["share"]
                nb = slice(max(0, j - 1), j + 2)
                def at(a, M_):
                    return float(np.nanmean(a[M_])) if M_.any() and np.isfinite(a[M_]).any() else np.nan
                base = base + [cc[0, j], cc[1, j], cc[2, j], cc[3, j], cc[4, j],
                               cc[0][Ac].max(), cc[2][Ac].min(), cc[3][Ac].max(),
                               at(cc[0], A0), at(cc[2], A0), at(cc[3], A0),
                               np.nanmin(vl[nb]) if np.isfinite(vl[nb]).any() else np.nan, at(vl, A0),
                               np.nanmax(ql[nb]) if np.isfinite(ql[nb]).any() else np.nan, at(ql, A0),
                               np.nanmax(sh[nb]) if np.isfinite(sh[nb]).any() else np.nan, at(sh, A0)]
            out.append((base + [0.0], Ac))              # replace
            out.append((base + [1.0], Ac | A0))         # add
    return out


def decode_sets(data_fit, s_fit, data_app, s_app):
    """BB's set decoder (production-like) -> per (panel, n): final set, stage-1 P1, s1, candidates."""
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
    out = {}
    for p_, d in data_app.items():
        if s_app not in d:
            continue
        D = d[s_app]
        apply_rank(rk, D)
        off = 0
        for (n, cand, reg, mass), g in zip(D["meta"], D["g"]):
            s1 = D["s1"][off:off + g]
            xp = D["Xp"][off:off + g]; nf = D["NF"][off:off + g]
            off += g
            Sb, rows = S2.pair_rows(D["P1"][n], s1, xp, nf, cand, reg)
            A = Sb.copy()
            if rows:
                pr = np.mean([m.predict(np.array([f for f, _, _ in rows], np.float32)) for m in du], 0)
                j = int(np.argmax(pr))
                if pr[j] > tau:
                    A = A | rows[j][1]
            out[(p_, n)] = (A, D["P1"][n], s1, cand)
        # windows without candidates (< 2 peaks): the ratio rule on P1
        for n in range(len(D["on"]["d"])):
            if (p_, n) not in out:
                out[(p_, n)] = (C.select(D["P1"][n], "ratio", 1.0), D["P1"][n], np.zeros(0), np.zeros(0, int))
    return out


def window_rows(early, sets, s):
    """per window: (key, u0, [(features, dU, set)])"""
    W = []
    K = int(os.environ.get("ONSET_EARLY_K", "0"))
    for p_, d in early.items():
        if s not in d:
            continue
        D = d[s]; on = D["on"]; cb = cbm_prob(s, p_)
        cc = cbm_concepts(s, p_, on)
        stc = M.panel_static(p_)
        vf, q95 = stc["vf"], stc["q95"]
        qd, nd = BC.month_share(M.load(p_, s), vf)
        for n in range(len(on["d"])):
            A0, P1, s1, cand = sets[(p_, n)]
            t, e = on["fut"][n, -1], on["efut"][n, -1]
            u0 = S2.iou_of(A0, t, e)
            rows = []
            dd = int(on["d"][n]); lo = max(0, dd - K)
            with np.errstate(all="ignore"):
                okh = on["ok"][n]
                vv = np.where(okh, on["v"][n] / vf, np.nan); qq = np.where(okh, on["qf"][n] / q95, np.nan)
                ctx = dict(c=cc[n], vlast=np.nanmean(vv[-3:], 0), qlast=np.nanmean(qq[-3:], 0),
                           share=(qd[lo:dd].sum(0) / np.maximum(nd[lo:dd].sum(0), 1)) if dd > lo
                           else np.full(len(vf), np.nan))
            if len(cand):
                for f, A in proposals(P1, A0, cb[n], cand, s1, ctx):
                    rows.append((f, S2.iou_of(A, t, e) - u0, A))
            W.append(((p_, n), u0, rows))
    return W


def pick_tau(X, y, fold, copy):
    """threshold from 5-fold out-of-fold predictions; folds by window, the decision per window copy (K)."""
    oof = np.zeros(len(y))
    for k in range(5):
        oof[fold == k] = np.mean([lgb.train({**PAR, "seed": sd}, lgb.Dataset(X[fold != k], y[fold != k]), 150)
                                  .predict(X[fold == k]) for sd in SEEDS], 0)
    best_row = {}
    for i, c in enumerate(copy):
        if c not in best_row or oof[i] > oof[best_row[c]]:
            best_row[c] = i
    b = np.array(list(best_row.values()))
    gain = {t: float(y[b][oof[b] > t].sum()) for t in S2.TAUS}
    best = max(gain.values())
    return max(t for t in S2.TAUS if gain[t] >= best - 1e-3) if best > 0 else np.inf


def main():
    full = prep(None)
    Ks = range(5)
    per = {}                        # (s_app, K) -> window rows
    for k in Ks:
        early = prep(k)
        for s_app in C.SPLITS:
            s_fit = [x for x in C.SPLITS if x != s_app][0]
            sets = decode_sets(full, s_fit, early, s_app)
            per[(s_app, k)] = window_rows(early, sets, s_app)
        print(f"  K={k} rows built", flush=True)
    import pickle
    with open(CA / "cbm_override_rows.pkl", "wb") as fh:          # for quick rule studies (cbm_rule.py)
        pickle.dump({k: [(key, u0, [(f, du, A.sum()) for f, du, A in rows]) for key, u0, rows in v]
                     for k, v in per.items()}, fh)
    for s_app in C.SPLITS:
        s_fit = [x for x in C.SPLITS if x != s_app][0]
        # fit on the other month's windows (all K pooled)
        Wf = [w for k in Ks for w in per[(s_fit, k)]]
        X = np.array([f for _, _, rows in Wf for f, _, _ in rows], np.float32)
        y = np.array([du for _, _, rows in Wf for _, du, _ in rows], np.float32)
        kid = {}
        for key, _, _ in Wf:                     # the same window under every K shares one fold
            kid.setdefault(key, len(kid))
        fold_of = np.random.default_rng(0).integers(0, 5, len(kid))
        fold = np.array([fold_of[kid[key]] for key, _, rows in Wf for _ in rows])
        copy = np.array([i for i, (_, _, rows) in enumerate(Wf) for _ in rows])
        tau = pick_tau(X, y, fold, copy)
        ms = [lgb.train({**PAR, "seed": sd}, lgb.Dataset(X, y), 150) for sd in SEEDS]
        base, new, nch = [], [], 0
        for k in Ks:
            for key, u0, rows in per[(s_app, k)]:
                u = u0
                if rows:
                    pr = np.mean([m.predict(np.array([f for f, _, _ in rows], np.float32)) for m in ms], 0)
                    j = int(np.argmax(pr))
                    if pr[j] > tau:
                        u = u0 + rows[j][1]; nch += 1
                base.append(u0); new.append(u)
        base, new = np.array(base), np.array(new)
        dd = (new - base).reshape(len(Ks), -1).mean(0)
        orc = np.mean([max([u0] + [u0 + du for _, du, _ in rows]) for k in Ks for _, u0, rows in per[(s_app, k)]])
        print(f"{s_app}: setdec {base.mean():.4f} -> +CBM proposals {new.mean():.4f} ({dd.mean():+.4f}, SE "
              f"{dd.std(ddof=1) / np.sqrt(len(dd)):.4f}); tau {tau:.3f}; changed {nch / len(Ks):.1f} windows / K; "
              f"oracle {orc:.4f}", flush=True)


if __name__ == "__main__":
    main()
