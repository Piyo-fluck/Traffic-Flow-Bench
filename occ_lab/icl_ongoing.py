"""A: a tabular in-context learner (TabICLv2, BSD-3) on the ongoing cells, conditioned on the SAME scored month.

For every day d of a scored month, the context is the released pseudo ongoing windows of days < d (official-like
pool, all panels of the split), the queries are day d's windows - no training on the month, no gradient: the
learner reads the month's own behaviour at inference. Only the uncertain cells (AU's final p in [LO, HI]) are
modelled; elsewhere AU stands.

    A2 stack   features: the 7 members' p (gs1, tfm8, cnn3, p_PS, occ_cnn, occ_tfm, stid), AU's p0 and final p,
               their +-2-link max, horizon, v/vf at the link over the last 3 slots and its +-1/+-2 neighbours,
               distance to the nearest queued link at T (both sides), queued share of the +-8 neighbourhood, time
    A1 raw     the same without any member / AU probability (a heterogeneous learner on history alone)
    control    LightGBM fitted on the same context rows (separates the in-context learner from 'same-month data')
Decision: AU's cells replaced by alpha * p + (1 - alpha) * p_AU (alpha 0.5, 1.0), tau 0.5; family IoU, all days and
days 1-5 (d <= 4), vs AU.

    .venv/Scripts/python.exe occ_lab/icl_ongoing.py
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta"))
os.environ.setdefault("HF_HOME", str(LAB.parent / ".hf_cache"))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

import common as C  # noqa: E402
import select_bank as SB  # noqa: E402
from online_adapter import ETA, GATE, global_features  # noqa: E402

F = C.F
LO, HI = 0.05, 0.95
CAP = 120_000
rng = np.random.default_rng(0)


def shiftmax(a, r):
    out = a.copy()
    for k in range(1, r + 1):
        out[..., k:] = np.maximum(out[..., k:], a[..., :-k]); out[..., :-k] = np.maximum(out[..., :-k], a[..., k:])
    return out


def cell_table(B, q_au, p0):
    """rows for the uncertain cells of one panel-split: (features stack, features raw, y, e, window, h, link)."""
    M = B["M"]                                     # (7, N, F, S)
    N, _, S = q_au.shape
    vr = B["vr"]                                   # (N, 3, S) v/vf at the last 3 slots (1.0 where unobserved)
    qT = vr[:, -1] <= 0.6
    pos = np.arange(S)
    dist_up = np.full((N, S), 99.0, np.float32); dist_dn = np.full((N, S), 99.0, np.float32)
    for n in range(N):
        j = np.flatnonzero(qT[n])
        if len(j):
            dd = pos[:, None] - j[None, :]
            dist_up[n] = np.where(dd >= 0, dd, 999).min(1).clip(max=99)
            dist_dn[n] = np.where(dd <= 0, -dd, 999).min(1).clip(max=99)
    share8 = np.stack([np.convolve(x.astype(float), np.ones(17) / 17, "same") for x in qT]).astype(np.float32)
    band = (q_au >= LO) & (q_au <= HI)
    n_i, h_i, s_i = np.nonzero(band)
    Mm = shiftmax(M, 2)
    hist = [vr[n_i, k, s_i] for k in range(3)]
    for off in (-2, -1, 1, 2):
        hist.append(vr[n_i, -1, np.clip(s_i + off, 0, S - 1)])
    raw = np.stack([h_i.astype(np.float32), *hist, dist_up[n_i, s_i], dist_dn[n_i, s_i], share8[n_i, s_i],
                    B["t0"][n_i] / 12.0], 1).astype(np.float32)
    mem = np.stack([*[M[k, n_i, h_i, s_i] for k in range(7)], *[Mm[k, n_i, h_i, s_i] for k in range(7)],
                    p0[n_i, h_i, s_i], q_au[n_i, h_i, s_i]], 1).astype(np.float32)
    return dict(stack=np.concatenate([mem, raw], 1), raw=raw, y=(B["y"][n_i, h_i, s_i] > 0.5).astype(int),
                e=B["e"][n_i, h_i, s_i] > 0.5, n=n_i, h=h_i, s=s_i, d=B["d"][n_i])


def main() -> None:
    warnings.simplefilter("ignore")
    from tabicl import TabICLClassifier
    gbm = lgb.Booster(model_file=str(LAB / "models" / "residual_lgb.txt"))

    def corr(base, P):
        r = gbm.predict(global_features(P, base).reshape(-1, 15)).reshape(base.shape).astype(np.float32)
        return np.clip(base + ETA * np.where(np.abs(r) >= GATE, r, 0.0), 0, 1)

    t_start = time.time()
    res = {}
    CROSS = "--cross" in sys.argv          # + the other scored month's cells in every context (days 1-5 have little)
    built = {}
    for s in C.SPLITS:
        per = {}
        for i, panel in enumerate(C.T2_PANELS):
            B = SB.bank_for(i, panel, s, "solid3", "")
            M0 = B["M"].copy()
            M0[3] = B["ps_ship"]
            B["M"] = M0
            p0 = np.tensordot(SB.WAU, M0, 1)
            q_au = corr(p0, B["P"])
            T = cell_table(B, q_au, p0)
            T["pid"] = np.full(len(T["y"]), i)
            per[panel] = (B, q_au, T)
        cat = {k: np.concatenate([per[p][2][k] for p in per]) for k in ("stack", "raw", "y", "e", "d", "pid")}
        built[s] = (per, cat)
    for s in C.SPLITS:
        per, cat = built[s]
        other = built[[x for x in C.SPLITS if x != s][0]][1]
        oe = np.flatnonzero(other["e"])
        print(f"[{s}] uncertain cells {len(cat['y']):,} (eligible {int(cat['e'].sum()):,}, queued "
              f"{int(cat['y'][cat['e']].sum()):,}) ({time.time() - t_start:.0f}s)", flush=True)
        preds = {k: np.full(len(cat["y"]), np.nan, np.float32) for k in ("icl stack", "icl raw", "lgb stack")}
        for d in np.unique(cat["d"]):
            q = np.flatnonzero(cat["d"] == d)
            c = np.flatnonzero((cat["d"] < d) & cat["e"])
            if not CROSS and (len(c) < 500 or cat["y"][c].sum() < 20):
                continue
            if len(c) > CAP:
                c = rng.choice(c, CAP, replace=False)
            Xc = {f: cat[f][c] for f in ("stack", "raw")}
            yc = cat["y"][c]
            if CROSS:
                o = rng.choice(oe, min(len(oe), CAP - len(c)), replace=False) if CAP > len(c) else oe[:0]
                Xc = {f: np.concatenate([Xc[f], other[f][o]]) for f in Xc}
                yc = np.concatenate([yc, other["y"][o]])
            for key, feats in (("icl stack", "stack"), ("icl raw", "raw")):
                if key == "icl raw" and CROSS:
                    continue
                m = TabICLClassifier(n_estimators=4, device="cuda", random_state=0)
                m.fit(Xc[feats], yc)
                preds[key][q] = m.predict_proba(cat[feats][q])[:, 1]
            lg = lgb.train({"objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 50,
                            "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1,
                            "num_threads": 8}, lgb.Dataset(Xc["stack"], yc), 300)
            preds["lgb stack"][q] = lg.predict(cat["stack"][q])
            print(f"    day {d}: context {len(yc):,}, queries {len(q):,} ({time.time() - t_start:.0f}s)", flush=True)
        # decisions
        off = 0
        for panel, (B, q_au, T) in per.items():
            k = len(T["y"])
            yb, eb, dd = B["y"] > 0.5, B["e"] > 0.5, B["d"]
            variants = {"AU": q_au}
            for key in preds:
                pk = preds[key][off:off + k]
                ok = np.isfinite(pk)
                if not ok.any():
                    continue
                for a in (0.5, 1.0):
                    q = q_au.copy()
                    q[T["n"][ok], T["h"][ok], T["s"][ok]] = a * pk[ok] + (1 - a) * q_au[T["n"][ok], T["h"][ok], T["s"][ok]]
                    variants[f"{key} a{a}"] = q
            for nm, q in variants.items():
                sc = C.iou(q >= 0.5, yb, eb)
                res.setdefault((s, nm, "all"), {})[panel] = sc.mean()
                res.setdefault((s, nm, "days 1-5"), {})[panel] = sc[dd <= 4].mean()
            off += k
        # cell-level quality on the modelled cells (eligible, with a prediction)
        ok = np.isfinite(preds["icl stack"]) & cat["e"]
        y = cat["y"][ok]
        au = np.concatenate([per[p][1][per[p][2]["n"], per[p][2]["h"], per[p][2]["s"]] for p in per])[ok]
        def bs(p):
            return float(np.mean((p - y) ** 2))
        print(f"[{s}] Brier on modelled cells ({ok.sum():,}): AU {bs(au):.4f}  " +
              "  ".join(f"{k} {bs(v[ok]):.4f}" for k, v in preds.items()), flush=True)
    names = ["AU"] + [f"{k} a{a}" for k in ("icl stack", "icl raw", "lgb stack") for a in (0.5, 1.0)]
    for sub in ("all", "days 1-5"):
        print(f"\n=== ongoing family IoU, {sub} ===")
        ref = {s: C.family_mean(res[(s, "AU", sub)]) for s in C.SPLITS}
        for nm in names:
            if (C.SPLITS[0], nm, sub) not in res:
                continue
            print(f"  {nm:<16}" + "".join(f"  {s[:4]} {C.family_mean(res[(s, nm, sub)]):.4f} "
                                          f"({C.family_mean(res[(s, nm, sub)]) - ref[s]:+.4f})" for s in C.SPLITS))


if __name__ == "__main__":
    main()
