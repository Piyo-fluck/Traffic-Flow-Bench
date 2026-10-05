"""ASDA-style moving-jam tracking (Kerner's ASDA/FOTO): room test against AU (note 163).

note 162 transported EVERY T-queue upstream -> precision 3-5 %. ASDA moves only what the history shows moving:
    clusters   runs of queued cells (observed) at slot T-k (k = 2, i.e. 10 min before T) and at T
    moving jam a T-cluster whose DOWNSTREAM front lies >= DMIN km upstream of the downstream front of the overlapping
               / nearest T-k cluster (a bottleneck-bound queue keeps its downstream front)
    speed      the measured downstream-front displacement / 10 min, clipped to 5..30 km/h
    forecast   the whole jam (both fronts) advected upstream by speed x 5(h+1) min
Measured against AU (tau 0.5): coverage of AU's missed 'new' cells (> 2 links from the observed T-queue), the cells
added outside AU's set and their precision, and AU's logit + beta on them (all days / days 1-5).

    .venv/Scripts/python.exe occ_lab/asda_room.py
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

import common as C  # noqa: E402
import select_bank as SB  # noqa: E402
from online_adapter import ETA, GATE, global_features  # noqa: E402

F = C.F
UP_HIGHER = {"D7_I10_E"}
DMINS = (0.3, 0.6, 1.0)
BETAS = (0.5, 1.0, 2.0)


def runs(mask):
    j = np.flatnonzero(mask)
    if not len(j):
        return []
    cut = np.flatnonzero(np.diff(j) > 1)
    return [(a[0], a[-1]) for a in np.split(j, cut + 1)]


def main() -> None:
    warnings.simplefilter("ignore")
    from tfb.data import load
    gbm = lgb.Booster(model_file=str(LAB / "models" / "residual_lgb.txt"))

    def corr(base, P):
        r = gbm.predict(global_features(P, base).reshape(-1, 15)).reshape(base.shape).astype(np.float32)
        return np.clip(base + ETA * np.where(np.abs(r) >= GATE, r, 0.0), 0, 1)

    st, res = {}, {}
    for i, panel in enumerate(C.T2_PANELS):
        ctx, stat, W = C.windows(i, panel)
        vf = np.asarray(ctx.vf, np.float32)
        mp = np.asarray(load(panel, "train")["milepost"], float)
        pos = np.concatenate([[0], np.cumsum(np.abs(np.diff(mp)))]) * 1.609
        up_lower = panel not in UP_HIGHER
        # 'upstream coordinate': grows upstream
        u = -pos if up_lower else pos
        for s in C.SPLITS:
            B = SB.bank_for(i, panel, s, "solid3", "")
            M0 = B["M"].copy(); M0[3] = B["ps_ship"]
            q = corr(np.tensordot(SB.WAU, M0, 1), B["P"])
            lg = np.log(np.clip(q, 1e-5, 1 - 1e-5)) - np.log1p(-np.clip(q, 1e-5, 1 - 1e-5))
            og = W[s]["og"]; y, e, dd = B["y"] > 0.5, B["e"] > 0.5, B["d"]
            ok, v = og["ok"], og["v"]
            qd = ok & (np.nan_to_num(v, nan=1e9) <= 0.6 * vf)
            N, _, S = y.shape
            ext = {dm: np.zeros((N, F, S), bool) for dm in DMINS}
            n_jams = {dm: 0 for dm in DMINS}
            for n in range(N):
                now, before = runs(qd[n, -1]), runs(qd[n, -3])
                if not now or not before:
                    continue
                for a, b in now:
                    dn_now = min(u[a], u[b])                      # downstream front = smallest upstream coordinate
                    # the T-k cluster overlapping this one, or the nearest one downstream of it
                    cands = [(a2, b2) for a2, b2 in before if min(u[a2], u[b2]) <= max(u[a], u[b])]
                    if not cands:
                        continue
                    a2, b2 = min(cands, key=lambda c: abs(min(u[c[0]], u[c[1]]) - dn_now))
                    disp = dn_now - min(u[a2], u[b2])             # > 0: moved upstream
                    lo_u, hi_u = min(u[a], u[b]), max(u[a], u[b])
                    for dm in DMINS:
                        if disp < dm:
                            continue
                        n_jams[dm] += 1
                        spd = np.clip(disp / (10 / 60), 5, 30)
                        for h in range(F):
                            sh = spd * 5 * (h + 1) / 60
                            ext[dm][n, h] |= (u >= lo_u + sh) & (u <= hi_u + sh)
                zone = np.zeros(S, bool)
                for j in np.flatnonzero(qd[n, -1]):
                    zone[max(0, j - 2):j + 3] = True
                fn_new = y[n] & e[n] & (q[n] < 0.5) & ~zone[None]
                t = st.setdefault((s, "fn_new"), [0]); t[0] += fn_new.sum()
                for dm in DMINS:
                    add = ext[dm][n] & (q[n] < 0.5) & e[n]
                    a_ = st.setdefault((s, dm), np.zeros(3))
                    a_[0] += (fn_new & ext[dm][n]).sum(); a_[1] += add.sum(); a_[2] += (add & y[n]).sum()
            for dm in DMINS:
                st.setdefault((s, dm, "jams"), [0])[0] += n_jams[dm]
            base = C.iou(q >= 0.5, y, e)
            res.setdefault((s, "AU"), {})[panel] = (base.mean(), base[dd <= 4].mean())
            for dm in DMINS:
                for b in BETAS:
                    sc = C.iou((lg + b * (ext[dm] & (q < 0.5))) >= 0.0, y, e)
                    res.setdefault((s, (dm, b)), {})[panel] = (sc.mean(), sc[dd <= 4].mean())
        print(f"  {panel}", flush=True)
    for s in C.SPLITS:
        nf = st[(s, "fn_new")][0]
        print(f"\n{s}: AU's missed new cells {nf:,}")
        for dm in DMINS:
            a = st[(s, dm)]
            print(f"  downstream front moved >= {dm} km: moving jams {st[(s, dm, 'jams')][0]:,}; covers {a[0] / max(nf, 1):.3f}"
                  f" of the missed; adds {int(a[1]):,} cells outside AU's set, precision {a[2] / max(a[1], 1):.3f}")
    ref = {s: [C.family_mean({p: res[(s, 'AU')][p][k] for p in C.T2_PANELS}) for k in (0, 1)] for s in C.SPLITS}
    print(f"\n{'dmin, beta':<12}{'all val':>9}{'all priv':>10}{'d1-5 val':>10}{'d1-5 priv':>11}")
    for dm in DMINS:
        for b in BETAS:
            v = [C.family_mean({p: res[(s, (dm, b))][p][k] for p in C.T2_PANELS}) - ref[s][k] for k in (0, 1) for s in C.SPLITS]
            print(f"{f'{dm}, {b}':<12}{v[0]:>+9.4f}{v[1]:>+10.4f}{v[2]:>+10.4f}{v[3]:>+11.4f}")


if __name__ == "__main__":
    main()
