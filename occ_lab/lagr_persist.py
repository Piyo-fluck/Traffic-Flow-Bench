"""Lagrangian persistence in physical distance: is there room for a Lagrangian ongoing member? (note 162)

Link spacing is 0.04-1.4 km, so waves have no constant speed in link units (note 162). Per window:
    Q_T       queued at the last history slot, or unobserved there and queued in the last 15 min
    LP_w(h)   Q_T carried UPSTREAM by w * 5(h+1) min in km (every cell between a queued cell and the point
              w * t km upstream of it), union Q_T itself; w in 5..25 km/h
Measured against AU (tau 0.5): the share of AU's missed 'new' cells (> 2 links from the observed T-queue) that the
extension covers, the precision of the cells it adds outside AU's set, and AU's logit + beta on those added cells
(beta chosen on one month, applied to the other; all days and days 1-5).

    .venv/Scripts/python.exe occ_lab/lagr_persist.py
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
WS = (5, 10, 15, 20, 25)
BETAS = (0.5, 1.0, 2.0, 3.0)
UP_HIGHER = {"D7_I10_E"}                      # note 159: upstream = higher link index only here


def lp_masks(qT, pos, up_lower):
    """qT (S,) bool, pos (S,) km -> {w: (F, S) bool}."""
    S = len(qT)
    out = {}
    j = np.flatnonzero(qT)
    for w in WS:
        m = np.zeros((F, S), bool)
        for h in range(F):
            D = w * 5 * (h + 1) / 60.0
            cov = qT.copy()
            for k in j:
                if up_lower:
                    cov |= (pos <= pos[k]) & (pos >= pos[k] - D)
                else:
                    cov |= (pos >= pos[k]) & (pos <= pos[k] + D)
            m[h] = cov
        out[w] = m
    return out


def main() -> None:
    warnings.simplefilter("ignore")
    gbm = lgb.Booster(model_file=str(LAB / "models" / "residual_lgb.txt"))

    def corr(base, P):
        r = gbm.predict(global_features(P, base).reshape(-1, 15)).reshape(base.shape).astype(np.float32)
        return np.clip(base + ETA * np.where(np.abs(r) >= GATE, r, 0.0), 0, 1)

    st = {}
    res = {}
    for i, panel in enumerate(C.T2_PANELS):
        ctx, stat, W = C.windows(i, panel)
        vf = np.asarray(ctx.vf, np.float32)
        mp = np.asarray(ctx.z["milepost"], float) if "milepost" in ctx.z else None
        if mp is None:
            from tfb.data import load
            mp = np.asarray(load(panel, "train")["milepost"], float)
        pos = np.concatenate([[0], np.cumsum(np.abs(np.diff(mp)))]) * 1.609
        up_lower = panel not in UP_HIGHER
        for s in C.SPLITS:
            B = SB.bank_for(i, panel, s, "solid3", "")
            M0 = B["M"].copy(); M0[3] = B["ps_ship"]
            q = corr(np.tensordot(SB.WAU, M0, 1), B["P"])
            lg = np.log(np.clip(q, 1e-5, 1 - 1e-5)) - np.log1p(-np.clip(q, 1e-5, 1 - 1e-5))
            og = W[s]["og"]; y, e, dd = B["y"] > 0.5, B["e"] > 0.5, B["d"]
            ok, v = og["ok"], og["v"]
            qd = ok & (np.nan_to_num(v, nan=1e9) <= 0.6 * vf)
            N, _, S = y.shape
            ext = {w: np.zeros((N, F, S), bool) for w in WS}
            for n in range(N):
                qobs = qd[n, -1]
                qT = qobs | (~ok[n, -1] & qd[n, -3:].any(0))
                if not qT.any():
                    continue
                m = lp_masks(qT, pos, up_lower)
                for w in WS:
                    ext[w][n] = m[w]
                # AU's missed new cells (> 2 links from the observed T-queue)
                zone = np.zeros(S, bool)
                for j in np.flatnonzero(qobs):
                    zone[max(0, j - 2):j + 3] = True
                fn_new = y[n] & e[n] & (q[n] < 0.5) & ~zone[None]
                t = st.setdefault((s, "fn_new"), [0]); t[0] += fn_new.sum()
                for w in WS:
                    add = m[w] & (q[n] < 0.5) & e[n]
                    a = st.setdefault((s, w), np.zeros(3))
                    a[0] += (fn_new & m[w]).sum(); a[1] += add.sum(); a[2] += (add & y[n]).sum()
            base = C.iou(q >= 0.5, y, e)
            res.setdefault((s, "AU"), {})[panel] = (base.mean(), base[dd <= 4].mean())
            for w in WS:
                for b in BETAS:
                    q2 = lg + b * (ext[w] & (q < 0.5))
                    sc = C.iou(q2 >= 0.0, y, e)
                    res.setdefault((s, (w, b)), {})[panel] = (sc.mean(), sc[dd <= 4].mean())
        print(f"  {panel} (pos span {pos[-1]:.1f} km)", flush=True)
    for s in C.SPLITS:
        nf = st[(s, "fn_new")][0]
        print(f"\n{s}: AU's missed new cells {nf:,}")
        for w in WS:
            a = st[(s, w)]
            print(f"  w {w:>2} km/h: covers {a[0] / max(nf, 1):.3f} of them; adds {int(a[1]):,} cells outside AU's set, "
                  f"precision {a[2] / max(a[1], 1):.3f}")
    ref = {s: [C.family_mean({p: res[(s, 'AU')][p][k] for p in C.T2_PANELS}) for k in (0, 1)] for s in C.SPLITS}
    print(f"\n{'w, beta':<10}{'all val':>9}{'all priv':>10}{'d1-5 val':>10}{'d1-5 priv':>11}")
    for w in WS:
        for b in BETAS:
            v = [C.family_mean({p: res[(s, (w, b))][p][k] for p in C.T2_PANELS}) - ref[s][k] for k in (0, 1) for s in C.SPLITS]
            print(f"{f'{w}, {b}':<10}{v[0]:>+9.4f}{v[1]:>+10.4f}{v[2]:>+10.4f}{v[3]:>+11.4f}")


if __name__ == "__main__":
    main()
