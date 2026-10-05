"""Idea 2 (Traffic4cast 2021 temporal domain adaptation), moved to where the run shift actually is.

T4C 2021 rescaled every pixel's INPUT by lambda = mean_2019 / mean_2020. Here the free-flow flow levels of a link
differ by only 2-6 % between the training and the scored months (and not where the queues moved); what shifts is
WHERE and WHEN queues occur. So the same per-location ratio is applied to the OUTPUT, per link x time of day:

    r_train(j, t)   queued share of link j (+-1 link) within +-30 min of t on the training days of the same day type
    r_month(j, t)   the same on the scored month's released days < d (and today's slots < T), n = cells seen
    w(j, t)         ((n r_month + kappa r_train) / (n + kappa) + eps) / (r_train + eps)
    p'              sigma(logit p_AU + beta log w)          (beta chosen on one month, applied to the other)
Note 140's prior shift was one global factor per run (AU stays calibrated); this one is local.

    .venv/Scripts/python.exe occ_lab/tda_output.py
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

import common as C  # noqa: E402
import select_bank as SB  # noqa: E402
from online_adapter import ETA, GATE, global_features  # noqa: E402

F = C.F
HW = 6                       # +-30 min
EPS = 0.01
KAPPAS = (200.0, 1000.0, 5000.0)
BETAS = (0.25, 0.5, 1.0)


def qshare(z, vf, days):
    """(len(days), T, S) queued flags and eligibility, both +-1-link dilated."""
    v = z["obs_speed"][days]
    ok = z["elig"][days] & np.isfinite(v)
    q = ok & (np.nan_to_num(v, nan=1e9) <= 0.6 * vf)
    qd, od = q.copy(), ok.copy()
    qd[..., 1:] |= q[..., :-1]; qd[..., :-1] |= q[..., 1:]
    od[..., 1:] |= ok[..., :-1]; od[..., :-1] |= ok[..., 1:]
    return qd.astype(np.float32), od.astype(np.float32)


def tod_sum(a):
    """(D, T, S) -> (T, S) sums over days and +-HW slots."""
    s = a.sum(0)
    c = np.cumsum(np.concatenate([np.zeros((1, s.shape[1])), s]), 0)
    T = s.shape[0]
    lo = np.clip(np.arange(T) - HW, 0, T); hi = np.clip(np.arange(T) + HW + 1, 0, T)
    return c[hi] - c[lo]


def main() -> None:
    warnings.simplefilter("ignore")
    from tfb.data import load
    gbm = lgb.Booster(model_file=str(LAB / "models" / "residual_lgb.txt"))

    def corr(base, P):
        r = gbm.predict(global_features(P, base).reshape(-1, 15)).reshape(base.shape).astype(np.float32)
        return np.clip(base + ETA * np.where(np.abs(r) >= GATE, r, 0.0), 0, 1)

    res = {}
    for i, panel in enumerate(C.T2_PANELS):
        ctx, stat, W = C.windows(i, panel)
        vf = np.asarray(ctx.vf, np.float32)
        zt = load(panel, "train")
        tr = {}
        for we in (False, True):
            days = np.flatnonzero((zt["weekday"] >= 5) == we)
            qd, od = qshare(zt, vf, days)
            tr[we] = tod_sum(qd) / np.maximum(tod_sum(od), 1)
        for s in C.SPLITS:
            B = SB.bank_for(i, panel, s, "solid3", "")
            M0 = B["M"].copy(); M0[3] = B["ps_ship"]
            q_au = corr(np.tensordot(SB.WAU, M0, 1), B["P"])
            z = W[s]["z"]
            og = W[s]["og"]
            qd_all, od_all = qshare(z, vf, np.arange(z["obs_speed"].shape[0]))
            wk = z["weekday"] >= 5
            N = len(og["d"])
            logw = {k: np.zeros(q_au.shape, np.float32) for k in KAPPAS}
            for n in range(N):
                d, t0 = int(og["d"][n]), int(og["t0"][n])
                same = np.flatnonzero((np.arange(d) >= 0) & (wk[:d] == wk[d]))
                num = tod_sum(qd_all[same]) if len(same) else 0.0
                den = tod_sum(od_all[same]) if len(same) else 0.0
                # today's slots before T
                qt, ot = qd_all[d:d + 1, :t0], od_all[d:d + 1, :t0]
                if t0 > 0:
                    pad = np.zeros((1, z["obs_speed"].shape[1] - t0, len(vf)), np.float32)
                    num = num + tod_sum(np.concatenate([qt, pad], 1)); den = den + tod_sum(np.concatenate([ot, pad], 1))
                slots = np.clip(t0 + np.arange(1, F + 1), 0, qd_all.shape[1] - 1)
                rt = tr[bool(wk[d])][slots]
                nn = np.asarray(den)[slots] if np.ndim(den) else np.zeros_like(rt)
                rm = np.asarray(num)[slots] / np.maximum(nn, 1) if np.ndim(num) else np.zeros_like(rt)
                for k in KAPPAS:
                    shr = (nn * rm + k * rt) / (nn + k)
                    logw[k][n] = np.log((shr + EPS) / (rt + EPS))
            yb, eb, dd = B["y"] > 0.5, B["e"] > 0.5, B["d"]
            lg = np.log(np.clip(q_au, 1e-5, 1 - 1e-5)) - np.log1p(-np.clip(q_au, 1e-5, 1 - 1e-5))
            sc = C.iou(q_au >= 0.5, yb, eb)
            res.setdefault((s, "AU", "all"), {})[panel] = sc.mean()
            res.setdefault((s, "AU", "d15"), {})[panel] = sc[dd <= 4].mean()
            for k in KAPPAS:
                for b in BETAS:
                    q = 1 / (1 + np.exp(-(lg + b * logw[k])))
                    sc = C.iou(q >= 0.5, yb, eb)
                    res.setdefault((s, (k, b), "all"), {})[panel] = sc.mean()
                    res.setdefault((s, (k, b), "d15"), {})[panel] = sc[dd <= 4].mean()
        print(f"  {panel}", flush=True)
    ref = {(s, sub): C.family_mean(res[(s, "AU", sub)]) for s in C.SPLITS for sub in ("all", "d15")}
    print(f"\n{'kappa, beta':<16}{'all val':>9}{'all priv':>10}{'d1-5 val':>10}{'d1-5 priv':>11}")
    for k in KAPPAS:
        for b in BETAS:
            v = [C.family_mean(res[(s, (k, b), sub)]) - ref[(s, sub)] for sub in ("all", "d15") for s in C.SPLITS]
            print(f"{f'{k:g}, {b}':<16}{v[0]:>+9.4f}{v[1]:>+10.4f}{v[2]:>+10.4f}{v[3]:>+11.4f}")
    # cross-month choice
    keys = [(k, b) for k in KAPPAS for b in BETAS]
    for sub in ("all", "d15"):
        pick = {s: max(keys, key=lambda kb: C.family_mean(res[(s, kb, sub)])) for s in C.SPLITS}
        v = C.family_mean(res[("validation", pick["private"], sub)]) - ref[("validation", sub)]
        p = C.family_mean(res[("private", pick["validation"], sub)]) - ref[("private", sub)]
        print(f"cross-month ({sub}): val <- {pick['private']} {v:+.4f}   priv <- {pick['validation']} {p:+.4f}")


if __name__ == "__main__":
    main()
