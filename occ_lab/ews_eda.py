"""Early-warning-signal EDA: before a first onset, does the future bottleneck show critical slowing down?

Training months, the h_oof first-onset events. For every link, statistics over the 12 released slots before T:
mean q/cap (the level), var / lag-1 autocorrelation of q/cap and of v/vf, and the lag-1 cross-correlation of q/cap
with the downstream neighbour. Target: the link is queued at T+30 (true, eligible). AUC of each statistic among the
event's links, overall and within the event's mean q/cap quintiles (does it add to the level?).

    .venv/Scripts/python.exe occ_lab/ews_eda.py
"""
import sys
import warnings
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB)); sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta")); sys.path.insert(0, str(LAB.parent))
import numpy as np  # noqa: E402

import common as C  # noqa: E402
import hidden_onset_recover as HR  # noqa: E402

W = 12


def ac1(x):
    x = x - np.nanmean(x, 0, keepdims=True)
    num = np.nansum(x[1:] * x[:-1], 0); den = np.nansum(x * x, 0)
    return np.where(den > 1e-12, num / np.maximum(den, 1e-12), np.nan)


def auc(s, y):
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    if y.all() or not y.any():
        return np.nan
    r = np.argsort(np.argsort(s)) + 1
    return (r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())


def main():
    warnings.simplefilter("ignore")
    from tfb.data import load
    rows = []
    for i, panel in enumerate(C.T2_PANELS):
        z = load(panel, "train")
        ctx = C.Ctx(panel, i)
        vf, cap = np.asarray(ctx.vf, np.float32), np.asarray(ctx.cap, np.float32)
        Hc = np.load(LAB / "cache" / f"h_oof_{panel}.npz")
        down = 1 if panel not in HR.UP_HIGHER else -1
        for n in range(len(Hc["d"])):
            y = Hc["fut"][n, -1] & Hc["efut"][n, -1]
            if not y.any():
                continue
            d, t0 = int(Hc["d"][n]), int(Hc["t0"][n])
            if t0 < W:
                continue
            q = z["obs_flow"][d, t0 - W:t0] / cap; v = z["obs_speed"][d, t0 - W:t0] / vf
            if np.isfinite(v).mean() < 0.5:
                continue
            qn = np.roll(q, -down, 1)
            qc = q - np.nanmean(q, 0); nc = qn - np.nanmean(qn, 0)
            xc = np.nansum(qc[1:] * nc[:-1], 0) / np.sqrt(np.nansum(qc ** 2, 0) * np.nansum(nc ** 2, 0) + 1e-12)
            st = dict(level=np.nanmean(q, 0), vmin=-np.nanmin(v, 0), var_q=np.nanvar(q, 0), ac1_q=ac1(q),
                      var_v=np.nanvar(v, 0), ac1_v=ac1(v), xcorr_dn=xc,
                      trend_q=np.nanmean(q[-4:], 0) - np.nanmean(q[:4], 0),
                      var_q_late=np.nanvar(q[-6:], 0) - np.nanvar(q[:6], 0))
            prior = y & False
            rows.append((panel, y, st, Hc["prob"][n]))
    keys = list(rows[0][2])
    print(f"events {len(rows)}")
    print(f"{'stat':<12}{'AUC':>8}{'AUC in level q5':>18}{'AUC | H top-10':>16}")
    for k in keys + ["H"]:
        a = []; b = []; c = []
        for panel, y, st, pH in rows:
            s = pH if k == "H" else st[k]
            a.append(auc(s, y))
            lv = st["level"]
            qs = np.nanquantile(lv, [0.8])
            m = lv >= qs[0]
            b.append(auc(s[m], y[m]) if m.sum() > 3 else np.nan)
            top = np.argsort(-pH)[:10]
            c.append(auc(s[top], y[top]))
        print(f"{k:<12}{np.nanmean(a):>8.3f}{np.nanmean(b):>18.3f}{np.nanmean(c):>16.3f}")


if __name__ == "__main__":
    main()
