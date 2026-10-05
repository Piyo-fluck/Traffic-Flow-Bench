"""Ongoing error anatomy of AU (the current best), by queue phenomenon (note 159; note 52 did this for model M).

Per window, the queue at T = queued cells of the last history slot (observed only). Every eligible future cell
(h, j) is labelled by what the truth does there, relative to the T-queue:
    stay            queued at T, queued at T+h
    grow-tail       not queued at T, queued at T+h, adjacent run (<= 2 links) on the UPSTREAM side of a T-queue
    grow-head       ... on the DOWNSTREAM side
    grow-new        queued at T+h, > 2 links from any T-queue (a new or fast-moving queue)
    clear-head      queued at T, not at T+h, at the downstream end of its T-queue run (outermost 2 links)
    clear-tail      ... at the upstream end
    clear-inner     ... inside the run
    far             never queued at T nor at T+h
Upstream per panel = the side toward which true queues grow more often (measured on the pool itself).
For each class: cells, FN / FP counts, share of the IoU loss (each window's 1 - IoU split over its error cells), and
AU's probability on the missed cells (near-miss p >= 0.3 vs invisible p < 0.1).

    .venv/Scripts/python.exe occ_lab/og_anatomy.py
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
CLS = ["stay", "grow-tail", "grow-head", "grow-new", "clear-head", "clear-tail", "clear-inner", "far"]


def runs(mask):
    j = np.flatnonzero(mask)
    if not len(j):
        return []
    cut = np.flatnonzero(np.diff(j) > 1)
    return [(a[0], a[-1]) for a in np.split(j, cut + 1)]


def main() -> None:
    warnings.simplefilter("ignore")
    gbm = lgb.Booster(model_file=str(LAB / "models" / "residual_lgb.txt"))

    def corr(base, P):
        r = gbm.predict(global_features(P, base).reshape(-1, 15)).reshape(base.shape).astype(np.float32)
        return np.clip(base + ETA * np.where(np.abs(r) >= GATE, r, 0.0), 0, 1)

    tot = {}
    for i, panel in enumerate(C.T2_PANELS):
        ctx, stat, W = C.windows(i, panel)
        vf = np.asarray(ctx.vf, np.float32)
        for s in C.SPLITS:
            B = SB.bank_for(i, panel, s, "solid3", "")
            M0 = B["M"].copy(); M0[3] = B["ps_ship"]
            q = corr(np.tensordot(SB.WAU, M0, 1), B["P"])
            og = W[s]["og"]
            y, e = B["y"] > 0.5, B["e"] > 0.5
            qT = og["ok"][:, -1] & (np.nan_to_num(og["v"][:, -1], nan=1e9) <= 0.6 * vf)
            N, _, S = y.shape
            # orientation: does the truth grow toward lower (-1) or higher (+1) link index?
            lo = hi = 0
            for n in range(N):
                for a, b in runs(qT[n]):
                    fut = y[n, -1]
                    lo += int(fut[max(0, a - 3):a].sum()); hi += int(fut[b + 1:b + 4].sum())
            up = -1 if lo >= hi else 1
            pred = q >= 0.5
            for n in range(N):
                lab = np.full((F, S), "far", object)
                near = np.zeros(S, bool)
                side = np.zeros(S, int)                      # -1 upstream of a run, +1 downstream, 0 none
                endpos = {}
                for a, b in runs(qT[n]):
                    near[max(0, a - 2):b + 3] = True
                    us, ds = (range(max(0, a - 2), a), range(b + 1, min(S, b + 3))) if up == -1 else \
                        (range(b + 1, min(S, b + 3)), range(max(0, a - 2), a))
                    for j in us: side[j] = -1
                    for j in ds: side[j] = 1
                    for j in range(a, b + 1):
                        if b - a < 2:
                            endpos[j] = "clear-head"
                        elif (up == -1 and j >= b - 1) or (up == 1 and j <= a + 1):
                            endpos[j] = "clear-head"
                        elif (up == -1 and j <= a + 1) or (up == 1 and j >= b - 1):
                            endpos[j] = "clear-tail"
                        else:
                            endpos[j] = "clear-inner"
                for h in range(F):
                    for j in range(S):
                        if qT[n, j]:
                            lab[h, j] = "stay" if y[n, h, j] else endpos[j]
                        elif y[n, h, j]:
                            lab[h, j] = ("grow-tail" if side[j] == -1 else "grow-head") if near[j] and side[j] else \
                                ("grow-new" if not near[j] else "grow-tail")
                ee = e[n]
                obsT = og["ok"][n, -1].copy()                  # observed at T (+-1 link)
                obsT[1:] &= og["ok"][n, -1][:-1]; obsT[:-1] &= og["ok"][n, -1][1:]
                hid = ~obsT
                # any history observation of the link in the last 15 min
                obs15 = og["ok"][n, -3:].any(0)
                for c in ("grow-new", "far"):
                    mk = (lab == c) & ee
                    fnc, fpc = (y[n] & ~pred[n] & mk), (pred[n] & ~y[n] & mk)
                    for nm, msk in (("hidden at T (+-1)", np.broadcast_to(hid, mk.shape)),
                                    ("no obs in 15 min", np.broadcast_to(~obs15, mk.shape))):
                        t2 = tot.setdefault((s, c, nm), np.zeros(4))
                        t2[0] += (mk & msk).sum(); t2[1] += mk.sum()
                        t2[2] += ((fnc | fpc) & msk).sum(); t2[3] += (fnc | fpc).sum()
                fn = y[n] & ~pred[n] & ee
                fp = pred[n] & ~y[n] & ee
                union = ((pred[n] | y[n]) & ee).sum()
                inter = (pred[n] & y[n] & ee).sum()
                loss = 1 - inter / union if union else 0.0
                nerr = fn.sum() + fp.sum()
                for c in CLS:
                    mk = (lab == c) & ee
                    t = tot.setdefault((s, c), np.zeros(8))
                    t[0] += mk.sum(); t[1] += (fn & mk).sum(); t[2] += (fp & mk).sum()
                    if nerr:
                        t[3] += loss * ((fn | fp) & mk).sum() / nerr
                    pm = q[n][fn & mk]
                    t[4] += (pm >= 0.3).sum(); t[5] += (pm < 0.1).sum()
                    t[6] += (d4 := float(B["d"][n] <= 4)) * ((fn | fp) & mk).sum()
                    t[7] += (y[n] & mk).sum()
                tl = tot.setdefault((s, "_windows"), np.zeros(8)); tl[0] += 1; tl[3] += loss
        print(f"  {panel} (upstream = {'lower' if up == -1 else 'higher'} index)", flush=True)
    for s in C.SPLITS:
        W_ = tot[(s, "_windows")]
        print(f"\n=== {s}: {int(W_[0])} windows, mean IoU loss {W_[3] / W_[0]:.4f} ===")
        print(f"{'class':<12}{'cells':>10}{'true':>9}{'FN':>8}{'FP':>8}{'err rate':>9}{'loss share':>11}"
              f"{'FN p>=.3':>9}{'FN p<.1':>9}")
        for c in CLS:
            t = tot[(s, c)]
            er = (t[1] + t[2]) / max(t[0], 1)
            print(f"{c:<12}{int(t[0]):>10,}{int(t[7]):>9,}{int(t[1]):>8,}{int(t[2]):>8,}{er:>9.3f}"
                  f"{t[3] / W_[3]:>11.3f}{t[4] / max(t[1], 1):>9.2f}{t[5] / max(t[1], 1):>9.2f}")
        for c in ("grow-new", "far"):
            for nm in ("hidden at T (+-1)", "no obs in 15 min"):
                t2 = tot[(s, c, nm)]
                print(f"   {c:<9} {nm:<18}: share of the class's cells {t2[0] / max(t2[1], 1):.3f}, "
                      f"of its errors {t2[2] / max(t2[3], 1):.3f}")


if __name__ == "__main__":
    main()
