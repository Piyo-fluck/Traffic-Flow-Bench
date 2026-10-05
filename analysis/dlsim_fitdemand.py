"""Note 169: v5 generator (released FD, receiving-capacity gates) with the censored demand fitted by simulation.

Parameters: per entry x 5 time blocks, a multiplier on the demand part that fills censored (queued) slots, plus
the same for the upstream boundary. Each round simulates the fit days; entry g (merging at sensored link b) moves by
the queued-share mismatch (true - sim) averaged over links b-3..b in each block: theta <- theta exp(eta mismatch /
scale), clipped to [0.5, 2.5]. Best round by fit-day IoU; evaluated on held-out days. Saves
work/dlsim_theta_<panel>.npz.

    .venv/Scripts/python.exe analysis/dlsim_fitdemand.py [panels]
"""
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "analysis"))
import numpy as np  # noqa: E402

from dlsim_gen4 import Gen4  # noqa: E402
from ctm_gen import CTMGen  # noqa: E402

USE_CTM = __import__("os").environ.get("CTM", "") == "1"
ACT = __import__("os").environ.get("ACT", "") == "1"                  # fit each head's day-activation share


def activation(q, blocks):
    """(n_links, 5): share of days with >= 2 head slots (link queued, next free) in each block; q list of (T, n)"""
    out = []
    for a in q:
        hd = np.zeros_like(a); hd[:, :-1] = a[:, :-1] & ~a[:, 1:]
        out.append(np.stack([hd[blocks == k].sum(0) >= 2 for k in range(5)], 1))
    return np.mean(out, 0)
from tfb.data import load  # noqa: E402

A_, B_ = 72, 240
LO = float(__import__("os").environ.get("THETA_LO", 0.5))
ITERS = int(__import__("os").environ.get("THETA_ITERS", 15))


def run(g, z, days, th, tu, vf):
    out = []
    for d in days:
        up, ent = g.censored_demand(z["true_flow"][d], z["true_speed"][d], int(z["weekday"][d]), th, tu)
        V, _ = g.simulate(up, ent)
        out.append((np.nan_to_num(V, nan=1e9) <= 0.6 * vf)[:, g.order])
    return out


def main():
    warnings.simplefilter("ignore")
    t0 = time.time()
    for panel in sys.argv[1:] or ["D12_I5_S"]:
        z = load(panel, "train")
        vf = np.asarray(z["free_speed"], float)
        g = CTMGen(panel, z) if USE_CTM else Gen4(panel, z)
        g.fit_profiles(z)
        nd = len(z["dates"])
        fit, ev = (np.arange(0, nd, 3)[:60], np.arange(1, nd, 6)[:30]) if USE_CTM else (np.arange(0, nd, 6)[:24], np.arange(3, nd, 6)[:16])
        truth = {d: (np.nan_to_num(z["true_speed"][d], nan=1e9) <= 0.6 * vf)[:, g.order] & np.isfinite(z["true_speed"][d])[:, g.order]
                 for d in np.concatenate([fit, ev])}
        ne = len(g.ent)
        th, tu = np.ones((ne, 5)), np.ones(5)
        blocks = g.BLOCKS

        def score(days, th, tu):
            sims = run(g, z, days, th, tu, vf)
            iou = np.mean([(a[A_:B_] & truth[d][A_:B_]).sum() / max((a[A_:B_] | truth[d][A_:B_]).sum(), 1) for a, d in zip(sims, days)])
            Ms = np.mean([a[A_:B_].mean(0) for a in sims], 0); Mt = np.mean([truth[d][A_:B_].mean(0) for d in days], 0)
            return iou, Ms.mean(), Mt.mean(), np.corrcoef(Ms, Mt)[0, 1], sims
        best = (-1e9, th.copy(), tu.copy())
        for it in range(ITERS):
            iou, ss, st, c, sims = score(fit, th, tu)
            obj = iou - abs(ss - st) if USE_CTM else iou                   # do not buy overlap with extra queue
            if obj > best[0]:
                best = (obj, th.copy(), tu.copy())
            if ACT:                                                         # head activation share mismatch
                at = activation([truth[d] for d in fit], blocks); asim = activation(sims, blocks)
                gap = at - asim
                keyh = (at.max(1) >= 0.05) | (asim.max(1) >= 0.05)
                act_err = np.abs(gap[keyh]).mean() if keyh.any() else 0.0
                obj = -act_err - abs(ss - st)
                if obj > best[0] or it == 0:
                    best = (obj, th.copy(), tu.copy())
                eta = 1.5 / (1 + it / 5)
                for i, b in enumerate(g.ent):
                    hs = gap[max(0, b - 1):min(g.n, b + 4)]
                    th[i] = np.clip(th[i] * np.exp(eta * hs.mean(0)), LO, 2.5)
                tu[:] = np.clip(tu * np.exp(eta * gap[:4].mean(0)), LO, 2.5)
                if it % 5 == 4 or it == 0:
                    print(f"  {panel} it {it + 1}: activation err {act_err:.3f} share {ss:.3f}/{st:.3f} IoU {iou:.3f} ({time.time() - t0:.0f}s)", flush=True)
                continue
            # per link x block queued-share mismatch
            mis = np.zeros((g.n, 5))
            for a, d in zip(sims, fit):
                for k in range(5):
                    m = blocks == k
                    mis[:, k] += truth[d][m].mean(0) - a[m].mean(0)
            mis /= len(fit)
            eta = 1.5 / (1 + it / 5)
            sc = max(np.abs(mis).max(), 1e-3)
            for i, b in enumerate(g.ent):
                th[i] = np.clip(th[i] * np.exp(eta * mis[max(0, b - 3):b + 1].mean(0) / sc), LO, 2.5)
            tu[:] = np.clip(tu * np.exp(eta * mis[:3].mean(0) / sc), LO, 2.5)
            if it % 5 == 4 or it == 0:
                print(f"  {panel} it {it + 1}: fit IoU {iou:.3f} share {ss:.3f}/{st:.3f} corr {c:.2f} ({time.time() - t0:.0f}s)", flush=True)
        iou, ss, st, c, sims_ev = score(ev, best[1], best[2])
        if ACT:
            at_e = activation([truth[d] for d in ev], blocks); as_e = activation(sims_ev, blocks)
            kh = (at_e.max(1) >= 0.05) | (as_e.max(1) >= 0.05)
            print(f"   held-out activation err {np.abs(at_e - as_e)[kh].mean():.3f}  heads true>=0.3: {int((at_e.max(1) >= 0.3).sum())}, sim>=0.3: {int((as_e.max(1) >= 0.3).sum())}, spurious (sim>=0.3, true<0.1) {int(((as_e.max(1) >= 0.3) & (at_e.max(1) < 0.1)).sum())}", flush=True)
        print(f"{panel:<10} BEST obj {best[0]:.3f} | held-out: share {ss:.3f} / true {st:.3f}  map corr {c:.2f}  daily IoU {iou:.3f}", flush=True)
        np.savez(ROOT / "work" / f"dlsim_theta_{panel}.npz", theta=best[1], theta_up=best[2])


if __name__ == "__main__":
    main()
