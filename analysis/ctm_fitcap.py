"""Note 169: with the demand fixed and unbiased (observed in free slots, volume-conserving fill, multipliers 1), fit
the CTM's bottleneck capacities - the receiving capacity of the first cell downstream of every sensored link.

Start: released capacities with the hidden-bottleneck values (CTMGen). Each round simulates the fit days; for
sensored link h the head occupancy (h queued, h+1 free) and queued share are compared with the truth and
c_h <- c_h exp(-eta [(H_true - H_sim) / s_H + 0.5 (dQ_h) / s_Q]), dQ_h = the queued-share mismatch at h minus at
h+1 (the extent just upstream of the bottleneck), clipped to [0.5, 1.1] x the released capacity of that cell.
Best round by fit-day IoU - |share gap|. Held-out: free-flow flow error, share, map corr, daily IoU.
Saves work/ctm_caps_<panel>.npy (cell capacities after each sensored link, travel order).

    CONSERVE_FILL=1 .venv/Scripts/python.exe analysis/ctm_fitcap.py [panels]
"""
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "analysis"))
import numpy as np  # noqa: E402

from ctm_gen import CTMGen  # noqa: E402
from tfb.data import load  # noqa: E402

A, B = 72, 240


def run(g, z, days):
    out = []
    for d in days:
        up, ent = g.censored_demand(z["true_flow"][d], z["true_speed"][d], int(z["weekday"][d]))
        V, Q = g.simulate(up, ent)
        out.append((V[:, g.order], Q[:, g.order]))
    return out


def stats(Vs, vf):
    qd = (np.nan_to_num(Vs, nan=1e9) <= 0.6 * vf)[A:B]
    head = np.zeros(qd.shape[1]); head[:-1] = (qd[:, :-1] & ~qd[:, 1:]).mean(0)
    return head, qd.mean(0), qd


def main():
    warnings.simplefilter("ignore")
    t0 = time.time()
    for panel in sys.argv[1:] or ["D12_I5_S"]:
        z = load(panel, "train")
        g = CTMGen(panel, z); g.fit_profiles(z)
        o, vf = g.order, g.vf
        nd = len(z["dates"])
        fit, ev = np.arange(0, nd, 3)[:50], np.arange(1, nd, 6)[:30]
        cells = g.sens_cell[:-1] + 1                                     # first cell after each sensored link
        base = np.array([g.cap[min(h + 1, g.n - 1)] for h in range(g.n - 1)], float)
        truth = {d: stats(z["true_speed"][d][:, o], vf) for d in np.concatenate([fit, ev])}
        Ht = np.mean([truth[d][0] for d in fit], 0); Qt = np.mean([truth[d][1] for d in fit], 0)

        def set_caps(c):
            g.cell_cap[cells] = c
            kc = c / g.cell_vf[cells]
            g.cell_w[cells] = c / np.maximum(g.cell_kj[cells] - kc, 1e-6)

        def evaluate(days):
            sims = run(g, z, days)
            iou, fe, ms, mt = [], [], [], []
            for (V, Q), d in zip(sims, days):
                a = stats(V, vf)[2]; b = truth[d][2]
                iou.append((a & b).sum() / max((a | b).sum(), 1)); ms.append(a.mean(0)); mt.append(b.mean(0))
                qt, vt = z["true_flow"][d][:, o], z["true_speed"][d][:, o]
                free = (vt > 0.9 * vf) & (V > 0.9 * vf)
                fe.append(np.nanmedian(np.where(free, (Q - qt) / np.maximum(qt, 1), np.nan)[60:252]))
            Ms, Mt = np.mean(ms, 0), np.mean(mt, 0)
            return np.mean(iou), Ms.mean(), Mt.mean(), np.corrcoef(Ms, Mt)[0, 1], np.nanmedian(fe), sims
        c = g.cell_cap[cells].copy()
        best = (-1e9, c.copy())
        for it in range(30):
            set_caps(c)
            iou, ss, st, corr, fe, sims = evaluate(fit)
            obj = iou - abs(ss - st)
            if obj > best[0]:
                best = (obj, c.copy())
            Hs = np.mean([stats(V, vf)[0] for V, _ in sims], 0); Qs = np.mean([stats(V, vf)[1] for V, _ in sims], 0)
            dH = (Ht - Hs)[:-1] / max(Ht.max(), 1e-3)
            dq = (Qt - Qs); dQ = (dq[:-1] - dq[1:]) / max(Qt.max(), 1e-3)
            eta = 0.6 / (1 + it / 8)
            c = np.clip(c * np.exp(-eta * (dH + 0.5 * np.clip(dQ, -1, 1))), 0.5 * base, 1.1 * base)
            if it % 5 == 0:
                print(f"  {panel} it {it + 1}: fit IoU {iou:.3f} share {ss:.3f}/{st:.3f} corr {corr:.2f} free-flow err {fe * 100:+.2f}% ({time.time() - t0:.0f}s)", flush=True)
        set_caps(best[1])
        iou, ss, st, corr, fe, _ = evaluate(ev)
        print(f"{panel:<10} BEST held-out: share {ss:.3f} / {st:.3f}  map corr {corr:.2f}  daily IoU {iou:.3f}  free-flow err {fe * 100:+.2f}%  "
              f"caps < 0.9 x released: {int((best[1] < 0.9 * base).sum())} / {len(base)}", flush=True)
        np.save(ROOT / "work" / f"ctm_caps_{panel}.npy", best[1])


if __name__ == "__main__":
    main()
