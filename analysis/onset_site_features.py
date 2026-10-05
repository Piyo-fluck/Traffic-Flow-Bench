"""Note 169 (evaluation): features built from the generator analysis to LOCATE the onset site among candidates.
Candidates per window: the ctrl CNN's top-5 sites, mutually > 1 link apart. Features per candidate c:
  first  - forward CTM (hidden bottlenecks): first slot T+1..T+30 at which c or its +-1 neighbours exceed the
           critical density (earlier = more likely; 8 if never)
  anom   - today's entry-demand anomaly upstream: mean over the 3 entries merging at or upstream of c of
           (last-3-slot conservation demand / train weekday profile at those slots)
  ttb    - time to breakdown: (effective capacity - last flow) / max(flow slope, eps), effective capacity =
           hidden-bottleneck capacity where identified
Reported for misses (true site among the candidates) and hits: share of windows where the feature puts the true
site first among the candidates; and a combined rule (CNN p x exp(-first/2)).
"""
import sys
import warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "analysis")); sys.path.insert(0, str(ROOT / "tta_lab" / "tta"))
import numpy as np
import common as C
from ctm_gen import CTMGen
from tfb.data import load
warnings.simplefilter("ignore")
F = 6


def forward_first(g, z, d, t0):
    q = z["obs_flow"][d].astype(float).copy(); v = z["obs_speed"][d].astype(float).copy()
    q[t0:] = np.nan; v[t0:] = np.nan
    wd = int(z["weekday"][d])
    up, ent = g.conservation_demand(q, v); up, ent = np.array(up, float), np.array(ent, float)
    last = slice(max(0, t0 - 3), t0)
    lu, le = up[last].mean(), ent[last].mean(0)
    pu, pe = g.U[wd], g.P[wd]
    bu, be = max(pu[last].mean(), 1.0), np.maximum(pe[last].mean(0), 1.0)
    U = np.array([lu * np.clip(pu[t] / bu, 0.5, 2.0) for t in range(t0, t0 + F + 1)])
    E = np.stack([le * np.clip(pe[t] / be, 0.5, 2.0) for t in range(t0, t0 + F + 1)])
    k0 = g.state_from_obs(z["obs_speed"][d, t0 - 1], z["obs_flow"][d, t0 - 1])
    g.simulate(U, E, n_slots=F + 1, k0=k0)
    kc = g.cell_cap / g.cell_vf
    over = (g.last_k[1:] > kc[None, :])[:, g.sens_cell]                    # (F, n) travel order
    first = np.where(over.any(0), over.argmax(0) + 1, F + 2).astype(float)
    out = np.full(g.S, F + 2.0); out[g.order] = first
    anom_e = le / be                                                         # per entry
    return out, anom_e


rows = {"miss": [], "hit": []}
for i, panel in enumerate(C.T2_PANELS):
    ctx, stat, W = C.windows(i, panel)
    g = CTMGen(panel, load(panel, "train"))
    g.fit_profiles(load(panel, "train"))
    rank = np.full(g.S, -1); rank[g.order] = np.arange(g.n)
    ceff = np.full(g.S, np.nan)
    for h in range(g.n - 1):
        ceff[g.order[h]] = g.cell_cap[g.sens_cell[h] + 1]
    ceff[g.order[-1]] = g.cell_cap[g.sens_cell[-1]]
    ent_rank = g.ent                                                          # travel-order index of the merge link
    for s in C.SPLITS:
        on, z = W[s]["on"], W[s]["z"]
        p = np.mean([np.load(ROOT / "work" / "leap_ctrl_on" / f"fam_leap_ctrl_s{k}_sup+rec_{s}_{panel}.npz")["on"] for k in (20, 21)], 0)
        for n in range(len(p)):
            y = on["fut"][n, -1] & on["efut"][n, -1]; e = on["efut"][n, -1]
            if not y.any():
                continue
            nt = y.copy(); nt[1:] |= y[:-1]; nt[:-1] |= y[1:]
            pp = np.where(e, p[n], -1).copy(); cands = []
            while len(cands) < 5 and pp.max() > -1:
                c = int(np.argmax(pp)); cands.append(c); pp[max(0, c - 1):c + 2] = -1
            truth_in = [bool(nt[c]) for c in cands]
            if not any(truth_in):
                continue
            kind = "hit" if truth_in[0] else "miss"
            first, anom_e = forward_first(g, z, int(on["d"][n]), int(on["t0"][n]))
            qh, okh = on["qf"][n], on["ok"][n]
            F1, A1, T1 = [], [], []
            for c in cands:
                F1.append(first[max(0, c - 1):c + 2].min())
                rc = rank[c]
                ups = [i_ for i_, b in enumerate(ent_rank) if rc - 3 <= b <= rc]
                A1.append(np.mean(anom_e[ups]) if ups else 1.0)
                x = np.where(okh[:, c], qh[:, c], np.nan); tt = np.arange(6); m = np.isfinite(x[-6:])
                sl = np.polyfit(tt[m], x[-6:][m], 1)[0] if m.sum() >= 3 else 0.0
                T1.append((ceff[c] - np.nanmean(x[-3:])) / max(sl, 1.0))
            F1, A1, T1 = np.array(F1), np.array(A1), np.array(T1)
            pc = np.array([p[n][c] for c in cands])
            tix = truth_in.index(True)
            first_rank = lambda sc: int(np.argmax(sc) == tix)  # noqa: E731
            rows[kind].append((first_rank(-F1), first_rank(A1), first_rank(-T1), first_rank(pc * np.exp(-F1 / 2)),
                               first_rank(pc * A1), len(cands)))
for k, v in rows.items():
    r = np.array(v, float)
    print(f"{k:<5} windows {len(r)} (true site among the CNN's top-5): true site ranked first by  CTM first-breakdown {r[:, 0].mean():.2f} | "
          f"entry anomaly {r[:, 1].mean():.2f} | time-to-breakdown {r[:, 2].mean():.2f} | CNN x exp(-first/2) {r[:, 3].mean():.2f} | CNN x anomaly {r[:, 4].mean():.2f} | chance {np.mean(1 / r[:, 5]):.2f}")
