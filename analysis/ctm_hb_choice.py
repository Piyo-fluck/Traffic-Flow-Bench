"""Note 169: per recurrent head with an entry in its gap, choose the hidden capacity 'discharge' (bottleneck after
the merge) or 'discharge - entry' (bottleneck before the merge) by train replication around that head (share of
days the head is active and queued share within +-5 links, 20 fit days); greedy over heads. Held-out evaluation.
Saves work/ctm_hb_choice_<panel>.npz."""
import os
import sys
import warnings
from pathlib import Path
os.environ["HB_RULE"] = "choice"
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "analysis"))
import numpy as np
from ctm_gen import CTMGen
from dlsim_fitcap import occupancy
from tfb.data import load
warnings.simplefilter("ignore")


def sims(g, z, days):
    out = []
    for d in days:
        up, ent = g.censored_demand(z["true_flow"][d], z["true_speed"][d], int(z["weekday"][d]))
        out.append(occupancy(g.simulate(up, ent)[0], np.asarray(z["free_speed"], float), g.order)[2])
    return out


def set_cap(g, h, c):
    cell = g.sens_cell[h] + 1
    g.cell_cap[cell] = c
    g.cell_w[cell] = c / max(g.cell_kj[cell] - c / g.cell_vf[cell], 1e-6)


for panel in sys.argv[1:]:
    z = load(panel, "train"); vf = np.asarray(z["free_speed"], float)
    g = CTMGen(panel, z); g.fit_profiles(z)
    opts = getattr(g, "hb_options", {})
    nd = len(z["dates"]); fit, ev = np.arange(0, nd, 7)[:20], np.arange(1, nd, 6)[:30]
    truth = {d: occupancy(z["true_speed"][d], vf, g.order)[2] for d in np.concatenate([fit, ev])}

    def local_err(sim_list, h):
        lo, hi = max(0, h - 5), min(g.n, h + 6)
        a = np.mean([s[:, lo:hi].mean(0) for s in sim_list], 0); b = np.mean([truth[d][:, lo:hi].mean(0) for d in fit], 0)
        ha = np.mean([(s[:, h] & ~s[:, min(h + 1, g.n - 1)]).any() for s in sim_list]); hb = np.mean([(truth[d][:, h] & ~truth[d][:, min(h + 1, g.n - 1)]).any() for d in fit])
        return np.abs(a - b).sum() + abs(ha - hb)
    choice = {}
    base_cells = g.cell_cap.copy(); base_w = g.cell_w.copy()

    def apply(h, nm, c):
        cell = g.sens_cell[h] + 1
        g.cell_cap[cell] = base_cells[cell]; g.cell_w[cell] = base_w[cell]; g.mcap[cell] = 1e9
        g.apply_choice([h], [c], [nm == "sub"])
    for h, (raw, sub) in sorted(opts.items()):
        errs = {}
        for nm, c in (("raw", raw), ("sub", sub)):
            apply(h, nm, c)
            errs[nm] = local_err(sims(g, z, fit), h)
        best = min(errs, key=errs.get)
        choice[h] = best
        apply(h, best, sub if best == "sub" else raw)
    S = sims(g, z, ev)
    Ms = np.mean([s.mean(0) for s in S], 0); Mt = np.mean([truth[d].mean(0) for d in ev], 0)
    iou = np.mean([(a & truth[d]).sum() / max((a | truth[d]).sum(), 1) for a, d in zip(S, ev)])
    print(f"{panel:<10} heads with a choice {len(opts)}: {sum(v == 'sub' for v in choice.values())} subtract | held-out share {Ms.mean():.3f}/{Mt.mean():.3f} map corr {np.corrcoef(Ms, Mt)[0, 1]:.2f} daily IoU {iou:.3f}", flush=True)
    np.savez(ROOT / "work" / f"ctm_hb_choice_{panel}.npz", heads=np.array(list(choice.keys())), sub=np.array([v == "sub" for v in choice.values()]),
             caps=np.array([opts[h][1] if choice[h] == "sub" else opts[h][0] for h in choice]))
