"""Note 169: why do generated onsets pile up at sites that real runs use only sometimes (D12_I5_N 239/114,
D7_I210_W downstream end)? Each generated day replays one train day (src_day) with run draws, so compare per
source day. Also a clean replay (observed demand of the day, no draws, no incidents) of all train days.
Per target site s: onset share (truth / clean replay / generated), what the truth did on the source days where
the replay puts the onset at s (truth onset site; was s queued at all that day), and the hidden capacity at s."""
import os
import sys
import warnings
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "analysis")); sys.path.insert(0, str(ROOT / "tta_lab" / "tta"))
for k, v in dict(FIFO="1", CONSERVE_FILL="1", HIDDEN_BN="1", HB_ALL="1", VIRT_LEN="3", HB_SUB_ENTRY="1", HB_RULE="choice", MAINLINE_CAP="1").items():
    os.environ.setdefault(k, v)
import numpy as np
from ctm_gen import CTMGen
from t2_eda03 import earliest_onset
from tfb.data import load
warnings.simplefilter("ignore")
F = 6


def day_sites(V, el, vf, rank):
    """list of (t, site) earliest-onset windows of one day; site = downstream-most queued link at t+F"""
    q = ((np.nan_to_num(V, nan=1e9) <= 0.6 * vf) & el)[None]
    out = []
    for t in np.flatnonzero(earliest_onset(q)[0]):
        y = np.flatnonzero(q[0, t + F]); y = y[rank[y] >= 0]
        if len(y):
            out.append((int(t), int(y[np.argmax(rank[y])])))
    return out


gdir = ROOT / "work" / os.environ.get("GEN_DIR", "genruns_v11")
for panel in sys.argv[1:] or ["D12_I5_N", "D7_I210_W"]:
    z = load(panel, "train"); vf = np.asarray(z["free_speed"], float)
    g = CTMGen(panel, z); g.fit_profiles(z)
    ch = np.load(ROOT / "work" / f"ctm_hb_choice_{panel}.npz"); g.apply_choice(ch["heads"], ch["caps"], ch["sub"])
    o = g.order; rank = np.full(len(vf), -1); rank[o] = np.arange(len(o))
    nd = len(z["dates"])
    tru = {d: day_sites(z["true_speed"][d], z["elig"][d] & np.isfinite(z["true_speed"][d]), vf, rank) for d in range(nd)}
    rep, repV = {}, {}
    for d in range(nd):
        up, ent = g.censored_demand(z["true_flow"][d], z["true_speed"][d], int(z["weekday"][d]))
        V, _ = g.simulate(up, ent)
        repV[d] = V; rep[d] = day_sites(V, z["elig"][d] & np.isfinite(V), vf, rank)
    gen = []                                                            # (src_day, t, site)
    for f in sorted(gdir.glob(f"{panel}_r*.npz")):
        c = np.load(f, allow_pickle=True)
        for i, d in enumerate(c["src_day"]):
            gen += [(int(d), t, s) for t, s in day_sites(c["true_speed"][i], c["elig"][i], vf, rank)]
    ht = Counter(s for d in tru for _, s in tru[d]); hr = Counter(s for d in rep for _, s in rep[d]); hg = Counter(s for _, _, s in gen)
    nt, nr, ng = sum(ht.values()), sum(hr.values()), sum(hg.values())
    targets = sorted(hg, key=lambda s: -(hg[s] / ng - ht.get(s, 0) / nt))[:3]
    print(f"\n{panel}: onsets truth {nt} clean replay {nr} generated {ng}")
    print("  top sites (rank along travel): " + ", ".join(f"{s}[r{rank[s]}] T{ht.get(s,0)/nt:.2f}/R{hr.get(s,0)/nr:.2f}/G{hg.get(s,0)/ng:.2f}" for s in sorted(set(list(ht)[:0]) | {s for s, _ in ht.most_common(4)} | {s for s, _ in hg.most_common(4)}, key=lambda s: rank[s])))
    for s in targets:
        h = int(rank[s]); cell = g.sens_cell[h] + 1 if h < g.n - 1 else None
        capinfo = (f"cell after s: cap {g.cell_cap[cell]:.0f} released {g.cap_released_cell[cell]:.0f} mcap {g.mcap[cell]:.0f} "
                   f"hidden-choice {'yes' if h in set(ch['heads'].tolist()) else 'no'}" if cell is not None else f"downstream end: out_cap {g.out_cap:.0f}")
        print(f"  target {s} (rank {h}/{g.n - 1}): truth {ht.get(s,0)/nt:.3f} clean replay {hr.get(s,0)/nr:.3f} generated {hg.get(s,0)/ng:.3f} | {capinfo}")
        # source days where the clean replay puts an onset at s
        days = [d for d in rep if any(x == s for _, x in rep[d])]
        tq = [bool(((np.nan_to_num(z['true_speed'][d][:, s], nan=1e9) <= 0.6 * vf[s]))[60:252].any()) for d in days]
        rq = [((np.nan_to_num(repV[d][:, s], nan=1e9) <= 0.6 * vf[s]))[60:252].mean() for d in days]
        tqm = [((np.nan_to_num(z['true_speed'][d][:, s], nan=1e9) <= 0.6 * vf[s]))[60:252].mean() for d in days]
        alt = Counter(x for d in days for _, x in tru[d][:1])
        print(f"    replay onset at s on {len(days)}/{nd} days; truth queues s at all on {np.mean(tq):.2f} of them; queued share of s 05-21h replay {np.mean(rq):.3f} truth {np.mean(tqm):.3f}")
        print(f"    truth first onset site on those days: " + ", ".join(f"{x}[r{rank[x]}] {n}" for x, n in alt.most_common(5)) + f" | none {sum(1 for d in days if not tru[d])}")
        # time: replay onset at s vs truth first onset
        dt = [(min(t for t, x in rep[d] if x == s) - tru[d][0][0]) / 12 for d in days if tru[d]]
        if dt:
            print(f"    replay onset time at s minus truth first onset: median {np.median(dt):+.1f} h [q10 {np.quantile(dt, .1):+.1f}, q90 {np.quantile(dt, .9):+.1f}]")
        # flow at s: truth daily max 5-min flow vs the capacity downstream
        qmax_t = [np.nanmax(z["true_flow"][d][60:252, s]) for d in range(nd)]
        qmax_r = [np.nanmax(np.load if False else _ for _ in [0]) for d in []]
        free_t = [np.nanquantile(np.where(z["true_speed"][d][60:252, s] > 0.9 * vf[s], z["true_flow"][d][60:252, s], np.nan), 0.95) for d in range(nd)]
        print(f"    truth flow at s: daily max median {np.nanmedian(qmax_t):.0f}, free-flow p95 median {np.nanmedian(free_t):.0f}")
