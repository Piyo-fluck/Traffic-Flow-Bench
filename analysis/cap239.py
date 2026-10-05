"""Note 169: how is the capacity at D12_I5_N head 239 determined, and is it right? Per link 234-243: released
capacity (mu), lanes, truth flow quantiles (all / free), and while head 239 is active (239 queued, 240 free):
flows and speeds 238-243. Also the sequence in which 236-242 first queue on each day."""
import os
import sys
import warnings
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "analysis"))
for k, v in dict(FIFO="1", CONSERVE_FILL="1", HIDDEN_BN="1", HB_ALL="1", VIRT_LEN="3", HB_SUB_ENTRY="1", HB_RULE="choice", MAINLINE_CAP="1").items():
    os.environ.setdefault(k, v)
import numpy as np
from ctm_gen import CTMGen
from tfb.data import load
warnings.simplefilter("ignore")
panel = "D12_I5_N"
z = load(panel, "train"); g = CTMGen(panel, z); g.fit_profiles(z)
opts = dict(getattr(g, "hb_options", {}))
o = g.order; vf = np.asarray(z["free_speed"], float)[o]
V, Q = z["true_speed"][:, :, o], z["true_flow"][:, :, o]; r = V / vf
L = list(range(234, 244))
print("link lanes  mu(released)  cellcap-after  | truth flow p50/p99 all | p99 free (v>0.9) | p99 at 0.6-0.9 | hb option raw/sub")
for l in L:
    x = Q[:, 60:252, l]; rv = r[:, 60:252, l]
    ca = g.cell_cap[g.sens_cell[l] + 1] if l < g.n - 1 else np.nan
    print(f" {l}  {z['lanes'][o][l]:.0f}   {g.mu[g.idx_map[g.cpos[g.sens_cell[l]]]]:6.0f}   {ca:6.0f}   | {np.nanquantile(x, .5):5.0f} / {np.nanquantile(x, .99):5.0f} | "
          f"{np.nanquantile(x[rv > 0.9], .99):5.0f} | {np.nanquantile(x[(rv > 0.6) & (rv <= 0.9)], .99) if ((rv > 0.6) & (rv <= 0.9)).sum() > 20 else np.nan:5.0f} | "
          f"{opts.get(l, '')}")
act = (r[:, :, 239] <= 0.6) & (r[:, :, 240] > 0.9)
print(f"\nwhile head 239 active ({act.sum()} slots): median flow / speed ratio per link")
print("  " + " ".join(f"{l}:{np.nanmedian(Q[:, :, l][act]):.0f}/{np.nanmedian(r[:, :, l][act]):.2f}" for l in range(236, 244)))
act2 = (r[:, :, 238] <= 0.6) & (r[:, :, 239] > 0.6)
print(f"while 238 queued and 239 not queued ({act2.sum()} slots):")
print("  " + " ".join(f"{l}:{np.nanmedian(Q[:, :, l][act2]):.0f}/{np.nanmedian(r[:, :, l][act2]):.2f}" for l in range(236, 244)))
seq = Counter()
for d in range(V.shape[0]):
    first = {}
    for l in range(236, 243):
        t = np.flatnonzero(r[d, 60:252, l] <= 0.6)
        if len(t):
            first[l] = t[0]
    if first:
        seq[tuple(sorted(first, key=lambda l: (first[l], -l))[:3])] += 1
print("\nfirst three links to queue (236-242), order: " + ", ".join(f"{s}:{c}" for s, c in seq.most_common(8)))
