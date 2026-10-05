"""Note 169: head 239 (D12_I5_N) capacity type. Truth: links 240-243 never exceed 6,518 (p99 of all slots, equal on
all four links, released caps 8,108 / 6,433 / 8,588 / 12,881) - a ceiling on the TOTAL flow after the merge in gap
239->240. The chosen option is 'sub' (mainline-only cap 5,328 = discharge 6,254 - entry profile). Replay all train
days with the head-239 variants and compare: free-flow p99 of q240, the queued-share of 239, onset sites."""
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
from onset_sites_util import day_sites
from tfb.data import load
warnings.simplefilter("ignore")
panel, H = "D12_I5_N", 239
variant = sys.argv[1]
z = load(panel, "train"); vf = np.asarray(z["free_speed"], float)
g = CTMGen(panel, z); g.fit_profiles(z)
ch = dict(np.load(ROOT / "work" / f"ctm_hb_choice_{panel}.npz"))
heads, caps, subs = ch["heads"].copy(), ch["caps"].copy().astype(float), ch["sub"].copy()
k = int(np.flatnonzero(heads == H)[0])
raw, sub = g.hb_options[H]
ap = 1.0
if variant.startswith("raw"):
    subs[k] = False; caps[k] = raw
if variant.startswith("sub"):
    subs[k] = True; caps[k] = sub
if "_ap" in variant:
    ap = float(variant.split("_ap")[1])
g.apply_choice(heads, caps, subs)
g.ap[g.sens_cell[H] + 1] = ap
o = g.order; rank = np.full(len(vf), -1); rank[o] = np.arange(len(o))
nd = len(z["dates"])
T, R, q240, q240t, qs239 = [], [], [], [], []
for d in range(nd):
    up, ent = g.censored_demand(z["true_flow"][d], z["true_speed"][d], int(z["weekday"][d]))
    V, Q = g.simulate(up, ent)
    el = z["elig"][d]
    T += [s for _, s in day_sites(z["true_speed"][d], el & np.isfinite(z["true_speed"][d]), vf, rank)]
    R += [s for _, s in day_sites(V, el & np.isfinite(V), vf, rank)]
    Vo, Qo = V[:, o], Q[:, o]
    fr = Vo[60:252, H + 1] > 0.9 * vf[o][H + 1]
    q240 += list(Qo[60:252, H + 1][fr])
    qs239.append((Vo[72:240, H] <= 0.6 * vf[o][H]).mean())
S = len(vf); ht = np.bincount(T, minlength=S) / len(T); hr = np.bincount(R, minlength=S) / len(R)
qt = z["true_flow"][:, 60:252, o[H + 1]]; vt = z["true_speed"][:, 60:252, o[H + 1]] / vf[o][H + 1]
tq239 = (z["true_speed"][:, 72:240, o[H]] <= 0.6 * vf[o][H]).mean()
print(f"{variant:<12} q240 free p99 replay {np.nanquantile(q240, .99):.0f} (truth {np.nanquantile(qt[vt > 0.9], .99):.0f}) | queued share 239 replay {np.mean(qs239):.3f} (truth {tq239:.3f}) | "
      f"onsets overlap {np.minimum(ht, hr).sum():.2f}, at 239 {hr[o[H]]:.2f}, 238 {hr[o[H - 1]]:.2f} (T {ht[o[H - 1]]:.2f}), 105 {hr[105]:.2f} (T {ht[105]:.2f})", flush=True)
