"""Note 169: D7_I405_N-only generator fix. Truth: head rank 347 (z349) active on 100 % of days, end (z351) onsets 0.77
with z349 also queued at T+30 on 81 % of them and z350 free between. v11 (choice: 347 sub) queues only z349,
v12 (rule: 347 raw x 1.067) only the end. Clean replay of every 2nd train day for variants of head 347's type / factor
and the end's out_ap: onset-site shares (z351, z349), co-queue share, head-347 and end active-day shares."""
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
from onset_sites_util import day_sites
from tfb.data import load
warnings.simplefilter("ignore")
panel, H = "D7_I405_N", 347
z = load(panel, "train"); vf = np.asarray(z["free_speed"], float)
base = CTMGen(panel, z); base.fit_profiles(z)
raw, sub = base.hb_options.get(H, (np.nan, np.nan))
rule = dict(np.load(ROOT / "work" / "ctm_rule0_D7_I405_N.npz"))
o = base.order; n = base.n; rank = np.full(len(vf), -1); rank[o] = np.arange(n)
days = np.arange(0, len(z["dates"]), 2)
print(f"head {H} options raw {raw:.0f} sub {sub:.0f}; rule heads {rule['heads'].tolist()} sub {rule['sub'].tolist()} ap cells {rule['ap_cells'].tolist()} ap {np.round(rule['ap'], 3).tolist()} out_cap {base.out_cap:.0f}")


def run(tag, h_type, h_ap, out_ap):
    g = CTMGen(panel, z); g.fit_profiles(z)
    heads, caps, subs = rule["heads"].copy(), rule["caps"].copy().astype(float), rule["sub"].copy()
    k = int(np.flatnonzero(heads == H)[0])
    subs[k] = h_type == "sub"; caps[k] = sub if h_type == "sub" else raw
    g.apply_choice(heads, caps, subs)
    g.ap[rule["ap_cells"]] = rule["ap"]
    g.ap[g.sens_cell[H] + 1] = h_ap
    g.out_ap = out_ap
    sites, co, a347, aend = [], [], [], []
    for d in days:
        up, ent = g.censored_demand(z["true_flow"][d], z["true_speed"][d], int(z["weekday"][d]))
        V = g.simulate(up, ent)[0]
        el = z["elig"][d] & np.isfinite(V)
        r = V[:, o] / vf[o]
        a347.append(((r[60:252, H] <= 0.6) & (r[60:252, H + 1] > 0.6)).sum() >= 2)
        aend.append((r[60:252, n - 1] <= 0.6).sum() >= 2)
        for t, s in day_sites(V, el, vf, rank):
            sites.append(s)
            if s == o[-1]:
                co.append(bool(r[t + 6, n - 3] <= 0.6))
    c = Counter(sites); m = len(sites)
    print(f"{tag:<24} onsets {m:3d} | z351 {c[o[-1]] / m:.2f} z349 {c[o[-3]] / m:.2f} z350 {c[o[-2]] / m:.2f} other {1 - (c[o[-1]] + c[o[-3]] + c[o[-2]]) / m:.2f} | "
          f"co-queue {np.mean(co) if co else float('nan'):.2f} | head347 days {np.mean(a347):.2f} end days {np.mean(aend):.2f}", flush=True)


# truth reference
sites, co, a347, aend = [], [], [], []
for d in days:
    V = z["true_speed"][d]; el = z["elig"][d] & np.isfinite(V); r = V[:, o] / vf[o]
    a347.append(((r[60:252, H] <= 0.6) & (r[60:252, H + 1] > 0.6)).sum() >= 2); aend.append((r[60:252, n - 1] <= 0.6).sum() >= 2)
    for t, s in day_sites(V, el, vf, rank):
        sites.append(s)
        if s == o[-1]:
            co.append(bool(np.nan_to_num(r[t + 6, n - 3], nan=9) <= 0.6))
c = Counter(sites); m = len(sites)
print(f"{'truth':<24} onsets {m:3d} | z351 {c[o[-1]] / m:.2f} z349 {c[o[-3]] / m:.2f} z350 {c[o[-2]] / m:.2f} other {1 - (c[o[-1]] + c[o[-3]] + c[o[-2]]) / m:.2f} | co-queue {np.mean(co):.2f} | head347 days {np.mean(a347):.2f} end days {np.mean(aend):.2f}", flush=True)
for tag, ht, ha, oa in [("v12 (raw x1.067, end 1.0)", "raw", 1.067, 1.0), ("raw x1.00", "raw", 1.0, 1.0), ("sub x1.00", "sub", 1.0, 1.0),
                        ("sub x1.00, end x1.05", "sub", 1.0, 1.05), ("sub x1.00, end x1.10", "sub", 1.0, 1.10),
                        ("sub x1.03, end x1.05", "sub", 1.03, 1.05), ("sub x1.03, end x1.10", "sub", 1.03, 1.10)]:
    run(tag, ht, ha, oa)
