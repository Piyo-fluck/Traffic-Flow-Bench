"""Note 169: clean replay of every train day (observed demand of the day, no draws, no incidents) - onset-site
distribution vs the truth of the same days: overlap, the replay's share at sites the truth rarely uses (< 1 %),
per-day site agreement (same first onset site on days where both have one), onset time, map corr.
Variants through the generator's environment switches (EXIT_TV, HB_RAW_AT_MERGE, OUT_CAP_MODE ...)."""
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
tag = os.environ.get("TAG", "base")
for panel in sys.argv[1:]:
    z = load(panel, "train"); vf = np.asarray(z["free_speed"], float)
    g = CTMGen(panel, z); g.fit_profiles(z)
    chf = ROOT / "work" / os.environ.get("CHOICE_FMT", "ctm_hb_choice_{}.npz").format(panel)
    if os.environ.get("RULE", "") == "1":                                # truth-only type rule (ctm_type_rule.py)
        chf = ROOT / "work" / os.environ.get("RULE_OUT", "ctm_rule_{}.npz").format(panel)
    ch = np.load(chf); g.apply_choice(ch["heads"], ch["caps"], ch["sub"])
    if os.environ.get("RULE", "") == "1" and os.environ.get("APRE", "") != "1":
        g.ap[ch["ap_cells"]] = ch["ap"]; g.out_ap = float(ch["out_ap"])
    if os.environ.get("APRE", "") == "1":                                # per-head breakdown factors
        a = np.load(ROOT / "work" / os.environ.get("APRE_OUT", "ctm_apre_{}.npz").format(panel)); g.ap[a["cells"]] = a["ap"]; g.out_ap = float(a["out_ap"])
    if os.environ.get("OD_PAIRS", "") == "1":                            # ramp-to-ramp classes
        import json
        pr = json.load(open(ROOT / "work" / f"od_pairs_{panel}.json"))
        if pr:
            g.set_od_pairs(z, [(j, h, float(os.environ.get("OD_B0", b))) for j, h, b, *_ in pr])
        if os.environ.get("OD_BF"):
            g.cls_betaf = np.full(len(pr), float(os.environ["OD_BF"]))
    o = g.order; rank = np.full(len(vf), -1); rank[o] = np.arange(len(o))
    nd = len(z["dates"])
    T, R, same, dt, mt, mr = [], [], [], [], [], []
    for d in range(nd):
        up, ent = g.censored_demand(z["true_flow"][d], z["true_speed"][d], int(z["weekday"][d]))
        V, _ = g.simulate(up, ent)
        el = z["elig"][d]
        st = day_sites(z["true_speed"][d], el & np.isfinite(z["true_speed"][d]), vf, rank)
        sr = day_sites(V, el & np.isfinite(V), vf, rank)
        T += [s for _, s in st]; R += [s for _, s in sr]
        if st and sr:
            same.append(st[0][1] == sr[0][1]); dt.append((sr[0][0] - st[0][0]) / 12)
        mt.append((np.nan_to_num(z["true_speed"][d], nan=1e9) <= 0.6 * vf)[72:240].mean(0))
        mr.append((np.nan_to_num(V, nan=1e9) <= 0.6 * vf)[72:240].mean(0))
    S = len(vf)
    ht = np.bincount(T, minlength=S) / max(len(T), 1); hr = np.bincount(R, minlength=S) / max(len(R), 1)
    top = [s for s, _ in Counter(R).most_common(3)]
    print(f"{tag:<14} {panel:<10} onsets T {len(T)} R {len(R)} | overlap {np.minimum(ht, hr).sum():.2f} | R at truth-rare {hr[ht < 0.01].sum():.2f} | "
          f"same first site {np.mean(same):.2f} | dt med {np.median(dt):+.2f}h | map corr {np.corrcoef(np.mean(mt, 0), np.mean(mr, 0))[0, 1]:.2f} | "
          f"R top " + " ".join(f"{s}:{hr[s]:.2f}(T{ht[s]:.2f})" for s in top), flush=True)
