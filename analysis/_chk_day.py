import os, sys, warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "analysis"))
for k, v in dict(FIFO="1", CONSERVE_FILL="1", HIDDEN_BN="1", HB_ALL="1", VIRT_LEN="3", HB_SUB_ENTRY="1", HB_RULE="choice", MAINLINE_CAP="1").items():
    os.environ.setdefault(k, v)
import numpy as np
from ctm_gen import CTMGen
from tfb.data import load
warnings.simplefilter("ignore")
panel = sys.argv[1]; L = [int(x) for x in sys.argv[2].split(",")]; days = [int(x) for x in sys.argv[3].split(",")]
z = load(panel, "train"); vf = np.asarray(z["free_speed"], float)
g = CTMGen(panel, z); g.fit_profiles(z)
ch = np.load(ROOT / "work" / f"ctm_hb_choice_{panel}.npz"); g.apply_choice(ch["heads"], ch["caps"], ch["sub"])
if os.environ.get("APRE", "") == "1":
    a = np.load(ROOT / "work" / os.environ.get("APRE_OUT", "ctm_apre_{}.npz").format(panel)); g.ap[a["cells"]] = a["ap"]; g.out_ap = float(a["out_ap"])
    print("ap:", dict(zip(a["cells"].tolist(), np.round(a["ap"], 3).tolist())), "out_ap", g.out_ap, "head_cell", getattr(g, "head_cell", {}))
if os.environ.get("OD_PAIRS", "") == "1":
    import json
    pr = json.load(open(ROOT / "work" / f"od_pairs_{panel}.json")); g.set_od_pairs(z, [(j, h, float(os.environ.get("OD_B0", b))) for j, h, b, *_ in pr])
    if os.environ.get("OD_BF"):
        g.cls_betaf = np.full(len(pr), float(os.environ["OD_BF"])); print("od pairs", pr)
o = g.order
ent_at = {int(b): i for i, b in enumerate(g.ent)}
print("entries:", {h: ent_at[h] for h in L if h in ent_at}, "f_exit:", {h: round(float(g.f_exit[h]), 3) for h in L if h < g.n - 1})
for d in days:
    up, ent = g.censored_demand(z["true_flow"][d], z["true_speed"][d], int(z["weekday"][d]))
    V, Q = g.simulate(up, ent)
    Vt, Qt = z["true_speed"][d], z["true_flow"][d]
    r = (Vt / vf)[:, o]; rr = (V / vf)[:, o]
    t1 = np.flatnonzero((r[60:252][:, L] <= 0.6).any(1)); t2 = np.flatnonzero((rr[60:252][:, L] <= 0.6).any(1))
    t0 = 60 + min(t1[0] if len(t1) else 999, t2[0] if len(t2) else 999)
    print(f"\nday {d}: first queue truth {(60 + t1[0]) / 12 if len(t1) else None} replay {(60 + t2[0]) / 12 if len(t2) else None}; cols " + " ".join(map(str, L)) + " | entries")
    for t in range(t0 - 12, t0 + 7):
        print(f" {t/12:5.2f} T " + " ".join(f"{r[t, l]:.2f}/{Qt[t, o[l]]:4.0f}" for l in L) + "\n       R " + " ".join(f"{rr[t, l]:.2f}/{Q[t, o[l]]:4.0f}" for l in L)
              + " | " + " ".join(f"{ent[t, ent_at[l]]:4.0f}" for l in L if l in ent_at))
