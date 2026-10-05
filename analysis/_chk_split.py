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
for panel, gaps in (("D12_I5_N", [238, 239]), ("D7_I210_W", [76, 77, 78, 79])):
    z = load(panel, "train"); vf = np.asarray(z["free_speed"], float)
    g = CTMGen(panel, z); g.fit_profiles(z)
    ch = np.load(ROOT / "work" / f"ctm_hb_choice_{panel}.npz"); g.apply_choice(ch["heads"], ch["caps"], ch["sub"])
    o = g.order; days = np.arange(0, len(z["dates"]), 5)
    Vt, Qt = z["true_speed"][days][:, :, o], z["true_flow"][days][:, :, o]
    Vr, Qr, E = [], [], []
    for d in days:
        up, ent = g.censored_demand(z["true_flow"][d], z["true_speed"][d], int(z["weekday"][d]))
        V, Q = g.simulate(up, ent); Vr.append(V[:, o]); Qr.append(Q[:, o]); E.append(ent)
    Vr, Qr, E = np.array(Vr), np.array(Qr), np.array(E)
    ent_at = {int(b): i for i, b in enumerate(g.ent)}
    print(f"\n{panel}", flush=True)
    for h in gaps:
        i_e = ent_at.get(h + 1)
        free = (Vt[:, :, h] > 0.9 * vf[o][h]) & (Vt[:, :, h + 1] > 0.9 * vf[o][h + 1])
        freer = free & (Vr[:, :, h] > 0.9 * vf[o][h]) & (Vr[:, :, h + 1] > 0.9 * vf[o][h + 1])
        print(f"  gap {h}->{h+1}: f_exit {g.f_exit[h]:.3f}, entry {'yes' if i_e is not None else 'no'}; lanes {z['lanes'][o][h]:.0f}->{z['lanes'][o][h+1]:.0f}")
        for a, b in ((60, 96), (96, 132), (132, 168), (168, 204), (204, 240)):
            m = free[:, a:b]; mr = freer[:, a:b]
            qa, qb = Qt[:, a:b, h][m], Qt[:, a:b, h + 1][m]
            e = E[:, a:b, i_e][m] if i_e is not None else 0 * qa
            rb = Qr[:, a:b, h + 1][mr]; tb = Qt[:, a:b, h + 1][mr]
            print(f"    {a/12:4.1f}-{b/12:4.1f}h: truth q_h {np.median(qa):5.0f} q_h+1 {np.median(qb):5.0f} | model entry {np.median(e):5.0f} "
                  f"| implied exit share (q_h + e - q_h+1)/q_h {np.median((qa + e - qb) / qa):.3f} | replay/truth q_h+1 in free flow {np.median(rb / tb):.3f}", flush=True)
