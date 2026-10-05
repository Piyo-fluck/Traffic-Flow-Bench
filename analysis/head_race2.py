"""Note 169: pair race. For an upstream head u and the downstream site s where the replay wrongly starts the queue:
per day (truth vs clean replay) the first time u and s drop below 0.6 vf, speed ratio v/vf at s when u first queues,
and flows (median over the 30 min before the first queue anywhere in [u, s]) along u..s plus the exit shares."""
import os
import sys
import warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "analysis"))
for k, v in dict(FIFO="1", CONSERVE_FILL="1", HIDDEN_BN="1", HB_ALL="1", VIRT_LEN="3", HB_SUB_ENTRY="1", HB_RULE="choice", MAINLINE_CAP="1").items():
    os.environ.setdefault(k, v)
import numpy as np
from ctm_gen import CTMGen
from tfb.data import load
warnings.simplefilter("ignore")
A, B = 60, 252
PAIRS = [("D12_I5_N", 238, 239), ("D12_I5_N", 105, 114), ("D7_I210_W", 77, 80)]

cache = {}
for panel, u, s in PAIRS:
    if panel not in cache:
        z = load(panel, "train"); vf = np.asarray(z["free_speed"], float)
        g = CTMGen(panel, z); g.fit_profiles(z)
        ch = np.load(ROOT / "work" / f"ctm_hb_choice_{panel}.npz"); g.apply_choice(ch["heads"], ch["caps"], ch["sub"])
        o = g.order; nd = len(z["dates"])
        Vt = z["true_speed"][:, :, o]; Qt = z["true_flow"][:, :, o]
        Vr = np.empty_like(Vt); Qr = np.empty_like(Qt)
        for d in range(nd):
            up, ent = g.censored_demand(z["true_flow"][d], z["true_speed"][d], int(z["weekday"][d]))
            V, Q = g.simulate(up, ent); Vr[d], Qr[d] = V[:, o], Q[:, o]
        cache[panel] = (z, g, vf[o], Vt, Qt, Vr, Qr)
    z, g, vfo, Vt, Qt, Vr, Qr = cache[panel]
    nd = Vt.shape[0]
    print(f"\n{panel} u={u} s={s} | exit share per link u..s (f_exit): " + " ".join(f"{g.f_exit[h]:.3f}" for h in range(u, min(s, g.n - 1) + 1))
          + f" | entries in gaps: {[int(b) for b in g.ent if u < b <= s]}")
    for tag, V, Q in (("truth ", Vt, Qt), ("replay", Vr, Qr)):
        r = V / vfo
        dt, vs_at_u, fl, both, ufirst = [], [], [], 0, 0
        for d in range(nd):
            qu = np.flatnonzero(r[d, A:B, u] <= 0.6); qs = np.flatnonzero(r[d, A:B, s] <= 0.6)
            if len(qu) and len(qs):
                both += 1; tu, ts = qu[0], qs[0]
                dt.append((ts - tu) / 12); ufirst += tu < ts
                vs_at_u.append(r[d, A + tu, s])
                t0 = A + min(tu, ts)
                fl.append(np.nanmedian(Q[d, max(t0 - 6, 0):t0, u:s + 1], 0))
        fl = np.nanmedian(np.array(fl), 0) if fl else []
        vs = np.array(vs_at_u)
        print(f"  {tag}: both queue on {both}/{nd} days, u first on {ufirst / max(both, 1):.2f} | t_s - t_u median {np.median(dt):+.2f} h | "
              f"v/vf at s when u first queues: <0.6 {np.mean(vs <= 0.6):.2f}, 0.6-0.9 {np.mean((vs > 0.6) & (vs <= 0.9)):.2f}, >0.9 {np.mean(vs > 0.9):.2f}")
        print(f"          flow 30 min before first queue u..s: " + " ".join(f"{x:.0f}" for x in fl))
    print(f"  capacities: cell after each sensored link u..s: " + " ".join(f"{g.cell_cap[g.sens_cell[h] + 1]:.0f}{'*' if g.mcap[g.sens_cell[h] + 1] < 1e8 else ''}" for h in range(u, min(s, g.n - 2) + 1))
          + (f" | out_cap {g.out_cap:.0f}" if s == g.n - 1 else ""))
