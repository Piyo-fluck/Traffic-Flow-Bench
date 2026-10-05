"""Note 169: the over-weighted onset sites (D12_I5_N 239, D7_I210_W downstream end) win a race against a neighbouring
upstream head in the clean replay. Per head h near the site (h queued, h+1 free, travel order): truth vs clean replay
share of days active, median first activation (05-21h), discharge at h+1 while active, the cell capacity after h,
and on days both are active in truth, which fires first."""
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
CFG = {"D12_I5_N": range(232, 257), "D7_I210_W": range(70, 81), "D12_I5_N_mid": range(100, 116)}

for key in sys.argv[1:] or ["D12_I5_N", "D7_I210_W", "D12_I5_N_mid"]:
    panel = key.replace("_mid", "")
    z = load(panel, "train"); vf = np.asarray(z["free_speed"], float)
    g = CTMGen(panel, z); g.fit_profiles(z)
    opts = dict(getattr(g, "hb_options", {}))
    ch = np.load(ROOT / "work" / f"ctm_hb_choice_{panel}.npz"); g.apply_choice(ch["heads"], ch["caps"], ch["sub"])
    chosen = {int(h): (bool(s), float(c)) for h, s, c in zip(ch["heads"], ch["sub"], ch["caps"])}
    o = g.order; nd = len(z["dates"]); vfo = vf[o]
    Vt = z["true_speed"][:, :, o]; Qt = z["true_flow"][:, :, o]
    Vr = np.empty_like(Vt); Qr = np.empty_like(Qt)
    for d in range(nd):
        up, ent = g.censored_demand(z["true_flow"][d], z["true_speed"][d], int(z["weekday"][d]))
        V, Q = g.simulate(up, ent); Vr[d], Qr[d] = V[:, o], Q[:, o]

    def heads(V):
        cg = np.isfinite(V) & (V <= 0.6 * vfo); fr = np.isfinite(V) & (V > 0.9 * vfo)
        hd = np.zeros_like(cg); hd[:, :, :-1] = cg[:, :, :-1] & fr[:, :, 1:]
        if True:                                                     # downstream end: queued = head
            hd[:, :, -1] = cg[:, :, -1]
        return hd[:, A:B]
    Ht, Hr = heads(Vt), heads(Vr)
    print(f"\n{key}: n={g.n} out_cap {g.out_cap:.0f}")
    print("  rank  z   | truth act first(h) disch | replay act first(h) disch | cell-after cap / released / mcap | hb opts raw/sub chosen")
    for h in CFG[key]:
        if h >= g.n:
            continue
        def st(H, Q):
            act = H[:, :, h].sum(1) >= 2
            first = [A + np.argmax(H[d, :, h]) for d in range(nd) if act[d]]
            dis = [np.median(Q[d, A:B][H[d, :, h], min(h + 1, g.n - 1)]) for d in range(nd) if act[d]]
            return act.mean(), (np.median(first) / 12 if first else np.nan), (np.median(dis) if dis else np.nan)
        a1, f1, d1 = st(Ht, Qt); a2, f2, d2 = st(Hr, Qr)
        if max(a1, a2) < 0.05:
            continue
        cell = g.sens_cell[h] + 1 if h < g.n - 1 else None
        ci = f"{g.cell_cap[cell]:5.0f} / {g.cap_released_cell[cell]:5.0f} / {min(g.mcap[cell], 99999):5.0f}" if cell is not None else "  (end)"
        op = f"{opts[h][0]:.0f}/{opts[h][1]:.0f} {'sub' if chosen.get(h, (0,))[0] else 'raw'}" if h in opts else ("hidden(no entry)" if cell is not None and g.cell_cap[cell] < 0.95 * g.cap_released_cell[cell] else "-")
        print(f"  {h:4d} {o[h]:4d} | {a1:.2f} {f1:5.1f} {d1:5.0f}     | {a2:.2f} {f2:5.1f} {d2:5.0f}      | {ci} | {op}")
