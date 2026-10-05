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
for panel, lo, hi in (("D7_I210_W", 74, 81), ("D12_I5_N", 234, 242), ("D12_I5_N", 104, 116)):
    z = load(panel, "train"); g = CTMGen(panel, z); g.fit_profiles(z)
    ch = np.load(ROOT / "work" / f"ctm_hb_choice_{panel}.npz"); g.apply_choice(ch["heads"], ch["caps"], ch["sub"])
    print(f"\n{panel}: sensored cells {g.sens_cell[lo:hi].tolist()}")
    for i, b in enumerate(g.ent):
        if lo < b <= hi:
            print(f"  entry gap {int(b)-1}->{int(b)}: merges into cell {g.ent_cell[i]}")
    last = g.sens_cell[min(hi, g.n) - 1] + 1
    for c in range(g.sens_cell[lo], min(last, len(g.cell_L))):
        tag = [f"S{h}" for h in range(g.n) if g.sens_cell[h] == c]
        print(f"  cell {c:4d} {''.join(tag):5s} L {g.cell_L[c]:.3f} lanes {g.cell_lanes[c]:.1f} cap {g.cell_cap[c]:6.0f} released {g.cap_released_cell[c]:6.0f} mcap {min(g.mcap[c],99999):6.0f} f {g.cell_f[c]:.3f}")
