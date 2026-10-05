"""Note 169: capacity type per hidden head with an entry in its gap. 'raw' = the ceiling is on the total after the
merge (flows at h+1 never exceed about the discharge, whatever the entry does); 'sub' = the ceiling is on the mainline
before the merge (flows at h+1 exceed the discharge by the entry's variation). Evidence: free-flow p99 of q_{h+1}
over the raw discharge (D12_I5_N 239: 6,518 / 6,254 = 1.04 and the replay with raw x 1.04-1.06 fixes the site race).
Prints the ratio next to the chosen option of work/ctm_hb_choice_<panel>.npz."""
import os
import sys
import warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "analysis")); sys.path.insert(0, str(ROOT / "tta_lab" / "tta"))
for k, v in dict(FIFO="1", CONSERVE_FILL="1", HIDDEN_BN="1", HB_ALL="1", VIRT_LEN="3", HB_SUB_ENTRY="1", HB_RULE="choice", MAINLINE_CAP="1").items():
    os.environ.setdefault(k, v)
import numpy as np
import common as C
from ctm_gen import CTMGen
from tfb.data import load
warnings.simplefilter("ignore")
for panel in sys.argv[1:] or C.T2_PANELS:
    z = load(panel, "train"); g = CTMGen(panel, z)
    opts = dict(getattr(g, "hb_options", {}))
    ch = np.load(ROOT / "work" / f"ctm_hb_choice_{panel}.npz")
    chosen = {int(h): bool(s) for h, s in zip(ch["heads"], ch["sub"])}
    o = g.order; vf = np.asarray(z["free_speed"], float)[o]
    V, Q = z["true_speed"][:, 60:252, o], z["true_flow"][:, 60:252, o]
    rows = []
    for h, (raw, sub) in sorted(opts.items()):
        fr = V[:, :, h + 1] > 0.9 * vf[h + 1]
        p99 = np.nanquantile(Q[:, :, h + 1][fr], 0.99)
        rows.append(f"{h}:{p99 / raw:.3f}{'S' if chosen.get(h) else 'R'}")
    print(f"{panel:<10} heads with an entry: ratio p99(free q_h+1)/raw, chosen S=sub R=raw | " + " ".join(rows), flush=True)
