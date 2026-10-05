"""Note 169: share of days on which each recurrent head is active (>= 4 queued-head slots), train truth vs the
generated runs (work/genruns_ctm_hb, all runs pooled, and the spread over runs)."""
import sys
import warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "analysis"))
import numpy as np
from dlsim_gen2 import corridor
from tfb.data import load
warnings.simplefilter("ignore")


def act(z, order, vf):
    v = z["true_speed"][:, :, order]
    vfo = vf[order]
    cg = np.isfinite(v) & (v <= 0.6 * vfo); fr = np.isfinite(v) & (v > 0.9 * vfo)
    hd = cg[:, :, :-1] & fr[:, :, 1:]
    return (hd.sum(1) >= 4)                                          # (days, heads)


for panel in sys.argv[1:] or ["D12_I5_N", "D12_I5_S", "D7_I405_N"]:
    z = load(panel, "train")
    order, *_ = corridor(panel, z)
    vf = np.asarray(z["free_speed"], float)
    at = act(z, order, vf).mean(0)
    runs = [act(dict(np.load(f, allow_pickle=True)), order, vf).mean(0)
            for f in sorted((ROOT / "work" / "genruns_ctm_hb").glob(f"{panel}_r*.npz"))]
    R = np.array(runs)
    top = np.argsort(-at)[:10]
    print(f"{panel}: head | true share of days | generated mean [min over runs, max over runs]")
    for h in sorted(top):
        print(f"   {h:4d} | {at[h]:.2f} | {R[:, h].mean():.2f} [{R[:, h].min():.2f}, {R[:, h].max():.2f}]")
    strong = R.mean(0) >= 0.3
    print(f"   heads active >= 30 % of generated days but < 10 % of true days: {int((strong & (at < 0.1)).sum())}")
