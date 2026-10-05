"""Note 169: onset sites of generated runs vs the three real runs (train truth; validation / private observed layer,
evaluation use). Per panel: share of onsets at train-rare sites (< 1 % of train onsets) in each real scored run and in
the generated runs, and the generated share at sites no real run uses (truly spurious)."""
import os
import sys
import warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "analysis")); sys.path.insert(0, str(ROOT / "tta_lab" / "tta"))
import numpy as np
import common as C
from dlsim_gen2 import corridor
from t2_eda03 import earliest_onset
from tfb.data import load
warnings.simplefilter("ignore")
F = 6


def sites_obs(z, order, vkey):
    v = z[vkey]; vf = np.asarray(z["free_speed"], float)
    el = z["elig"] & np.isfinite(v)
    q = (np.nan_to_num(v, nan=1e9) <= 0.6 * vf) & el
    dd, tt = np.nonzero(earliest_onset(q))
    rank = np.full(q.shape[2], -1); rank[order] = np.arange(len(order))
    out = []
    for d, t in zip(dd, tt):
        y = np.flatnonzero(q[d, t + F]); y = y[rank[y] >= 0]
        if len(y):
            out.append(int(y[np.argmax(rank[y])]))
    return np.array(out, int)


gdir = ROOT / "work" / os.environ.get("GEN_DIR", "genruns_v11")
for p in C.T2_PANELS:
    z = load(p, "train"); order, *_ = corridor(p, z); S = len(z["link_ids"])
    H = {"train": sites_obs(z, order, "true_speed")}
    for s in ("validation", "private"):
        H[s] = sites_obs(load(p, s), order, "obs_speed")
    H["gen"] = np.concatenate([sites_obs(dict(np.load(f, allow_pickle=True)), order, "true_speed") for f in sorted(gdir.glob(f"{p}_r*.npz"))])
    Hn = {k: np.bincount(v, minlength=S) / max(len(v), 1) for k, v in H.items()}
    rare = Hn["train"] < 0.01
    used = (Hn["train"] >= 0.01) | (Hn["validation"] > 0) | (Hn["private"] > 0)
    print(f"{p:<10} at train-rare sites: val {Hn['validation'][rare].sum():.2f} pri {Hn['private'][rare].sum():.2f} gen {Hn['gen'][rare].sum():.2f} | "
          f"gen at sites no real run uses {Hn['gen'][~used].sum():.2f} | gen overlap with train {np.minimum(Hn['gen'], Hn['train']).sum():.2f}", flush=True)
