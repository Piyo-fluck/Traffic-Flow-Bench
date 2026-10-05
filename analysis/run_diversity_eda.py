"""What varies between runs? Per panel, the per-link queued share (v <= 0.6 vf, eligible observed cells, 06-20 h)
for every train month and the two scored months; correlation matrix between months, and total queued share."""
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tta_lab" / "tta"))
import numpy as np  # noqa: E402

import common as C  # noqa: E402
from tfb.data import load  # noqa: E402

warnings.simplefilter("ignore")
for i, panel in enumerate(C.T2_PANELS):
    vf = np.asarray(C.Ctx(panel, i).vf, np.float32)
    maps, names, share = [], [], []
    for split in ("train", "validation", "private"):
        z = load(panel, split)
        mon = np.array([str(d)[:7] for d in z["dates"]])
        v = z["obs_speed"][:, 72:240]
        ok = z["elig"][:, 72:240] & np.isfinite(v)
        q = ok & (np.nan_to_num(v, nan=1e9) <= 0.6 * vf)
        for m in np.unique(mon):
            k = mon == m
            mp = q[k].sum((0, 1)) / np.maximum(ok[k].sum((0, 1)), 1)
            maps.append(mp); names.append(m[2:] if split == "train" else split[:3]); share.append(q[k].sum() / ok[k].sum())
    M = np.corrcoef(np.array(maps))
    print(f"\n{panel}  queued share: " + " ".join(f"{n}:{s:.3f}" for n, s in zip(names, share)))
    print("      " + " ".join(f"{n:>6}" for n in names))
    for n, row in zip(names, M):
        print(f"{n:>6} " + " ".join(f"{x:6.2f}" for x in row))
