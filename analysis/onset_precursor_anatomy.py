"""Note 169: which precursor does the TRUE onset follow? Earliest-onset windows, travel order. For each window,
per link: q/cap (own), q/cap_down (next sensored downstream link's capacity), q/min cap over the next 2 downstream
links, and the number of unsensored corridor links to the next sensor. AUC (within window) of each for the links
queued at T+30; for real train truth and for CTM-generated runs."""
import sys
import warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "analysis")); sys.path.insert(0, str(ROOT / "tta_lab" / "tta"))
import numpy as np
import pandas as pd
from dlsim_gen2 import corridor
from t2_eda03 import earliest_onset, labels
from tfb.data import RELEASE, load
import common as C
warnings.simplefilter("ignore")
F = 6


def auc_within(s, y):
    a, b = s[y], s[~y]
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if not len(a) or not len(b):
        return np.nan
    return (a[:, None] > b[None, :]).mean() + 0.5 * (a[:, None] == b[None, :]).mean()


def run(z, order, spos, cap):
    capo = cap[order]
    cdown = np.append(capo[1:], capo[-1]); cdown2 = np.minimum(cdown, np.append(capo[2:], [capo[-1]] * 2))
    nuns = np.append(np.diff(spos) - 1, 0)
    q, el = labels(z)
    dd, tt = np.nonzero(earliest_onset(q))
    res = {k: [] for k in ("q/cap", "q/cap_down", "q/min cap_down2", "unsensored gap", "cap_down/cap")}
    for d, t0 in zip(dd, tt):
        y = q[d, t0 + F][order]
        if not y.any() or y.all():
            continue
        s = np.nanmean(z["true_flow"][d, t0 - 3:t0][:, order], 0)
        for k, v in (("q/cap", s / capo), ("q/cap_down", s / cdown), ("q/min cap_down2", s / cdown2),
                     ("unsensored gap", nuns.astype(float)), ("cap_down/cap", -cdown / capo)):
            res[k].append(auc_within(v, y))
    return {k: np.nanmean(v) for k, v in res.items()}


for panel in C.T2_PANELS:
    z = load(panel, "train")
    order, spos, *_ = corridor(panel, z)
    ids = [str(x) for x in z["link_ids"]]
    cap = pd.read_csv(RELEASE / "corridors" / panel / "network" / "fd_parameters.csv").set_index("link_id").loc[ids].capacity_vph.values
    r = run(z, order, spos, cap)
    gs = [run(dict(np.load(f, allow_pickle=True)), order, spos, cap) for f in sorted((ROOT / "work" / __import__("os").environ.get("GEN_DIR", "genruns_ctm")).glob(f"{panel}_r*.npz"))[:3]]
    g = {k: np.nanmean([x[k] for x in gs]) for k in r}
    print(f"{panel:<10} " + " | ".join(f"{k} {r[k]:.2f}/{g[k]:.2f}" for k in r), flush=True)
