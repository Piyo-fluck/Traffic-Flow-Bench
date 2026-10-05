"""Note 169: why do generated runs hurt training? Compare onset precursors, real train truth vs CTM-generated runs.

Earliest-onset windows (t2_eda03.earliest_onset). For each window, the links queued at T+30 (onset links) vs the
other links: history (T-60..T-5) statistics at the last 3 slots: q / cap, v / vf, and the q / cap trend. Reported:
AUC of q/cap_last3 for picking the onset links (within window), median q/cap and v/vf at onset links, share of
windows whose onset link is the window's max-q/cap link, and the number of windows per day.
"""
import sys
import warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "analysis")); sys.path.insert(0, str(ROOT / "tta_lab" / "tta"))
import numpy as np
import pandas as pd
from t2_eda03 import earliest_onset, labels
from tfb.data import RELEASE, load
import common as C
warnings.simplefilter("ignore")
H, F = 12, 6


def stats(z, cap, vf):
    q, el = labels(z)
    dd, tt = np.nonzero(earliest_onset(q))
    aucs, qc_on, v_on, top, n = [], [], [], [], 0
    for d, t0 in zip(dd, tt):
        y = q[d, t0 + F]
        if not y.any() or y.all():
            continue
        qh = z["true_flow"][d, t0 - 3:t0] / cap; vh = z["true_speed"][d, t0 - 3:t0] / vf
        s = np.nanmean(qh, 0)
        ok = np.isfinite(s)
        if ok.sum() < 5:
            continue
        a, b = s[ok & y], s[ok & ~y]
        if len(a) and len(b):
            aucs.append((a[:, None] > b[None, :]).mean() + 0.5 * (a[:, None] == b[None, :]).mean())
        qc_on.append(np.nanmean(s[y])); v_on.append(np.nanmean(np.nanmean(vh, 0)[y]))
        top.append(bool(y[np.nanargmax(np.where(ok, s, -1))]))
        n += 1
    return np.mean(aucs), np.nanmedian(qc_on), np.nanmedian(v_on), np.mean(top), n / len(z["dates"])


for i, panel in enumerate(C.T2_PANELS):
    z = load(panel, "train")
    ids = [str(x) for x in z["link_ids"]]
    cap = pd.read_csv(RELEASE / "corridors" / panel / "network" / "fd_parameters.csv").set_index("link_id").loc[ids].capacity_vph.values
    vf = np.asarray(z["free_speed"], float)
    r = stats(z, cap, vf)
    g = [stats(dict(np.load(f, allow_pickle=True)), cap, vf) for f in sorted((ROOT / "work" / "genruns_ctm").glob(f"{panel}_r*.npz"))[:4]]
    gm = np.nanmean(np.array(g, float), 0)
    print(f"{panel:<10} real: AUC q/cap {r[0]:.3f} onset q/cap {r[1]:.2f} v/vf {r[2]:.2f} top-link {r[3]:.2f} windows/day {r[4]:.2f} | "
          f"gen: AUC {gm[0]:.3f} q/cap {gm[1]:.2f} v/vf {gm[2]:.2f} top-link {gm[3]:.2f} windows/day {gm[4]:.2f}", flush=True)
