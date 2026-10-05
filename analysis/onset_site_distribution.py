"""Note 169: where and when does the first queue of the day occur, real train truth vs generated runs
(genruns_ctm_hb)? Earliest-onset windows (t2_eda03.earliest_onset); site = most downstream queued link at T+30
(travel order). Per panel: the top real and top generated sites with their shares, the overlap (sum of min shares),
the share of generated onsets at sites that host < 1 % of real onsets, and the onset time (T) distribution."""
import sys
import warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "analysis")); sys.path.insert(0, str(ROOT / "tta_lab" / "tta"))
import numpy as np
import common as C
from dlsim_gen2 import corridor
from t2_eda03 import earliest_onset, labels
from tfb.data import load
warnings.simplefilter("ignore")
F = 6


def sites(z, order):
    q, el = labels(z)
    dd, tt = np.nonzero(earliest_onset(q))
    rank = np.full(q.shape[2], -1); rank[order] = np.arange(len(order))
    out, times = [], []
    for d, t in zip(dd, tt):
        y = np.flatnonzero(q[d, t + F])
        y = y[rank[y] >= 0]
        if len(y):
            out.append(int(y[np.argmax(rank[y])])); times.append(t)
    return np.array(out), np.array(times)


def main():
    for panel in C.T2_PANELS:
        z = load(panel, "train")
        order, *_ = corridor(panel, z)
        S = len(z["link_ids"])
        rs, rt = sites(z, order)
        gs, gt = [], []
        for f in sorted((ROOT / "work" / __import__("os").environ.get("GEN_DIR", "genruns_ctm_hb")).glob(f"{panel}_r*.npz")):
            a, b = sites(dict(np.load(f, allow_pickle=True)), order)
            gs.append(a); gt.append(b)
        gs, gt = np.concatenate(gs), np.concatenate(gt)
        hr = np.bincount(rs, minlength=S) / max(len(rs), 1); hg = np.bincount(gs, minlength=S) / max(len(gs), 1)
        overlap = np.minimum(hr, hg).sum()
        novel = hg[hr < 0.01].sum()
        tr = np.argsort(-hr)[:4]; tg = np.argsort(-hg)[:4]
        print(f"{panel:<10} onsets real {len(rs)} gen {len(gs)} | overlap of site distributions {overlap:.2f} | generated onsets at sites with < 1 % of real: {novel:.2f}")
        print(f"   real top sites: " + ", ".join(f"{int(s)} {hr[s]:.2f} (gen {hg[s]:.2f})" for s in tr))
        print(f"   gen  top sites: " + ", ".join(f"{int(s)} {hg[s]:.2f} (real {hr[s]:.2f})" for s in tg))
        print(f"   onset time T (h): real median {np.median(rt) / 12:.1f} [q10 {np.quantile(rt, .1) / 12:.1f}, q90 {np.quantile(rt, .9) / 12:.1f}] | "
              f"gen median {np.median(gt) / 12:.1f} [q10 {np.quantile(gt, .1) / 12:.1f}, q90 {np.quantile(gt, .9) / 12:.1f}]", flush=True)


if __name__ == "__main__":
    main()
