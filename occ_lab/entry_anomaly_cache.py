"""Note 169: cache per lab onset window and sensored link the upstream entry-demand anomaly features:
today = mean over the 3 entries merging at or upstream of the link of (last-3-slot demand / train profile),
run = mean over 5 such entries of the scored run's earlier-day free-flow demand / train profile (06-11 h).
occ_lab/cache/entanom_<split>_<panel>.npz: today, run (N_on, S), common.windows order."""
import sys
import warnings
from pathlib import Path
LAB = Path(__file__).resolve().parent
ROOT = LAB.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "analysis")); sys.path.insert(0, str(ROOT / "tta_lab" / "tta"))
import numpy as np
import common as C
from dlsim_gen2 import Gen
from tfb.data import load
warnings.simplefilter("ignore")


def per_link(g, rank, vals, k):
    out = np.ones(g.S)
    for c in range(g.S):
        rc = rank[c]
        ups = [i for i, b in enumerate(g.ent) if b <= rc][-k:]
        if ups:
            out[c] = np.mean(vals[ups])
    return out


def main():
    from entry_anomaly_robust import anomaly
    from run_entry_ratio import run_ratio
    for i, panel in enumerate(C.T2_PANELS):
        zt = load(panel, "train")
        g = Gen(panel, zt); g.fit_profiles(zt)
        rank = np.full(g.S, -1); rank[g.order] = np.arange(g.n)
        ctx, stat, W = C.windows(i, panel)
        for s in C.SPLITS:
            on, z = W[s]["on"], W[s]["z"]
            T_, R_ = [], []
            for n in range(len(on["d"])):
                d, t0 = int(on["d"][n]), int(on["t0"][n])
                an = anomaly(g, z["obs_flow"][d], z["obs_speed"][d], t0, int(z["weekday"][d]), 3)
                rr = run_ratio(g, z, d, g.P)
                T_.append(per_link(g, rank, an, 3)); R_.append(per_link(g, rank, rr if rr is not None else np.ones(len(g.ent)), 5))
            np.savez(LAB / "cache" / f"entanom_{s}_{panel}.npz", today=np.array(T_, np.float32).reshape(-1, g.S),
                     run=np.array(R_, np.float32).reshape(-1, g.S))
            print(f"  {panel} {s}: {len(T_)}", flush=True)


if __name__ == "__main__":
    main()
