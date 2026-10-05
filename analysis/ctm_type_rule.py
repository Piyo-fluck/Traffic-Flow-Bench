"""Note 169: capacity type and pre-breakdown factor per head from the truth alone (no replication fit).
Heads = sensored links h queued (< 0.6 vf) with h+1 free (> 0.9 vf) on >= 15 % of train days (05-21 h), plus the
downstream end when the corridor has a boundary capacity. Discharge D_h = median over days of the median flow at h+1
while active; pre-breakdown P_h = p99 of the free-flow flow at h+1 (the ceiling a head holds before it breaks).
  heads with an entry in the gap (raw/sub options): P / raw <= RATIO_MAX -> 'raw' (ceiling on the total after the
  merge) with ap = P / raw; else 'sub' (the entry adds above the ceiling: mainline-only cap), ap 1
  other heads: ap = P / D on the cell after h when it is <= RATIO_MAX (else 1)
  downstream end: out_ap = p99 free flow at the end / out_cap (<= 1.3)
Saves work/ctm_rule_<panel>.npz (heads, caps, sub, ap_cells, ap, out_ap)."""
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
RMAX = float(os.environ.get("RATIO_MAX", 1.10))


def main():
    for panel in sys.argv[1:] or C.T2_PANELS:
        z = load(panel, "train"); g = CTMGen(panel, z)
        opts = dict(getattr(g, "hb_options", {}))
        o = g.order; vf = np.asarray(z["free_speed"], float)[o]
        V, Q = z["true_speed"][:, 60:252, o], z["true_flow"][:, 60:252, o]
        nd = V.shape[0]; n = g.n
        heads, caps, subs, apc, apv, info = [], [], [], [], [], []
        for h in range(n - 1):
            act = (V[:, :, h] <= 0.6 * vf[h]) & (V[:, :, h + 1] > float(os.environ.get("HB_HEAD_FREE", 0.9)) * vf[h + 1])
            days = act.sum(1) >= 2
            if days.sum() < max(5, 0.15 * nd):
                continue
            fr = V[:, :, h + 1] > 0.9 * vf[h + 1]
            P = np.nanquantile(Q[:, :, h + 1][fr], 0.99)
            D = np.median([np.nanmedian(Q[d, :, h + 1][act[d]]) for d in np.flatnonzero(days)])
            if h in opts:
                raw, sub = opts[h]
                r = P / raw
                if r <= RMAX:
                    heads.append(h); caps.append(raw); subs.append(False)
                    apc.append(g.sens_cell[h] + 1); apv.append(max(r, 1.0)); info.append(f"{h}:R{r:.3f}")
                else:
                    heads.append(h); caps.append(sub); subs.append(True); info.append(f"{h}:S{r:.2f}")
            else:
                r = P / D
                if 1.0 < r <= RMAX and os.environ.get("AP_OTHER", "1") == "1":
                    apc.append(g.sens_cell[h] + 1); apv.append(r); info.append(f"{h}:a{r:.3f}")
        out_ap = 1.0
        if g.out_cap < 1e8:
            fr = V[:, :, n - 1] > 0.9 * vf[n - 1]
            out_ap = float(np.clip(np.nanquantile(Q[:, :, n - 1][fr], 0.99) / g.out_cap, 1.0, 1.3))
            info.append(f"end:{out_ap:.3f}")
        print(f"{panel:<10} " + " ".join(info), flush=True)
        np.savez(ROOT / "work" / os.environ.get("RULE_OUT", "ctm_rule_{}.npz").format(panel), heads=np.array(heads, int), caps=np.array(caps, float), sub=np.array(subs, bool),
                 ap_cells=np.array(apc, int), ap=np.array(apv, float), out_ap=out_ap)


if __name__ == "__main__":
    main()
