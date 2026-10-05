"""Note 169: per-head breakdown factor. A head h (h queued, h+1 free; the downstream end: queued) passes a flow
before it breaks down that differs from its queue discharge (truth: D12_I5_N 105 0.97, 239 1.02, D7_I210_W end
1.19 x). Statistic per head: median over days of the max flow at the discharge sensor in the 3 free slots before
the day's first activation, over the median discharge while active. The CTM without a factor has its own value of
the same statistic (noise, sampling), so calibrate: ap <- ap x ratio_truth / ratio_replay (heads the replay never
activates: ap x 0.97), 4 rounds on half the train days. Saves work/ctm_apre_<panel>.npz (cells, ap, out_ap)."""
import os
import sys
import warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "analysis"))
for k, v in dict(FIFO="1", CONSERVE_FILL="1", HIDDEN_BN="1", HB_ALL="1", VIRT_LEN="3", HB_SUB_ENTRY="1", HB_RULE="choice", MAINLINE_CAP="1").items():
    os.environ.setdefault(k, v)
import numpy as np
from ctm_gen import CTMGen
from tfb.data import load
warnings.simplefilter("ignore")
A, B = 60, 252
HEAD_FREE = float(os.environ.get("HEAD_FREE", 0.6))                  # h+1 not queued (0.9: free)


def head_ratios(V, Q, vfo, heads, n):
    """V, Q (days, T, n) travel order; per head: (ratio, active days)"""
    r = V / vfo
    out = {}
    for h in heads:
        pre, dis = [], []
        for d in range(V.shape[0]):
            if h < n - 1:
                act = (r[d, :, h] <= 0.6) & (r[d, :, h + 1] > HEAD_FREE); m = h + 1
            else:
                act = r[d, :, h] <= 0.6; m = h
            a = np.flatnonzero(act[A:B])
            if len(a) < 2:
                continue
            t = A + a[0]
            w = Q[d, max(t - 3, 0):t, m]; fr = r[d, max(t - 3, 0):t, m] > 0.9
            if not np.isfinite(w[fr]).any():
                continue
            pre.append(np.nanmax(w[fr])); dis.append(np.nanmedian(Q[d, A:B, m][act[A:B]]))
        out[h] = (np.median(pre) / np.median(dis) if len(dis) >= 5 else np.nan, len(dis))
    return out


def main():
    for panel in sys.argv[1:]:
        z = load(panel, "train"); vf = np.asarray(z["free_speed"], float)
        g = CTMGen(panel, z); g.fit_profiles(z)
        if os.environ.get("RULE", "") == "1":                            # start from the truth-only type rule
            ch = np.load(ROOT / "work" / f"ctm_rule_{panel}.npz"); g.apply_choice(ch["heads"], ch["caps"], ch["sub"])
            g.ap[ch["ap_cells"]] = ch["ap"]; g.out_ap = float(ch["out_ap"])
        else:
            ch = np.load(ROOT / "work" / f"ctm_hb_choice_{panel}.npz"); g.apply_choice(ch["heads"], ch["caps"], ch["sub"])
        o = g.order; vfo = vf[o]; n = g.n; nd = len(z["dates"])
        days = np.arange(0, nd, 2)
        Vt, Qt = z["true_speed"][days][:, :, o], z["true_flow"][days][:, :, o]
        heads = list(range(n - 1)) + ([n - 1] if g.out_cap < 1e8 else [])
        rt = head_ratios(Vt, Qt, vfo, heads, n)
        heads = [h for h in heads if rt[h][1] >= max(5, 0.15 * len(days))]
        hc = getattr(g, "head_cell", {})
        cells = {h: (hc.get(h, g.sens_cell[h] + 1) if h < n - 1 else None) for h in heads}
        print(f"  end head: ratio {rt.get(n - 1, (np.nan, 0))[0]:.3f} days {rt.get(n - 1, (np.nan, 0))[1]} | out_cap {g.out_cap:.0f}", flush=True)
        print(f"{panel}: heads {heads}; truth ratios " + " ".join(f"{h}:{rt[h][0]:.3f}" for h in heads), flush=True)
        for it in range(4):
            Vr = np.empty_like(Vt); Qr = np.empty_like(Qt)
            for i, d in enumerate(days):
                up, ent = g.censored_demand(z["true_flow"][d], z["true_speed"][d], int(z["weekday"][d]))
                V, Q = g.simulate(up, ent); Vr[i], Qr[i] = V[:, o], Q[:, o]
            rr = head_ratios(Vr, Qr, vfo, heads, n)
            line = []
            for h in heads:
                cur = g.ap[cells[h]] if cells[h] is not None else g.out_ap
                if rr[h][1] >= 5 and np.isfinite(rr[h][0]):
                    new = cur * rt[h][0] / rr[h][0]
                else:
                    new = cur * (1.0 if os.environ.get("NEVER_ACT", "") == "keep" else 0.97)
                new = float(np.clip(new, 0.85, 1.3))
                if cells[h] is not None:
                    g.ap[cells[h]] = new
                else:
                    g.out_ap = new
                line.append(f"{h}:{rr[h][0]:.3f}/{rr[h][1]}->{new:.3f}")
            print(f"  it {it}: replay ratio/days -> ap  " + " ".join(line), flush=True)
        cl = [h for h in heads if cells[h] is not None]
        if os.environ.get("RULE", "") == "1":                            # keep the rule's other ap cells too
            rc = {int(c): float(a) for c, a in zip(ch["ap_cells"], ch["ap"])}
            for h in cl:
                rc[int(cells[h])] = float(g.ap[cells[h]])
            np.savez(ROOT / "work" / os.environ.get("APRE_OUT", "ctm_apre_{}.npz").format(panel), cells=np.array(list(rc), int),
                     ap=np.array(list(rc.values())), out_ap=g.out_ap)
            continue
        np.savez(ROOT / "work" / os.environ.get("APRE_OUT", "ctm_apre_{}.npz").format(panel), cells=np.array([cells[h] for h in cl], int),
                 ap=np.array([g.ap[cells[h]] for h in cl]), out_ap=g.out_ap)


if __name__ == "__main__":
    main()
