"""Note 169 (b): ramp-to-ramp paths. For each exit gap h->h+1 and each entry j merging up to 6 gaps upstream
(gap j-1 -> j, j < h: e_j must not be built from q_h itself, whose own noise does not pass the exit): high-pass (minus 1 h mean) fluctuations in free slots; the share of an entry-j fluctuation
that passes the exit, s_e = slope(q_{h+1} <- e_j) / slope(q_h <- e_j), vs the share of a mainline fluctuation
arriving from upstream of j, s_m = slope(q_{h+1} <- q_{j-1}) / slope(q_h <- q_{j-1}). With turning fractions both
equal 1 - F_h; s_e << s_m means vehicles entering at j leave at h (paths the node split does not know)."""
import sys
import warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "analysis")); sys.path.insert(0, str(ROOT / "tta_lab" / "tta"))
import numpy as np
from scipy.ndimage import uniform_filter1d
import common as C
from dlsim_gen2 import corridor
from dlsim_gen4 import Gen4
from tfb.data import load
warnings.simplefilter("ignore")


def hp(x):
    out = np.full_like(x, np.nan)
    for d in range(x.shape[0]):
        y = x[d].copy(); ok = np.isfinite(y)
        if ok.sum() < 50:
            continue
        y[~ok] = np.interp(np.flatnonzero(~ok), np.flatnonzero(ok), y[ok])
        out[d] = y - uniform_filter1d(y, 13, mode="nearest")
    return out


def slope(y, x, m):
    a, b = x[m], y[m]; ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 300:
        return np.nan, np.nan
    return np.polyfit(a[ok], b[ok], 1)[0], np.corrcoef(a[ok], b[ok])[0, 1]


def main():
    for panel in sys.argv[1:] or C.T2_PANELS:
        z = load(panel, "train")
        g = Gen4(panel, z)
        o = g.order; vf = np.asarray(z["free_speed"], float)[o]
        V, Q = z["true_speed"][:, :, o], z["true_flow"][:, :, o]
        fr = (V / vf > 0.9) & np.isfinite(Q)
        fr[:, :60] = False; fr[:, 252:] = False
        Hq = {}
        def hq(l):
            if l not in Hq:
                Hq[l] = hp(Q[:, :, l])
            return Hq[l]
        ents = sorted(int(b) for b in g.ent)
        found = []
        for h in range(g.n - 1):
            F = g.f_exit[h]
            if F <= 0.02:
                continue
            for j in [j for j in ents if h - 6 < j < h and j >= 1]:
                m = fr[:, :, j - 1] & fr[:, :, j] & fr[:, :, h] & fr[:, :, h + 1]
                e = hq(j) - (1 - g.f_exit[j - 1]) * hq(j - 1)
                a1, c1 = slope(hq(h), e, m); a2, _ = slope(hq(h + 1), e, m)
                b1, _ = slope(hq(h), hq(j - 1), m); b2, _ = slope(hq(h + 1), hq(j - 1), m)
                if not np.isfinite(a1) or a1 < 0.2 or c1 < 0.3 or not np.isfinite(b1) or b1 < 0.2:
                    continue
                se, sm = a2 / a1, b2 / b1
                found.append((h, j, F, se, sm, c1))
        print(f"{panel}: exit gaps with an upstream entry tested {len(found)}")
        for h, j, F, se, sm, c1 in found:
            flag = " <<" if se < sm - 0.2 else ""
            print(f"  exit {h}->{h + 1} (F {F:.2f}, 1-F {1 - F:.2f}) entry at {j}: pass share entry {se:.2f} | mainline {sm:.2f}{flag}")


if __name__ == "__main__":
    main()
