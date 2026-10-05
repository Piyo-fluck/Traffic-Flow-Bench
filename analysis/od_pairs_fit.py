"""Note 169 (b)(i): ramp-to-ramp pairs and their share beta from high-pass flow responses in free slots (see
ramp_to_ramp_scan.py): beta = 1 - s_e / s_m for pairs with s_e < s_m - 0.2 (entry j at least one gap upstream of
exit h). Saves work/od_pairs_<panel>.json. With --check, also replays every train day with the pairs (CTM, v11
settings + OD classes) and prints the replay's own s_e / s_m at the same pairs (does the mechanism reproduce?)."""
import json
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
from ramp_to_ramp_scan import hp, slope
from tfb.data import load
warnings.simplefilter("ignore")


def pair_stats(V, Q, vf, g, pairs=None):
    """V, Q travel order (days, T, n). Returns [(h, j, s_e, s_m, corr)]"""
    fr = (V / vf > 0.9) & np.isfinite(Q)
    fr[:, :60] = False; fr[:, 252:] = False
    cache = {}
    def hq(l):
        if l not in cache:
            cache[l] = hp(Q[:, :, l])
        return cache[l]
    ents = sorted(int(b) for b in g.ent)
    cand = pairs if pairs is not None else [(h, j) for h in range(g.n - 1) if g.f_exit[h] > 0.02 for j in ents if h - 6 < j < h and j >= 1]
    out = []
    for h, j in cand:
        m = fr[:, :, j - 1] & fr[:, :, j] & fr[:, :, h] & fr[:, :, h + 1]
        e = hq(j) - (1 - g.f_exit[j - 1]) * hq(j - 1)
        a1, c1 = slope(hq(h), e, m); a2, _ = slope(hq(h + 1), e, m)
        b1, _ = slope(hq(h), hq(j - 1), m); b2, _ = slope(hq(h + 1), hq(j - 1), m)
        if not np.isfinite(a1) or a1 < 0.2 or c1 < 0.3 or not np.isfinite(b1) or b1 < 0.2:
            out.append((h, j, np.nan, np.nan, c1)); continue
        out.append((h, j, a2 / a1, b2 / b1, c1))
    return out


def main():
    from ctm_gen import CTMGen
    check = "--check" in sys.argv
    panels = [a for a in sys.argv[1:] if not a.startswith("--")] or C.T2_PANELS
    for panel in panels:
        z = load(panel, "train")
        g = CTMGen(panel, z); g.fit_profiles(z)
        o = g.order; vf = np.asarray(z["free_speed"], float)[o]
        st = pair_stats(z["true_speed"][:, :, o], z["true_flow"][:, :, o], vf, g)
        pairs = [(int(j), int(h), float(np.clip(1 - se / sm, 0, 0.95)), float(se), float(sm)) for h, j, se, sm, c in st
                 if np.isfinite(se) and se < sm - 0.2]
        json.dump(pairs, open(ROOT / "work" / f"od_pairs_{panel}.json", "w"))
        print(f"{panel}: pairs (entry j, exit h, beta, s_e, s_m): " + "; ".join(f"{j}->{h} beta {b:.2f} ({se:.2f}/{sm:.2f})" for j, h, b, se, sm in pairs), flush=True)
        if "--calib" in sys.argv and pairs:
            # calibrate beta so that the replay's s_e / s_m equals the truth's, pair by pair (secant, 3 steps);
            # the CTM without classes does not give s_e / s_m = 1 itself (D12_I5_N 239: 0.70)
            ch = np.load(ROOT / "work" / f"ctm_hb_choice_{panel}.npz"); g.apply_choice(ch["heads"], ch["caps"], ch["sub"])
            nd = len(z["dates"]); days = np.arange(0, nd, 2)
            def ratios(betas):
                g.set_od_pairs(z, [(j, h, b) for (j, h, *_), b in zip(pairs, betas)])
                Vr = np.empty((len(days), 288, g.n)); Qr = np.empty_like(Vr)
                for i, d in enumerate(days):
                    up, ent = g.censored_demand(z["true_flow"][d], z["true_speed"][d], int(z["weekday"][d]))
                    V, Q = g.simulate(up, ent); Vr[i], Qr[i] = V[:, o], Q[:, o]
                return np.array([se / sm for h, j, se, sm, _ in pair_stats(Vr, Qr, vf, g, [(h, j) for j, h, *_ in pairs])])
            target = np.array([se / sm for _, _, _, se, sm in pairs])
            b0 = np.zeros(len(pairs)); r0 = ratios(b0)
            b1 = np.array([p[2] for p in pairs]); r1 = ratios(b1)
            print(f"  calib: target {np.round(target, 2)} | beta 0 -> {np.round(r0, 2)} | beta {np.round(b1, 2)} -> {np.round(r1, 2)}", flush=True)
            for it in range(3):
                with np.errstate(all="ignore"):
                    b2 = np.where(np.abs(r1 - r0) > 1e-3, b1 + (target - r1) * (b1 - b0) / (r1 - r0), b1)
                b2 = np.clip(np.nan_to_num(b2, nan=0.0), 0, 0.95)
                r2 = ratios(b2)
                print(f"  calib it {it}: beta {np.round(b2, 3)} -> ratio {np.round(r2, 2)}", flush=True)
                b0, r0, b1, r1 = b1, r1, b2, r2
            pairs = [(j, h, float(b), se, sm) for (j, h, _, se, sm), b in zip(pairs, b1)]
            json.dump(pairs, open(ROOT / "work" / f"od_pairs_{panel}.json", "w"))
        if check and pairs:
            ch = np.load(ROOT / "work" / f"ctm_hb_choice_{panel}.npz"); g.apply_choice(ch["heads"], ch["caps"], ch["sub"])
            nd = len(z["dates"])
            for tag in ("no classes", "OD classes"):
                if tag == "OD classes":
                    g.set_od_pairs(z, [(j, h, b) for j, h, b, _, _ in pairs])
                Vr = np.empty((nd, 288, g.n)); Qr = np.empty_like(Vr)
                for d in range(nd):
                    up, ent = g.censored_demand(z["true_flow"][d], z["true_speed"][d], int(z["weekday"][d]))
                    V, Q = g.simulate(up, ent); Vr[d], Qr[d] = V[:, o], Q[:, o]
                sr = pair_stats(Vr, Qr, vf, g, [(h, j) for j, h, *_ in pairs])
                print(f"  replay {tag}: " + "; ".join(f"{j}->{h} s_e {se:.2f} s_m {sm:.2f}" for h, j, se, sm, _ in sr), flush=True)


if __name__ == "__main__":
    main()
