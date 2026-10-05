"""Note 169: robustness of the upstream entry-demand anomaly. (1) lab scored windows with the ctrl CNN, entries
k = 1/3/5 and span 3/6 slots; (2) train months with the out-of-fold H (occ_lab/cache/h_oof_<panel>.npz) as the
model, using the train truth flows. Candidates = the model's top-5 sites > 1 link apart; share of windows where
the anomaly ranks the true site first, misses vs hits; chance = 1 / #candidates."""
import sys
import warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "analysis")); sys.path.insert(0, str(ROOT / "tta_lab" / "tta"))
import numpy as np
import common as C
from dlsim_gen2 import Gen
from tfb.data import load
warnings.simplefilter("ignore")
F = 6


def cands_of(p, e):
    pp = np.where(e, p, -1).copy(); out = []
    while len(out) < 5 and pp.max() > -1:
        c = int(np.argmax(pp)); out.append(c); pp[max(0, c - 1):c + 2] = -1
    return out


def anomaly(g, q_day, v_day, t0, wd, span):
    q = q_day.astype(float).copy(); v = v_day.astype(float).copy(); q[t0:] = np.nan; v[t0:] = np.nan
    qs = q[:, g.order]
    import pandas as pd
    qs = pd.DataFrame(qs).interpolate(axis=1, limit_direction="both").ffill().bfill().to_numpy()
    e = qs[:, g.ent] - (1 - g.f_exit[g.ent - 1]) * qs[:, g.ent - 1]
    sl = slice(max(0, t0 - span), t0)
    return np.clip(e[sl], 0, None).mean(0) / np.maximum(g.P[wd][sl].mean(0), 1.0)


def score(g, rank, cands, truth_first_idx, an, k):
    vals = []
    for c in cands:
        rc = rank[c]
        ups = [i for i, b in enumerate(g.ent) if b <= rc][-k:]
        vals.append(np.mean(an[ups]) if ups else 1.0)
    return int(np.argmax(vals) == truth_first_idx)



def main():
    res = {}
    for i, panel in enumerate(C.T2_PANELS):
        zt = load(panel, "train")
        g = Gen(panel, zt); g.fit_profiles(zt)
        rank = np.full(g.S, -1); rank[g.order] = np.arange(g.n)
        # (1) scored-month lab windows, ctrl CNN
        ctx, stat, W = C.windows(i, panel)
        for s in C.SPLITS:
            on, z = W[s]["on"], W[s]["z"]
            p = np.mean([np.load(ROOT / "work" / "leap_ctrl_on" / f"fam_leap_ctrl_s{k}_sup+rec_{s}_{panel}.npz")["on"] for k in (20, 21)], 0)
            for n in range(len(p)):
                y = on["fut"][n, -1] & on["efut"][n, -1]; e = on["efut"][n, -1]
                if not y.any():
                    continue
                nt = y.copy(); nt[1:] |= y[:-1]; nt[:-1] |= y[1:]
                cs = cands_of(p[n], e); ti = [bool(nt[c]) for c in cs]
                if not any(ti):
                    continue
                kind = "hit" if ti[0] else "miss"
                d, t0 = int(on["d"][n]), int(on["t0"][n])
                for span in (3, 6):
                    an = anomaly(g, z["obs_flow"][d], z["obs_speed"][d], t0, int(z["weekday"][d]), span)
                    for k in (1, 3, 5):
                        res.setdefault(("lab", kind, span, k), []).append((score(g, rank, cs, ti.index(True), an, k), 1 / len(cs)))
        # (2) train months, out-of-fold H
        Hc = np.load(ROOT / "occ_lab" / "cache" / f"h_oof_{panel}.npz")
        for n in range(len(Hc["d"])):
            ph = Hc["prob"][n]
            if not np.isfinite(ph).all():
                continue
            y = Hc["fut"][n, -1] & Hc["efut"][n, -1]; e = Hc["efut"][n, -1]
            if not y.any():
                continue
            nt = y.copy(); nt[1:] |= y[:-1]; nt[:-1] |= y[1:]
            cs = cands_of(ph, e); ti = [bool(nt[c]) for c in cs]
            if not any(ti):
                continue
            kind = "hit" if ti[0] else "miss"
            d, t0 = int(Hc["d"][n]), int(Hc["t0"][n])
            for span in (3, 6):
                an = anomaly(g, zt["true_flow"][d], zt["true_speed"][d], t0, int(zt["weekday"][d]), span)
                for k in (1, 3, 5):
                    res.setdefault(("train", kind, span, k), []).append((score(g, rank, cs, ti.index(True), an, k), 1 / len(cs)))
        print(f"  {panel}", flush=True)
    for key in sorted(res):
        r = np.array(res[key], float)
        print(f"{key[0]:<5} {key[1]:<4} span {key[2]} entries {key[3]}: n {len(r):4d}  anomaly ranks the true site first {r[:, 0].mean():.3f}  (chance {r[:, 1].mean():.3f})")


if __name__ == "__main__":
    main()
