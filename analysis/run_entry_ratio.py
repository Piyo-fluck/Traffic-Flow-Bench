"""Note 169 (evaluation): the run-level entry-demand ratio as onset site evidence. For a lab window on day d of a
scored month: per entry, the free-flow entry demand over the run's earlier days (all days < d, weekdays matched when
possible; released before T) divided by the train weekday profile, 06-11 h. Candidate score = mean over the k
entries merging at or upstream of the candidate. Same test as onset_site_features (ctrl CNN top-5 candidates)."""
import sys
import warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "analysis")); sys.path.insert(0, str(ROOT / "tta_lab" / "tta"))
import numpy as np
import common as C
from dlsim_gen2 import Gen
from entry_anomaly_robust import anomaly, cands_of, score
from tfb.data import load
warnings.simplefilter("ignore")
AM = slice(72, 132)


def run_ratio(g, z, d, P_train):
    """per entry: mean free-flow demand over days < d / train profile (06-11 h, weekday-matched)"""
    if d == 0:
        return None
    v, q = z["obs_speed"][:d][:, :, g.order], z["obs_flow"][:d][:, :, g.order]
    free = np.isfinite(v) & (v > 0.85 * g.vf) & np.isfinite(q)
    e = q[:, :, g.ent] - (1 - g.f_exit[g.ent - 1]) * q[:, :, g.ent - 1]
    ok = free[:, :, g.ent] & free[:, :, g.ent - 1]
    wd = z["weekday"][:d]
    num = np.zeros(len(g.ent)); den = np.zeros(len(g.ent))
    for i in range(d):
        m = ok[i][AM]
        num += np.where(m, np.clip(e[i][AM], 0, None), 0).sum(0)
        den += np.where(m, P_train[wd[i]][AM], 0).sum(0)
    return num / np.maximum(den, 1.0)



def main():
    res = {}
    for i, panel in enumerate(C.T2_PANELS):
        zt = load(panel, "train")
        g = Gen(panel, zt); g.fit_profiles(zt)
        Ptr = g.P.copy()
        rank = np.full(g.S, -1); rank[g.order] = np.arange(g.n)
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
                rr = run_ratio(g, z, d, Ptr)
                if rr is None:
                    continue
                g.P = Ptr
                today = anomaly(g, z["obs_flow"][d], z["obs_speed"][d], t0, int(z["weekday"][d]), 3)
                for k in (1, 3, 5):
                    res.setdefault((kind, "run-level", k), []).append((score(g, rank, cs, ti.index(True), rr, k), 1 / len(cs)))
                    res.setdefault((kind, "today", k), []).append((score(g, rank, cs, ti.index(True), today, k), 1 / len(cs)))
                    res.setdefault((kind, "run x today", k), []).append((score(g, rank, cs, ti.index(True), rr * today, k), 1 / len(cs)))
        print(f"  {panel}", flush=True)
    for key in sorted(res):
        r = np.array(res[key], float)
        print(f"{key[0]:<4} {key[1]:<12} entries {key[2]}: n {len(r):4d}  ranks the true site first {r[:, 0].mean():.3f} (chance {r[:, 1].mean():.3f})")


if __name__ == "__main__":
    main()
