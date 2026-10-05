"""Note 169 (evaluation): a forward CTM from the observed state at T decides the habitual-bottleneck race?
Per lab onset window: cell densities from the last history slot (state_from_obs), demand for T..T+30 = the last
3 slots' level (conservation estimate on the day's flows before T only) x the train weekday profile shape; the CTM
(hidden bottlenecks) runs 7 slots. Score per sensored link = simulated density / critical density at T+30. Race:
true site vs competitor (as onset_race.py), misses vs hits; top-1 of the CTM score alone and of ctrl p + w x CTM."""
import sys
import warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "analysis")); sys.path.insert(0, str(ROOT / "tta_lab" / "tta"))
import numpy as np
import common as C
from ctm_gen import CTMGen
from tfb.data import load
warnings.simplefilter("ignore")
F = 6


def forward(g, z, d, t0):
    q = z["obs_flow"][d].astype(float).copy(); v = z["obs_speed"][d].astype(float).copy()
    q[t0:] = np.nan; v[t0:] = np.nan
    wd = int(z["weekday"][d])
    up, ent = g.conservation_demand(q, v)
    up, ent = np.array(up, float), np.array(ent, float)
    last = slice(max(0, t0 - 3), t0)
    lu, le = up[last].mean(), ent[last].mean(0)
    pu, pe = g.U[wd], g.P[wd]
    bu, be = max(pu[last].mean(), 1.0), np.maximum(pe[last].mean(0), 1.0)
    U = np.array([lu * np.clip(pu[t] / bu, 0.5, 2.0) for t in range(t0, t0 + F + 1)])
    E = np.stack([le * np.clip(pe[t] / be, 0.5, 2.0) for t in range(t0, t0 + F + 1)])
    k0 = g.state_from_obs(z["obs_speed"][d, t0 - 1], z["obs_flow"][d, t0 - 1])
    g.simulate(U, E, n_slots=F + 1, k0=k0)
    kc = g.cell_cap / g.cell_vf
    ratio = (g.last_k[F] / kc)[g.sens_cell]                                # travel order
    sc = np.full(g.S, np.nan); sc[g.order] = ratio
    return np.nan_to_num(sc)


def main():
    res = {"miss": [], "hit": []}
    top = {"ctrl": [], "ctm": [], "blend 0.1": [], "blend 0.3": []}
    for i, panel in enumerate(C.T2_PANELS):
        ctx, stat, W = C.windows(i, panel)
        zt = load(panel, "train")
        g = CTMGen(panel, zt)
        g.fit_profiles(zt)
        for s in C.SPLITS:
            on, z = W[s]["on"], W[s]["z"]
            p = np.mean([np.load(ROOT / "work" / "leap_ctrl_on" / f"fam_leap_ctrl_s{k}_sup+rec_{s}_{panel}.npz")["on"] for k in (20, 21)], 0)
            for n in range(len(p)):
                y = on["fut"][n, -1] & on["efut"][n, -1]; e = on["efut"][n, -1]
                if not y.any():
                    continue
                sc = forward(g, z, int(on["d"][n]), int(on["t0"][n]))
                nt = y.copy(); nt[1:] |= y[:-1]; nt[:-1] |= y[1:]
                j = int(np.argmax(np.where(e, p[n], -1)))
                kind = "hit" if nt[j] else "miss"
                if kind == "hit":
                    pp = np.where(e & ~nt, p[n], -1)
                    if pp.max() >= 0:
                        j2 = int(np.argmax(pp))
                        res[kind].append(sc[y].max() > sc[j2])
                else:
                    res[kind].append(sc[y].max() > sc[j])
                scn = sc / max(sc.max(), 1e-9)
                for nm, q in (("ctrl", p[n]), ("ctm", sc), ("blend 0.1", p[n] + 0.1 * scn), ("blend 0.3", p[n] + 0.3 * scn)):
                    top[nm].append(bool(nt[int(np.argmax(np.where(e, q, -1)))]))
            print(f"  {panel} {s}", flush=True)
    for k, v in res.items():
        print(f"{k:<5} windows {len(v)}: forward CTM ranks the true site above the competitor in {np.mean(v):.2f}")
    for k, v in top.items():
        print(f"top-1 hit {k:<10} {np.mean(v):.4f} (n {len(v)})")


if __name__ == "__main__":
    main()
