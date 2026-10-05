"""Note 169 (b): why does D12_I5_N 239 still take onsets with the breakdown factors and EXIT_TV? Per day where the
replay's first onset is at 239: the implied exit share of gap 239->240 in truth, F = 1 - (q240 - e) / q239 (e = the
entry's weekday profile), before and after the onset, whether EXIT_TV changed F in the replay, the flows at 239 in
truth and replay at the replay onset, and the truth's own first onset site."""
import os
import sys
import warnings
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "analysis"))
for k, v in dict(FIFO="1", CONSERVE_FILL="1", HIDDEN_BN="1", HB_ALL="1", VIRT_LEN="3", HB_SUB_ENTRY="1", HB_RULE="choice",
                 MAINLINE_CAP="1", EXIT_TV="1", EXIT_TV_RHO="1.0").items():
    os.environ.setdefault(k, v)
import numpy as np
from ctm_gen import CTMGen
from onset_sites_util import day_sites
from tfb.data import load
warnings.simplefilter("ignore")
panel, H = "D12_I5_N", 239
z = load(panel, "train"); vf = np.asarray(z["free_speed"], float)
g = CTMGen(panel, z); g.fit_profiles(z)
ch = np.load(ROOT / "work" / f"ctm_hb_choice_{panel}.npz"); g.apply_choice(ch["heads"], ch["caps"], ch["sub"])
a = np.load(ROOT / "work" / "ctm_apre3_D12_I5_N.npz"); g.ap[a["cells"]] = a["ap"]; g.out_ap = float(a["out_ap"])
o = g.order; rank = np.full(len(vf), -1); rank[o] = np.arange(len(o))
i_e = {int(b): i for i, b in enumerate(g.ent)}[H + 1]
F0 = g.f_exit[H]
rows = []
for d in range(len(z["dates"])):
    up, ent = g.censored_demand(z["true_flow"][d], z["true_speed"][d], int(z["weekday"][d]))
    Ft = g.f_t[:, H].copy() if g.f_t is not None else np.full(288, F0)
    V, Q = g.simulate(up, ent)
    el = z["elig"][d]
    sr = day_sites(V, el & np.isfinite(V), vf, rank); st = day_sites(z["true_speed"][d], el & np.isfinite(z["true_speed"][d]), vf, rank)
    if not sr or sr[0][1] != o[H]:
        continue
    t = sr[0][0]                                                    # replay onset origin; queue appears by t+6
    q, v = z["true_flow"][d][:, o], z["true_speed"][d][:, o] / vf[o]
    P = g.P[int(z["weekday"][d])][:, i_e]
    with np.errstate(all="ignore"):
        Fi = 1 - (q[:, H + 1] - P) / q[:, H]
    w0, w1 = slice(t - 6, t + 1), slice(t + 1, t + 13)             # 30 min before / 1 h after
    rq = (V[:, o][:, H] / vf[o][H] <= 0.6)
    tq = np.flatnonzero(rq[t:t + 12])
    t_b = t + (tq[0] if len(tq) else 6)
    rows.append(dict(d=d, t=t, truth_site=(st[0][1] if st else -1), truth_t=(st[0][0] if st else -1),
                     Fi_pre=np.nanmedian(Fi[w0]), Fi_post=np.nanmedian(Fi[w1]), Ft_pre=np.median(Ft[w0]), Ft_post=np.median(Ft[w1]),
                     tv_pre=np.mean(Ft[w0] > F0 + 1e-6), q239_t=np.nanmax(q[t_b - 2:t_b + 1, H]), q239_r=Q[:, o][t_b - 2:t_b + 1, H].max(),
                     v239_t=np.nanmin(v[t_b:t_b + 3, H]), v240_t=np.nanmin(v[w1, H + 1]), up_t=np.nanmax(q[t_b - 2:t_b + 1, H - 3]),
                     up_r=Q[:, o][t_b - 2:t_b + 1, H - 3].max()))
R = {k: np.array([r[k] for r in rows]) for k in rows[0]}
n = len(rows)
print(f"{panel}: replay first onset at 239 on {n} days; fixed exit share {F0:.3f}")
print("  truth first onset site on those days: " + ", ".join(f"{s}:{c}" for s, c in Counter(R['truth_site'].tolist()).most_common(5)))
print(f"  truth onset time - replay onset time: median {np.median((R['truth_t'] - R['t'])[R['truth_t'] >= 0]) / 12:+.2f} h")
surge = R["Fi_pre"] > F0 + 0.05
print(f"  truth implied exit share 30 min before: median {np.nanmedian(R['Fi_pre']):.3f} [q10 {np.nanquantile(R['Fi_pre'], .1):.3f}, q90 {np.nanquantile(R['Fi_pre'], .9):.3f}]; "
      f"after: median {np.nanmedian(R['Fi_post']):.3f} [q90 {np.nanquantile(R['Fi_post'], .9):.3f}]")
print(f"  days with a surge (implied F > {F0 + .05:.3f} before): {surge.sum()}/{n}; EXIT_TV raised F before the onset on {np.mean(R['tv_pre'] > 0):.2f} of days (share of slots {np.mean(R['tv_pre']):.2f})")
print(f"  replay F used: before {np.median(R['Ft_pre']):.3f}, after {np.median(R['Ft_post']):.3f}")
for nm, m in (("surge", surge), ("no surge", ~surge)):
    if m.sum() == 0:
        continue
    print(f"  [{nm}] n={m.sum()}: at the replay breakdown truth q239 {np.median(R['q239_t'][m]):.0f} vs replay {np.median(R['q239_r'][m]):.0f}; "
          f"truth min v/vf at 239 then {np.median(R['v239_t'][m]):.2f}; truth 240 min v/vf next hour {np.median(R['v240_t'][m]):.2f}; "
          f"flow 3 links upstream truth {np.median(R['up_t'][m]):.0f} replay {np.median(R['up_r'][m]):.0f}")
