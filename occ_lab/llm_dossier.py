"""Blind judging study (note 172): can a reasoning model decide production's close two-way onset calls better than
production? Builds one text dossier per lab close call (K = 0 condition), with data released at or before T only.

Close call: after BB's set decoder stage 1 (production-like, rankers fitted on the other month), the 2nd candidate's
region mass >= 0.5 x the top candidate's. 20 windows per month (fixed seed). The two candidates are shown as X / Y in
random order. The key (truth region, production's top) goes to key.json, which the judge must not open before judging.

    ONSET_SHARE_K=1 .venv/Scripts/python.exe occ_lab/llm_dossier.py
"""
import json
import sys
import warnings
from pathlib import Path

LAB = Path(__file__).resolve().parent
ROOT = LAB.parent
for p in (LAB, LAB.parent / "tta_lab" / "tta", ROOT, ROOT / "analysis"):
    sys.path.insert(0, str(p))
import numpy as np  # noqa: E402

import common as C  # noqa: E402
from ctm_gen import CTMGen  # noqa: E402
from onset_setdec_fit import SEEDS, fit_rank, apply_rank  # noqa: E402
from setdec_early_check import prep  # noqa: E402
from tfb.data import load  # noqa: E402
from tfb.physics import load_ramps  # noqa: E402

warnings.simplefilter("ignore")
OUT = ROOT / "work" / "llm_judge"
F = 6
N_PER = 20


def forward_first(g, z, d, t0):
    q = z["obs_flow"][d].astype(float).copy(); v = z["obs_speed"][d].astype(float).copy()
    q[t0:] = np.nan; v[t0:] = np.nan
    wd = int(z["weekday"][d])
    up, ent = g.conservation_demand(q, v); up, ent = np.array(up, float), np.array(ent, float)
    last = slice(max(0, t0 - 3), t0)
    lu, le = up[last].mean(), ent[last].mean(0)
    pu, pe = g.U[wd], g.P[wd]
    bu, be = max(pu[last].mean(), 1.0), np.maximum(pe[last].mean(0), 1.0)
    U = np.array([lu * np.clip(pu[t] / bu, 0.5, 2.0) for t in range(t0, t0 + F + 1)])
    E = np.stack([le * np.clip(pe[t] / be, 0.5, 2.0) for t in range(t0, t0 + F + 1)])
    k0 = g.state_from_obs(z["obs_speed"][d, t0 - 1], z["obs_flow"][d, t0 - 1])
    g.simulate(U, E, n_slots=F + 1, k0=k0)
    kc = g.cell_cap / g.cell_vf
    dens = (g.last_k[1:] / kc[None, :])[:, g.sens_cell]                     # (F, n) travel order
    out = np.full((F, g.S), np.nan); out[:, g.order] = dens
    return out, le / be


def hhmm(t):
    return f"{t * 5 // 60:02d}:{t * 5 % 60:02d}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dossiers").mkdir(exist_ok=True)
    full = prep(None)
    rk = {s: [fit_rank([d[s] for d in full.values() if s in d], sd) for sd in SEEDS] for s in C.SPLITS}
    early = prep(0)
    pool = {s: [] for s in C.SPLITS}
    for p_, d in early.items():
        for s, D in d.items():
            other = [x for x in C.SPLITS if x != s][0]
            apply_rank(rk[other], D)
            off = 0
            for (n, cand, reg, mass), g_ in zip(D["meta"], D["g"]):
                y = D["y"][off:off + g_]; off += g_
                if not y.any():
                    continue
                m1 = np.bincount(reg, weights=D["P1"][n], minlength=len(cand))
                o = np.argsort(-m1)
                if len(o) < 2 or m1[o[1]] < 0.5 * m1[o[0]]:
                    continue
                pool[s].append((p_, n, int(o[0]), int(o[1]), int(np.argmax(y))))
    rng = np.random.default_rng(171)
    chosen = {s: [pool[s][i] for i in sorted(rng.choice(len(pool[s]), min(N_PER, len(pool[s])), replace=False))]
              for s in C.SPLITS}
    key = {}
    gens = {}
    for s in C.SPLITS:
        for (p_, n, top, sec, tru) in chosen[s]:
            D = early[p_][s]; on = D["on"]
            meta = {m[0]: m for m in D["meta"]}
            _, cand, reg, mass = meta[n]
            if p_ not in gens:
                zt = load(p_, "train")
                g = CTMGen(p_, zt); g.fit_profiles(zt)
                rank = np.full(g.S, -1); rank[g.order] = np.arange(g.n)
                ceff = np.full(g.S, np.nan)
                for h in range(g.n - 1):
                    ceff[g.order[h]] = g.cell_cap[g.sens_cell[h] + 1]
                ceff[g.order[-1]] = g.cell_cap[g.sens_cell[-1]]
                vf = np.asarray(zt["free_speed"], np.float32)
                okt = zt["elig"] & np.isfinite(zt["obs_speed"])
                qt = okt & (np.nan_to_num(zt["obs_speed"], nan=1e9) <= 0.6 * vf)
                habit = qt[:, 72:144].any(1).mean(0)                         # train days with a queue 06-12 h
                rt = load_ramps(p_, "train", zt)
                gens[p_] = (g, rank, ceff, vf, habit, rt)
            g, rank, ceff, vf, habit, rt = gens[p_]
            z = load(p_, s)
            d, t0 = int(on["d"][n]), int(on["t0"][n])
            dens, anom_e = forward_first(g, z, d, t0)
            m1 = np.bincount(reg, weights=D["P1"][n], minlength=len(cand))
            order = [top, sec] if rng.random() < 0.5 else [sec, top]
            wid = f"{s[:3]}_{p_}_{n:03d}"
            key[wid] = dict(X=order[0], Y=order[1], production_top="X" if order[0] == top else "Y",
                            truth="X" if tru == order[0] else ("Y" if tru == order[1] else "other"))
            L = [f"WINDOW {wid}", f"corridor {p_}, forecast origin T = {hhmm(t0)}, "
                 f"{'weekend' if z['weekday'][d] >= 5 else 'weekday'}, day {d} of the run (no earlier day of this run is used)",
                 "Question: the first queue of the day appears at T+30 min. Which candidate site is it at, X or Y?",
                 "Links are listed in travel order (upstream first). v = speed / free speed, q = flow / train 95th pct free flow.",
                 "Each series: the 12 five-minute slots T-60 .. T-5 min.", ""]
            ents = np.asarray(g.ent)
            for lab_, c in zip(("X", "Y"), order):
                j = int(cand[c]); rc = int(rank[j])
                L.append(f"--- candidate {lab_}: link {j} (travel position {rc} of {g.n})")
                L.append(f"  production model: region probability share {m1[c] / m1.sum():.2f}, peak link probability {D['P1'][n][j]:.2f}")
                L.append(f"  training-run habit: queued on {habit[j]:.0%} of training days (06-12 h) at this link")
                L.append(f"  site discharge capacity (train) {ceff[j]:.0f} veh/h")
                ups = [i_ for i_, b in enumerate(ents) if rc - 3 <= b <= rc]
                if ups:
                    L.append("  upstream on-ramp demand today / training profile at this time: "
                             + ", ".join(f"{anom_e[i_]:.2f}" for i_ in ups))
                seq = [g.order[r] for r in range(max(0, rc - 3), min(g.n, rc + 4))]
                for k in seq:
                    okk = on["ok"][n][:, k]
                    vs = " ".join(f"{on['v'][n][h, k] / vf[k]:.2f}" if okk[h] else " -- " for h in range(12))
                    qs = " ".join(f"{on['qf'][n][h, k] / max(ceff[k], 1):.2f}" if okk[h] and np.isfinite(ceff[k]) else " -- "
                                  for h in range(12))
                    mark = " <== candidate" if k == j else ""
                    L.append(f"  link {k:>3} (pos {rank[k]:>3}) v: {vs}{mark}")
                    L.append(f"  {'':>17} q/cap: {qs}")
                fd = dens[:, j]
                L.append("  physics (CTM forward from T, demand held at today's level): density / critical at T+5..T+30: "
                         + " ".join(f"{x:.2f}" for x in fd))
                L.append("")
            (OUT / "dossiers" / f"{wid}.txt").write_text("\n".join(L), encoding="utf-8")
        print(f"  {s}: {len(chosen[s])} dossiers", flush=True)
    (OUT / "key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    print(f"wrote {len(key)} dossiers to {OUT / 'dossiers'}; key.json written (do not open before judging)")


if __name__ == "__main__":
    main()
