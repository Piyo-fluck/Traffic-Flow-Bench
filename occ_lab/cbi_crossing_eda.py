"""CBI/QVDF-style bottleneck activation evidence for the onset candidates, with the generator's fitted service rate.

Per candidate peak j of note 152's set decoder (scored-month lab onset windows; evaluation only):
    lambda(t)  arrival = flow at j's upstream sensored neighbour (travel order from path_set, note 169), history
    lambda^    quadratic fit over the last 12 history slots, extrapolated to T+5 .. T+30
    mu         service rate at j = min(fitted effective capacity of j, of its downstream neighbour)
               (work/dlsim_caps_<panel>.npy, fitted on train by analysis/dlsim_fitcap.py); control: released cap
    E(h)       cumulative excess sum_{s<=h} (lambda^_s - mu) / 12  [veh], h = 2, 4, 6 slots
    first      first slot with lambda^ > mu (7 if none);  mu / C
Discrimination: within each window, pairwise AUC between the best candidate (max utility U, U > 0) and the others,
for each feature, against the production evidence (the candidate's blend mass) and a random placebo.

    .venv/Scripts/python.exe occ_lab/cbi_crossing_eda.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB)); sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta")); sys.path.insert(0, str(LAB.parent / "analysis"))
sys.path.insert(0, str(LAB.parent))
import numpy as np  # noqa: E402

import common as C  # noqa: E402
import onset_rerank as RR  # noqa: E402
from dlsim_gen2 import corridor  # noqa: E402

ROOT = LAB.parent


def feats(v, qf, ok, j, up, mu, cap):
    q = np.where(ok[:, up], qf[:, up], np.nan)
    t = np.arange(len(q))
    m = np.isfinite(q)
    if m.sum() < 4:
        return [np.nan] * 6
    c = np.polyfit(t[m], q[m], 2)
    lam = np.clip(np.polyval(c, len(q) - 1 + np.arange(1, 7)), 0, None)
    ex = lam - mu
    E = np.cumsum(ex) / 12
    first = int(np.argmax(ex > 0)) + 1 if (ex > 0).any() else 7
    return [E[1], E[3], E[5], first, ex.max() / mu, mu / cap]


def pair_auc(f, U, g):
    num, den, off = 0.0, 0.0, 0
    for k in g:
        u, x = U[off:off + k], f[off:off + k]; off += k
        if k < 2 or u.max() <= 0:
            continue
        b = int(np.argmax(u))
        for i in range(k):
            if i == b or u[i] >= u[b] or not (np.isfinite(x[i]) and np.isfinite(x[b])):
                continue
            num += (x[b] > x[i]) + 0.5 * (x[b] == x[i]); den += 1
    return num / max(den, 1), int(den)


def main():
    warnings.simplefilter("ignore")
    panels = [p for p in C.T2_PANELS if (ROOT / "work" / f"dlsim_caps_{p}.npy").exists()]
    print("panels with fitted caps:", panels, flush=True)
    data = RR.prepare()
    names = ["E10", "E20", "E30", "first", "maxex", "mu/C"]
    rng = np.random.default_rng(0)
    res = {}
    from tfb.data import load
    for panel in panels:
        z = load(panel, "train")
        order, *_ = corridor(panel, z)
        rank = np.full(len(z["link_ids"]), -1); rank[order] = np.arange(len(order))
        mu_o = np.load(ROOT / "work" / f"dlsim_caps_{panel}.npy")
        i = C.T2_PANELS.index(panel)
        cap = np.asarray(C.Ctx(panel, i).cap, np.float32)
        mu = np.full(len(cap), np.nan); mu[order] = mu_o
        for s, D in data[panel].items():
            on = D["on"]
            F_fit, F_rel, mass = [], [], []
            for n, cand, reg, ms in D["meta"]:
                for r, j in enumerate(cand):
                    rk = rank[j]
                    upz = order[rk - 1] if rk > 0 else j
                    dnz = order[rk + 1] if 0 <= rk < len(order) - 1 else j
                    m_fit = np.nanmin([mu[j], mu[dnz]])
                    m_rel = min(cap[j], cap[dnz])
                    F_fit.append(feats(on["v"][n], on["qf"][n], on["ok"][n], j, upz, m_fit, cap[j]))
                    F_rel.append(feats(on["v"][n], on["qf"][n], on["ok"][n], j, upz, m_rel, cap[j]))
                    mass.append(ms[r])
            F_fit, F_rel = np.array(F_fit, float), np.array(F_rel, float)
            for k, nm in enumerate(names):
                sign = -1 if nm == "first" else 1
                for tag, Fm in (("fitted mu", F_fit), ("released cap", F_rel)):
                    a, npair = pair_auc(sign * Fm[:, k], D["U"], D["g"])
                    res.setdefault((tag, nm), []).append((a, npair))
            res.setdefault(("production", "mass"), []).append(pair_auc(np.array(mass), D["U"], D["g"]))
            res.setdefault(("placebo", "random"), []).append(pair_auc(rng.random(len(mass)), D["U"], D["g"]))
    print("\npairwise AUC (best candidate vs the others), pair-weighted over panels x splits:")
    for key, v in res.items():
        a = np.array([x[0] for x in v]); w = np.array([x[1] for x in v])
        print(f"  {key[0]:<13} {key[1]:<6} AUC {np.sum(a * w) / max(w.sum(), 1):.3f}  pairs {w.sum()}")


if __name__ == "__main__":
    main()
