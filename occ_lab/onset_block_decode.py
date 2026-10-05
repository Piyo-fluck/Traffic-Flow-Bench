"""Onset idea 3: shape-aware decoding of the T+30 onset set.

The ratio rule picks the top-k links by probability, the expected IoU computed as if the links were independent. A
real onset queue is a block around its bottleneck: the links of one region queue together or not at all. With the
marginals p kept exactly, the dependence is put back as a Gaussian copula per candidate region (note 129's peaks /
regions): latent_i = rho Z_r + sqrt(1 - rho^2) e_i, Y_i = Phi(latent_i) <= p_i; regions independent. rho = 0 is the
independent model (~ the ratio rule), rho = 1 the comonotone one (a region's queue is a level set of p). The decision
maximises the Monte-Carlo expected IoU over per-region level sets {i in r: p_i >= theta_r} (or nothing), by
coordinate ascent from the ratio rule's set; option 'hull' fills a level set's gaps (contiguous block).

rho is chosen on one scored month and applied to the other. Evaluated on the production probabilities and on note
145's prior4-swapped ones (the current onset candidate).

    .venv/Scripts/python.exe occ_lab/onset_block_decode.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta"))
sys.path.insert(0, str(LAB.parent))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
from scipy.special import ndtr  # noqa: E402

import bottleneck_capacity as BK  # noqa: E402
import bottleneck_cls as BC  # noqa: E402
import common as C  # noqa: E402

F = C.F
M = 3000
RHOS = (0.0, 0.5, 0.8, 0.9, 0.95, 1.0)
PMIN = 0.003


def blocks(mask):
    """number of contiguous runs and their lengths."""
    m = np.concatenate([[0], mask.astype(np.int8), [0]])
    st, en = np.flatnonzero(np.diff(m) == 1), np.flatnonzero(np.diff(m) == -1)
    return len(st), en - st


def decode(p, rho, hull=False, seed=0):
    S = len(p)
    ratio = C.select(p, "ratio", 1.0)
    idx = np.flatnonzero((p >= PMIN) | ratio)
    if len(idx) < 2:
        return ratio
    cand = BC.peaks(p)
    if len(cand) == 0:
        return ratio
    reg_all = np.abs(np.arange(S)[:, None] - cand[None, :]).argmin(1)
    reg = reg_all[idx]
    pi = p[idx]
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((M, len(cand)))
    E = rng.standard_normal((M, len(idx)))
    lat = rho * Z[:, reg] + np.sqrt(max(1 - rho ** 2, 0.0)) * E
    Y = ndtr(lat) <= pi[None]                                           # (M, n)
    Yf = Y.astype(np.float32)
    ysum = Yf.sum(1)

    def eiou(A):
        a = A.astype(np.float32)
        inter = Yf @ a
        union = a.sum() + ysum - inter
        return float(np.mean(np.where(union > 0, inter / np.maximum(union, 1), 0.0)))

    # per-region options: nothing, or a level set (optionally its hull)
    opts = []
    for r in range(len(cand)):
        mem = np.flatnonzero(reg == r)
        o = [np.zeros(len(idx), bool)]
        if len(mem):
            lv = np.unique(pi[mem])[::-1][:20]
            for th in lv:
                A = np.zeros(len(idx), bool)
                sel = mem[pi[mem] >= th]
                if hull:
                    lo, hi = idx[sel].min(), idx[sel].max()
                    sel = mem[(idx[mem] >= lo) & (idx[mem] <= hi)]
                A[sel] = True
                o.append(A)
        opts.append(o)
    # start: the ratio rule's set restricted to idx, split into the regions' parts
    cur = [np.zeros(len(idx), bool) for _ in cand]
    r0 = ratio[idx]
    for r in range(len(cand)):
        cur[r] = r0 & (reg == r)
    best = eiou(np.any(cur, 0))
    for _ in range(3):
        improved = False
        for r in range(len(cand)):
            for A in opts[r]:
                trial = cur.copy()
                trial[r] = A
                v = eiou(np.any(trial, 0))
                if v > best + 1e-9:
                    best, cur, improved = v, trial, True
        if not improved:
            break
    out = np.zeros(S, bool)
    out[idx[np.any(cur, 0)]] = True
    if not out.any():
        return ratio
    return out


def iou_of(pred_last, on):
    pr = np.zeros(on["fut"].shape, bool)
    pr[:, F - 1] = pred_last
    return C.iou(pr, on["fut"], on["efut"])


def main() -> None:
    warnings.simplefilter("ignore")
    data = {}
    for i, panel in enumerate(C.T2_PANELS):
        B = BC.build(i, panel)
        ctx, stat, W = C.windows(i, panel)
        vf = np.asarray(ctx.vf, np.float32)
        Pr = BK.prior_stats(panel, vf)
        for s, D in B.items():
            A = BK.day_arrays(W[s]["z"], vf)
            D["Xp"] = np.concatenate([D["X"], BK.prior_features(D, Pr), BK.capacity_features_trunc(D, A)], 1)
        data[panel] = B
        print(f"  {panel}", flush=True)
    # the current candidate's probabilities: prior4 swap, cross-month (seed 0)
    par = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 30,
           "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "num_threads": 8,
           "seed": 0}
    for s_fit, s_app in (("validation", "private"), ("private", "validation")):
        X = np.concatenate([d[s_fit]["Xp"] for d in data.values() if s_fit in d])
        y = np.concatenate([d[s_fit]["y"] for d in data.values() if s_fit in d])
        m = lgb.train(par, lgb.Dataset(X, y), 200)
        for d in data.values():
            if s_app not in d:
                continue
            D = d[s_app]
            prob = m.predict(D["Xp"])
            P2 = D["base"].copy()
            off = 0
            for (n, cand, reg, mass), g in zip(D["meta"], D["g"]):
                P2[n] = BC.swap(D["base"][n], cand, reg, mass, prob[off:off + g]); off += g
            D["swapped"] = P2
    # shape EDA: the truth vs the ratio rule's set
    nb_t, ln_t, nb_p, sz_p, sz_t = [], [], [], [], []
    for d in data.values():
        for D in d.values():
            on = D["on"]
            for n in range(len(on["d"])):
                t = on["fut"][n, -1] & on["efut"][n, -1]
                k, L = blocks(t); nb_t.append(k); ln_t += list(L); sz_t.append(t.sum())
                pr = C.select(D["swapped"][n], "ratio", 1.0)
                k2, _ = blocks(pr); nb_p.append(k2); sz_p.append(pr.sum())
    nb_t, nb_p = np.array(nb_t), np.array(nb_p)
    print(f"truth at T+30: blocks per window mean {nb_t.mean():.2f} (1 block {np.mean(nb_t == 1):.2f}), "
          f"block length median {np.median(ln_t):.0f} (p90 {np.percentile(ln_t, 90):.0f}), cells mean {np.mean(sz_t):.1f}")
    print(f"ratio rule:    blocks per window mean {nb_p.mean():.2f} (1 block {np.mean(nb_p == 1):.2f}), "
          f"cells mean {np.mean(sz_p):.1f}")
    # decoding
    res = {}
    for src in ("base", "swapped"):
        for hull in (False, True):
            for rho in RHOS:
                for s in C.SPLITS:
                    fam = {}
                    for p_, d in data.items():
                        if s not in d:
                            continue
                        D = d[s]
                        pred = np.stack([decode(q, rho, hull) for q in D[src]])
                        fam[p_] = iou_of(pred, D["on"]).mean()
                    res[(src, hull, rho, s)] = C.family_mean(fam)
            ref = {s: C.family_mean({p_: C.onset_iou(d[s][src], d[s]["on"]).mean() for p_, d in data.items() if s in d})
                   for s in C.SPLITS}
            print(f"\n[{src}{' + hull' if hull else ''}]  ratio rule: val {ref['validation']:.4f}  priv {ref['private']:.4f}")
            for rho in RHOS:
                v, pr = (res[(src, hull, rho, s)] - ref[s] for s in C.SPLITS)
                print(f"  rho {rho:<5} val {v:+.4f}  priv {pr:+.4f}  mean {(v + pr) / 2:+.4f}", flush=True)
            # cross-month choice of rho
            pick = {s: max(RHOS, key=lambda r: res[(src, hull, r, s)]) for s in C.SPLITS}
            v = res[(src, hull, pick["private"], "validation")] - ref["validation"]
            pr = res[(src, hull, pick["validation"], "private")] - ref["private"]
            print(f"  cross-month rho (val <- {pick['private']}, priv <- {pick['validation']}): val {v:+.4f}  "
                  f"priv {pr:+.4f}  mean {(v + pr) / 2:+.4f}")


if __name__ == "__main__":
    main()
