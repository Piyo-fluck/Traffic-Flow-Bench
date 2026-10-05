"""Three cheap diagnostics of the ongoing / onset residual structure (user-supplied proposal, 2026-09-26).

1  temporal events: on official-like ongoing truth, the share of eligible (window, link) sequences over the six
   horizon steps with at most one switch; and what projecting AT's p onto the best <=1-switch sequence per cell
   (maximum likelihood under p, i.e. a first-passage / survival shape) does to the family IoU
2  spatial bias: per day, each link's mean residual (y - p) - AT's ongoing p over the horizon, and H's onset p at
   T+30 - correlated with the next available day's, against a link-shuffled control
3  low-rank run shift: per day, the mean residual matrix R_d (6 x links); the energy share of its top 1/2/4 singular
   components, and whether R_d's rank-k projection predicts R_{d+1} better than zero (explained share of R_{d+1}'s
   energy), against the same with R_d link-shuffled

    .venv/Scripts/python.exe occ_lab/structure_diag.py
"""
from __future__ import annotations

import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta"))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

import common as C  # noqa: E402
from online_adapter import ETA, GATE, global_features, load_split  # noqa: E402

TAG = "ssl_w192ssl_avg4"          # AT's p_PS
rng = np.random.default_rng(0)


def best_one_switch(p, e):
    """(F,) probabilities -> (F,) bool: the <=1-switch sequence (constant, 0..01..1, 1..10..0) of maximum
    log-likelihood under independent Bernoulli(p) over the eligible steps."""
    q = np.clip(p, 1e-5, 1 - 1e-5)
    l1, l0 = np.where(e, np.log(q), 0), np.where(e, np.log(1 - q), 0)
    F = len(p)
    best, arg = -1e18, None
    for k in range(F + 1):                      # first k steps one value, the rest the other
        for a, b in ((l0, l1), (l1, l0)):
            s = a[:k].sum() + b[k:].sum()
            if s > best:
                best, arg = s, (k, a is l1)
    k, first_is_one = arg
    out = np.zeros(F, bool)
    out[:k] = first_is_one
    out[k:] = not first_is_one
    return out


def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 5 and a[m].std() > 0 and b[m].std() > 0 else np.nan


def main() -> None:
    gbm = lgb.Booster(model_file=str(LAB / "models" / "residual_lgb.txt"))
    sw, fam = [], {}
    c_og, c_og_sh, c_on, c_on_sh = [], [], [], []
    ek = {1: [], 2: [], 4: []}
    pr = {k: [] for k in (1, 2, 4)}
    pr_sh = {k: [] for k in (1, 2, 4)}
    for i, panel in enumerate(C.T2_PANELS):
        ctx, stat, W = C.windows(i, panel)
        sp = load_split(i, panel, TAG)
        for s, D in sp.items():
            blend = D["base"]
            Fe = global_features(D["P"], blend)
            rh = gbm.predict(Fe.reshape(-1, Fe.shape[-1])).reshape(blend.shape).astype(np.float32)
            p = np.clip(blend + ETA * np.where(np.abs(rh) >= GATE, rh, 0.0), 0, 1)
            y, e, d = D["y"], D["e"], D["d"]
            # 1 - switches on truth, cells eligible at all six steps
            full = e.all(1)                                             # (N, S)
            yy = y.transpose(0, 2, 1)[full]                              # (cells, F)
            sw.append((yy[:, 1:] != yy[:, :-1]).sum(1))
            proj = np.zeros_like(y)
            for n in range(len(p)):
                for j in range(p.shape[2]):
                    proj[n, :, j] = best_one_switch(p[n, :, j], e[n, :, j])
            for nm, q in (("AT (tau 0.50)", p >= 0.5), ("AT projected to <=1 switch", proj)):
                fam.setdefault((s, nm), {})[panel] = C.iou(q, y, e).mean()
            # 2/3 - per-day residual structure (ongoing)
            days = np.unique(d)
            R = {}
            for dd in days:
                m = d == dd
                r = np.where(e[m], y[m].astype(np.float32) - p[m], np.nan)
                with np.errstate(invalid="ignore"):
                    R[int(dd)] = np.nanmean(r, 0)                       # (F, S)
            for a, b in zip(days[:-1], days[1:]):
                Ra, Rb = R[int(a)], R[int(b)]
                va, vb = np.nanmean(Ra, 0), np.nanmean(Rb, 0)
                c_og.append(corr(va, vb)); c_og_sh.append(corr(rng.permutation(va), vb))
                A, B = np.nan_to_num(Ra), np.nan_to_num(Rb)
                u, sv, vt = np.linalg.svd(A, full_matrices=False)
                tot = (sv ** 2).sum()
                Ash = A[:, rng.permutation(A.shape[1])]
                us, ss, vts = np.linalg.svd(Ash, full_matrices=False)
                eb = (B ** 2).sum()
                for k in (1, 2, 4):
                    ek[k].append((sv[:k] ** 2).sum() / max(tot, 1e-12))
                    Ak = (u[:, :k] * sv[:k]) @ vt[:k]
                    Ak_s = (us[:, :k] * ss[:k]) @ vts[:k]
                    pr[k].append(1 - ((B - Ak) ** 2).sum() / max(eb, 1e-12))
                    pr_sh[k].append(1 - ((B - Ak_s) ** 2).sum() / max(eb, 1e-12))
            # 2 - onset spatial bias at T+30 with H's probability
            on, ref = W[s]["on"], W[s]["ref"]
            if len(on["d"]) > 1:
                ydays = np.unique(on["d"])
                rows = {}
                for dd in ydays:
                    m = on["d"] == dd
                    yo = on["fut"][m][:, -1].astype(np.float32)
                    eo = on["efut"][m][:, -1]
                    with np.errstate(invalid="ignore"):
                        rows[int(dd)] = np.nanmean(np.where(eo, yo - ref["on_H"][m], np.nan), 0)
                for a, b in zip(ydays[:-1], ydays[1:]):
                    c_on.append(corr(rows[int(a)], rows[int(b)]))
                    c_on_sh.append(corr(rng.permutation(rows[int(a)]), rows[int(b)]))
        print(f"  {panel}", flush=True)
    sw = np.concatenate(sw)
    print(f"\n1. truth, ongoing cells eligible at all six steps: {len(sw):,}")
    for k in range(4):
        print(f"   switches = {k}: {np.mean(sw == k):.4f}")
    print(f"   P(switches <= 1) = {np.mean(sw <= 1):.4f}")
    print("   family IoU:")
    for nm in ("AT (tau 0.50)", "AT projected to <=1 switch"):
        v = [C.family_mean(fam[(s, nm)]) for s in C.SPLITS]
        print(f"     {nm:<28} " + "  ".join(f"{s} {x:.4f}" for s, x in zip(C.SPLITS, v)) + f"  mean {np.mean(v):.4f}")
    f = lambda a: f"{np.nanmean(a):+.4f} (n={np.isfinite(a).sum()})"  # noqa: E731
    print(f"\n2. next-day correlation of per-link mean residuals")
    print(f"   ongoing (AT p):   {f(np.array(c_og))}   link-shuffled control {f(np.array(c_og_sh))}")
    print(f"   onset (H, T+30):  {f(np.array(c_on))}   link-shuffled control {f(np.array(c_on_sh))}")
    print(f"\n3. per-day mean residual matrix (6 x links), ongoing")
    for k in (1, 2, 4):
        print(f"   rank {k}: energy share {np.mean(ek[k]):.3f}   explains next day's energy "
              f"{np.mean(pr[k]):+.4f} (shuffled {np.mean(pr_sh[k]):+.4f})")


if __name__ == "__main__":
    main()
