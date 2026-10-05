"""Decide the ongoing set as CONTIGUOUS SEGMENTS, not by a flat threshold.

Note 121 put a +0.0762 oracle on the ongoing decision layer - keep the posterior's per-cell order, change only
how many cells each window takes - and a regression for the budget failed. But note 31 already named the reason
the analytic rule fails there: **ongoing cells are strongly correlated in space and time**, so the
independent-Bernoulli expected-IoU calculus is wrong. A budget estimator that ignores the correlation was
attacking the symptom.

The correlation has a known shape. A queue is a contiguous run of links, and it persists across the horizon. So
put that in the decision instead of in the budget: for each window and horizon step, choose the subset of the
scored links maximising

    sum_s x_s (logit p_s - theta)  -  lambda * (number of runs of ones)

which is a two-state dynamic program over the corridor. theta is the threshold; lambda is what it costs to open
a new queue. lambda = 0 reduces exactly to a flat threshold at sigmoid(theta), so the flat rule is inside the
family and the sweep can only be read as an improvement over it if it beats the lambda = 0 column.

`--persist` adds the temporal half: a link already chosen at the previous horizon step is charged a reduced
opening cost, which is the same prior along time.

Tuned on the OOF tail (`cache_oof.py`, earlier than every scored origin, the surface the corrector was fitted
on) and read on both scored months.

    python occ_lab/segment_decision.py --tag s20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta"))
sys.path.insert(0, str(LAB.parent / "analysis"))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

import common as C  # noqa: E402
from online_adapter import ETA, GATE, global_features, load_split, mean_of  # noqa: E402
from residual_diag import TAU  # noqa: E402

CACHE = LAB / "cache"
W_PS = 0.60


def lg(p):
    q = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q))


def segment(a, keep, lam, open_cost=None):
    """Two-state DP over the last axis: maximise sum(x*a) - lam * (runs). `a` is (B, S), `keep` (B, S) bool.

    `open_cost` (B, S), when given, replaces lam for the cost of STARTING a run at that link - the temporal
    prior passes a smaller cost where the previous horizon step was already on."""
    B, S = a.shape
    oc = np.full((B, S), lam, float) if open_cost is None else open_cost
    av = np.where(keep, a, 0.0)
    NEG = -1e18
    f0 = np.zeros(B)
    f1 = np.where(keep[:, 0], av[:, 0] - oc[:, 0], NEG)
    b0 = np.zeros((B, S), bool)                               # came from state 1 at s-1
    b1 = np.zeros((B, S), bool)                               # stayed on
    for s in range(1, S):
        n0 = np.maximum(f0, f1)
        b0[:, s] = f1 > f0
        stay = f1
        start = f0 - oc[:, s]
        b1[:, s] = stay > start
        n1 = np.where(keep[:, s], np.maximum(stay, start) + av[:, s], NEG)
        f0, f1 = n0, n1
    x = np.zeros((B, S), bool)
    st = f1 > f0
    for s in range(S - 1, -1, -1):
        x[:, s] = st
        st = np.where(st, b1[:, s], b0[:, s]) if s else st
    return x & keep


def decide(p, e, theta, lam, persist=0.0):
    """Segment every (window, horizon) line; with persist, walk the horizon and discount continuations."""
    N, Fh, S = p.shape
    a = lg(p) - theta
    out = np.zeros_like(e)
    prev = np.zeros((N, S), bool)
    for h in range(Fh):
        oc = np.full((N, S), lam, float)
        if persist:
            oc = np.where(prev, lam * (1.0 - persist), lam)
        x = segment(a[:, h], e[:, h], lam, oc)
        out[:, h] = x
        prev = x
    return out


def oof_panel(panel, gbm):
    f, fs, fp = (CACHE / f"og_oof_{panel}.npz", CACHE / f"stid_oof_{panel}.npz",
                 CACHE / f"pseudo_oof_{panel}.npz")
    if not (f.exists() and fs.exists() and fp.exists()):
        return None
    z, zp = np.load(f), np.load(fp)
    P = {"gs1": z["gs1"].astype(np.float32), "tfm8": z["tfm8"].astype(np.float32),
         "cnn3": z["cnn3"].astype(np.float32), "occ_cnn": mean_of(z, "occ_cnn"),
         "occ_tfm": mean_of(z, "occ_tfm"), "stid": mean_of(np.load(fs), "stid")}
    old = 0.2 * P["gs1"] + 0.6 * P["tfm8"] + 0.2 * P["cnn3"]
    base = ((1 - W_PS) * old + W_PS * zp["pseudo"].astype(np.float32)).astype(np.float32)
    F = global_features(P, base)
    rh = gbm.predict(F.reshape(-1, F.shape[-1])).reshape(base.shape).astype(np.float32)
    return np.clip(base + ETA * np.where(np.abs(rh) >= GATE, rh, 0.0), 0, 1), z["y"], z["e"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="s20")
    ap.add_argument("--thetas", type=float, nargs="+", default=[-0.4, -0.2, 0.0, 0.2])
    ap.add_argument("--lams", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--persists", type=float, nargs="+", default=[0.0])
    ap.add_argument("--skip-oof", action="store_true")
    args = ap.parse_args()
    gbm = lgb.Booster(model_file=str(LAB / "models" / "residual_lgb.txt"))
    grid = [(t, l, q) for t in args.thetas for l in args.lams for q in args.persists]

    if not args.skip_oof:
        acc = {g: [] for g in grid}
        acc["flat"] = []
        for panel in C.T2_PANELS:
            r = oof_panel(panel, gbm)
            if r is None:
                continue
            pg, y, e = r
            acc["flat"].append(C.iou(pg >= TAU, y, e).mean())
            for g in grid:
                acc[g].append(C.iou(decide(pg, e, *g), y, e).mean())
            print(f"  OOF {panel}", flush=True)
        base = float(np.mean(acc["flat"]))
        print(f"\nOOF tail, ongoing IoU (panel mean); the flat threshold is {base:.4f}\n")
        print(f"{'theta':>7}{'lambda':>8}{'persist':>9}{'IoU':>9}{'vs flat':>9}")
        for g in sorted(grid, key=lambda g: -float(np.mean(acc[g])))[:14]:
            v = float(np.mean(acc[g]))
            print(f"{g[0]:>7.2f}{g[1]:>8.2f}{g[2]:>9.2f}{v:>9.4f}{v - base:>+9.4f}")
        best = max(grid, key=lambda g: float(np.mean(acc[g])))
        print(f"\nOOF picks theta {best[0]:g}, lambda {best[1]:g}, persist {best[2]:g}", flush=True)
        grid = [best] + [g for g in grid if g != best]

    fam, fam_e = {}, {}
    for i, panel in enumerate(C.T2_PANELS):
        sp = load_split(i, panel, args.tag)
        for s, D in sp.items():
            base_p, y, e, dd = D["base"], D["y"], D["e"], D["d"]
            F = global_features(D["P"], base_p)
            rh = gbm.predict(F.reshape(-1, F.shape[-1])).reshape(base_p.shape).astype(np.float32)
            pg = np.clip(base_p + ETA * np.where(np.abs(rh) >= GATE, rh, 0.0), 0, 1)
            days = np.unique(dd)
            early = np.isin(dd, days[1:5])
            v0 = C.iou(pg >= TAU, y, e)
            fam.setdefault((s, "AS"), {})[panel] = v0.mean()
            fam_e.setdefault((s, "AS"), {})[panel] = v0[early].mean()
            for g in grid:
                v = C.iou(decide(pg, e, *g), y, e)
                nm = f"th {g[0]:g} lam {g[1]:g} ps {g[2]:g}"
                fam.setdefault((s, nm), {})[panel] = v.mean()
                fam_e.setdefault((s, nm), {})[panel] = v[early].mean()
        print(f"  {panel}", flush=True)

    names = ["AS"] + [f"th {g[0]:g} lam {g[1]:g} ps {g[2]:g}" for g in grid]
    for tag, F_ in (("whole month", fam), ("at most 4 earlier days", fam_e)):
        print(f"\nongoing family IoU, segment decision, {tag}\n")
        print(f"{'setting':<22}" + "".join(f"{s:>13}" for s in C.SPLITS) + f"{'mean':>10}{'vs AS':>9}")
        g0 = None
        for nm in names:
            v = [C.family_mean(F_[(s, nm)]) for s in C.SPLITS]
            m = float(np.mean(v))
            if g0 is None:
                g0 = m
            print(f"{nm:<22}" + "".join(f"{x:>13.4f}" for x in v) + f"{m:>10.4f}{m - g0:>+9.4f}")


if __name__ == "__main__":
    main()
