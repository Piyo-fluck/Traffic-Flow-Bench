"""Condition the ongoing posterior on the fact that the window was SELECTED as an official ongoing window.

The organizers do not take an arbitrary ongoing window. They take one whose true future is not already solved by
holding the last observed queue constant:

    S = 1  iff  IoU(Y, Y_persist) <= 0.9      (`maximum_persistence_iou`, notes 24-25)

with Y_persist the last observed queued state per link, broadcast over the horizon - computable at test time
from the released history alone, and the scored cell set is the sample submission's own rows, so both sides of
the statistic are known. The window being in the file is therefore EVIDENCE ABOUT ITS LABEL, and the quantity to
report is not P(Y | X) but

    P(Y | X, S = 1)  proportional to  P(Y | X) 1[IoU(Y, Y_persist) <= 0.9].

Our pipeline reports P(Y | X). Under an independent-Bernoulli posterior the decision is the per-cell MAP, and
the constrained MAP is reached by flipping the CHEAPEST cells - the ones closest to the threshold - until the
constraint holds. Only two kinds of flip lower the statistic: switching on a cell the persistence answer leaves
off (extension), or switching off a cell it holds on (retraction). Each flip costs |logit p - logit tau|, so the
greedy order is exactly "flip the most uncertain cells first", and it stops as soon as S = 1.

This is NOT note 118's class-conditional threshold, which moved every cell of a class and bought +0.0003. The
constraint binds only on the windows that violate it, and leaves the rest untouched.

Reported: how often our answer violates the constraint at all (if it never does, there is nothing here), what
those windows score, and the ongoing family IoU after the projection at several caps.

    python occ_lab/selector_posterior.py --tag s20
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
from online_adapter import ETA, GATE, global_features, load_split  # noqa: E402
from residual_diag import TAU  # noqa: E402


def persist_of(og, vf):
    """Y_persist: the last OBSERVED queued state of each link in the history, held over the horizon."""
    vq = og["ok"] & (np.nan_to_num(og["v"], nan=1e9) <= 0.6 * vf)
    idx = np.where(og["ok"][:, ::-1], np.arange(og["ok"].shape[1])[::-1][None, :, None], -1).max(1)
    N, S = idx.shape
    last = np.zeros((N, S), bool)
    m = idx >= 0
    jj = np.broadcast_to(np.arange(N)[:, None], idx.shape)
    ss = np.broadcast_to(np.arange(S), idx.shape)
    last[m] = vq[jj[m], idx[m], ss[m]]
    return np.broadcast_to(last[:, None, :], og["fut"].shape)


def stat(a, b, e):
    """IoU of two binary tensors inside the scored cells, per window."""
    p, q = a & e, b & e
    ax = tuple(range(1, a.ndim))
    u = (p | q).sum(ax)
    return np.where(u == 0, 1.0, (p & q).sum(ax) / np.maximum(u, 1))


def topk_oracle(p, y, e):
    """Keep the posterior's ORDER and take the true number of queued cells: the budget oracle."""
    out = np.zeros_like(e)
    for j in range(len(p)):
        idx = np.flatnonzero(e[j].ravel())
        k = int((y[j] & e[j]).sum())
        if k == 0 or not len(idx):
            continue
        o = idx[np.argsort(-p[j].ravel()[idx])][:k]
        f = out[j].ravel()
        f[o] = True
        out[j] = f.reshape(out[j].shape)
    return out


def tau_oracle(p, y, e):
    """The best per-window threshold: the ceiling of ANY rule that only moves tau window by window."""
    out = np.zeros_like(e)
    for j in range(len(p)):
        idx = np.flatnonzero(e[j].ravel())
        if not len(idx):
            continue
        pv, yv = p[j].ravel()[idx], (y[j] & e[j]).ravel()[idx]
        o = np.argsort(-pv)
        hit = np.cumsum(yv[o])
        tot = int(yv.sum())
        k = np.arange(1, len(o) + 1)
        iou = hit / np.maximum(k + tot - hit, 1)
        kb = int(np.argmax(iou)) + 1 if tot else 0
        f = out[j].ravel()
        f[idx[o[:kb]]] = True
        out[j] = f.reshape(out[j].shape)
    return out


def project(p, yhat, per, e, cap, allow="both", ref=None, only=None):
    """The constrained MAP: flip the cells closest to the threshold until IoU(Y, Y_persist) <= cap.

    `allow` restricts the move set to extension or retraction, which separates the two ways of leaving the
    persistence answer. `ref` is the tensor the constraint is measured against - the control passes a
    window-permuted persistence, so the same budget of flips is spent for a reason that carries no selection
    information."""
    out = yhat.copy()
    con = per if ref is None else ref
    s = stat(out, con, e)
    capv = np.full(len(s), cap, float) if np.isscalar(cap) else np.asarray(cap, float)
    bad = np.flatnonzero((s > capv) if only is None else ((s > capv) & only))
    flips = 0
    for j in bad:
        y, pe, ee, pp, cn = out[j], per[j], e[j], p[j], con[j]
        add = ee & ~pe & ~y                                   # extension: off in both, persistence says no
        rem = ee & pe & y                                     # retraction: on in both
        if allow == "add":
            rem = np.zeros_like(rem)
        elif allow == "rem":
            add = np.zeros_like(add)
        cost = np.where(add, TAU - pp, np.where(rem, pp - TAU, np.inf))
        order = np.argsort(cost, axis=None)
        k = 0
        while stat(y[None], cn[None], ee[None])[0] > capv[j] and k < order.size:
            c = np.unravel_index(order[k], y.shape)
            k += 1
            if not np.isfinite(cost[c]):
                break
            y[c] = ~y[c]
            flips += 1
        out[j] = y
    return out, len(bad), flips


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="s20")
    ap.add_argument("--caps", type=float, nargs="+", default=[0.90, 0.85, 0.80])
    ap.add_argument("--allows", nargs="+", default=["both", "add", "rem"])
    ap.add_argument("--pushes", type=float, nargs="*", default=[],
                    help="second stage: the windows that VIOLATED 0.9 are pushed all the way to this value "
                         "instead of stopping at the boundary. The truth's own mean is about 0.65, so the "
                         "conditional mass is nowhere near the constraint surface")
    ap.add_argument("--tilts", type=float, nargs="*", default=[],
                    help="the marginal-level version: logit p + lam (1 - 2 Y_persist), then the threshold. "
                         "Note 118 measured its class-conditional twin at +0.0003")
    ap.add_argument("--oracle", action="store_true",
                    help="the ceiling of the whole persistence-distance axis: project each window to its TRUE "
                         "IoU(Y, Y_persist) instead of to the 0.9 constraint. It uses the label, so it bounds "
                         "the axis rather than scoring on it")
    ap.add_argument("--control", action="store_true",
                    help="also project against a window-permuted persistence: the same budget of flips, spent "
                         "for a reason that carries no selection information")
    ap.add_argument("--early", type=int, default=4)
    args = ap.parse_args()
    gbm = lgb.Booster(model_file=str(LAB / "models" / "residual_lgb.txt"))
    fam, fam_e, diag = {}, {}, []
    for i, panel in enumerate(C.T2_PANELS):
        ctx, stat_, Wd = C.windows(i, panel)
        sp = load_split(i, panel, args.tag)
        for s, D in sp.items():
            og = Wd[s]["og"]
            base, y, e, dd = D["base"], D["y"], D["e"], D["d"]
            per = persist_of(og, ctx.vf)
            F = global_features(D["P"], base)
            rh = gbm.predict(F.reshape(-1, F.shape[-1])).reshape(base.shape).astype(np.float32)
            pg = np.clip(base + ETA * np.where(np.abs(rh) >= GATE, rh, 0.0), 0, 1)
            yhat = pg >= TAU
            days = np.unique(dd)
            early = np.isin(dd, days[1:args.early + 1])
            s_true, s_pred = stat(y, per, e), stat(yhat, per, e)
            v0 = C.iou(yhat, y, e)
            bad = s_pred > 0.9
            ax = (1, 2)
            _ = s_true
            diag.append((panel, s, len(y), float((~(y & e).any(ax)).mean()),
                         float((~(yhat & e).any(ax)).mean()),
                         float(s_pred.mean()), float(bad.mean()),
                         float(v0[bad].mean()) if bad.any() else np.nan,
                         float(v0[~bad].mean()) if (~bad).any() else np.nan))
            fam.setdefault((s, "AS"), {})[panel] = v0.mean()
            fam_e.setdefault((s, "AS"), {})[panel] = v0[early].mean()
            for cap in args.caps:
                for al in args.allows:
                    yp, nb, nf = project(pg, yhat, per, e, cap, al)
                    v = C.iou(yp, y, e)
                    fam.setdefault((s, f"cap {cap:g} {al}"), {})[panel] = v.mean()
                    fam_e.setdefault((s, f"cap {cap:g} {al}"), {})[panel] = v[early].mean()
            for pu in args.pushes:
                yp, _, _ = project(pg, yhat, per, e, pu, "both", None, bad)
                v = C.iou(yp, y, e)
                fam.setdefault((s, f"push {pu:g}"), {})[panel] = v.mean()
                fam_e.setdefault((s, f"push {pu:g}"), {})[panel] = v[early].mean()
            for la in args.tilts:
                pc = np.clip(pg, 1e-6, 1 - 1e-6)
                pt = 1 / (1 + np.exp(-(np.log(pc / (1 - pc)) + la * (1 - 2 * per.astype(np.float32)))))
                v = C.iou(pt >= TAU, y, e)
                fam.setdefault((s, f"tilt {la:g}"), {})[panel] = v.mean()
                fam_e.setdefault((s, f"tilt {la:g}"), {})[panel] = v[early].mean()
            if args.oracle:
                for nm, yp in (("oracle |Y| top-k", topk_oracle(pg, y, e)),
                               ("oracle per-window tau", tau_oracle(pg, y, e))):
                    v = C.iou(yp, y, e)
                    fam.setdefault((s, nm), {})[panel] = v.mean()
                    fam_e.setdefault((s, nm), {})[panel] = v[early].mean()
                yp, _, _ = project(pg, yhat, per, e, s_true)
                v = C.iou(yp, y, e)
                fam.setdefault((s, "oracle cap"), {})[panel] = v.mean()
                fam_e.setdefault((s, "oracle cap"), {})[panel] = v[early].mean()
            if args.control:
                rr = np.random.default_rng(5).permutation(len(y))
                yp, nb, nf = project(pg, yhat, per, e, args.caps[0], "both", per[rr])
                v = C.iou(yp, y, e)
                fam.setdefault((s, "control"), {})[panel] = v.mean()
                fam_e.setdefault((s, "control"), {})[panel] = v[early].mean()
        ctx.z = ctx.q = ctx.el = None
        print(f"  {panel}", flush=True)

    print(f"\n{'panel':<11}{'split':<11}{'win':>6}{'Y empty':>9}{'Yhat empty':>12}"
          f"{'IoU(Yhat,per)':>15}{'>0.9':>8}{'score if >0.9':>15}{'score else':>12}")
    for r in diag:
        print(f"{r[0]:<11}{r[1]:<11}{r[2]:>6}{r[3]:>9.3f}{r[4]:>12.3f}{r[5]:>15.4f}{r[6]:>8.3f}"
              f"{r[7]:>15.4f}{r[8]:>12.4f}")

    names = ["AS"] + [f"cap {c:g} {a}" for c in args.caps for a in args.allows] \
        + [f"push {c:g}" for c in args.pushes] + [f"tilt {c:g}" for c in args.tilts] \
        + (["oracle cap", "oracle |Y| top-k", "oracle per-window tau"] if args.oracle else []) + (["control"] if args.control else [])
    for tag, F_ in (("whole month", fam), (f"at most {args.early} earlier days", fam_e)):
        print(f"\nongoing family IoU after the threshold, {tag}\n")
        print(f"{'setting':<18}" + "".join(f"{s:>13}" for s in C.SPLITS) + f"{'mean':>10}{'vs AS':>9}")
        g0 = None
        for nm in names:
            if (C.SPLITS[0], nm) not in F_:
                continue
            v = [C.family_mean(F_[(s, nm)]) for s in C.SPLITS]
            m = float(np.mean(v))
            if g0 is None:
                g0 = m
            print(f"{nm:<18}" + "".join(f"{x:>13.4f}" for x in v) + f"{m:>10.4f}{m - g0:>+9.4f}")


if __name__ == "__main__":
    main()
