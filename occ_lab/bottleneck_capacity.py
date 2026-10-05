"""Onset idea 2: learned discharge-capacity headroom per candidate bottleneck.

A breakdown happens where the arriving demand reaches the flow the link actually breaks down at - not the nominal
cap (ctx.cap) but a site-specific value that the same run reveals. All from data released at or before T (slot T
itself excluded):

    run capacity        per link, from earlier days and today's slots < T:
                          q95_free   95th percentile of flow in free flow (v >= 0.7 vf)
                          q_brk      median of the highest flow in the 3 slots before an observed breakdown
                                     (v >= 0.7 vf for 3 slots, then v <= 0.6 vf), and the number of such breakdowns
    demand now          today's flow in the last 3 slots (max over +-2 links), its 30-min slope
    demand at T+30      demand now x the run's flow ramp at this time of day (median over earlier days of
                        q(T+30) / q(T) at the link, free-flow slots only) - demand is forecast, capacity is learned
    headroom            demand now / q95_free, demand now / q_brk, forecast / q95_free, forecast / q_brk
    capacity step       q95_free at the candidate / the smaller of its two sides' q95_free (a drop in capacity
                        along the road is where the queue forms; orientation-free, both sides)
    in-window rank      each headroom minus the window's best candidate (the competition is between candidates)

All ratios are run-invariant (note 128: link identities do not survive the run shift). Added to note 129's
candidate classifier, cross-month, over LightGBM seeds 0-5 (note 144: single-seed differences are noise).

    .venv/Scripts/python.exe occ_lab/bottleneck_capacity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta"))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

import bottleneck_cls as BC  # noqa: E402
import common as C  # noqa: E402

F = C.F
SEEDS = range(6)
NB = 2


def day_arrays(z, vf):
    v, q = z["obs_speed"], z["obs_flow"]
    ok = z["elig"] & np.isfinite(v) & np.isfinite(q)
    vv = np.where(ok, v / vf, np.nan)
    qq = np.where(ok, q, np.nan)
    free = ok & (vv >= 0.7)
    # breakdown at slot t: free for the 3 previous slots, queued at t
    brk = np.zeros_like(ok)
    pre = np.full(ok.shape, np.nan, np.float32)
    f3 = free[:, :-3] & free[:, 1:-2] & free[:, 2:-1]
    cong = ok[:, 3:] & (vv[:, 3:] <= 0.6)
    brk[:, 3:] = f3 & cong
    with np.errstate(all="ignore"):
        pre[:, 3:] = np.nanmax(np.stack([qq[:, :-3], qq[:, 1:-2], qq[:, 2:-1]]), 0)
    return dict(qq=qq, free=free, brk=brk, pre=pre)


def nbmax(a):
    S = len(a)
    out = np.full(S, np.nan, np.float32)
    for j in range(S):
        w = a[max(0, j - NB):j + NB + 1]
        if np.isfinite(w).any():
            out[j] = np.nanmax(w)
    return out


def cap_rows(A, d, t0, qh, okh, cand):
    """Capacity rows of ONE window: A = day_arrays of the released month, d / t0 its day and origin, qh / okh its
    (H, S) flow history (slots t0-H .. t0-1), cand its candidate peaks."""
    qq, free, brk, pre = A["qq"], A["free"], A["brk"], A["pre"]
    S = qq.shape[2]
    # released flows: earlier days whole, today strictly before slot T
    qf_rel = np.concatenate([qq[:d].reshape(-1, S), qq[d, :t0]])
    fr_rel = np.concatenate([free[:d].reshape(-1, S), free[d, :t0]])
    bk_rel = np.concatenate([brk[:d].reshape(-1, S), brk[d, :t0]])
    pr_rel = np.concatenate([pre[:d].reshape(-1, S), pre[d, :t0]])
    hq = np.where(okh, qh, np.nan)
    with np.errstate(all="ignore"):
        q95 = np.nanpercentile(np.where(fr_rel, qf_rel, np.nan), 95, axis=0)
        nbrk = bk_rel.sum(0).astype(np.float32)
        qb = np.nanmedian(np.where(bk_rel, pr_rel, np.nan), axis=0)
        now = np.nanmean(hq[-3:], 0)
        before = np.nanmean(hq[-9:-6], 0)
        # the run's flow ramp T -> T+30 at this time of day (earlier days, free at both ends)
        if d:
            t1 = min(t0 + F - 1, qq.shape[1] - 1)
            a0, a1 = qq[:d, t0 - 1], qq[:d, t1]
            okr = free[:d, t0 - 1] & free[:d, t1] & (a0 > 0)
            ramp = np.nanmedian(np.where(okr, a1 / a0, np.nan), 0)
        else:
            ramp = np.full(S, np.nan)
    q95n, qbn, nown = nbmax(q95), nbmax(qb), nbmax(now)
    slope = nbmax(now - before)
    fore = nown * np.where(np.isfinite(ramp), nbmax(ramp), 1.0)
    with np.errstate(all="ignore"):
        h = np.stack([nown / q95n, nown / qbn, fore / q95n, fore / qbn], 1)       # (S, 4)
        left = np.array([np.nanmax(q95[max(0, j - 4):max(0, j - 1)]) if j > 1 else np.nan for j in range(S)])
        right = np.array([np.nanmax(q95[j + 2:j + 5]) if j + 2 < S else np.nan for j in range(S)])
        step = q95 / np.fmin(left, right)
        hc = h[cand]
        rel = hc - np.nanmax(hc, 0, keepdims=True)
    rows = []
    for r, j in enumerate(cand):
        rows.append([*hc[r], *rel[r], step[j], slope[j] / max(q95n[j], 1e-3) if np.isfinite(q95n[j]) else np.nan,
                     np.nansum(nbrk[max(0, j - NB):j + NB + 1]), ramp[j]])
    return np.asarray(rows, np.float32)


def capacity_features(D, A):
    on = D["on"]
    return np.concatenate([cap_rows(A, int(on["d"][n]), int(on["t0"][n]), on["qf"][n], on["ok"][n], cand)
                           for n, cand, reg, mass in D["meta"]])


def main() -> None:
    data = {}
    for i, panel in enumerate(C.T2_PANELS):
        B = BC.build(i, panel)
        ctx, stat, W = C.windows(i, panel)
        vf = np.asarray(ctx.vf, np.float32)
        for s, D in B.items():
            D["Xc"] = capacity_features(D, day_arrays(W[s]["z"], vf))
            assert len(D["Xc"]) == len(D["X"])
        data[panel] = B
        print(f"  {panel}", flush=True)
    # does headroom alone rank the true bottleneck? (within-window, candidates with a label)
    names = ["now/q95", "now/q_brk", "fore/q95", "fore/q_brk"]
    for c, nm in enumerate(names):
        hit = tot = 0
        for d in data.values():
            for D in d.values():
                off = 0
                for g in D["g"]:
                    x, yy = D["Xc"][off:off + g, c], D["y"][off:off + g]
                    off += g
                    if yy.any() and np.isfinite(x).sum() >= 2:
                        tot += 1; hit += int(np.nanargmax(x) == np.argmax(yy))
        print(f"  {nm:<11} alone: top-1 {hit / max(tot, 1):.3f} (n {tot}; production mass top-1 ~0.80)")
    par = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 30,
           "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "num_threads": 8}
    FEATS = ("note 129 features", "+ capacity headroom")
    fam, acc, cnt = {}, {}, {}
    for sd in SEEDS:
        for feats in FEATS:
            def X_of(D):
                return D["X"] if feats == FEATS[0] else np.concatenate([D["X"], D["Xc"]], 1)
            for s_fit, s_app in (("validation", "private"), ("private", "validation")):
                X = np.concatenate([X_of(d[s_fit]) for d in data.values() if s_fit in d])
                y = np.concatenate([d[s_fit]["y"] for d in data.values() if s_fit in d])
                m = lgb.train({**par, "seed": sd}, lgb.Dataset(X, y), 200)
                hit = tot = 0
                for panel, d in data.items():
                    if s_app not in d:
                        continue
                    D = d[s_app]
                    prob = m.predict(X_of(D))
                    on = D["on"]
                    b_sc = C.onset_iou(D["base"], on)
                    fam.setdefault((s_app, "production", 0), {})[panel] = b_sc.mean()
                    P2 = D["base"].copy()
                    off = 0
                    for (n, cand, reg, mass), gsize in zip(D["meta"], D["g"]):
                        pc, yy = prob[off:off + gsize], D["y"][off:off + gsize]
                        off += gsize
                        if yy.any():
                            tot += 1; hit += int(np.argmax(pc) == np.argmax(yy))
                        P2[n] = BC.swap(D["base"][n], cand, reg, mass, pc)
                    sc = C.onset_iou(P2, on)
                    fam.setdefault((s_app, feats, sd), {})[panel] = sc.mean()
                    c = cnt.setdefault((feats, sd), [0, 0])
                    c[0] += int(((b_sc == 0) & (sc > 0)).sum()); c[1] += int(((b_sc > 0) & (sc == 0)).sum())
                acc[(s_app, feats, sd)] = hit / max(tot, 1)
        print(f"  seed {sd}: " + "  ".join(
            f"{f[:6]} " + " ".join(f"{C.family_mean(fam[(s, f, sd)]) - C.family_mean(fam[(s, 'production', 0)]):+.4f}"
                                   for s in C.SPLITS) for f in FEATS), flush=True)
    ref = {s: C.family_mean(fam[(s, "production", 0)]) for s in C.SPLITS}
    print(f"\nmean over {len(SEEDS)} seeds (swap onset IoU vs production; top-1; rescue/spoil)")
    for f in FEATS:
        dv = [np.mean([C.family_mean(fam[(s, f, sd)]) - ref[s] for sd in SEEDS]) for s in C.SPLITS]
        sdv = [np.std([C.family_mean(fam[(s, f, sd)]) - ref[s] for sd in SEEDS]) for s in C.SPLITS]
        a = [np.mean([acc[(s, f, sd)] for sd in SEEDS]) for s in C.SPLITS]
        rs = np.mean([cnt[(f, sd)][0] for sd in SEEDS]); sp = np.mean([cnt[(f, sd)][1] for sd in SEEDS])
        print(f"  {f:<22} val {dv[0]:+.4f}±{sdv[0]:.4f}  priv {dv[1]:+.4f}±{sdv[1]:.4f}  mean {np.mean(dv):+.4f}"
              f"   top-1 {a[0]:.3f} / {a[1]:.3f}   {rs:.1f} / {sp:.1f}")
    # which capacity features the model uses (last seed, full fit on both months)
    X = np.concatenate([np.concatenate([D["X"], D["Xc"]], 1) for d in data.values() for D in d.values()])
    y = np.concatenate([D["y"] for d in data.values() for D in d.values()])
    m = lgb.train(par, lgb.Dataset(X, y), 200)
    gain = m.feature_importance("gain"); gain = gain / gain.sum()
    cn = names + [f"rel {x}" for x in names] + ["cap step", "slope/q95", "n breakdowns", "ramp"]
    print("  capacity features' gain share: " + ", ".join(f"{a} {g:.3f}" for a, g in zip(cn, gain[-len(cn):])))
    print(f"  total {gain[-len(cn):].sum():.3f}")


if __name__ == "__main__":
    main()


# ------------------------------------------------------------------------------------------------ training-month prior
def prior_stats(panel, vf, last_days=None):
    """The same capacity quantities from the TRAINING months (real sensors, same road): per link q95_free, q_brk,
    breakdowns per day, and per slot the flow ramp T -> T+30. Official onset windows sit on days 1-5, where the
    scored month has too few earlier days to learn capacity from; the training months hold ~270 days of the same
    sites."""
    from tfb.data import load
    zt = load(panel, "train")
    z = {k: zt[k][-last_days:] if last_days else zt[k] for k in ("obs_speed", "obs_flow", "elig")}
    A = day_arrays(z, vf)
    qq, free, brk, pre = A["qq"], A["free"], A["brk"], A["pre"]
    S = qq.shape[2]
    with np.errstate(all="ignore"):
        q95 = np.nanpercentile(np.where(free, qq, np.nan).reshape(-1, S), 95, axis=0)
        qb = np.nanmedian(np.where(brk, pre, np.nan).reshape(-1, S), axis=0)
        rate = brk.sum((0, 1)) / qq.shape[0]
        a0, a1 = qq[:, :-(F - 1) - 1], qq[:, F:]                         # q(t-1), q(t+F-1) for t = 1 .. 288-F
        okr = free[:, :-(F - 1) - 1] & free[:, F:] & (a0 > 0)
        R = np.full((qq.shape[1], S), np.nan, np.float32)
        R[1:1 + a0.shape[1]] = np.nanmedian(np.where(okr, a1 / a0, np.nan), 0)
    return dict(q95=q95, qb=qb, rate=rate, R=R)


def prior_rows(Pr, t0, qh, okh, cand):
    """Headroom of ONE window against the training-month capacity (today's demand from its own history)."""
    hq = np.where(okh, qh, np.nan)
    with np.errstate(all="ignore"):
        now = np.nanmean(hq[-3:], 0)
        before = np.nanmean(hq[-9:-6], 0)
    q95n, qbn, nown = nbmax(Pr["q95"]), nbmax(Pr["qb"]), nbmax(now)
    ramp = nbmax(Pr["R"][min(t0, len(Pr["R"]) - 1)])
    fore = nown * np.where(np.isfinite(ramp), ramp, 1.0)
    S = len(now)
    q95 = Pr["q95"]
    with np.errstate(all="ignore"):
        h = np.stack([nown / q95n, nown / qbn, fore / q95n, fore / qbn], 1)
        left = np.array([np.nanmax(q95[max(0, j - 4):max(0, j - 1)]) if j > 1 else np.nan for j in range(S)])
        right = np.array([np.nanmax(q95[j + 2:j + 5]) if j + 2 < S else np.nan for j in range(S)])
        step = q95 / np.fmin(left, right)
        hc = h[cand]
        rel = hc - np.nanmax(hc, 0, keepdims=True)
        slope = nbmax(now - before)
    rows = []
    for r, j in enumerate(cand):
        rows.append([*hc[r], *rel[r], step[j], slope[j] / max(q95n[j], 1e-3) if np.isfinite(q95n[j]) else np.nan,
                     np.nansum(Pr["rate"][max(0, j - NB):j + NB + 1]), ramp[j]])
    return np.asarray(rows, np.float32)


def prior_features(D, Pr):
    on = D["on"]
    return np.concatenate([prior_rows(Pr, int(on["t0"][n]), on["qf"][n], on["ok"][n], cand)
                           for n, cand, reg, mass in D["meta"]])


def capacity_features_trunc(D, A, K=4):
    """capacity_features with only the last K earlier days of the month released (the official windows sit on
    days 1-5: at most 4 earlier days) - the lab then sees what an official window sees."""
    on = D["on"]
    out = []
    for n, cand, reg, mass in D["meta"]:
        d, t0 = int(on["d"][n]), int(on["t0"][n])
        lo = max(0, d - K)
        As = {k: v[lo:d + 1] for k, v in A.items()}
        out.append(cap_rows(As, d - lo, t0, on["qf"][n], on["ok"][n], cand))
    return np.concatenate(out)
