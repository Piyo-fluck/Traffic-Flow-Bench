"""Onset idea 4: candidate-specific breakdown-time likelihood.

An onset window asks WHICH site breaks down first at T+30. The sites differ in when they break down: one fails at
07:10 on most weekdays, another at 16:30. Per candidate peak j of a window (time T+30, weekday/weekend):

    first-onset likelihood   over the day's first-onset events (first_onsets: the panel's first queue of the day),
                             kernel-weighted by |t_e - (T+30)| (sigma 15 / 30 / 60 min): the share whose queued
                             cells touch j +-2 links; also weekday/weekend-matched (sigma 30), and unweighted
    site hazard              over days on which j (+-2) was free at T-5 min: the share breaking down at j within
                             +-15 min of T+30 (breakdown = 3 free slots, then v <= 0.6 vf)
    gap                      each minus the window's best candidate

Sources: the TRAINING months (~270 days, same sensors; site behaviour transfers, note 145 corr 0.82) and the scored
month's last <= 4 earlier days (what an official window sees on days 1-5). Evaluated with note 129's classifier and
note 145's prior4 capacity, cross-month, LightGBM seeds 0-5.

    .venv/Scripts/python.exe occ_lab/bottleneck_timing.py
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

import bottleneck_capacity as BK  # noqa: E402
import bottleneck_cls as BC  # noqa: E402
import common as C  # noqa: E402

F = C.F
NB = 2
SIG = (3, 6, 12)
K_DAYS = 4


def near(cells):
    """(E, S) queued cells -> (E, S) 'touches link j +-NB'."""
    out = cells.copy()
    for r in range(1, NB + 1):
        out[:, r:] |= cells[:, :-r]; out[:, :-r] |= cells[:, r:]
    return out


def events_of(z, vf, blank=None):
    """The day's first-onset events: day, time of the first queue (slot), weekend, near-cells (E, S)."""
    on = C.first_onsets(vf, z, blank=blank)
    t = on["t0"] + F
    return dict(d=on["d"], t=t, we=z["weekday"][on["d"]] >= 5, nc=near(on["cells"]))


def site_arrays(z, vf):
    A = BK.day_arrays(z, vf)
    S = A["qq"].shape[2]
    free, brk = A["free"], A["brk"]
    fr, bk = free.copy(), brk.copy()
    for r in range(1, NB + 1):                                       # +-NB links
        fr[..., r:] |= free[..., :-r]; fr[..., :-r] |= free[..., r:]
        bk[..., r:] |= brk[..., :-r]; bk[..., :-r] |= brk[..., r:]
    cb = np.concatenate([np.zeros((bk.shape[0], 1, S), np.int32), np.cumsum(bk, 1, dtype=np.int32)], 1)
    return dict(fr=fr, cb=cb)


def likelihood(E, T30, we, mask=None):
    """(S, len(SIG) + 2): time-kernel shares, weekend-matched sigma 6, unweighted."""
    S = E["nc"].shape[1]
    m = np.ones(len(E["t"]), bool) if mask is None else mask
    if not m.any():
        return np.full((S, len(SIG) + 2), np.nan, np.float32)
    nc, t, w = E["nc"][m].astype(np.float32), E["t"][m], E["we"][m]
    out = []
    for s in SIG:
        k = np.exp(-0.5 * ((t - T30) / s) ** 2)
        out.append(k @ nc / max(k.sum(), 1e-6) if k.sum() > 1e-3 else np.full(S, np.nan))
    k = np.exp(-0.5 * ((t - T30) / 6) ** 2) * (w == we)
    out.append(k @ nc / k.sum() if k.sum() > 1e-3 else np.full(S, np.nan))
    out.append(nc.mean(0))
    return np.stack(out, 1).astype(np.float32)


def hazard(SA, T30, days):
    """(S,): among the given days with the site free at T-1, the share breaking down within T30 +- 3 slots."""
    fr, cb = SA["fr"][days], SA["cb"][days]
    t_free = max(T30 - F - 1, 0)
    lo, hi = max(T30 - 3, 0), min(T30 + 4, cb.shape[1] - 1)
    ok = fr[:, t_free]
    hit = (cb[:, hi] - cb[:, lo]) > 0
    n = ok.sum(0)
    with np.errstate(all="ignore"):
        return np.where(n > 0, (hit & ok).sum(0) / n, np.nan).astype(np.float32)


def timing_rows(prior, month, T30, we, d, cand):
    """prior = (events, site arrays) of the training months; month = (events, site arrays) of the scored month."""
    Et, St = prior
    Em, Sm = month
    lo = max(0, d - K_DAYS)
    L_t = likelihood(Et, T30, we)
    H_t = hazard(St, T30, np.arange(St["fr"].shape[0]))
    L_m = likelihood(Em, T30, we, (Em["d"] >= lo) & (Em["d"] < d))
    H_m = hazard(Sm, T30, np.arange(lo, d)) if d > lo else np.full(L_t.shape[0], np.nan, np.float32)
    X = np.concatenate([L_t, H_t[:, None], L_m, H_m[:, None]], 1)[cand]          # (K, 12)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gap = X - np.nanmax(X, 0, keepdims=True)
    return np.concatenate([X, gap], 1)


def prior_trust(prior, month, d):
    """How well the training-month prior (time-kernel sigma 12 and unweighted) has named this month's first
    bottleneck on the released earlier days (last <= K_DAYS): hit rate of its arg-max (+-NB links) and the mean
    likelihood share of the true site. nan on the first day."""
    Et, _ = prior
    Em, _ = month
    lo = max(0, d - K_DAYS)
    idx = np.flatnonzero((Em["d"] >= lo) & (Em["d"] < d))
    if not len(idx):
        return np.full(4, np.nan, np.float32)
    h12 = h0 = s12 = s0 = 0.0
    for e in idx:
        L = likelihood(Et, int(Em["t"][e]), bool(Em["we"][e]))
        tru = Em["nc"][e]
        if not tru.any():
            continue
        for c, ss in ((2, "12"), (4, "0")):
            x = L[:, c]
            if not np.isfinite(x).any():
                continue
            hit = float(tru[int(np.nanargmax(x))])
            sh = float(np.nanmax(np.where(tru, x, np.nan)) / max(np.nanmax(x), 1e-6))
            if ss == "12":
                h12 += hit; s12 += sh
            else:
                h0 += hit; s0 += sh
    n = len(idx)
    return np.array([h12 / n, s12 / n, h0 / n, s0 / n], np.float32)


def timing_features(D, prior, month, z):
    on = D["on"]
    out = []
    for n, cand, reg, mass in D["meta"]:
        d = int(on["d"][n])
        r = timing_rows(prior, month, int(on["t0"][n]) + F, bool(z["weekday"][d] >= 5), d, cand)
        tr = prior_trust(prior, month, d)
        # trust columns and trust x train likelihood (sigma 12 gap, unweighted gap)
        out.append(np.concatenate([r, np.repeat(tr[None], len(cand), 0),
                                   (r[:, [14, 16]] * tr[[0, 2]][None])], 1))
    return np.concatenate(out)


def prior_of(panel, vf):
    from tfb.data import load
    zt = load(panel, "train")
    zt = {k: zt[k] for k in ("obs_speed", "obs_flow", "elig", "weekday")}
    return events_of(zt, vf), site_arrays(zt, vf)


def main() -> None:
    data = {}
    for i, panel in enumerate(C.T2_PANELS):
        B = BC.build(i, panel)
        ctx, stat, W = C.windows(i, panel)
        vf = np.asarray(ctx.vf, np.float32)
        prior = prior_of(panel, vf)
        Pr = BK.prior_stats(panel, vf)
        for s, D in B.items():
            z = W[s]["z"]
            # the month's own first-onset events: exactly the released lab pool of this split
            on = D["on"]
            month = (dict(d=on["d"], t=on["t0"] + F, we=z["weekday"][on["d"]] >= 5,
                          nc=near(on["fut"][:, -1] & on["efut"][:, -1])), site_arrays(z, vf))
            D["Xt"] = timing_features(D, prior, month, z)
            A = BK.day_arrays(z, vf)
            D["Xp"] = np.concatenate([BK.prior_features(D, Pr), BK.capacity_features_trunc(D, A)], 1)
        data[panel] = B
        print(f"  {panel}: {len(prior[0]['t'])} training first-onset events", flush=True)
    # alone: does the likelihood rank the true bottleneck?
    names = [f"L_train s{s}" for s in SIG] + ["L_train we", "L_train all", "H_train"] + \
            [f"L_month s{s}" for s in SIG] + ["L_month we", "L_month all", "H_month"]
    for c, nm in enumerate(names):
        hit = tot = 0
        for d in data.values():
            for D in d.values():
                off = 0
                for g in D["g"]:
                    x, yy = D["Xt"][off:off + g, c], D["y"][off:off + g]
                    off += g
                    if yy.any() and np.isfinite(x).sum() >= 2 and np.nanmax(x) > np.nanmin(x):
                        tot += 1; hit += int(np.nanargmax(x) == np.argmax(yy))
        print(f"  {nm:<14} alone: top-1 {hit / max(tot, 1):.3f} (n {tot})")
    par = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 30,
           "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "num_threads": 8}
    ref = {s: C.family_mean({p: C.onset_iou(d[s]["base"], d[s]["on"]).mean() for p, d in data.items() if s in d})
           for s in C.SPLITS}
    COLS = {"Xt_tr": [2, 4, 14, 16], "Xt_mo": [6, 7, 18, 19], "Xt_all1": [4, 16],     # compact subsets of Xt
            "Xt_trust": [2, 4, 14, 16, 24, 25, 26, 27, 28, 29], "Xt_mo_trust": [2, 4, 6, 7, 14, 16, 18, 19, 24, 25, 26, 27, 28, 29]}
    for sp_ in C.SPLITS:
        hit = tot = 0
        for d in data.values():
            if sp_ not in d:
                continue
            D = d[sp_]; off = 0
            for g in D["g"]:
                x, yy = D["Xt"][off:off + g, 4], D["y"][off:off + g]; off += g
                if yy.any() and np.nanmax(x) > np.nanmin(x):
                    tot += 1; hit += int(np.nanargmax(x) == np.argmax(yy))
        print(f"  L_train all alone, {sp_}: top-1 {hit / max(tot, 1):.3f} (n {tot})")
    for d in data.values():
        for D in d.values():
            for k, c in COLS.items():
                D[k] = D["Xt"][:, c]
            # hard gate: the training prior is shown only where it named >= half of this month's released first
            # bottlenecks (last <= 4 days); otherwise (and on day 0) it is missing
            g_ = ~(D["Xt"][:, 24] >= 0.5)
            Xg = D["Xt"][:, [2, 4, 14, 16]].copy(); Xg[g_] = np.nan
            D["Xt_gate"] = Xg
            D["Xt_gate_mo"] = np.concatenate([Xg, D["Xt"][:, [6, 7, 18, 19]]], 1)
    VAR = {"note 129": [], "+ prior4 capacity (145)": ["Xp"], "+ train L (4)": ["Xt_tr"], "+ month L (4)": ["Xt_mo"],
           "+ train L + trust": ["Xt_tr" if False else "Xt_trust"], "+ train/month L + trust": ["Xt_mo_trust"],
           "+ prior4 + train L + trust": ["Xp", "Xt_trust"], "+ prior4 + month L (4)": ["Xp", "Xt_mo"],
           "+ prior4 + train/month L + trust": ["Xp", "Xt_mo_trust"],
           "+ prior4 + gated train L": ["Xp", "Xt_gate"], "+ prior4 + gated train L + month L": ["Xp", "Xt_gate_mo"]}
    for nm, keys in VAR.items():
        res = {s: [] for s in C.SPLITS}
        rs = sp = 0
        for sd in range(6):
            for s_fit, s_app in (("validation", "private"), ("private", "validation")):
                def Xo(D):
                    return np.concatenate([D["X"]] + [D[k] for k in keys], 1)
                X = np.concatenate([Xo(d[s_fit]) for d in data.values() if s_fit in d])
                y = np.concatenate([d[s_fit]["y"] for d in data.values() if s_fit in d])
                m = lgb.train({**par, "seed": sd}, lgb.Dataset(X, y), 200)
                fam = {}
                for p, d in data.items():
                    if s_app not in d:
                        continue
                    D = d[s_app]
                    prob = m.predict(Xo(D))
                    P2 = D["base"].copy()
                    off = 0
                    for (n, cand, reg, mass), g in zip(D["meta"], D["g"]):
                        P2[n] = BC.swap(D["base"][n], cand, reg, mass, prob[off:off + g]); off += g
                    sc, b = C.onset_iou(P2, D["on"]), C.onset_iou(D["base"], D["on"])
                    fam[p] = sc.mean()
                    rs += int(((b == 0) & (sc > 0)).sum()); sp += int(((b > 0) & (sc == 0)).sum())
                res[s_app].append(C.family_mean(fam) - ref[s_app])
        v, pr = np.mean(res["validation"]), np.mean(res["private"])
        print(f"{nm:<34} val {v:+.4f}±{np.std(res['validation']):.4f}  priv {pr:+.4f}±{np.std(res['private']):.4f}"
              f"  mean {(v + pr) / 2:+.4f}   rescue/spoil {rs / 6:.1f}/{sp / 6:.1f}", flush=True)


if __name__ == "__main__":
    main()
