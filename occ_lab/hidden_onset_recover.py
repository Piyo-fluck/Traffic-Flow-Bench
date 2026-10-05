"""Idea (1): recover the official onset that the blank hides, from what the release shows AFTER the blank.

An official onset window on day d hides its own horizon and the next hour (18 slots). For the NEXT day that event
is in the past and the released layer around it is usable - the queue is visible again 65 min after the breakdown.
A bottleneck queue keeps its DOWNSTREAM front at the bottleneck while the tail grows upstream, so the head of the
post-blank queue points at the hidden bottleneck.

Simulated on the training months (truth known): every first-onset event (first_onsets on the observations, one per
day), blank slots t0+1 .. t0+F+12, first visible slot a = t0+F+13 (and the next 3 slots). Candidate bottlenecks =
the downstream ends of the observed queue runs at a..a+3 (runs merged across 1-link gaps). Methods:

    M2  H only        the pre-T prediction (out-of-fold H, occ_lab/h_oof.py): arg-max at T+30 - no bridge
    M0  longest run   the head of the longest post-blank run
    M1  head x H      heads ranked by H's max within +-2 links of the head
    M3  learned       LightGBM on candidate heads (run length, earliest visible slot, head speed, H near the head,
                      distance to H's arg-max, time), leave-one-month-out
Hit = the candidate lies within +-2 links of a true queued cell at T+30. Top-1 and top-3.

    .venv/Scripts/python.exe occ_lab/hidden_onset_recover.py
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

import common as C  # noqa: E402

F = C.F
CACHE = LAB / "cache"
UP_HIGHER = {"D7_I10_E"}                  # note 159: upstream = higher index here, lower elsewhere


def runs(mask, gap=1):
    j = np.flatnonzero(mask)
    if not len(j):
        return []
    cut = np.flatnonzero(np.diff(j) > gap + 1)
    return [(a[0], a[-1]) for a in np.split(j, cut + 1)]


def cand_rows(pH, qd, vr, d, t0, down_is_high):
    """candidate heads and their feature rows for one event (None if no post-blank queue)."""
    a = t0 + F + 13
    if a + 3 >= qd.shape[1]:
        return None
    post = qd[d, a:a + 4]
    cands = {}
    for k in range(4):
        for lo, hi in runs(post[k]):
            head = hi if down_is_high else lo
            c = cands.setdefault(head, dict(L=0, first=k))
            c["L"] = max(c["L"], hi - lo + 1); c["first"] = min(c["first"], k)
    if not cands:
        return None
    heads = np.array(sorted(cands))
    Ls = np.array([cands[h]["L"] for h in heads]); firsts = np.array([cands[h]["first"] for h in heads])
    hnear = np.array([pH[max(0, h - 2):h + 3].max() for h in heads])
    jH = int(np.argmax(pH))
    rows = []
    for h, L, fk, hn in zip(heads, Ls, firsts, hnear):
        with np.errstate(all="ignore"):
            vh = np.nanmean(vr[d, a:a + 4, h])
            w = vr[d, t0 - 3:t0, max(0, h - 2):h + 3]
            vpre = np.nanmin(w) if np.isfinite(w).any() else np.nan
        rows.append([L, fk, vh, hn, np.log(hn + 1e-6), abs(h - jH), hn / max(pH.max(), 1e-9), vpre, t0 / 12.0,
                     len(heads), L / max(Ls.max(), 1)])
    return heads, np.asarray(rows, np.float32)


def fit_ranker(before="2031-01"):
    """the M3 ranker fitted on the training-month events of months < before (out of fold H)."""
    from tfb.data import load
    X, y, g = [], [], []
    for i, panel in enumerate(C.T2_PANELS):
        z = load(panel, "train")
        vf = np.asarray(C.Ctx(panel, i).vf, np.float32)
        Hc = np.load(CACHE / f"h_oof_{panel}.npz")
        v, ok = z["obs_speed"], z["elig"] & np.isfinite(z["obs_speed"])
        qd = ok & (np.nan_to_num(v, nan=1e9) <= 0.6 * vf)
        vr = np.where(ok, v / vf, np.nan)
        for n in range(len(Hc["d"])):
            if str(Hc["month"][n]) >= before or not np.isfinite(Hc["prob"][n]).all():
                continue
            true = Hc["fut"][n, -1] & Hc["efut"][n, -1]
            if not true.any():
                continue
            nt = true.copy()
            for r in (1, 2):
                nt[r:] |= true[:-r]; nt[:-r] |= true[r:]
            cr = cand_rows(Hc["prob"][n], qd, vr, int(Hc["d"][n]), int(Hc["t0"][n]), panel not in UP_HIGHER)
            if cr is None:
                continue
            heads, Xr = cr
            X.append(Xr); y.append(nt[heads].astype(int)); g.append(len(heads))
    par = {"objective": "lambdarank", "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 30,
           "verbose": -1, "num_threads": 8}
    return lgb.train(par, lgb.Dataset(np.concatenate(X), np.concatenate(y), group=np.asarray(g)), 200)


def recover(model, pH, qd, vr, d, t0, down_is_high, width=1):
    """(S,) bool pseudo T+30 label: the recovered head +-width, or None when nothing is visible after the blank."""
    cr = cand_rows(pH, qd, vr, d, t0, down_is_high)
    if cr is None:
        return None
    heads, X = cr
    h = int(heads[int(np.argmax(model.predict(X)))])
    lab = np.zeros(len(pH), bool)
    lab[max(0, h - width):h + width + 1] = True
    return lab


def scored_check(model):
    """the scored months' lab onset events: production blend arg-max vs the bridge (ranker fitted on train)."""
    import bottleneck_cls as BC
    res = {}
    for i, panel in enumerate(C.T2_PANELS):
        ctx, stat, W = C.windows(i, panel)
        vf = np.asarray(ctx.vf, np.float32)
        B = BC.build(i, panel)
        for s in C.SPLITS:
            if s not in B:
                continue
            on, z = W[s]["on"], W[s]["z"]
            v, ok = z["obs_speed"], z["elig"] & np.isfinite(z["obs_speed"])
            qd = ok & (np.nan_to_num(v, nan=1e9) <= 0.6 * vf)
            vr = np.where(ok, v / vf, np.nan)
            for key, P in (("blend", B[s]["base"]), ("H", W[s]["ref"]["on_H"])):
                for n in range(len(on["d"])):
                    true = on["fut"][n, -1] & on["efut"][n, -1]
                    if not true.any():
                        continue
                    nt = true.copy()
                    for r in (1, 2):
                        nt[r:] |= true[:-r]; nt[:-r] |= true[r:]
                    base_hit = bool(nt[int(np.argmax(P[n]))])
                    cr = cand_rows(P[n], qd, vr, int(on["d"][n]), int(on["t0"][n]), panel not in UP_HIGHER)
                    if cr is None:
                        br = base_hit; has = False
                    else:
                        heads, X = cr
                        br = bool(nt[heads[int(np.argmax(model.predict(X)))]]); has = True
                    r_ = res.setdefault((s, key), [0, 0, 0, 0, 0])
                    r_[0] += 1; r_[1] += base_hit; r_[2] += br; r_[3] += has
                    r_[4] += int(has and br and not base_hit) - int(has and base_hit and not br)
    print()
    print("scored months, lab onset events (hit = within +-2 links of a true T+30 queued cell):")
    for (s, key), r in sorted(res.items()):
        print(f"  {s:<10} {key:<6} events {r[0]}  arg-max {r[1] / r[0]:.3f}  bridge {r[2] / r[0]:.3f}  "
              f"(post-blank queue in {r[3] / r[0]:.2f}; net rescued {r[4]:+d})")


def main() -> None:
    warnings.simplefilter("ignore")
    from tfb.data import load
    rows, labels, groups, meta, base = [], [], [], [], {"M2": [], "M0": [], "M1": []}
    for i, panel in enumerate(C.T2_PANELS):
        z = load(panel, "train")
        ctx = C.Ctx(panel, i)
        vf = np.asarray(ctx.vf, np.float32)
        Hc = np.load(CACHE / f"h_oof_{panel}.npz")
        mon = Hc["month"]
        v, ok = z["obs_speed"], z["elig"] & np.isfinite(z["obs_speed"])
        qd = ok & (np.nan_to_num(v, nan=1e9) <= 0.6 * vf)
        vr = np.where(ok, v / vf, np.nan)
        S = len(vf)
        down_is_high = panel not in UP_HIGHER             # downstream = higher index when upstream = lower
        for n in range(len(Hc["d"])):
            pH = Hc["prob"][n]
            if not np.isfinite(pH).all():
                continue
            d, t0 = int(Hc["d"][n]), int(Hc["t0"][n])
            true = Hc["fut"][n, -1] & Hc["efut"][n, -1]
            if not true.any():
                continue
            near_true = true.copy()
            for r in (1, 2):
                near_true[r:] |= true[:-r]; near_true[:-r] |= true[r:]
            a = t0 + F + 13
            if a + 3 >= qd.shape[1]:
                continue
            post = qd[d, a:a + 4]                          # (4, S)
            cands = {}
            for k in range(4):
                for lo, hi in runs(post[k]):
                    head = hi if down_is_high else lo
                    L = hi - lo + 1
                    c = cands.setdefault(head, dict(L=0, first=k))
                    c["L"] = max(c["L"], L); c["first"] = min(c["first"], k)
            jH = int(np.argmax(pH))
            base["M2"].append(bool(near_true[jH]))
            if not cands:
                base["M0"].append(False); base["M1"].append(False)
                continue
            heads = np.array(sorted(cands))
            Ls = np.array([cands[h]["L"] for h in heads]); firsts = np.array([cands[h]["first"] for h in heads])
            hnear = np.array([pH[max(0, h - 2):h + 3].max() for h in heads])
            base["M0"].append(bool(near_true[heads[np.argmax(Ls)]]))
            base["M1"].append(bool(near_true[heads[np.argmax(hnear)]]))
            for h, L, fk, hn in zip(heads, Ls, firsts, hnear):
                with np.errstate(all="ignore"):
                    vh = np.nanmean(vr[d, a:a + 4, h])
                    vpre = np.nanmin(vr[d, t0 - 3:t0, max(0, h - 2):h + 3]) if np.isfinite(vr[d, t0 - 3:t0, max(0, h - 2):h + 3]).any() else np.nan
                rows.append([L, fk, vh, hn, np.log(hn + 1e-6), abs(h - jH), hn / max(pH.max(), 1e-9), vpre, t0 / 12.0,
                             len(heads), L / max(Ls.max(), 1)])
                labels.append(int(near_true[h]))
            groups.append(len(heads)); meta.append((panel, mon[n]))
        print(f"  {panel}: events {len(base['M2'])}", flush=True)
    X, y, g = np.asarray(rows, np.float32), np.asarray(labels), np.asarray(groups)
    offs = np.r_[0, np.cumsum(g)[:-1]]
    months = np.array([m for _, m in meta])
    print(f"\nevents {len(base['M2'])}, with a post-blank queue {len(g)} "
          f"({len(g) / len(base['M2']):.2f}); candidate heads per event {g.mean():.1f}")
    print(f"top-1 over all events:  M2 H only {np.mean(base['M2']):.3f}  M0 longest run {np.mean(base['M0']):.3f}"
          f"  M1 head x H {np.mean(base['M1']):.3f}")
    # M3 learned, leave-one-month-out; top-1 / top-3 on the events that have candidates
    par = {"objective": "lambdarank", "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 30,
           "verbose": -1, "num_threads": 8}
    hit1 = hit3 = any3 = h1m1 = 0
    for m in np.unique(months):
        tr, te = np.flatnonzero(months != m), np.flatnonzero(months == m)
        idx_tr = np.concatenate([np.arange(offs[k], offs[k] + g[k]) for k in tr])
        mdl = lgb.train(par, lgb.Dataset(X[idx_tr], y[idx_tr], group=g[tr]), 200)
        for k in te:
            sl = slice(offs[k], offs[k] + g[k])
            sc = mdl.predict(X[sl]); yy = y[sl]
            order = np.argsort(-sc)
            hit1 += yy[order[0]]; hit3 += yy[order[:3]].max(); any3 += yy.max()
            h1m1 += yy[np.argmax(X[sl, 3])]
    ne = len(g)
    print(f"events with candidates ({ne}): M1 head x H top-1 {h1m1 / ne:.3f} | M3 learned top-1 {hit1 / ne:.3f}, "
          f"top-3 {hit3 / ne:.3f} | some candidate correct {any3 / ne:.3f}")
    full = lgb.train(par, lgb.Dataset(X, y, group=g), 200)
    scored_check(full)


if __name__ == "__main__":
    main()
