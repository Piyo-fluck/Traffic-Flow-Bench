"""Can the day's own evidence, before T, say that the queue will form somewhere other than the usual place?

Note 51 located the onset loss exactly. On 12% of onset windows the answer scores IoU 0, those windows carry the
LARGEST top1-top2 margin - the model is confidently wrong - and the link it picks sits a median of 26 links away
from the true one. The true bottleneck is inside the top-3 candidates 98-99% of the time, so the candidate set is
not the problem. What is missing is the evidence that today is not a usual day.

Our models cannot hold that evidence: `window_history.parquet` is 60 minutes, and every member reads only that.
The organizers allow every released observation at or before T, which is the whole morning, the previous days and
the ramp flows up to T. This asks whether that material separates the habitual bottleneck from a rare one.

Setup, per panel:
  candidates   the peaks of the first-onset histogram over the FIT days only, at least 3 links apart, top 10,
               with every link assigned to the nearest one (note 51's Voronoi partition)
  windows      `first_onsets` over all train days, truth restricted to eligibility - the coverage that matches
               the official one (note 108)
  label        the candidate whose cell holds most of the true onset cells at T+30
  features     the habitual prior, plus the day's state near the candidate at T-5, over the last hour, the last
               three hours and since 04:00, the upstream approach, the attached on-ramp flow, the recurrence of
               that candidate on the last days, and the day's demand level against the same weekday's fit mean
  split        fit on days < FIT_END, read on days >= FIT_END

Everything is timestamped at or before T-5: slot T itself and the whole horizon are excluded, ramps included.

    python occ_lab/onset_bottleneck_rerank.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta"))
sys.path.insert(0, str(LAB.parent / "analysis"))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

import common as C  # noqa: E402
from t2_first_onset_eval import first_onsets  # noqa: E402
from t2_m1_predict import onset_X  # noqa: E402
from tfb.t1eval import FIT_END  # noqa: E402

N_CAND, MIN_SEP, DAWN = 10, 3, 48                              # 04:00


def candidates(cells, S):
    """The peaks of the first-onset histogram, at least MIN_SEP links apart."""
    h = cells.sum(0).astype(float)
    h = h + 0.5 * (np.roll(h, 1) + np.roll(h, -1))
    picked = []
    for s in np.argsort(-h):
        if h[s] <= 0:
            break
        if all(abs(s - p) >= MIN_SEP for p in picked):
            picked.append(int(s))
        if len(picked) >= N_CAND:
            break
    if not picked:
        picked = [S // 2]
    cand = np.array(sorted(picked))
    vor = np.argmin(np.abs(np.arange(S)[:, None] - cand[None, :]), 1)
    return cand, vor, h


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=300)
    args = ap.parse_args()
    rows, labs, grp, ev, pri_col, mscore, wcol = [], [], [], [], [], [], []
    t0_ = time.time()
    for i, panel in enumerate(C.T2_PANELS):
        ctx = C.Ctx(panel, i)
        ctx.fitd = np.ones_like(ctx.fitd)
        z = ctx.z
        S = ctx.S
        el = z["elig"]
        v = np.where(el, z["true_speed"], np.nan) / np.asarray(ctx.vf, np.float32)
        q = np.where(el, z["true_flow"], np.nan) / np.asarray(ctx.cap, np.float32)
        ramp = C.load_ramps(panel, "train", z)
        on_q = ramp["on_q"]
        w = first_onsets(ctx.vf, z, truth=True, blank=None)
        is_ev = z["dates"][w["d"]] >= FIT_END
        hic = LAB / "cache" / f"onset_hicov_{panel}.npz"
        MP = None
        if hic.exists():
            ref = np.load(hic)
            if len(ref["y"]) == int(is_ev.sum()):
                MP = 0.35 * ref["H"] + 0.35 * ref["ad9b"]     # the AP blend's frozen half, on the tail
        ev_pos = np.cumsum(is_ev) - 1
        # a WINDOW model trained on the fit days only: the 60-minute view, same features as production H
        fitd = z["dates"] < FIT_END
        tabs = {n: np.stack([xs[(ctx.g == g) & fitd].mean(0) for g in range(int(ctx.g.max()) + 1)])
                for n, xs in (("q", ctx.xs_q), ("on", ctx.xs_on))}
        pri_fn = lambda n, key, t: tabs[n][key[:, 1], np.clip(t, 0, tabs[n].shape[1] - 1)]  # noqa: E731
        gg = (z["weekday"][w["d"]] >= 5).astype(int)
        kk = np.stack([w["d"], gg], 1)
        _, XW = onset_X(ctx, w["v"], w["qf"], w["ok"], w["t0"], gg, pri_fn, kk)
        yw = w["cells"].astype(np.float32)
        fw = ~is_ev
        wm = lgb.train({"objective": "binary", "learning_rate": 0.05, "num_leaves": 31,
                        "min_data_in_leaf": 60, "feature_fraction": 0.8, "verbose": -1,
                        "num_threads": 8},
                       lgb.Dataset(XW[fw].reshape(-1, XW.shape[-1]), label=yw[fw].ravel()),
                       num_boost_round=300)
        PW = wm.predict(XW.reshape(-1, XW.shape[-1])).reshape(XW.shape[:2])
        cand, vor, hist = candidates(w["cells"][~is_ev], S)
        K = len(cand)
        pri = np.array([hist[vor == k].sum() for k in range(K)], float)
        pri = pri / max(pri.sum(), 1e-9)
        cells_of = [np.flatnonzero(vor == k) for k in range(K)]

        # the candidate that owns each window's true onset cells, and the onset candidate of each day
        own = np.array([np.bincount(vor[np.flatnonzero(c)], minlength=K).argmax() if c.any() else -1
                        for c in w["cells"]])
        day_cand = {int(d): int(o) for d, o in zip(w["d"], own) if o >= 0}

        def band(a, d, lo, hi, links):
            sl = a[d, max(lo, 0):max(hi, 1)][:, links]
            return float(np.nanmean(sl)) if np.isfinite(sl).any() else 0.0

        for j in range(len(w["d"])):
            d, t0 = int(w["d"][j]), int(w["t0"][j])
            if own[j] < 0:
                continue
            gmean = band(q, d, t0 - 36, t0, np.arange(S))
            same = [int(x) for x in np.flatnonzero((z["weekday"] == z["weekday"][d]) &
                                                   (z["dates"] < FIT_END))]
            ref = float(np.nanmean([band(q, dd, t0 - 36, t0, np.arange(S)) for dd in same[-8:]])) if same else 0.0
            prev = [day_cand.get(d - k, -1) for k in (1, 2, 3, 7)]
            for k in range(K):
                L = cells_of[k]
                up = np.arange(max(cand[k] - 6, 0), cand[k])
                f = [pri[k], float(np.argsort(-pri).tolist().index(k)), len(L),
                     band(v, d, t0 - 1, t0, L), band(v, d, t0 - 12, t0, L),
                     band(v, d, t0 - 36, t0, L), band(v, d, DAWN, t0, L),
                     band(v, d, t0 - 6, t0, L) - band(v, d, t0 - 12, t0 - 6, L),
                     band(q, d, t0 - 1, t0, L), band(q, d, t0 - 12, t0, L),
                     band(q, d, t0 - 36, t0, L), band(q, d, DAWN, t0, L),
                     band(q, d, t0 - 6, t0, L) - band(q, d, t0 - 12, t0 - 6, L),
                     band(v, d, t0 - 12, t0, up) if len(up) else 0.0,
                     float(on_q[d, max(t0 - 12, 0):t0][:, L].mean()) if len(L) else 0.0,
                     float(on_q[d, max(t0 - 36, 0):t0][:, L].mean()) if len(L) else 0.0,
                     float(prev[0] == k), float(k in prev[:3]), float(k in prev),
                     gmean, gmean - ref, float(t0) / 12, float(z["weekday"][d]),
                     band(v, d, t0 - 12, t0, np.arange(S))]
                rows.append(f + [float(PW[j][L].sum()), float(PW[j][L].max())])
                wcol.append(float(PW[j][L].sum()))
                mscore.append(float(MP[ev_pos[j]][L].sum()) if (MP is not None and is_ev[j]) else np.nan)
                labs.append(int(own[j] == k))
                grp.append((i, j))
                ev.append(bool(is_ev[j]))
                pri_col.append(pri[k])
        print(f"  {panel}: {len(w['d'])} onset windows, {K} candidates, "
              f"eval {int(is_ev.sum())} ({time.time() - t0_:.0f}s)", flush=True)
        ctx.z = ctx.q = ctx.el = None

    X = np.asarray(rows, np.float64)
    y = np.asarray(labs)
    ev = np.asarray(ev)
    pri_col = np.asarray(pri_col)
    key = np.array([g[0] * 10 ** 6 + g[1] for g in grp])
    NAMES = ["prior", "prior_rank", "n_links", "v_T5", "v_1h", "v_3h", "v_dawn", "v_trend",
             "q_T5", "q_1h", "q_3h", "q_dawn", "q_trend", "v_upstream", "ramp_1h", "ramp_3h",
             "was_yesterday", "in_last3", "in_last7", "day_flow", "day_flow_vs_ref", "hour",
             "weekday", "corridor_v_1h", "window_sum", "window_max"]
    print(f"\n{len(np.unique(key)):,} windows, {len(y):,} (window, candidate) rows, "
          f"{int(ev.sum()):,} eval rows\n")
    par = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 40,
           "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1,
           "num_threads": 8}
    wcol = np.asarray(wcol)
    day_cols = [i for i, n in enumerate(NAMES) if n not in ("window_sum", "window_max")]
    win_cols = [i for i, n in enumerate(NAMES) if n in ("prior", "prior_rank", "window_sum", "window_max")]

    def top1(score):
        k, yy = key[ev], y[ev]
        return float(np.mean([bool(yy[k == u][np.argmax(score[k == u])]) for u in np.unique(k)])),             len(np.unique(k))

    def fit_top1(cols):
        m_ = lgb.train(par, lgb.Dataset(X[~ev][:, cols], label=y[~ev],
                                        feature_name=[NAMES[c] for c in cols]),
                       num_boost_round=args.rounds)
        return top1(m_.predict(X[ev][:, cols]))[0], m_

    a_pri, n = top1(pri_col[ev])
    a_win, _ = top1(wcol[ev])
    a_wfit, _ = fit_top1(win_cols)
    a_day, _ = fit_top1(day_cols)
    a_all, mdl = fit_top1(list(range(len(NAMES))))
    p = mdl.predict(X[ev])
    print(f"top-1 bottleneck accuracy on the held-out days ({n} windows), everything fitted on days "
          f"before FIT_END only\n")
    print(f"  habitual prior alone                      {a_pri:.4f}")
    print(f"  60-minute window model alone              {a_win:.4f}")
    print(f"  prior + window model, refitted            {a_wfit:.4f}")
    print(f"  the day's evidence (no window model)      {a_day:.4f}")
    print(f"  both                                      {a_all:.4f}   "
          f"{a_all - max(a_wfit, a_day):+.4f} over the better half")
    imp = sorted(zip(NAMES, mdl.feature_importance("gain")), key=lambda t: -t[1])[:10]
    print("\n  gain: " + ", ".join(f"{k} {v:.0f}" for k, v in imp))

    k, yy = key[ev], y[ev]
    rare = []
    for u in np.unique(k):
        m = k == u
        rare.append((bool(yy[m][np.argmax(pri_col[ev][m])]), bool(yy[m][np.argmax(p[m])])))
    rare = np.array(rare)
    n_rare = int((~rare[:, 0]).sum())
    print(f"\n  windows the habitual answer gets wrong: {n_rare} of {len(rare)} "
          f"({n_rare / len(rare):.3f})")
    if n_rare:
        print(f"  of those, the day's evidence recovers {int(rare[~rare[:, 0], 1].sum())} "
              f"({rare[~rare[:, 0], 1].mean():.3f})")
    kept = rare[rare[:, 0], 1].mean() if rare[:, 0].any() else float("nan")
    print(f"  of the ones it already had right, it keeps {kept:.3f}")


if __name__ == "__main__":
    main()
