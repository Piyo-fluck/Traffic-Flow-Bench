"""Task 2 onset: the capacity-margin and bottleneck-structure features rebuilt with the CORRECT link orientation
(note 49). The old implementation (analysis/t2_struct_noid.py) treated index s+k as downstream in every panel, and
detected queue heads (for the effective discharge capacity) as "link s free, link s-1 queued". Both are wrong in the
S/W panels, whose link order runs downstream -> upstream.

Orientation sign per panel: +1 = downstream is the higher index, -1 = the lower index.
    old       +1 everywhere (the old implementation)
    rule      +1 for N/E panels, -1 for S/W panels (the teammate's EDA)
    measured  the sign measured in note 49 (differs from the rule on D7_I10_E, D7_I10_W, D12_I5_S)
Features added to H's onset features, per orientation:
    effective capacity ceff = train median flow at queue heads (link free, its UPSTREAM neighbour queued), >= 20 events
    st_*  lane drop / capacity ratio / on-ramp / merge ahead (downstream), lane gain behind (upstream), distance to the
          next downstream bottleneck, ceff / released cap
    mg_*  last-30-min flow / ceff at the link, 1 link and max over 4 links downstream, flow vs the tightest ceff within
          4 links downstream, projected to T+30 by the profile growth, and the history trend
Paired LightGBM in H's configuration (prior dropout, 5 bags), 3 seeds; the variants share the prior-dropout draws.
Scored on the blank-aware official-like onset windows: LightGBM alone on masked and on T1-filled histories, and in
M's blend (0.7 LightGBM + 0.3 CNN x3, cached CNN probabilities, masked).

    .venv/Scripts/python.exe analysis/t2_margin_oriented.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "scripts"))
import t2_prior_robust  # noqa: E402
from t2_analog_prior import filled, rehist  # noqa: E402
from t2_demand_stress import load_tq  # noqa: E402
from t2_eda02 import T2_PANELS  # noqa: E402
from t2_eda03 import F  # noqa: E402
from t2_expected_iou import select  # noqa: E402
from t2_first_onset_eval import first_onsets, official_blank  # noqa: E402
from t2_m1_models import Ctx, hist_from_truth  # noqa: E402
from t2_m1_predict import feats_on, table_prior  # noqa: E402
from t2_pseudo_eval import iou  # noqa: E402
from t2_shift_remedies import PARAMS  # noqa: E402
from tfb.data import CACHE, load  # noqa: E402
from tfb.features import interp_along  # noqa: E402
from tfb.physics import load_ramps  # noqa: E402
from tfb.t1eval import family_mean  # noqa: E402

SEEDS, N_BAGS, MAX_WIN = (0, 1, 2), 5, 2000
rng = np.random.default_rng(29)
MEASURED = {"D7_I10_E": -1, "D7_I10_W": 1, "D7_I210_E": 1, "D7_I210_W": -1,
            "D7_I405_N": 1, "D7_I405_S": -1, "D12_I5_N": 1, "D12_I5_S": 1}
ORIENT = {"old": lambda p: 1, "rule": lambda p: 1 if p.endswith(("_N", "_E")) else -1, "measured": lambda p: MEASURED[p]}
VARIANTS = ("base",) + tuple(ORIENT)
TAGS = ["t2_cnn_lov", "t2_cnn_lov_s1", "t2_cnn_lov_s2"]
CONDS = [("validation", 14), ("validation", 31), ("private", 14), ("private", 31)]


def fmean(d):
    d = {k: v for k, v in d.items() if np.isfinite(v)}
    return family_mean(d) if d else float("nan")


def at_o(x, k, sgn, axis=-1):
    """Value k links DOWNSTREAM (k < 0: upstream) along `axis`, edge-padded; sgn = +1 if downstream = higher index."""
    S = x.shape[axis]
    return np.take(x, np.clip(np.arange(S) + sgn * k, 0, S - 1), axis=axis)


def effective_capacity(ctx, sgn) -> np.ndarray:
    z = ctx.z
    v, q = z["true_speed"], z["true_flow"]
    ok = z["elig"] & np.isfinite(v) & np.isfinite(q)
    qd = ok & (np.nan_to_num(v, nan=1e9) <= 0.6 * ctx.vf)
    fr = ok & (np.nan_to_num(v, nan=0) > 0.85 * ctx.vf)
    head = np.zeros_like(ok)
    if sgn > 0:   # upstream neighbour of s is s-1
        head[:, :, 1:] = fr[:, :, 1:] & qd[:, :, :-1]
    else:         # upstream neighbour of s is s+1
        head[:, :, :-1] = fr[:, :, :-1] & qd[:, :, 1:]
    n = head.sum((0, 1))
    dis = np.array([np.median(q[:, :, s][head[:, :, s]]) if n[s] >= 20 else np.nan for s in range(ctx.S)])
    return np.where(np.isfinite(dis), dis, ctx.cap).astype(np.float32)


def struct_vecs(ctx, ceff, sgn, ramps) -> dict:
    S = ctx.S
    on, off = ramps["on_att"].astype(np.float32), ramps["off_att"].astype(np.float32)
    lanes, cap = np.asarray(ctx.lanes, np.float32), np.asarray(ctx.cap, np.float32)
    bneck = (at_o(ceff, 1, sgn) < 0.95 * ceff) | (at_o(on, 1, sgn) > 0)
    dist, nxt = np.full(S, 99, np.float32), None
    for s in (range(S - 1, -1, -1) if sgn > 0 else range(S)):   # walk from the downstream end
        if bneck[s]:
            nxt = s
        dist[s] = 99 if nxt is None else abs(nxt - s)
    return {
        "st_lane_drop1": lanes - at_o(lanes, 1, sgn),
        "st_lane_drop4": lanes - np.min([at_o(lanes, k, sgn) for k in range(1, 5)], 0),
        "st_lane_gain_up1": lanes - at_o(lanes, -1, sgn),
        "st_cap_ratio1": at_o(ceff, 1, sgn) / ceff,
        "st_cap_ratio4": np.min([at_o(ceff, k, sgn) for k in range(1, 5)], 0) / ceff,
        "st_on": on, "st_off": off, "st_on_dn1": at_o(on, 1, sgn),
        "st_merge_dn": np.max([at_o(on, k, sgn) for k in range(3)], 0),
        "st_dist_bneck": dist, "st_ceff_ratio": ceff / cap,
    }


def margins(ceff, tq, qf, ok, t0, wd, sgn) -> dict:
    with np.errstate(all="ignore"):
        q3 = np.nanmean(np.where(ok[:, -3:], qf[:, -3:], np.nan), 1)
        qa = np.nanmean(np.where(ok[:, :3], qf[:, :3], np.nan), 1)
    have = np.isfinite(q3)
    qfill, _ = interp_along(np.where(have, q3, np.nan), have, axis=1)
    qfill = np.nan_to_num(qfill, nan=0.0)
    tight = np.min([at_o(ceff, k, sgn) for k in range(5)], 0)
    growth = tq[wd, np.clip(t0 + F, 0, tq.shape[1] - 1)] / np.maximum(tq[wd, t0 - 1], 1.0)
    r = qfill / ceff
    return {
        "mg_q_own": r, "mg_q_dn1": at_o(r, 1, sgn), "mg_dn_max4": np.max([at_o(r, k, sgn) for k in range(1, 5)], 0),
        "mg_tight": qfill / tight, "mg_tight_proj": qfill / tight * growth,
        "mg_trend": np.nan_to_num((q3 - qa) / ceff, nan=0.0),
    }


def extra(o, tq, qf, ok, t0, wd):
    """(names, (N, S, K)) structure + margin columns for orientation state o = (sgn, ceff, sv)."""
    sgn, ceff, sv = o
    N, S = qf.shape[0], qf.shape[2]
    cols = {k: np.broadcast_to(np.asarray(v, np.float32)[None], (N, S)) for k, v in sv.items()}
    cols.update(margins(ceff, tq, qf, ok, t0, wd, sgn))
    names = sorted(cols)
    return names, np.stack([np.asarray(cols[k], np.float32) for k in names], -1)


def main() -> None:
    t_start = time.time()
    Xb = {(s, b): [] for s in SEEDS for b in range(N_BAGS)}
    Ex = {v: [] for v in ORIENT}
    ys, ctxs, names_on, names_ex = [], [], None, None
    for i, p in enumerate(T2_PANELS):
        ctx = Ctx(p, i)
        ctx.fitd = np.ones_like(ctx.fitd)
        z = ctx.z
        ctx.tq = load_tq(p)
        ramps = load_ramps(p, "train", z)
        ctx.o = {}
        for var, fn in ORIENT.items():
            sgn = fn(p)
            ceff = effective_capacity(ctx, sgn)
            ctx.o[var] = (sgn, ceff, struct_vecs(ctx, ceff, sgn, ramps))
        d, t0 = ctx.eo
        if len(d) > MAX_WIN:
            s = np.sort(rng.choice(len(d), MAX_WIN, replace=False))
            d, t0 = d[s], t0[s]
        v, qf, ok = hist_from_truth(ctx, d, t0)
        wd = z["weekday"][d].astype(int)
        for var in ORIENT:
            names_ex, E = extra(ctx.o[var], ctx.tq, qf, ok, t0, wd)
            Ex[var].append(E.reshape(-1, E.shape[-1]))
        ys.append(ctx.q[d, t0 + F].reshape(-1))
        for sd in SEEDS:
            t2_prior_robust.rng = np.random.default_rng(1000 * (sd + 1) + i)
            for b in range(N_BAGS):
                names_on, X = feats_on(ctx, False, v, qf, ok, t0, ctx.g[d], t2_prior_robust.dropout_prior(ctx), d, z["weekday"][d])
                Xb[(sd, b)].append(X.reshape(-1, X.shape[-1]))
        c_old, c_rule = ctx.o["old"][1], ctx.o["rule"][1]
        print(f"[{p}] windows {len(d)}; sign old/rule/measured {[ctx.o[k][0] for k in ORIENT]}; ceff old vs rule differ "
              f">3% on {int((np.abs(c_old / c_rule - 1) > 0.03).sum())}/{ctx.S} links ({time.time() - t_start:.0f}s)", flush=True)
        ctx.z = {k: z[k] for k in ("link_ids", "weekday", "dates")}
        ctx.q = ctx.el = None
        ctxs.append(ctx)
    y_all = np.concatenate(ys)
    E_all = {k: np.concatenate(v) for k, v in Ex.items()}
    models = {}
    for sd in SEEDS:
        for var in VARIANTS:
            ms = []
            for b in range(N_BAGS):
                X = np.concatenate(Xb[(sd, b)])
                if var != "base":
                    X = np.concatenate([X, E_all[var]], 1)
                nm = names_on + (names_ex if var != "base" else [])
                ms.append(lgb.train(dict(PARAMS, num_leaves=63, seed=b + 100 * sd), lgb.Dataset(X, y_all, feature_name=nm), 600))
            models[(sd, var)] = ms
            if var != "base":
                imp = ms[0].feature_importance("gain")
                print(f"trained seed {sd} {var} ({time.time() - t_start:.0f}s); margin+struct share of gain "
                      f"{imp[-len(names_ex):].sum() / imp.sum():.3f}", flush=True)
    del Xb
    res: dict = {}
    for ctx in ctxs:
        for split in ("validation", "private"):
            z = load(ctx.panel, split)
            on = first_onsets(ctx.vf, z, blank=official_blank(ctx.panel, split, z))
            if not len(on["d"]):
                continue
            c = np.load(CACHE / f"t2probs_{split}_{ctx.panel}.npz")
            assert (c["on_d"] == on["d"]).all()
            cn = np.mean([np.load(CACHE / f"t2cnn_{t}_{split}_{ctx.panel}.npz")["on_p"] for t in TAGS], 0)
            grp = (z["weekday"][on["d"]] >= 5).astype(int)
            key = np.stack([on["d"], grp], 1)
            wd = z["weekday"][on["d"]].astype(int)
            for hname, (hv, hq, ho) in (("masked", (on["v"], on["qf"], on["ok"])),
                                         ("filled", rehist(filled(z, ctx.panel, split), on["d"], on["t0"]))):
                _, X = feats_on(ctx, False, hv, hq, ho, on["t0"], grp, table_prior(ctx), key, wd)
                exs = {var: extra(ctx.o[var], ctx.tq, hq, ho, on["t0"], wd)[1] for var in ORIENT}
                for (sd, var), ms in models.items():
                    Xo = np.concatenate([X, exs[var]], -1) if var != "base" else X
                    p = np.mean([m.predict(Xo.reshape(-1, Xo.shape[-1])) for m in ms], 0).reshape(Xo.shape[:2])
                    modes = [("alone", p)] + ([("M-blend", 0.7 * p + 0.3 * cn)] if hname == "masked" else [])
                    for mode, pp in modes:
                        pr = np.zeros(on["fut"].shape, bool)
                        pr[:, F - 1] = np.stack([select(q, "ratio", 1.0) for q in pp])
                        s = iou(pr, on["fut"], on["efut"])
                        for sp, dmax in CONDS:
                            if sp == split:
                                m = on["d"] < dmax
                                res.setdefault((hname, mode, var, sd), {}).setdefault((sp, dmax), {})[ctx.panel] = s[m].mean() if m.any() else np.nan
        print(f"  scored {ctx.panel} ({time.time() - t_start:.0f}s)", flush=True)
    for hname, mode in (("masked", "alone"), ("filled", "alone"), ("masked", "M-blend")):
        print(f"\n== onset, {mode}, {hname} histories: val d1-14 | val all | priv d1-14 | priv all "
              f"(mean of 3 seeds; delta vs paired base, seeds > 0) ==")
        for var in VARIANTS:
            cells = []
            for cnd in CONDS:
                vals = [fmean(res[(hname, mode, var, sd)][cnd]) for sd in SEEDS]
                base = [fmean(res[(hname, mode, "base", sd)][cnd]) for sd in SEEDS]
                dl = [a - b for a, b in zip(vals, base)]
                cells.append(f"{np.mean(vals):.3f}" + ("" if var == "base" else f" ({np.mean(dl):+.4f}, {sum(x > 0 for x in dl)}/3)"))
            print(f"  {var:<9} " + " | ".join(cells))
        # per-panel view (all days, both splits pooled as the mean of the two splits), rule and measured vs old
        print("  per panel, all days, mean of val & priv and of seeds (old | rule | measured | base):")
        for ctx in ctxs:
            row = []
            for var in ("old", "rule", "measured", "base"):
                vals = [res[(hname, mode, var, sd)].get((sp, 31), {}).get(ctx.panel, np.nan) for sd in SEEDS for sp in ("validation", "private")]
                row.append(f"{np.nanmean(vals):.3f}")
            print(f"    {ctx.panel:<10} sign rule {ORIENT['rule'](ctx.panel):+d} measured {MEASURED[ctx.panel]:+d}: " + " | ".join(row))
    print(f"total {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
