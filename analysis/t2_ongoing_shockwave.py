"""Task 2 ONGOING: shockwave features for the queue boundary (the model-side lever identified in notes 52/53).

Note 52: 70% of the ongoing errors are missed cells; the largest bucket is queue GROWTH (33-40% of the cells the
queue newly reaches), then dissolution (15-17%), and the last two horizon steps carry 43-45% of the loss.
Note 53: no threshold rule fixes this (per-cell thresholds add nothing over a global tau), so the probabilities
themselves must improve on boundary cells.

The queue boundary is a shockwave. Its speed follows Rankine-Hugoniot, w = (q1 - q2) / (k1 - k2), from the flow and
density on the two sides, and it is also directly observable as the movement of the queue edge over the history.
Both are computed here from the history only (T-60 .. T-5), and neither needs to know which index side is upstream
(notes 49/50: the orientation rules do not transfer), because the sign is taken TOWARDS the link being scored.

Per window x link (static):
    sw_r        signed distance in links to the boundary: + to the nearest queued link if free, - to the nearest
                free link if queued (i.e. how deep inside the queue the link sits)
    sw_appr3/6  the boundary's measured approach speed towards this link over the last 15 / 30 min (links per slot)
    sw_eta      slots until the boundary reaches the link at the measured speed
    sw_rh       |Rankine-Hugoniot speed| at the nearest boundary, converted to links per 5-min slot with length_km
    sw_toward   +1 if the arriving flow exceeds the discharge at that boundary (the shock travels towards the link)
    sw_qratio   arriving flow / discharge flow at the boundary;  sw_dqcap  their difference over capacity
    sw_kratio   density ratio across the boundary;  sw_len  the link's length_km
Per window x horizon step x link:
    sw_margin_h   sw_r - sw_appr3 * h        (< 0: the measured edge has passed this link by step h)
    sw_rh_marg_h  sw_r - sw_toward * sw_rh * h   (the same with the physical shock speed)
    sw_arr_h      1 if sw_eta <= h
    sw_frac_h     sw_r / (sw_appr3 * h)      (how much of the gap the edge closes by step h)

Paired experiment, production I configuration (demand-stress features, no priors, LightGBM 2000 rounds / 127
leaves), 2 seeds: base vs base + shockwave, on identical sampled rows. Scored on the blank-aware official-like
ongoing windows: LightGBM alone and in M's blend (0.5 * LGBM + 0.5 * CNN x3) at tau 0.60 and 0.50, with paired
bootstrap CIs, plus the note-52 error classes to check the mechanism (does the growth error rate fall?).

    .venv/Scripts/python.exe analysis/t2_ongoing_shockwave.py
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
from t2_demand_stress import load_tq, ongoing_Xs  # noqa: E402
from t2_eda02 import T2_PANELS, candidates  # noqa: E402
from t2_eda03 import F  # noqa: E402
from t2_first_onset_eval import official_blank  # noqa: E402
from t2_m1_models import Ctx, hist_from_truth  # noqa: E402
from t2_m1_predict import zero_prior  # noqa: E402
from t2_official_like_eval import ongoing_official  # noqa: E402
from t2_pseudo_eval import iou  # noqa: E402
from t2_shift_remedies import PARAMS  # noqa: E402
from tfb.data import CACHE, load  # noqa: E402
from tfb.physics import fd_params  # noqa: E402

TAGS = ["t2_cnn_lov", "t2_cnn_lov_s1", "t2_cnn_lov_s2"]
SEEDS, ROUNDS, LEAVES, MAX_OG, CHUNK = (0, 1), 2000, 127, 150_000, 400
TAUS = (0.60, 0.50)
W_L = 0.5
SW_STATIC = ["sw_r", "sw_appr3", "sw_appr6", "sw_eta", "sw_rh", "sw_toward", "sw_qratio", "sw_dqcap", "sw_kratio", "sw_len"]
SW_PERH = ["sw_margin_h", "sw_rh_marg_h", "sw_arr_h", "sw_frac_h"]
SW = SW_STATIC + SW_PERH
rng = np.random.default_rng(5)
boot = np.random.default_rng(17)


def ci(delta: np.ndarray, n_boot: int = 2000) -> str:
    idx = boot.integers(0, len(delta), (n_boot, len(delta)))
    m = delta[idx].mean(1)
    return f"{delta.mean():+.4f} [{np.percentile(m, 2.5):+.4f}, {np.percentile(m, 97.5):+.4f}]"


def nearest_true(m: np.ndarray):
    """(..., S) bool -> (distance, index) of the nearest True along the last axis (99 / -1 where none)."""
    S = m.shape[-1]
    ar = np.arange(S)
    prev = np.maximum.accumulate(np.where(m, ar, -1), axis=-1)
    nxt = np.flip(np.minimum.accumulate(np.flip(np.where(m, ar, S), -1), -1), -1)
    dp = np.where(prev >= 0, ar - prev, 10 ** 4)
    dn = np.where(nxt < S, nxt - ar, 10 ** 4)
    take_p = dp <= dn
    d = np.minimum(dp, dn)
    idx = np.where(take_p, prev, np.minimum(nxt, S - 1))
    return np.where(d > 99, 99, d).astype(np.float32), np.where(d > 99, -1, idx)


def ffill_time(x: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """Forward-fill along the time axis (axis 1) so that unobserved slots do not break a queue run."""
    out = np.zeros_like(x)
    cur = np.zeros((x.shape[0], x.shape[2]), x.dtype)
    seen = np.zeros((x.shape[0], x.shape[2]), bool)
    for t in range(x.shape[1]):
        cur = np.where(ok[:, t], x[:, t], cur)
        seen |= ok[:, t]
        out[:, t] = np.where(seen, cur, x[:, t])
    return out


def shock_feats(ctx, length, v, qf, ok) -> tuple[np.ndarray, np.ndarray]:
    """(N, S, len(SW_STATIC)) and (N, F, S, len(SW_PERH))."""
    N, H, S = v.shape
    qflag = ok & (np.nan_to_num(v, nan=1e9) <= 0.6 * ctx.vf)
    qf_fill = ffill_time(qflag.astype(np.float32), ok) > 0.5                      # (N, H, S)
    r_t = np.empty((N, H, S), np.float32)
    for t in range(H):
        d_q, _ = nearest_true(qf_fill[:, t])
        d_f, _ = nearest_true(~qf_fill[:, t])
        r_t[:, t] = np.where(qf_fill[:, t], -d_f, d_q)
    r = r_t[:, -1]
    appr3 = (r_t[:, -4] - r) / 3.0 if H >= 4 else np.zeros_like(r)
    appr6 = (r_t[:, -7] - r) / 6.0 if H >= 7 else appr3
    eta = np.where(appr3 > 1e-3, np.clip(r / np.maximum(appr3, 1e-3), 0, 99), 99).astype(np.float32)
    # ---- Rankine-Hugoniot at the nearest boundary, from the last observed state
    with np.errstate(all="ignore"):
        idx_last = np.where(ok, np.arange(H)[None, :, None], -1).max(1)
        take = np.clip(idx_last, 0, None)[:, None, :]
        q_l = np.take_along_axis(qf, take, 1)[:, 0]
        v_l = np.take_along_axis(v, take, 1)[:, 0]
        q_l = np.where(idx_last >= 0, q_l, np.nan)
        v_l = np.where(idx_last >= 0, v_l, np.nan)
        k_l = q_l / np.maximum(v_l, 1.0)
    last_q = qf_fill[:, -1]
    _, j_cong = nearest_true(last_q)                                              # nearest queued link
    _, j_free = nearest_true(~last_q)                                             # nearest free link
    j_b = np.where(last_q, j_free, j_cong)                                        # the boundary link on the other side
    ar = np.broadcast_to(np.arange(S), (N, S))
    step = np.sign(j_b - ar).astype(int)
    step = np.where(step == 0, 1, step)
    j_other = np.clip(j_b - step, 0, S - 1)                                       # the link just before the boundary
    g = lambda a, j: np.take_along_axis(a, np.clip(j, 0, S - 1), 1)               # noqa: E731
    q_here, k_here = g(q_l, np.where(last_q, ar, j_b)), g(k_l, np.where(last_q, ar, j_b))       # congested side
    q_free, k_free = g(q_l, np.where(last_q, j_b, j_other)), g(k_l, np.where(last_q, j_b, j_other))
    with np.errstate(all="ignore"):
        w = (q_free - q_here) / (k_free - k_here)                                 # km/h, sign vs the index axis
        rh = np.abs(w) * (5.0 / 60.0) / np.maximum(length[None, :], 0.05)         # links per slot
        toward = np.sign(np.nan_to_num(q_free - q_here, nan=0.0)).astype(np.float32)
        qratio = q_free / np.maximum(q_here, 1.0)
        dqcap = (q_free - q_here) / np.maximum(ctx.cap[None, :], 1.0)
        kratio = k_free / np.maximum(k_here, 1e-3)
    rh = np.clip(np.nan_to_num(rh, nan=0.0), 0, 10).astype(np.float32)
    stat = np.stack([r, appr3, appr6, eta, rh, toward,
                     np.nan_to_num(qratio, nan=1.0).clip(0, 10), np.nan_to_num(dqcap, nan=0.0).clip(-5, 5),
                     np.nan_to_num(kratio, nan=1.0).clip(0, 20),
                     np.broadcast_to(length[None, :], (N, S))], -1).astype(np.float32)
    per = np.empty((N, F, S, len(SW_PERH)), np.float32)
    for h in range(1, F + 1):
        per[:, h - 1, :, 0] = r - appr3 * h
        per[:, h - 1, :, 1] = r - toward * rh * h
        per[:, h - 1, :, 2] = (eta <= h).astype(np.float32)
        per[:, h - 1, :, 3] = np.clip(r / np.maximum(appr3 * h, 1e-3), -99, 99)
    return stat, per


def add_shock(Xh: np.ndarray, stat: np.ndarray, per: np.ndarray) -> np.ndarray:
    s = np.broadcast_to(stat[:, None], (stat.shape[0], F) + stat.shape[1:])
    return np.concatenate([Xh, s, per], -1)


def main() -> None:
    t0_all = time.time()
    parts: list = []
    ctxs, tqs, lens = [], {}, {}
    names_og = None
    for i, panel in enumerate(T2_PANELS):
        ctx = Ctx(panel, i)
        ctx.fitd = np.ones_like(ctx.fitd)
        z = ctx.z
        tqs[panel] = load_tq(panel)
        lens[panel] = fd_params(panel, z["link_ids"])["length"].astype(np.float32)
        c = candidates(z)
        d, t0 = c["d"][c["ongoing"]], c["t0"][c["ongoing"]]
        xs, es, ys = [], [], []
        for j in range(0, len(d), CHUNK):
            dd, tt = d[j:j + CHUNK], t0[j:j + CHUNK]
            v, qf, ok = hist_from_truth(ctx, dd, tt)
            names_og, Xh, base = ongoing_Xs(ctx, v, qf, ok, tt, ctx.g[dd], zero_prior(ctx), dd,
                                            z["weekday"][dd], tqs[panel], True)
            stat, per = shock_feats(ctx, lens[panel], v, qf, ok)
            E = np.concatenate([np.broadcast_to(stat[:, None], (len(dd), F) + stat.shape[1:]), per], -1)
            y = ctx.q[dd[:, None], tt[:, None] + np.arange(1, F + 1)]
            e = ctx.el[dd[:, None], tt[:, None] + np.arange(1, F + 1)]
            # one keep mask for both matrices: sample_og draws 2% of the cells at random, so calling it twice
            # would select different rows and break the pairing (and the row counts) between X and E
            near = np.minimum(base["dist_slow_up"], base["dist_slow_dn"]) <= 10
            keep = np.flatnonzero(e & (np.broadcast_to(near[:, None, :], y.shape) | y | (rng.random(y.shape) < 0.02)))
            xs.append(Xh.reshape(-1, Xh.shape[-1])[keep].astype(np.float32))
            es.append(E.reshape(-1, E.shape[-1])[keep].astype(np.float32))
            ys.append(y.reshape(-1)[keep])
        X, E, y = np.concatenate(xs), np.concatenate(es), np.concatenate(ys)
        if len(y) > MAX_OG:
            k = rng.choice(len(y), MAX_OG, replace=False)
            X, E, y = X[k], E[k], y[k]
        parts.append((X, E, y))
        ctx.z = {k: z[k] for k in ("link_ids", "weekday", "dates")}
        ctx.q = ctx.el = None
        ctxs.append(ctx)
        print(f"[{panel}] {len(y)} ongoing rows ({time.time() - t0_all:.0f}s)", flush=True)
    y_all = np.concatenate([p[2] for p in parts])
    models = {}
    for var in ("base", "shock"):
        X_all = np.concatenate([p[0] for p in parts]) if var == "base" else \
            np.concatenate([np.concatenate([p[0], p[1]], 1) for p in parts])
        nm = names_og + (SW if var == "shock" else [])
        for sd in SEEDS:
            m = lgb.train(dict(PARAMS, num_leaves=LEAVES, seed=sd), lgb.Dataset(X_all, y_all, feature_name=nm), ROUNDS)
            models[(var, sd)] = m
            if var == "shock":
                imp = m.feature_importance("gain")
                top = ", ".join(f"{nm[j]} {imp[j] / imp.sum():.3f}" for j in np.argsort(-imp)[:5])
                print(f"  shock seed {sd}: shockwave share of gain {imp[-len(SW):].sum() / imp.sum():.3f}; top: {top}", flush=True)
        del X_all
        print(f"trained {var} ({time.time() - t0_all:.0f}s)", flush=True)
    del parts
    res: dict = {}
    for ctx in ctxs:
        for split in ("validation", "private"):
            z = load(ctx.panel, split)
            og, _ = ongoing_official(ctx, z, official_blank(ctx.panel, split, z))
            n = len(og["d"])
            if not n:
                continue
            c = np.load(CACHE / f"t2probs_{split}_{ctx.panel}.npz")
            assert (c["og_d"] == og["d"]).all()
            cn = np.mean([np.load(CACHE / f"t2cnn_{t}_{split}_{ctx.panel}.npz")["og_p"] for t in TAGS], 0)
            truth, elig = og["fut"], og["efut"]
            vis_q = og["ok"] & (np.nan_to_num(og["v"], nan=1e9) <= 0.6 * ctx.vf)
            idx_last = np.where(og["ok"], np.arange(og["ok"].shape[1])[None, :, None], -1).max(1)
            base_q = np.take_along_axis(vis_q, np.clip(idx_last, 0, None)[:, None, :], 1)[:, 0] & (idx_last >= 0)
            b3 = np.broadcast_to(base_q[:, None], truth.shape)
            cls = {"stay": b3 & truth, "dissolve": b3 & ~truth & elig, "grow": ~b3 & truth, "far": ~b3 & ~truth & elig}
            ps = {}
            for j in range(0, n, CHUNK):
                sl = slice(j, j + CHUNK)
                grp = (z["weekday"][og["d"][sl]] >= 5).astype(int)
                key = np.stack([og["d"][sl], grp], 1)
                _, Xh, _ = ongoing_Xs(ctx, og["v"][sl], og["qf"][sl], og["ok"][sl], og["t0"][sl], grp,
                                      zero_prior(ctx), key, z["weekday"][og["d"][sl]].astype(int), tqs[ctx.panel], True)
                stat, per = shock_feats(ctx, lens[ctx.panel], og["v"][sl], og["qf"][sl], og["ok"][sl])
                Xs = add_shock(Xh, stat, per)
                for (var, sd), m in models.items():
                    XX = Xh if var == "base" else Xs
                    p = m.predict(XX.reshape(-1, XX.shape[-1])).reshape(XX.shape[:3])
                    ps.setdefault((var, sd), []).append(p)
            for (var, sd), pl in ps.items():
                p = np.concatenate(pl)
                for tau in TAUS:
                    for mode, pp in (("alone", p), ("M-blend", W_L * p + (1 - W_L) * cn)):
                        pr = pp >= tau
                        r = res.setdefault((mode, tau, var, sd), {})
                        r.setdefault(split, []).append(iou(pr, truth, elig))
                        if tau == TAUS[0]:
                            for cname, cm in cls.items():
                                e = ((pr & elig & ~truth) | (truth & ~(pr & elig))) & cm
                                r[f"{split}_{cname}"] = r.get(f"{split}_{cname}", 0) + e.sum()
                                r[f"{split}_{cname}_n"] = r.get(f"{split}_{cname}_n", 0) + cm.sum()
        print(f"  scored {ctx.panel} ({time.time() - t0_all:.0f}s)", flush=True)
    for split in ("validation", "private"):
        print(f"\n=== {split}: ongoing IoU on official-like windows (mean of {len(SEEDS)} seeds) ===")
        for mode in ("alone", "M-blend"):
            for tau in TAUS:
                b = [np.concatenate(res[(mode, tau, "base", sd)][split]) for sd in SEEDS]
                s = [np.concatenate(res[(mode, tau, "shock", sd)][split]) for sd in SEEDS]
                d = [x - y for x, y in zip(s, b)]
                print(f"  {mode:<8} tau {tau:.2f}: base {np.mean([x.mean() for x in b]):.4f} -> shock "
                      f"{np.mean([x.mean() for x in s]):.4f}  delta {ci(np.mean(d, 0))} "
                      f"(seeds {[f'{x.mean():+.4f}' for x in d]})")
        print("  error rate by class at tau 0.60, alone (base -> shock):")
        for cname in ("grow", "dissolve", "stay", "far"):
            b = np.mean([res[("alone", TAUS[0], "base", sd)][f"{split}_{cname}"] /
                         max(res[("alone", TAUS[0], "base", sd)][f"{split}_{cname}_n"], 1) for sd in SEEDS])
            s = np.mean([res[("alone", TAUS[0], "shock", sd)][f"{split}_{cname}"] /
                         max(res[("alone", TAUS[0], "shock", sd)][f"{split}_{cname}_n"], 1) for sd in SEEDS])
            print(f"    {cname:<9} {b:.4f} -> {s:.4f}  ({s - b:+.4f})")
    print(f"total {time.time() - t0_all:.0f}s")


if __name__ == "__main__":
    main()
