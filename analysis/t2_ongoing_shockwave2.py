"""Task 2 ONGOING: shockwave features v2 — both sides of every link tracked separately (growth AND dissolution).

Note 54: v1 (`t2_ongoing_shockwave.py`) cut the GROW error rate by 0.023-0.027 and gained +0.004...+0.006 in M's
blend, but left DISSOLUTION untouched. The reason is structural: v1 keyed everything on the *nearest* boundary, so
for a link inside a queue it could be looking at either the tail or the head, and the two have opposite dynamics.

v2 tracks the LOW-index and HIGH-index side of each link separately, which needs no orientation assumption
(notes 49/50) and describes growth and dissolution symmetrically:
    u1, u2      signed distance to the boundary on each side (+ = distance to the queue when the link is free,
                - = distance to the free road when the link is queued)
    v1, v2      how fast each side moves towards the link (links per slot) over the last 15 and 30 min
    run         the length of the queue run the link belongs to (or the nearest one) and its rate of change
    q/k/v at the queued and free link bounding each side, over capacity / free speed: the supply, the discharge
                and the recovery that drive the shock
per horizon step h: the projected margin of each side (u - v*h), whether either side has passed the link, and the
profile-based arrival ratio at the queue tail.

Variants (paired, identical sampled rows, production I configuration, 2 seeds):
    base    the production ongoing features (demand stress, no priors)
    v1      + the note-54 shockwave block
    v2      + v1 + the two-sided block
Scored on the blank-aware official-like ongoing windows: LightGBM alone and in M's blend at tau 0.60 and 0.50, with
paired bootstrap CIs and the note-52 error classes.

    .venv/Scripts/python.exe analysis/t2_ongoing_shockwave2.py
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
from t2_demand_stress import load_tq, ongoing_Xs, stress  # noqa: E402
from t2_eda02 import T2_PANELS, candidates  # noqa: E402
from t2_eda03 import F  # noqa: E402
from t2_first_onset_eval import official_blank  # noqa: E402
from t2_m1_models import Ctx, hist_from_truth  # noqa: E402
from t2_m1_predict import zero_prior  # noqa: E402
from t2_official_like_eval import ongoing_official  # noqa: E402
from t2_ongoing_shockwave import SW, shock_feats  # noqa: E402
from t2_pseudo_eval import iou  # noqa: E402
from t2_shift_remedies import PARAMS  # noqa: E402
from tfb.data import CACHE, load  # noqa: E402
from tfb.physics import fd_params  # noqa: E402

TAGS = ["t2_cnn_lov", "t2_cnn_lov_s1", "t2_cnn_lov_s2"]
SEEDS, ROUNDS, LEAVES, MAX_OG, CHUNK = (0, 1), 2000, 127, 150_000, 400
TAUS = (0.60, 0.50)
W_L = 0.5
S2_STATIC = ["s2_u1", "s2_u2", "s2_v1_3", "s2_v2_3", "s2_v1_6", "s2_v2_6", "s2_run", "s2_run_v",
             "s2_q_cong", "s2_q_free", "s2_k_cong", "s2_k_free", "s2_v_free", "s2_depth"]
S2_PERH = ["s2_m1_h", "s2_m2_h", "s2_any_h", "s2_tail_pqc_h"]
S2 = S2_STATIC + S2_PERH
rng = np.random.default_rng(5)
boot = np.random.default_rng(17)


def ci(delta: np.ndarray, n_boot: int = 2000) -> str:
    idx = boot.integers(0, len(delta), (n_boot, len(delta)))
    m = delta[idx].mean(1)
    return f"{delta.mean():+.4f} [{np.percentile(m, 2.5):+.4f}, {np.percentile(m, 97.5):+.4f}]"


def side_dists(m: np.ndarray):
    """(..., S) bool -> distance to the nearest True on the low-index side and on the high-index side (99 if none),
    plus the index of that True (-1 if none)."""
    S = m.shape[-1]
    ar = np.arange(S)
    prev = np.maximum.accumulate(np.where(m, ar, -1), axis=-1)
    nxt = np.flip(np.minimum.accumulate(np.flip(np.where(m, ar, S), -1), -1), -1)
    d_lo = np.where(prev >= 0, ar - prev, 99).astype(np.float32)
    d_hi = np.where(nxt < S, nxt - ar, 99).astype(np.float32)
    return d_lo, d_hi, prev, np.minimum(nxt, S - 1)


def ffill_time(x: np.ndarray, ok: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x)
    cur = np.zeros((x.shape[0], x.shape[2]), x.dtype)
    seen = np.zeros((x.shape[0], x.shape[2]), bool)
    for t in range(x.shape[1]):
        cur = np.where(ok[:, t], x[:, t], cur)
        seen |= ok[:, t]
        out[:, t] = np.where(seen, cur, x[:, t])
    return out


def two_sided(ctx, v, qf, ok, tq, t0, wd) -> tuple[np.ndarray, np.ndarray]:
    """(N, S, len(S2_STATIC)) and (N, F, S, len(S2_PERH))."""
    N, H, S = v.shape
    qflag = ok & (np.nan_to_num(v, nan=1e9) <= 0.6 * ctx.vf)
    q_fill = ffill_time(qflag.astype(np.float32), ok) > 0.5
    u1_t = np.empty((N, H, S), np.float32)
    u2_t = np.empty((N, H, S), np.float32)
    for t in range(H):
        qm = q_fill[:, t]
        dq_lo, dq_hi, _, _ = side_dists(qm)
        df_lo, df_hi, _, _ = side_dists(~qm)
        u1_t[:, t] = np.where(qm, -df_lo, dq_lo)
        u2_t[:, t] = np.where(qm, -df_hi, dq_hi)
    u1, u2 = u1_t[:, -1], u2_t[:, -1]
    v1_3 = (u1_t[:, -4] - u1) / 3.0 if H >= 4 else np.zeros_like(u1)
    v2_3 = (u2_t[:, -4] - u2) / 3.0 if H >= 4 else np.zeros_like(u2)
    v1_6 = (u1_t[:, -7] - u1) / 6.0 if H >= 7 else v1_3
    v2_6 = (u2_t[:, -7] - u2) / 6.0 if H >= 7 else v2_3
    qm = q_fill[:, -1]
    run = np.where(qm, -u1 - u2 + 1, 0.0).astype(np.float32)                  # length of the link's own queue run
    qm0 = q_fill[:, -4] if H >= 4 else q_fill[:, -1]
    run0 = np.where(qm0, -(u1_t[:, -4]) - (u2_t[:, -4]) + 1, 0.0)
    run_v = ((run - run0) / 3.0).astype(np.float32)
    with np.errstate(all="ignore"):
        idx_last = np.where(ok, np.arange(H)[None, :, None], -1).max(1)
        take = np.clip(idx_last, 0, None)[:, None, :]
        q_l = np.where(idx_last >= 0, np.take_along_axis(qf, take, 1)[:, 0], np.nan)
        v_l = np.where(idx_last >= 0, np.take_along_axis(v, take, 1)[:, 0], np.nan)
        k_l = q_l / np.maximum(v_l, 1.0)
    dq_lo, dq_hi, i_q_lo, i_q_hi = side_dists(qm)
    df_lo, df_hi, i_f_lo, i_f_hi = side_dists(~qm)
    g = lambda a, j: np.take_along_axis(a, np.clip(j, 0, S - 1), 1)           # noqa: E731
    near_q = np.where(dq_lo <= dq_hi, i_q_lo, i_q_hi)                         # nearest queued link
    near_f = np.where(df_lo <= df_hi, i_f_lo, i_f_hi)                         # nearest free link
    with np.errstate(all="ignore"):
        q_cong, k_cong = g(q_l, near_q) / np.maximum(ctx.cap[None], 1.0), g(k_l, near_q) / np.maximum(ctx.kcrit[None], 1e-3)
        q_free, k_free = g(q_l, near_f) / np.maximum(ctx.cap[None], 1.0), g(k_l, near_f) / np.maximum(ctx.kcrit[None], 1e-3)
        v_free = g(v_l, near_f) / np.maximum(ctx.vf[None], 1.0)
    depth = np.where(qm, np.minimum(-u1, -u2), 0.0).astype(np.float32)        # links from the closest end
    cols = [u1, u2, v1_3, v2_3, v1_6, v2_6, run, run_v,
            np.nan_to_num(q_cong, nan=0.0).clip(0, 3), np.nan_to_num(q_free, nan=0.0).clip(0, 3),
            np.nan_to_num(k_cong, nan=0.0).clip(0, 10), np.nan_to_num(k_free, nan=0.0).clip(0, 10),
            np.nan_to_num(v_free, nan=1.0).clip(0, 2), depth]
    stat = np.stack([np.asarray(c, np.float32) for c in cols], -1)
    st = stress(ctx, tq, qf, ok, t0, wd)                                      # pqc (N, F, S): profile flow / capacity
    tail = np.where(dq_lo <= dq_hi, i_q_lo, i_q_hi)                           # the queue link the shock starts from
    per = np.empty((N, F, S, len(S2_PERH)), np.float32)
    for h in range(1, F + 1):
        m1, m2 = u1 - v1_3 * h, u2 - v2_3 * h
        per[:, h - 1, :, 0] = m1
        per[:, h - 1, :, 1] = m2
        per[:, h - 1, :, 2] = ((m1 < 0) | (m2 < 0)).astype(np.float32)
        per[:, h - 1, :, 3] = np.take_along_axis(st["pqc"][:, h - 1], np.clip(tail, 0, S - 1), 1)
    return stat, per


def build(ctx, length, tq, v, qf, ok, t0, grp, key, wd):
    names, Xh, base = ongoing_Xs(ctx, v, qf, ok, t0, grp, zero_prior(ctx), key, wd, tq, True)
    s1, p1 = shock_feats(ctx, length, v, qf, ok)
    s2, p2 = two_sided(ctx, v, qf, ok, tq, t0, wd)
    bc = lambda a: np.broadcast_to(a[:, None], (a.shape[0], F) + a.shape[1:])  # noqa: E731
    E1 = np.concatenate([bc(s1), p1], -1)
    E2 = np.concatenate([bc(s2), p2], -1)
    return names, Xh, E1, E2, base


def main() -> None:
    t0_all = time.time()
    parts: list = []
    ctxs, tqs, lens, names_og = [], {}, {}, None
    for i, panel in enumerate(T2_PANELS):
        ctx = Ctx(panel, i)
        ctx.fitd = np.ones_like(ctx.fitd)
        z = ctx.z
        tqs[panel] = load_tq(panel)
        lens[panel] = fd_params(panel, z["link_ids"])["length"].astype(np.float32)
        c = candidates(z)
        d, t0 = c["d"][c["ongoing"]], c["t0"][c["ongoing"]]
        xs, e1s, e2s, ys = [], [], [], []
        for j in range(0, len(d), CHUNK):
            dd, tt = d[j:j + CHUNK], t0[j:j + CHUNK]
            v, qf, ok = hist_from_truth(ctx, dd, tt)
            names_og, Xh, E1, E2, base = build(ctx, lens[panel], tqs[panel], v, qf, ok, tt, ctx.g[dd], dd,
                                               z["weekday"][dd])
            y = ctx.q[dd[:, None], tt[:, None] + np.arange(1, F + 1)]
            e = ctx.el[dd[:, None], tt[:, None] + np.arange(1, F + 1)]
            near = np.minimum(base["dist_slow_up"], base["dist_slow_dn"]) <= 10
            keep = np.flatnonzero(e & (np.broadcast_to(near[:, None, :], y.shape) | y | (rng.random(y.shape) < 0.02)))
            xs.append(Xh.reshape(-1, Xh.shape[-1])[keep].astype(np.float32))
            e1s.append(E1.reshape(-1, E1.shape[-1])[keep].astype(np.float32))
            e2s.append(E2.reshape(-1, E2.shape[-1])[keep].astype(np.float32))
            ys.append(y.reshape(-1)[keep])
        X, E1, E2, y = np.concatenate(xs), np.concatenate(e1s), np.concatenate(e2s), np.concatenate(ys)
        if len(y) > MAX_OG:
            k = rng.choice(len(y), MAX_OG, replace=False)
            X, E1, E2, y = X[k], E1[k], E2[k], y[k]
        parts.append((X, E1, E2, y))
        ctx.z = {k: z[k] for k in ("link_ids", "weekday", "dates")}
        ctx.q = ctx.el = None
        ctxs.append(ctx)
        print(f"[{panel}] {len(y)} ongoing rows ({time.time() - t0_all:.0f}s)", flush=True)
    y_all = np.concatenate([p[3] for p in parts])
    models = {}
    for var in ("base", "v1", "v2"):
        cols = {"base": (0,), "v1": (0, 1), "v2": (0, 1, 2)}[var]
        X_all = np.concatenate([np.concatenate([p[c] for c in cols], 1) for p in parts])
        nm = names_og + (SW if var != "base" else []) + (S2 if var == "v2" else [])
        for sd in SEEDS:
            m = lgb.train(dict(PARAMS, num_leaves=LEAVES, seed=sd), lgb.Dataset(X_all, y_all, feature_name=nm), ROUNDS)
            models[(var, sd)] = m
            if var == "v2" and sd == SEEDS[0]:
                imp = m.feature_importance("gain")
                print(f"  v2: shockwave share {imp[len(names_og):len(names_og) + len(SW)].sum() / imp.sum():.3f}, "
                      f"two-sided share {imp[-len(S2):].sum() / imp.sum():.3f}; top: "
                      + ", ".join(f"{nm[j]} {imp[j] / imp.sum():.3f}" for j in np.argsort(-imp)[:6]), flush=True)
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
            ps: dict = {}
            for j in range(0, n, CHUNK):
                sl = slice(j, j + CHUNK)
                grp = (z["weekday"][og["d"][sl]] >= 5).astype(int)
                key = np.stack([og["d"][sl], grp], 1)
                _, Xh, E1, E2, _ = build(ctx, lens[ctx.panel], tqs[ctx.panel], og["v"][sl], og["qf"][sl], og["ok"][sl],
                                         og["t0"][sl], grp, key, z["weekday"][og["d"][sl]].astype(int))
                mats = {"base": Xh, "v1": np.concatenate([Xh, E1], -1), "v2": np.concatenate([Xh, E1, E2], -1)}
                for (var, sd), m in models.items():
                    XX = mats[var]
                    p = m.predict(XX.reshape(-1, XX.shape[-1])).reshape(XX.shape[:3])
                    ps.setdefault((var, sd), []).append(p)
            for (var, sd), pl in ps.items():
                p = np.concatenate(pl)
                for tau in TAUS:
                    for mode, pp in (("alone", p), ("M-blend", W_L * p + (1 - W_L) * cn)):
                        pr = pp >= tau
                        r = res.setdefault((mode, tau, var, sd), {})
                        r.setdefault(split, []).append(iou(pr, truth, elig))
                        if tau == TAUS[0] and mode == "alone":
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
                line = f"  {mode:<8} tau {tau:.2f}: base {np.mean([x.mean() for x in b]):.4f}"
                for var in ("v1", "v2"):
                    s = [np.concatenate(res[(mode, tau, var, sd)][split]) for sd in SEEDS]
                    d = [x - y for x, y in zip(s, b)]
                    line += f" | {var} {np.mean([x.mean() for x in s]):.4f} {ci(np.mean(d, 0))}"
                print(line, flush=True)
        print("  error rate by class at tau 0.60, alone (base -> v1 -> v2):")
        for cname in ("grow", "dissolve", "stay", "far"):
            vals = []
            for var in ("base", "v1", "v2"):
                vals.append(np.mean([res[("alone", TAUS[0], var, sd)][f"{split}_{cname}"] /
                                     max(res[("alone", TAUS[0], var, sd)][f"{split}_{cname}_n"], 1) for sd in SEEDS]))
            print(f"    {cname:<9} {vals[0]:.4f} -> {vals[1]:.4f} -> {vals[2]:.4f}  (v2 vs base {vals[2] - vals[0]:+.4f})")
    print(f"total {time.time() - t0_all:.0f}s")


if __name__ == "__main__":
    main()
