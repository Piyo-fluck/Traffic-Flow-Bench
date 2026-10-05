"""Task 2 ONGOING: a specialist for the +-4 link band around the queue visible at T (the user's track (1), revised).

Note 61 measured the ceiling of the boundary formulation and corrected its own first attempt:
    93% of the true queue cells sit within 4 links of the queue visible at T (d0 84%, d1-2 7%, d3-4 2.5%);
    ORACLE "every true cell within 4 links, nothing beyond" = 0.930 vs R 0.850, i.e. +0.0121 total;
    but ORACLE d0 alone = 0.815, BELOW R, so boundary tracking must NOT replace the per-cell model.

So this is not a replacement but a band specialist: train only on the cells within +-4 links of the T queue, give
the model the boundary geometry explicitly (distance to the queue, inside/outside, both sides' measured speeds and
the Rankine-Hugoniot projection from notes 54), and blend it into R's probability ON THAT BAND ONLY. R keeps the
interior of the queue and the far field, where note 61 says it is already at or beyond the boundary oracle.

    base features   production I configuration (demand stress, no priors)
    + SW            the one-sided shockwave block (note 54 v1)
    + S2            the two-sided block (note 54 v2: each side's distance, speed, run length, flows across it)
    + band geometry b_dist (links to the nearest queued link at T), b_in (queued at T), b_step (horizon index)

Evaluated on the blank-aware official-like ongoing windows against R (0.25 gs1 + 0.50 tfm x4 + 0.25 cnn x3,
tau 0.45), blending weights on the band only, with paired bootstrap CIs in both months, plus the note-52 error
classes restricted to the band.

    .venv/Scripts/python.exe analysis/t2_ongoing_band_specialist.py
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
from t2_demand_stress import load_tq  # noqa: E402
from t2_eda02 import T2_PANELS, candidates  # noqa: E402
from t2_eda03 import F  # noqa: E402
from t2_first_onset_eval import official_blank  # noqa: E402
from t2_m1_models import Ctx, hist_from_truth  # noqa: E402
from t2_official_like_eval import ongoing_official  # noqa: E402
from t2_ongoing_boundary_oracle import CNN3, TFM4, group_mean, last_state  # noqa: E402
from t2_ongoing_shockwave import SW, nearest_true  # noqa: E402
from t2_ongoing_shockwave2 import CHUNK, LEAVES, MAX_OG, ROUNDS, S2, SEEDS, build  # noqa: E402
from t2_pseudo_eval import iou  # noqa: E402
from t2_shift_remedies import PARAMS  # noqa: E402
from tfb.data import CACHE, load  # noqa: E402
from tfb.physics import fd_params  # noqa: E402

BAND = 4                      # the band the specialist owns, in links
TAUS = (0.40, 0.45, 0.50)
WEIGHTS = (1.0, 0.7, 0.5, 0.3, 0.0)     # weight on R inside the band (1.0 = R unchanged)
GEO = ["b_dist", "b_in", "b_step"]
rng = np.random.default_rng(11)
boot = np.random.default_rng(29)


def ci(delta: np.ndarray, n_boot: int = 2000) -> str:
    idx = boot.integers(0, len(delta), (n_boot, len(delta)))
    m = delta[idx].mean(1)
    return f"{delta.mean():+.4f} [{np.percentile(m, 2.5):+.4f}, {np.percentile(m, 97.5):+.4f}]"


def geometry(base: np.ndarray, shape) -> tuple[np.ndarray, np.ndarray]:
    """(N, F, S, 3) band-geometry columns and the (N, F, S) band mask."""
    d, _ = nearest_true(base)
    d3 = np.broadcast_to(d[:, None], shape).astype(np.float32)
    in3 = np.broadcast_to(base[:, None], shape).astype(np.float32)
    st3 = np.broadcast_to(np.arange(1, F + 1, dtype=np.float32)[None, :, None], shape)
    return np.stack([d3, in3, st3], -1), d3 <= BAND


def main() -> None:
    t0 = time.time()
    parts, ctxs, tqs, lens, names = [], [], {}, {}, None
    for i, panel in enumerate(T2_PANELS):
        ctx = Ctx(panel, i)
        ctx.fitd = np.ones_like(ctx.fitd)
        z = ctx.z
        tqs[panel], lens[panel] = load_tq(panel), fd_params(panel, z["link_ids"])["length"].astype(np.float32)
        c = candidates(z)
        d, t0w = c["d"][c["ongoing"]], c["t0"][c["ongoing"]]
        xs, ys = [], []
        for j in range(0, len(d), CHUNK):
            dd, tt = d[j:j + CHUNK], t0w[j:j + CHUNK]
            v, qf, ok = hist_from_truth(ctx, dd, tt)
            nm, Xh, E1, E2, _ = build(ctx, lens[panel], tqs[panel], v, qf, ok, tt, ctx.g[dd], dd, z["weekday"][dd])
            names = nm + SW + S2 + GEO
            vis_q = ok & (np.nan_to_num(v, nan=1e9) <= 0.6 * ctx.vf)
            base = last_state(vis_q, ok)
            G, band = geometry(base, (len(dd), F, ctx.S))
            X = np.concatenate([Xh, E1, E2, G], -1)
            y = ctx.q[dd[:, None], tt[:, None] + np.arange(1, F + 1)]
            e = ctx.el[dd[:, None], tt[:, None] + np.arange(1, F + 1)]
            keep = np.flatnonzero(e & band)                    # the band owns the cells; no extra sampling
            xs.append(X.reshape(-1, X.shape[-1])[keep].astype(np.float32))
            ys.append(y.reshape(-1)[keep])
        X, y = np.concatenate(xs), np.concatenate(ys)
        if len(y) > MAX_OG:
            k = rng.choice(len(y), MAX_OG, replace=False)
            X, y = X[k], y[k]
        parts.append((X, y))
        ctx.z = {k: z[k] for k in ("link_ids", "weekday", "dates")}
        ctx.q = ctx.el = None
        ctxs.append(ctx)
        print(f"[{panel}] band rows {len(y)} (positives {y.mean():.3f}) ({time.time() - t0:.0f}s)", flush=True)
    X_all = np.concatenate([p[0] for p in parts]); y_all = np.concatenate([p[1] for p in parts])
    del parts
    models = [lgb.train(dict(PARAMS, num_leaves=LEAVES, seed=sd), lgb.Dataset(X_all, y_all, feature_name=names), ROUNDS)
              for sd in SEEDS]
    imp = models[0].feature_importance("gain")
    print(f"trained {len(models)} seeds ({time.time() - t0:.0f}s); band geometry share {imp[-len(GEO):].sum() / imp.sum():.3f}, "
          f"boundary blocks {imp[len(names) - len(GEO) - len(SW) - len(S2):len(names) - len(GEO)].sum() / imp.sum():.3f}; top: "
          + ", ".join(f"{names[j]} {imp[j] / imp.sum():.3f}" for j in np.argsort(-imp)[:6]), flush=True)
    del X_all, y_all
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
            g = np.load(CACHE / f"t2gs_{split}_{ctx.panel}.npz")
            pR = 0.25 * g["og_p1"] + 0.50 * group_mean(TFM4, split, ctx.panel) + 0.25 * group_mean(CNN3, split, ctx.panel)
            truth, elig = og["fut"], og["efut"]
            vis_q = og["ok"] & (np.nan_to_num(og["v"], nan=1e9) <= 0.6 * ctx.vf)
            base = last_state(vis_q, og["ok"])
            G, band = geometry(base, truth.shape)
            p_sp = np.zeros(truth.shape, np.float32)
            for j in range(0, n, CHUNK):
                sl = slice(j, j + CHUNK)
                grp = (z["weekday"][og["d"][sl]] >= 5).astype(int)
                key = np.stack([og["d"][sl], grp], 1)
                _, Xh, E1, E2, _ = build(ctx, lens[ctx.panel], tqs[ctx.panel], og["v"][sl], og["qf"][sl], og["ok"][sl],
                                         og["t0"][sl], grp, key, z["weekday"][og["d"][sl]].astype(int))
                XX = np.concatenate([Xh, E1, E2, G[sl]], -1)
                m = band[sl]
                flat = XX.reshape(-1, XX.shape[-1])[m.reshape(-1)]
                if len(flat):
                    p = np.mean([mm.predict(flat) for mm in models], 0)
                    buf = np.zeros(m.size, np.float32)
                    buf[m.reshape(-1)] = p
                    p_sp[sl] = buf.reshape(m.shape)
            for w in WEIGHTS:
                pp = np.where(band, w * pR + (1 - w) * p_sp, pR)
                for tau in TAUS:
                    r = res.setdefault((w, tau), {})
                    r.setdefault(split, []).append(iou(pp >= tau, truth, elig))
                    if tau == 0.45:
                        pr = (pp >= tau) & elig
                        err = ((pr & ~truth) | (truth & ~pr)) & band
                        r[f"{split}_err"] = r.get(f"{split}_err", 0) + int(err.sum())
                        r[f"{split}_n"] = r.get(f"{split}_n", 0) + int((band & elig).sum())
        print(f"  scored {ctx.panel} ({time.time() - t0:.0f}s)", flush=True)
    for split in ("validation", "private"):
        print(f"\n=== {split}: ongoing IoU, specialist blended on the +-{BAND} link band only ===")
        base_iou = np.concatenate(res[(1.0, 0.45)][split])
        for w in WEIGHTS:
            cells = []
            for tau in TAUS:
                v = np.concatenate(res[(w, tau)][split])
                cells.append(f"tau {tau:.2f}: {v.mean():.4f} {ci(v - base_iou)}")
            er = res[(w, 0.45)][f"{split}_err"] / max(res[(w, 0.45)][f"{split}_n"], 1)
            print(f"  R weight {w:.1f} (band error rate {er:.4f})  " + " | ".join(cells), flush=True)
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
