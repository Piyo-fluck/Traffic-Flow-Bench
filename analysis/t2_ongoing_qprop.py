"""Task 2 ongoing: add queue-propagation features (when will the queue reach / leave each link?).

In an ongoing window the queue head sits at a bottleneck and the tail moves upstream or downstream as a shock
wave. The current ongoing features see only the queue's position (distance to the nearest slow link in the last
slots) and speed trends, not how fast the queue front is moving. Per window and link, from the last 6 history
slots of visible queue flags (v <= 0.6 vf):
    dq_dn / dq_up        links to the nearest queued link downstream / upstream at T-5
    slope_dn             how fast dq_dn shrinks (links per slot): a queue approaching from downstream
    eta_dn               slots until it arrives, dq_dn / slope_dn (capped at 99)
    tail_up              for a queued link: links to the nearest visible free link upstream (distance to the tail)
    slope_tail, eta_clear  how fast that distance shrinks, and the slots until the link clears
and per horizon step h: arrive_h = eta_dn <= h+1, clear_h = eta_clear <= h+1, and the margins h+1-eta.
Two no-prior stress ongoing models (= I) are trained on identical sampled rows, without and with these features,
and scored on the blank-aware official-like ongoing windows (tau 0.65). Days 1-14 and all, validation and
private, family mean.

    py analysis/t2_ongoing_qprop.py
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
from t2_eda02 import T2_PANELS, candidates  # noqa: E402
from t2_eda03 import F, H  # noqa: E402
from t2_first_onset_eval import official_blank  # noqa: E402
from t2_m1_models import Ctx  # noqa: E402
from t2_m1_predict import feats_og, zero_prior  # noqa: E402
from t2_official_like_eval import ongoing_official  # noqa: E402
from t2_pseudo_eval import iou  # noqa: E402
from t2_shift_remedies import PARAMS  # noqa: E402
from tfb.data import load  # noqa: E402
from tfb.features import neighbors_along  # noqa: E402
from tfb.t1eval import family_mean  # noqa: E402

MAX_ROWS, K_SLOTS = 400_000, 6
DAY_SETS = {"d1-14": 14, "all": 31}
QP_NAMES = ["qp_dq_dn", "qp_dq_up", "qp_slope_dn", "qp_eta_dn", "qp_tail_up", "qp_slope_tail", "qp_eta_clear",
            "qp_arrive_h", "qp_clear_h", "qp_arrive_margin", "qp_clear_margin"]
rng = np.random.default_rng(7)


def fmean(d):
    d = {k: v for k, v in d.items() if np.isfinite(v)}
    return family_mean(d) if d else float("nan")


def qp_features(vf, v, ok) -> np.ndarray:
    """(N, H, S) history -> (N, F, S, len(QP_NAMES)) queue-propagation features."""
    vq = ok & (np.nan_to_num(v, nan=1e9) <= 0.6 * vf)
    free = ok & ~vq
    N, _, S = v.shape
    pos = np.broadcast_to(np.arange(S, dtype=np.float32), (N, S))
    dn, up, tail = [], [], []
    for t in range(H - K_SLOTS, H):
        m, fr = vq[:, t], free[:, t]
        _, _, dp, dnn = neighbors_along(np.where(m, pos, np.nan), m, axis=1)
        _, _, dpf, _ = neighbors_along(np.where(fr, pos, np.nan), fr, axis=1)
        dn.append(np.nan_to_num(dnn, nan=99.0)); up.append(np.nan_to_num(dp, nan=99.0))
        tail.append(np.where(m, np.nan_to_num(dpf, nan=99.0), 0.0))
    dn, up, tail = np.stack(dn, 1), np.stack(up, 1), np.stack(tail, 1)       # (N, K, S)
    slope_dn = np.clip((dn[:, 0] - dn[:, -1]) / (K_SLOTS - 1), -5, 5)
    slope_tail = np.clip((tail[:, 0] - tail[:, -1]) / (K_SLOTS - 1), -5, 5)
    eta_dn = np.where(dn[:, -1] == 0, 0.0, np.where(slope_dn > 0.05, dn[:, -1] / np.maximum(slope_dn, 0.05), 99.0))
    queued = vq[:, -1]
    eta_clear = np.where(queued & (slope_tail > 0.05), tail[:, -1] / np.maximum(slope_tail, 0.05), 99.0)
    eta_dn, eta_clear = np.minimum(eta_dn, 99.0), np.minimum(eta_clear, 99.0)
    hh = np.arange(1, F + 1, dtype=np.float32)[None, :, None]                   # h+1 in slots
    per_w = [dn[:, -1], up[:, -1], slope_dn, eta_dn, tail[:, -1], slope_tail, eta_clear]
    E = [np.broadcast_to(x[:, None, :], (N, F, S)) for x in per_w]
    E += [(eta_dn[:, None, :] <= hh).astype(np.float32), (eta_clear[:, None, :] <= hh).astype(np.float32),
          hh - eta_dn[:, None, :], hh - eta_clear[:, None, :]]
    return np.stack(E, -1).astype(np.float32)


def main() -> None:
    t_start = time.time()
    Xa, Xb, ys, ctxs, names = [], [], [], [], None
    for i, p in enumerate(T2_PANELS):
        ctx = Ctx(p, i)
        ctx.fitd = np.ones_like(ctx.fitd)
        z = ctx.z
        c = candidates(z)
        d, t0 = c["d"][c["ongoing"]], c["t0"][c["ongoing"]]
        pa, pb, yl = [], [], []
        for j in range(0, len(d), 400):
            dd, tt = d[j:j + 400], t0[j:j + 400]
            hi = tt[:, None] + np.arange(-H, 0)
            v, qf, ok = z["true_speed"][dd[:, None], hi], z["true_flow"][dd[:, None], hi], ctx.el[dd[:, None], hi]
            n, Xh, base = feats_og(ctx, True, v, qf, ok, tt, ctx.g[dd], zero_prior(ctx), dd, z["weekday"][dd])
            names = n
            y = ctx.q[dd[:, None], tt[:, None] + np.arange(1, F + 1)]
            e = ctx.el[dd[:, None], tt[:, None] + np.arange(1, F + 1)]
            near = np.minimum(base["dist_slow_up"], base["dist_slow_dn"]) <= 10
            keep = np.flatnonzero(e & (np.broadcast_to(near[:, None, :], y.shape) | y | (rng.random(y.shape) < 0.02)))
            qp = qp_features(ctx.vf, v, ok)
            pa.append(Xh.reshape(-1, Xh.shape[-1])[keep]); pb.append(qp.reshape(-1, qp.shape[-1])[keep])
            yl.append(y.reshape(-1)[keep])
        yp = np.concatenate(yl)
        sub = rng.choice(len(yp), MAX_ROWS, replace=False) if len(yp) > MAX_ROWS else np.arange(len(yp))
        Xa.append(np.concatenate(pa)[sub]); Xb.append(np.concatenate(pb)[sub]); ys.append(yp[sub])
        ctx.z = {k: z[k] for k in ("link_ids", "weekday")}
        ctx.q = ctx.el = None
        ctxs.append(ctx)
        print(f"[{p}] rows {len(sub):,} ({time.time() - t_start:.0f}s)", flush=True)
    y = np.concatenate(ys)
    A = np.concatenate(Xa)
    m_base = lgb.train(dict(PARAMS, num_leaves=127), lgb.Dataset(A, y, feature_name=names), 3300)
    print(f"trained base (stress) ({time.time() - t_start:.0f}s)", flush=True)
    B = np.concatenate([A, np.concatenate(Xb)], 1)
    del A, Xa
    m_qp = lgb.train(dict(PARAMS, num_leaves=127), lgb.Dataset(B, y, feature_name=names + QP_NAMES), 3300)
    imp = m_qp.feature_importance("gain")
    top = ", ".join(f"{(names + QP_NAMES)[j]} {imp[j] / imp.sum():.3f}" for j in np.argsort(-imp)[:8])
    print(f"trained +qprop ({time.time() - t_start:.0f}s); top gain: {top}", flush=True)
    del B, Xb
    res: dict = {}
    for ctx in ctxs:
        for split in ("validation", "private"):
            z = load(ctx.panel, split)
            og, _ = ongoing_official(ctx, z, official_blank(ctx.panel, split, z))
            if not len(og["d"]):
                continue
            grp = (z["weekday"][og["d"]] >= 5).astype(int)
            key = np.stack([og["d"], grp], 1)
            wdo = z["weekday"][og["d"]].astype(int)
            sa, sb = [], []
            for j in range(0, len(og["d"]), 200):
                b = slice(j, j + 200)
                _, Xh, _ = feats_og(ctx, True, og["v"][b], og["qf"][b], og["ok"][b], og["t0"][b], grp[b], zero_prior(ctx), key[b], wdo[b])
                qp = qp_features(ctx.vf, og["v"][b], og["ok"][b])
                pa = m_base.predict(Xh.reshape(-1, Xh.shape[-1])).reshape(Xh.shape[:3])
                pb = m_qp.predict(np.concatenate([Xh, qp], -1).reshape(-1, Xh.shape[-1] + qp.shape[-1])).reshape(Xh.shape[:3])
                sa.append(iou(pa >= 0.65, og["fut"][b], og["efut"][b])); sb.append(iou(pb >= 0.65, og["fut"][b], og["efut"][b]))
            sa, sb = np.concatenate(sa), np.concatenate(sb)
            for ds, dmax in DAY_SETS.items():
                m = og["d"] < dmax
                r = res.setdefault((split, ds), {"base": {}, "qprop": {}})
                r["base"][ctx.panel] = sa[m].mean() if m.any() else np.nan
                r["qprop"][ctx.panel] = sb[m].mean() if m.any() else np.nan
        print(f"  scored {ctx.panel} ({time.time() - t_start:.0f}s)", flush=True)
    for (split, ds), r in res.items():
        a, b = fmean(r["base"]), fmean(r["qprop"])
        diffs = [r["qprop"][p] - r["base"][p] for p in r["base"] if np.isfinite(r["base"][p]) and np.isfinite(r["qprop"][p])]
        print(f"== {split} {ds}: ongoing IoU base {a:.3f} -> +qprop {b:.3f} ({b - a:+.3f}; better/worse "
              f"{sum(x > 1e-9 for x in diffs)}/{sum(x < -1e-9 for x in diffs)})")
    print(f"total {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
