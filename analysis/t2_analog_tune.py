"""Task 2 onset: improve the state-conditioned analog map itself (note 47). No model re-training: each map variant is
scored with the probabilities already cached for the same blank-aware official-like onset windows.

Scores per variant (onset IoU, val d1-14 / val all / priv d1-14 / priv all, family mean):
    map masked     the map alone, masked histories
    map filled     the map alone, T1-filled (official-like) histories
    M+map          0.7 * M + 0.3 * map on masked histories, M = 0.7 H + 0.3 CNN (production probabilities)
    base+map (f)   0.7 * LGBM + 0.3 * map on filled histories (the re-trained H of t2_analog_prior, cached)
Baseline variant 'ref' = note 47's map: pca 8 on the last-30-min log flow ratio, time of day, K 20, alpha 2, library =
all train earliest-onset windows of the same weekday type, shrunk to the library frequency. Variants change one thing.

    .venv/Scripts/python.exe analysis/t2_analog_tune.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "scripts"))
from t2_analog_prior import filled, log_ratio, rehist, train_profile  # noqa: E402
from t2_eda02 import T2_PANELS  # noqa: E402
from t2_eda03 import F, H  # noqa: E402
from t2_expected_iou import select  # noqa: E402
from t2_first_onset_eval import first_onsets, official_blank, smooth_links  # noqa: E402
from t2_m1_models import Ctx, hist_from_truth  # noqa: E402
from t2_m1_predict import train_tables  # noqa: E402
from t2_pseudo_eval import iou  # noqa: E402
from tfb.data import CACHE, load  # noqa: E402
from tfb.t1eval import family_mean  # noqa: E402

TAGS = ["t2_cnn_lov", "t2_cnn_lov_s1", "t2_cnn_lov_s2"]
REF = dict(npc=8, key="flow6", k=20, alpha=2.0, lib="all", tod="feat", smooth=False, shrink="freq", tw=1.0)
VARIANTS = {
    "ref": {},
    "K10": dict(k=10), "K40": dict(k=40),
    "alpha0.5": dict(alpha=0.5), "alpha5": dict(alpha=5.0),
    "pca4": dict(npc=4), "pca16": dict(npc=16),
    "flow12 (hour)": dict(key="flow12"), "flow+speed": dict(key="flowspeed"),
    "lib morning": dict(lib="morning"),
    "tod window 1h": dict(tod="window"), "tod weight 3": dict(tw=3.0),
    "smooth map": dict(smooth=True),
    "shrink to tod prior": dict(shrink="table"),
}
CONDS = [("validation", 14), ("validation", 31), ("private", 14), ("private", 31)]
# Sweep 2 (`... t2_analog_tune.py sweep2`): base = sweep 1's best (K 40); structural changes to the key and kernel.
VARIANTS2 = {
    "ref": {},
    "K60": dict(k=60), "K80": dict(k=80),
    "bw x0.5": dict(bw=0.5), "bw x2": dict(bw=2.0),
    "tod weight 0.5": dict(tw=0.5), "tod weight 3": dict(tw=3.0),
    "xcov4": dict(proj="xcov", npc=4), "xcov8": dict(proj="xcov", npc=8), "xcov16": dict(proj="xcov", npc=16),
    "xcov8 flow+speed": dict(proj="xcov", npc=8, key="flowspeed"),
    "xcov8 K60": dict(proj="xcov", npc=8, k=60),
}
if len(sys.argv) > 1 and sys.argv[1] == "sweep2":
    REF, VARIANTS = dict(REF, k=40), VARIANTS2


def fmean(d):
    d = {k: v for k, v in d.items() if np.isfinite(v)}
    return family_mean(d) if d else float("nan")


def state(ctx, prof, v, qf, ok, t0, grp, kind):
    """(N, D) raw state vector before PCA."""
    r = log_ratio(prof, qf, ok, t0, grp)
    with np.errstate(all="ignore"):
        parts = [np.nanmean(r[:, -6:], 1)]
        if kind == "flow12":
            parts.append(np.nanmean(r[:, :6], 1))
        if kind == "flowspeed":
            parts.append(np.nanmean(np.where(ok, v / ctx.vf, np.nan)[:, -6:], 1) - 1.0)
    return np.nan_to_num(np.concatenate(parts, 1), nan=0.0).astype(np.float32)


def build(cfg, lib_raw, lib_t0, npc, lib_y=None):
    """Key function. proj 'pca': principal directions of the state. proj 'xcov': supervised directions, the leading
    left singular vectors of X^T Y (state x onset map cross-covariance; a PLS-style reduced-rank key) -- the directions
    of today's demand state that best explain WHERE the onset happens in the library."""
    mu = lib_raw.mean(0)
    if cfg.get("proj", "pca") == "xcov":
        u, _, _ = np.linalg.svd((lib_raw - mu).T @ (lib_y - lib_y.mean(0)), full_matrices=False)
        comps = u[:, :npc].T
    else:
        _, _, vt = np.linalg.svd(lib_raw - mu, full_matrices=False)
        comps = vt[:npc]
    kf = lambda X, t0: np.concatenate([(X - mu) @ comps.T, (t0 / 12.0)[:, None]], 1)  # noqa: E731
    return kf


def amap(cfg, kf, L, q_raw, q_t0, q_grp, q_shrink):
    lk = kf(L["raw"], L["t0"])
    wv = np.ones(lk.shape[1], np.float32)
    wv[-1] = cfg["tw"]                       # time-of-day weight, applied AFTER standardisation (sweep-1 bug fixed)
    sd = (lk.std(0) + 1e-6) / wv
    qk = kf(q_raw, q_t0)
    out = np.empty((len(q_raw), L["y"].shape[1]), np.float32)
    for n in range(len(q_raw)):
        ok = L["grp"] == q_grp[n]
        if cfg["lib"] == "morning":
            ok &= (L["t0"] >= 48) & (L["t0"] < 168)
        if cfg["tod"] == "window":
            ok &= np.abs(L["t0"] - q_t0[n]) <= 12
        idx = np.flatnonzero(ok)
        if len(idx) < 3:
            idx = np.flatnonzero(L["grp"] == q_grp[n])
        d = np.sqrt((((lk[idx] - qk[n]) / sd) ** 2).sum(1))
        o = np.argsort(d)[:cfg["k"]]
        dn = d[o]
        w = np.exp(-(dn / max(cfg.get("bw", 1.0) * float(np.median(dn)), 1e-6)) ** 2)
        out[n] = (w @ L["y"][idx[o]] + cfg["alpha"] * q_shrink[n]) / (w.sum() + cfg["alpha"])
    if cfg["smooth"]:
        out = smooth_links(out) / 2.0
    return out


def score(p, fut, efut):
    pr = np.zeros(fut.shape, bool)
    pr[:, F - 1] = np.stack([select(q, "ratio", 1.0) for q in p])
    return iou(pr, fut, efut)


def main() -> None:
    t_start = time.time()
    res: dict = {}
    for i, panel in enumerate(T2_PANELS):
        ctx = Ctx(panel, i)
        ctx.fitd = np.ones_like(ctx.fitd)
        d, t0 = ctx.eo
        v, qf, ok = hist_from_truth(ctx, d, t0)
        y = ctx.q[d, t0 + F].astype(np.float32)
        grp = ctx.g[d].astype(int)
        prof = train_profile(ctx)
        tab_on = train_tables(ctx)["on"]                                         # (2, T, S) onset-start prior
        pg = y.mean(0)
        lib_raw = {k: state(ctx, prof, v, qf, ok, t0, grp, k) for k in ("flow6", "flow12", "flowspeed")}
        queries = {}
        for split in ("validation", "private"):
            z = load(panel, split)
            on = first_onsets(ctx.vf, z, blank=official_blank(panel, split, z))
            if not len(on["d"]):
                continue
            c = np.load(CACHE / f"t2probs_{split}_{panel}.npz")
            assert (c["on_d"] == on["d"]).all()
            m_prob = 0.7 * c["on_p"] + 0.3 * np.mean([np.load(CACHE / f"t2cnn_{t}_{split}_{panel}.npz")["on_p"] for t in TAGS], 0)
            fb = np.load(CACHE / f"t2analog_filled_{split}_{panel}.npz")
            qgrp = (z["weekday"][on["d"]] >= 5).astype(int)
            hist = {"masked": (on["v"], on["qf"], on["ok"]), "filled": rehist(filled(z, panel, split), on["d"], on["t0"])}
            queries[split] = (on, qgrp, m_prob, fb["p_base"], hist)
        for name, change in VARIANTS.items():
            cfg = dict(REF, **change)
            kf = build(cfg, lib_raw[cfg["key"]], t0, cfg["npc"], lib_y=y)
            L = dict(raw=lib_raw[cfg["key"]], t0=t0, y=y, grp=grp)
            for split, (on, qgrp, m_prob, p_base_f, hist) in queries.items():
                for hname, (hv, hq, ho) in hist.items():
                    q_raw = state(ctx, prof, hv, hq, ho, on["t0"], qgrp, cfg["key"])
                    if cfg["shrink"] == "table":
                        tp = tab_on[qgrp, np.clip(on["t0"] + F, 0, tab_on.shape[1] - 1)]
                        shrink = tp / np.maximum(tp.sum(1, keepdims=True), 1e-9) * pg.sum()
                    else:
                        shrink = np.broadcast_to(pg, (len(on["d"]), ctx.S))
                    mp = amap(cfg, kf, L, q_raw, on["t0"], qgrp, shrink)
                    outs = {f"map {hname}": mp}
                    if hname == "masked":
                        outs["M+map"] = 0.7 * m_prob + 0.3 * mp
                    else:
                        outs["base+map (f)"] = 0.7 * p_base_f + 0.3 * mp
                    for metric, p in outs.items():
                        s = score(p, on["fut"], on["efut"])
                        for sp, dmax in CONDS:
                            if sp == split:
                                m = on["d"] < dmax
                                res.setdefault((name, metric), {}).setdefault((sp, dmax), {})[panel] = s[m].mean() if m.any() else np.nan
            # references without any map, once per panel
            if name == "ref":
                for split, (on, qgrp, m_prob, p_base_f, hist) in queries.items():
                    for metric, p in (("M+map", m_prob), ("base+map (f)", p_base_f)):
                        s = score(p, on["fut"], on["efut"])
                        for sp, dmax in CONDS:
                            if sp == split:
                                m = on["d"] < dmax
                                res.setdefault(("(no map)", metric), {}).setdefault((sp, dmax), {})[panel] = s[m].mean() if m.any() else np.nan
        print(f"[{panel}] ({time.time() - t_start:.0f}s)", flush=True)
    for metric in ("map masked", "map filled", "M+map", "base+map (f)"):
        print(f"\n== {metric}: val d1-14 | val all | priv d1-14 | priv all  (delta vs ref) ==")
        ref = res[("ref", metric)]
        names = (["(no map)"] if ("(no map)", metric) in res else []) + list(VARIANTS)
        rows = []
        for name in names:
            r = res[(name, metric)]
            vals = [fmean(r[c]) for c in CONDS]
            dl = [fmean(r[c]) - fmean(ref[c]) for c in CONDS]
            rows.append((name, vals, dl))
        for name, vals, dl in rows:
            print(f"  {name:<20} " + " | ".join(f"{v:.3f} ({x:+.3f})" for v, x in zip(vals, dl)) + f"  mean {np.mean(vals):.4f}")
    print(f"total {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
