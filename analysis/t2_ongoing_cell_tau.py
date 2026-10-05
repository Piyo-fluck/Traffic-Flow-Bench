"""Task 2 ONGOING: a per-CELL decision threshold, keyed on what is known at prediction time.

Note 52: 70% of the ongoing errors are missed cells, the largest bucket is queue GROWTH (33-40% of the cells the
queue newly reaches are missed), and the last two horizon steps carry 43-45% of the loss. A single threshold per
window (note 53, learnable gain +0.014...+0.018) cannot express that: it moves every cell of the window together.

Here the threshold depends on three things known at prediction time:
    h     the horizon step (T+5 .. T+30)
    base  whether the link is queued at T in the visible history
    dist  distance in links to the nearest edge of the queue visible at T (0, 1, 2, 3-4, 5+)
"grow" cells are exactly `base = 0` with a small `dist`, so a lower threshold there directly buys back the missed
growth, and a higher threshold on old interior cells keeps dissolution in check. 60 groups in total.

Thresholds are fitted by coordinate ascent on the mean window IoU of ONE split and applied to the OTHER (both
directions), so every number is out-of-sample. Decomposition of the rule: h only / (base, dist) only / full.
A production version must be fitted on the train run; this measures whether it is worth building.

    .venv/Scripts/python.exe analysis/t2_ongoing_cell_tau.py
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
from t2_eda02 import T2_PANELS  # noqa: E402
from t2_eda03 import F  # noqa: E402
from t2_first_onset_eval import official_blank  # noqa: E402
from t2_m1_models import Ctx  # noqa: E402
from t2_official_like_eval import ongoing_official  # noqa: E402
from t2_pseudo_eval import iou  # noqa: E402
from tfb.data import CACHE, load  # noqa: E402

TAGS = ["t2_cnn_lov", "t2_cnn_lov_s1", "t2_cnn_lov_s2"]
TAU, W_L = 0.60, 0.5
TAUS = np.round(np.arange(0.20, 0.91, 0.05), 2)
DEDGE = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 99)]
N_D = len(DEDGE)
PASSES = 3
boot = np.random.default_rng(31)


def ci(delta: np.ndarray, n_boot: int = 2000) -> str:
    idx = boot.integers(0, len(delta), (n_boot, len(delta)))
    m = delta[idx].mean(1)
    return f"{delta.mean():+.4f} [{np.percentile(m, 2.5):+.4f}, {np.percentile(m, 97.5):+.4f}]"


def last_state(vis_q, ok):
    has = ok.any(1)
    idx = ok.shape[1] - 1 - np.argmax(ok[:, ::-1], axis=1)
    return np.take_along_axis(vis_q, idx[:, None], axis=1)[:, 0] & has


def edge_dist(base: np.ndarray) -> np.ndarray:
    n, S = base.shape
    edge = np.zeros_like(base)
    edge[:, :-1] |= base[:, :-1] != base[:, 1:]
    edge[:, 1:] |= base[:, 1:] != base[:, :-1]
    d = np.full((n, S), 99, np.int16)
    ar = np.arange(S)
    for j in range(n):
        e = np.flatnonzero(edge[j])
        if len(e):
            d[j] = np.abs(ar[:, None] - e[None, :]).min(1)
    return d


def collect(split_only=None) -> dict:
    """Flatten every eligible-or-true ongoing cell of both splits, with its group id and window id."""
    out: dict = {}
    for i, panel in enumerate(T2_PANELS):
        ctx = Ctx(panel, i)
        ctx.fitd = np.ones_like(ctx.fitd)
        for split in ("validation", "private"):
            z = load(panel, split)
            og, _ = ongoing_official(ctx, z, official_blank(panel, split, z))
            n = len(og["d"])
            if not n:
                continue
            c = np.load(CACHE / f"t2probs_{split}_{panel}.npz")
            assert (c["og_d"] == og["d"]).all()
            pC = np.mean([np.load(CACHE / f"t2cnn_{t}_{split}_{panel}.npz")["og_p"] for t in TAGS], 0)
            pM = (W_L * c["og_p"] + (1 - W_L) * pC).astype(np.float32)
            truth, elig = og["fut"], og["efut"]
            vis_q = og["ok"] & (np.nan_to_num(og["v"], nan=1e9) <= 0.6 * ctx.vf)
            base = last_state(vis_q, og["ok"])
            db = np.zeros(base.shape, np.int8)
            d = edge_dist(base)
            for k, (lo, hi) in enumerate(DEDGE):
                db[(d >= lo) & (d <= hi)] = k
            g_cell = (base.astype(np.int16) * N_D + db)                              # (N, S)
            g = (np.arange(F)[None, :, None] * (2 * N_D) + g_cell[:, None, :])       # (N, F, S)
            s = out.setdefault(split, {k: [] for k in ("w", "g", "p", "t", "e", "iou_w", "nwin")})
            off = s["nwin"][-1] if s["nwin"] else 0
            w = np.broadcast_to(np.arange(n)[:, None, None] + off, truth.shape)
            keep = elig | truth
            s["w"].append(w[keep].astype(np.int32)); s["g"].append(g[keep].astype(np.int16))
            s["p"].append(pM[keep]); s["t"].append(truth[keep]); s["e"].append(elig[keep])
            s["iou_w"].append(iou(pM >= TAU, truth, elig))
            s["nwin"].append(off + n)
    for split, s in out.items():
        out[split] = {k: np.concatenate(v) for k, v in s.items() if k not in ("nwin",)} | {"nwin": s["nwin"][-1]}
    return out


def tables(s: dict, n_g: int):
    """A[g, t, w] = true positives, B[g, t, w] = false positives, for every group and threshold."""
    W = s["nwin"]
    A = np.zeros((n_g, len(TAUS), W), np.int32)
    B = np.zeros((n_g, len(TAUS), W), np.int32)
    ok_t = s["e"] & s["t"]
    ok_f = s["e"] & ~s["t"]
    for g in range(n_g):
        m = s["g"] == g
        if not m.any():
            continue
        pw, pp = s["w"][m], s["p"][m]
        mt, mf = ok_t[m], ok_f[m]
        for k, t in enumerate(TAUS):
            sel = pp >= t
            A[g, k] = np.bincount(pw[sel & mt], minlength=W)
            B[g, k] = np.bincount(pw[sel & mf], minlength=W)
    T_w = np.bincount(s["w"][s["t"]], minlength=W)
    return A, B, T_w


def score(A, B, T_w, idx) -> np.ndarray:
    inter = A[np.arange(len(idx)), idx].sum(0)
    fp = B[np.arange(len(idx)), idx].sum(0)
    union = T_w + fp
    return np.where(union == 0, 1.0, inter / np.maximum(union, 1))


def fit_groups(A, B, T_w, groups: np.ndarray, n_g: int, start: int) -> np.ndarray:
    """Coordinate ascent over the threshold index of each group class (groups maps group id -> class)."""
    idx = np.full(n_g, start, int)
    n_cls = int(groups.max()) + 1
    for _ in range(PASSES):
        moved = False
        for cl in range(n_cls):
            members = np.flatnonzero(groups == cl)
            best, best_v = idx[members[0]], score(A, B, T_w, idx).mean()
            for k in range(len(TAUS)):
                trial = idx.copy()
                trial[members] = k
                v = score(A, B, T_w, trial).mean()
                if v > best_v + 1e-9:
                    best_v, best, moved = v, k, True
            idx[members] = best
        if not moved:
            break
    return idx


def main() -> None:
    t0 = time.time()
    n_g = F * 2 * N_D
    data = collect()
    print(f"collected: " + ", ".join(f"{k} {v['nwin']} windows / {len(v['p'])} cells" for k, v in data.items())
          + f" ({time.time() - t0:.0f}s)", flush=True)
    tab = {k: tables(v, n_g) for k, v in data.items()}
    print(f"tables built ({time.time() - t0:.0f}s)", flush=True)
    gid = np.arange(n_g)
    h_of, bd_of = gid // (2 * N_D), gid % (2 * N_D)
    schemes = {"tau per step": h_of, "tau per (base, dist)": bd_of, "tau per (step, base, dist)": gid}
    prod = int(np.argmin(np.abs(TAUS - TAU)))
    for fit, app in (("validation", "private"), ("private", "validation")):
        Af, Bf, Tf = tab[fit]
        Aa, Ba, Ta = tab[app]
        base_iou = score(Aa, Ba, Ta, np.full(n_g, prod))
        print(f"\n=== fitted on {fit}, applied to {app} ({len(base_iou)} windows) ===")
        print(f"  production tau 0.60            IoU {base_iou.mean():.4f}")
        g1 = fit_groups(Af, Bf, Tf, np.zeros(n_g, int), n_g, prod)
        v = score(Aa, Ba, Ta, g1)
        print(f"  global tau {TAUS[g1[0]]:.2f}                IoU {v.mean():.4f}  delta {ci(v - base_iou)}")
        for name, cls in schemes.items():
            idx = fit_groups(Af, Bf, Tf, cls, n_g, prod)
            v = score(Aa, Ba, Ta, idx)
            print(f"  {name:<30} IoU {v.mean():.4f}  delta {ci(v - base_iou)}", flush=True)
            if name == "tau per (step, base, dist)":
                for h in range(F):
                    row = [f"{TAUS[idx[h * 2 * N_D + b * N_D + d]]:.2f}" for b in (0, 1) for d in range(N_D)]
                    print(f"      +{5 * (h + 1):2d}min  outside(d0..d5+) {' '.join(row[:N_D])} | inside {' '.join(row[N_D:])}")
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
