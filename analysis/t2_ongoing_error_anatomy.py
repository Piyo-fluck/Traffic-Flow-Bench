"""Task 2 ONGOING: where the loss sits (the counterpart of notes 44/51, which analysed onset only).

Ongoing is half of S_queue and has never been decomposed. M predicts it as 0.5*LightGBM(I) + 0.5*CNN x3 at tau 0.60
over the six horizon steps T+5 .. T+30. Scored on the blank-aware official-like ongoing windows.

Measured, per split, pooled over panels:
  1. loss split: false positives vs false negatives, and the per-window IoU distribution (which windows carry it);
  2. by horizon step: does the loss come from the far end of the horizon?
  3. by cell class relative to the queue visible at T (what the model can see):
        stay     queued at T and still queued      -> FN = we dropped a persisting queue
        dissolve queued at T, gone in the horizon  -> FP = we kept a queue that cleared
        grow     not queued at T, queued later     -> FN = we missed an extension / a new queue
        far      not queued at T and never queued  -> FP = spurious queue somewhere else
     This separates "queue extent precision" from "whole-queue" errors.
  4. by distance to the nearest edge of the T queue: errors at the ends (soft, learnable) vs away from them;
  5. ceilings: persistence, the global tau sweep, an ORACLE per-window tau, an ORACLE per-window blend weight alpha,
     and an ORACLE per-horizon-step tau. These bound any gate / threshold model for ongoing.
Deltas carry a paired bootstrap CI over windows.

    .venv/Scripts/python.exe analysis/t2_ongoing_error_anatomy.py
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
TAU, W_L = 0.60, 0.5                      # production ongoing rule
TAUS = np.round(np.arange(0.20, 0.91, 0.05), 2)
ALPHAS = np.round(np.arange(0, 1.01, 0.1), 2)
DBINS = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 99)]
boot = np.random.default_rng(7)


def ci(delta: np.ndarray, n_boot: int = 2000) -> str:
    if not len(delta):
        return "n/a"
    idx = boot.integers(0, len(delta), (n_boot, len(delta)))
    m = delta[idx].mean(1)
    return f"{delta.mean():+.4f} [{np.percentile(m, 2.5):+.4f}, {np.percentile(m, 97.5):+.4f}]"


def last_state(vis_q: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """(N, S) the queue state at the last observed history slot of each link."""
    has = ok.any(1)
    idx = ok.shape[1] - 1 - np.argmax(ok[:, ::-1], axis=1)                  # (N, S)
    out = np.take_along_axis(vis_q, idx[:, None], axis=1)[:, 0]
    return out & has


def edge_dist(base: np.ndarray) -> np.ndarray:
    """(N, S) distance in links to the nearest boundary of the queue visible at T (a cell where base changes)."""
    n, S = base.shape
    edge = np.zeros_like(base)
    edge[:, :-1] |= base[:, :-1] != base[:, 1:]
    edge[:, 1:] |= base[:, 1:] != base[:, :-1]
    d = np.full((n, S), 99, np.int16)
    for j in range(n):
        e = np.flatnonzero(edge[j])
        if len(e):
            d[j] = np.abs(np.arange(S)[:, None] - e[None, :]).min(1)
    return d


def main() -> None:
    t0_all = time.time()
    acc: dict = {}
    for i, panel in enumerate(T2_PANELS):
        ctx = Ctx(panel, i)
        ctx.fitd = np.ones_like(ctx.fitd)
        for split in ("validation", "private"):
            z = load(panel, split)
            og, pers = ongoing_official(ctx, z, official_blank(panel, split, z))
            n = len(og["d"])
            if not n:
                continue
            c = np.load(CACHE / f"t2probs_{split}_{panel}.npz")
            assert (c["og_d"] == og["d"]).all(), "cached ongoing windows differ"
            pL = c["og_p"]                                                   # (N, F, S)
            pC = np.mean([np.load(CACHE / f"t2cnn_{t}_{split}_{panel}.npz")["og_p"] for t in TAGS], 0)
            pM = W_L * pL + (1 - W_L) * pC
            truth, elig = og["fut"], og["efut"]                              # (N, F, S)
            pred = pM >= TAU
            vis_q = og["ok"] & (np.nan_to_num(og["v"], nan=1e9) <= 0.6 * ctx.vf)
            base = last_state(vis_q, og["ok"])                               # (N, S) queue at T
            b3 = np.broadcast_to(base[:, None], truth.shape)
            p_ok = pred & elig
            fp, fn = p_ok & ~truth, truth & ~p_ok
            cls = {"stay": b3 & truth, "dissolve": b3 & ~truth & elig, "grow": ~b3 & truth, "far": ~b3 & ~truth & elig}
            d = np.broadcast_to(edge_dist(base)[:, None], truth.shape)
            r = acc.setdefault(split, {k: 0.0 for k in
                                       ["n", "fp", "fn", "inter", "union", "t_cells", "b_cells"]})
            r["n"] += n
            r["fp"] += fp.sum(); r["fn"] += fn.sum()
            r["inter"] += (p_ok & truth).sum(); r["union"] += (p_ok | truth).sum()
            r["t_cells"] += truth.sum(); r["b_cells"] += (b3 & elig).sum()
            for name, m in cls.items():
                r[f"cls_{name}"] = r.get(f"cls_{name}", 0) + m.sum()
                r[f"fp_{name}"] = r.get(f"fp_{name}", 0) + (fp & m).sum()
                r[f"fn_{name}"] = r.get(f"fn_{name}", 0) + (fn & m).sum()
            for h in range(F):
                r[f"h_iou{h}"] = r.get(f"h_iou{h}", []) + [iou(pred[:, h:h + 1], truth[:, h:h + 1], elig[:, h:h + 1])]
                r[f"h_fp{h}"] = r.get(f"h_fp{h}", 0) + fp[:, h].sum()
                r[f"h_fn{h}"] = r.get(f"h_fn{h}", 0) + fn[:, h].sum()
            for lo, hi in DBINS:
                m = (d >= lo) & (d <= hi)
                r[f"d_fp{lo}"] = r.get(f"d_fp{lo}", 0) + (fp & m).sum()
                r[f"d_fn{lo}"] = r.get(f"d_fn{lo}", 0) + (fn & m).sum()
            iou_m = iou(pred, truth, elig)
            r.setdefault("iou", []).append(iou_m)
            r.setdefault("pers", []).append(pers)
            # ---- ceilings
            by_tau = np.stack([iou(pM >= t, truth, elig) for t in TAUS])      # (T, N)
            r.setdefault("tau", []).append(by_tau)
            r.setdefault("tau_best", []).append(by_tau.max(0))
            by_a = np.stack([iou(a * pL + (1 - a) * pC >= TAU, truth, elig) for a in ALPHAS])
            r.setdefault("alpha", []).append(by_a)
            r.setdefault("alpha_best", []).append(by_a.max(0))
            hs = np.zeros(truth.shape, bool)                                  # oracle tau per horizon step (global)
            for h in range(F):
                sc = [iou(pM[:, h:h + 1] >= t, truth[:, h:h + 1], elig[:, h:h + 1]).mean() for t in TAUS]
                hs[:, h] = pM[:, h] >= TAUS[int(np.argmax(sc))]
                r[f"h_tau{h}"] = r.get(f"h_tau{h}", []) + [TAUS[int(np.argmax(sc))]]
            r.setdefault("htau_iou", []).append(iou(hs, truth, elig))
        print(f"[{panel}] done ({time.time() - t0_all:.0f}s)", flush=True)
    for split, r in acc.items():
        cat = lambda k: np.concatenate(r[k])                                  # noqa: E731
        iou_m, pers = cat("iou"), cat("pers")
        err = r["fp"] + r["fn"]
        print(f"\n=== {split}: {int(r['n'])} official-like ongoing windows ===")
        print(f"  M (0.5 LGBM + 0.5 CNN, tau {TAU}) IoU {iou_m.mean():.4f} | persistence {pers.mean():.4f} | "
              f"micro IoU {r['inter'] / r['union']:.4f}")
        print(f"  true queue cells {int(r['t_cells'])}, queue visible at T {int(r['b_cells'])}; "
              f"errors: FP {r['fp'] / err:.3f} / FN {r['fn'] / err:.3f} of {int(err)}")
        q = np.percentile(iou_m, [10, 25, 50, 75, 90])
        loss = 1 - iou_m
        o = np.argsort(-loss)
        print(f"  per-window IoU deciles: p10 {q[0]:.3f} p25 {q[1]:.3f} p50 {q[2]:.3f} p75 {q[3]:.3f} p90 {q[4]:.3f}; "
              f"worst 10% of windows hold {loss[o[:len(o) // 10]].sum() / loss.sum():.2f} of the loss; "
              f"windows with IoU < 0.5: {(iou_m < 0.5).mean():.3f}")
        print("  --- 1) by cell class relative to the queue visible at T (share of all errors) ---")
        for name in ("stay", "dissolve", "grow", "far"):
            cells = r[f"cls_{name}"]
            e = r[f"fp_{name}"] + r[f"fn_{name}"]
            kind = "FN (dropped)" if name in ("stay", "grow") else "FP (spurious)"
            print(f"    {name:<9} cells {int(cells):8d}  errors {int(e):7d} = {e / err:.3f} of all errors  "
                  f"({kind}, error rate within the class {e / max(cells, 1):.3f})")
        print("  --- 2) by horizon step (T+5 .. T+30) ---")
        print("    IoU  " + "  ".join(f"+{5 * (h + 1)}min {np.concatenate(r[f'h_iou{h}']).mean():.3f}" for h in range(F)))
        print("    FP/FN share " + "  ".join(f"+{5 * (h + 1)} {(r[f'h_fp{h}'] + r[f'h_fn{h}']) / err:.3f}" for h in range(F)))
        print("  --- 3) by distance to the nearest edge of the T queue ---")
        print("    " + "  ".join(f"d{lo}-{hi}: {(r[f'd_fp{lo}'] + r[f'd_fn{lo}']) / err:.3f}" for lo, hi in DBINS))
        print("  --- 4) ceilings (paired delta vs M, 95% bootstrap CI) ---")
        tau_all = np.concatenate(r["tau"], 1)                                 # (T, N)
        best_t = int(np.argmax(tau_all.mean(1)))
        print("    global tau:  " + "  ".join(f"{t:.2f} {tau_all[k].mean():.4f}" for k, t in enumerate(TAUS)))
        print(f"    best global tau {TAUS[best_t]:.2f}: IoU {tau_all[best_t].mean():.4f}  delta {ci(tau_all[best_t] - iou_m)}")
        print(f"    per-step tau (chosen globally per step {[float(np.mean(r[f'h_tau{h}'])) for h in range(F)]}): "
              f"IoU {cat('htau_iou').mean():.4f}  delta {ci(cat('htau_iou') - iou_m)}")
        print(f"    ORACLE per-window tau: IoU {cat('tau_best').mean():.4f}  delta {ci(cat('tau_best') - iou_m)}")
        a_all = np.concatenate(r["alpha"], 1)
        print("    global alpha: " + "  ".join(f"{a:.1f} {a_all[k].mean():.4f}" for k, a in enumerate(ALPHAS)))
        print(f"    ORACLE per-window alpha: IoU {cat('alpha_best').mean():.4f}  delta {ci(cat('alpha_best') - iou_m)}")
        print(f"    persistence: delta {ci(pers - iou_m)}")
    print(f"total {time.time() - t0_all:.0f}s")


if __name__ == "__main__":
    main()
