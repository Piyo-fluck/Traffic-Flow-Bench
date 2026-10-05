"""Task 2 data-route AUDIT (measurement only; nothing here is for submissions without the user's decision).

Our onset is information-limited (notes 27, 34-39): misses are windows where the queue appears at links no
model ranks high. The information that could close the gap to the top (~0.92) sits in data we excluded on
purpose. Two routes are measured:

A. Same-day data after the blanked hour. Official windows blank the horizon and the following hour (T+5 ... T+90).
   Later same-day observations (T+95 ...) remain in the released Task 1 layer. For each official-like pseudo window:
       later_q[s] = link s is visibly queued at any slot T+95 .. T+150 (same day)
   Blends with the cached model probabilities (onset: H with the expected-IoU ratio rule; ongoing: I with tau 0.65):
       onset   score = p + beta * later_q (beta 0.25 / 1.0), k from the expected-IoU rule on p
       ongoing pred  = (p >= tau) | (later_q & p >= 0.3)
   These are future information for an online forecast. The data rule (t2-data-usage-boundary) forbids them.
D. Ineligible horizon cells. MASK_SPEC blanks the horizon on ELIGIBLE cells only. Count, in the released masked
   layer, how many horizon cells of the OFFICIAL windows still carry a speed value (and how many show a queue).
   That would be a direct read of the answer; it is counted, not used.

    py analysis/t2_route_audit.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "scripts"))
from t2_eda02 import T2_PANELS  # noqa: E402
from t2_eda03 import F  # noqa: E402
from t2_expected_iou import select  # noqa: E402
from t2_first_onset_eval import first_onsets, official_blank  # noqa: E402
from t2_official_like_eval import ongoing_official  # noqa: E402
from t2_pseudo_eval import iou  # noqa: E402
from tfb.data import CACHE, RELEASE, T, load  # noqa: E402
from tfb.t1eval import family_mean  # noqa: E402

LATE = (F + 13, F + 24)  # slots T+95 .. T+150 relative to the origin T (t0)
DAY_SETS = {"d1-14": 14, "all": 31}


def fmean(d):
    d = {k: v for k, v in d.items() if np.isfinite(v)}
    return family_mean(d) if d else float("nan")


def later_queue(q_vis: np.ndarray, d: np.ndarray, t0: np.ndarray) -> np.ndarray:
    lo, hi = t0 + LATE[0], np.minimum(t0 + LATE[1], T - 1)
    out = np.zeros((len(d), q_vis.shape[2]), bool)
    for j in range(len(d)):
        if lo[j] <= hi[j]:
            out[j] = q_vis[d[j], lo[j]:hi[j] + 1].any(0)
    return out


class _C:
    def __init__(self, vf):
        self.vf = vf


def main() -> None:
    res: dict = {}
    d_rows = []
    for panel in T2_PANELS:
        for split in ("validation", "private"):
            z = load(panel, split)
            vf = z["free_speed"]
            blank = official_blank(panel, split, z)
            c = np.load(CACHE / f"t2probs_{split}_{panel}.npz")
            vis = z["elig"] & np.isfinite(z["obs_speed"])
            q_vis = vis & (np.nan_to_num(z["obs_speed"], nan=1e9) <= 0.6 * vf)
            # ---- route A: onset
            on = first_onsets(vf, z, blank=blank)
            assert len(on["d"]) == len(c["on_d"])
            if len(on["d"]):
                lq = later_queue(q_vis, on["d"], on["t0"]).astype(float)
                p = c["on_p"]
                variants = {"model": None, "later_only": None, "beta0.25": 0.25, "beta1.0": 1.0}
                for name, beta in variants.items():
                    pr = np.zeros(on["fut"].shape, bool)
                    for j in range(len(p)):
                        sel = select(p[j], "ratio", 1.0)
                        k = int(sel.sum())
                        if name == "model":
                            pr[j, F - 1] = sel
                        elif name == "later_only":
                            sc = lq[j] + 1e-3 * p[j]
                            pr[j, F - 1] = np.argsort(np.argsort(-sc)) < k
                        else:
                            sc = p[j] + beta * lq[j]
                            pr[j, F - 1] = np.argsort(np.argsort(-sc)) < k
                    s = iou(pr, on["fut"], on["efut"])
                    for ds, dmax in DAY_SETS.items():
                        m = on["d"] < dmax
                        res.setdefault(("onset", split, ds), {}).setdefault(name, {})[panel] = s[m].mean() if m.any() else np.nan
            # ---- route A: ongoing
            ctx = _C(vf)
            og, _ = ongoing_official(ctx, z, blank)
            assert len(og["d"]) == len(c["og_d"])
            if len(og["d"]):
                lq = later_queue(q_vis, og["d"], og["t0"])[:, None, :]
                p = c["og_p"]
                for name, pr in {"model": p >= 0.65, "or_later": (p >= 0.65) | (lq & (p >= 0.3))}.items():
                    s = iou(pr, og["fut"], og["efut"])
                    for ds, dmax in DAY_SETS.items():
                        m = og["d"] < dmax
                        res.setdefault(("ongoing", split, ds), {}).setdefault(name, {})[panel] = s[m].mean() if m.any() else np.nan
            # ---- route D: ineligible cells left in the official horizons
            W = pd.read_csv(RELEASE / "task2" / panel / split / "window_index.csv")
            d_of = {dt: i for i, dt in enumerate(z["dates"])}
            n_cells = n_inel_val = n_inel_q = 0
            for w in W.itertuples():
                di = d_of[str(w.date)]
                t = pd.Timestamp(w.forecast_origin)
                t0 = t.hour * 12 + t.minute // 5
                sl = slice(t0 + 1, t0 + F + 1)
                sp = z["obs_speed"][di, sl]
                inel = ~z["elig"][di, sl]
                n_cells += sp.size
                n_inel_val += int((inel & np.isfinite(sp)).sum())
                n_inel_q += int((inel & np.isfinite(sp) & (np.nan_to_num(sp, nan=1e9) <= 0.6 * vf)).sum())
            d_rows.append((panel, split, n_cells, n_inel_val, n_inel_q))
        print(f"[{panel}] done", flush=True)
    for (kind, split, ds), r in res.items():
        base = r["model"]
        line = []
        for name, d in r.items():
            diffs = [d[p] - base[p] for p in d if np.isfinite(d[p]) and np.isfinite(base[p])]
            line.append(f"{name} {fmean(d):.3f} ({fmean(d) - fmean(base):+.3f}; {sum(x > 1e-9 for x in diffs)}/{sum(x < -1e-9 for x in diffs)})")
        print(f"== A {kind} {split} {ds}: " + "  ".join(line))
    tot = np.array([r[2:] for r in d_rows]).sum(0)
    print(f"\n== D: official horizon cells {tot[0]:,}; ineligible cells still carrying a speed value {tot[1]:,} "
          f"({tot[1] / max(tot[0], 1):.3%}); of those queued {tot[2]:,}")
    for r in d_rows:
        print(f"   {r[0]:<10} {r[1]:<10} horizon cells {r[2]:>6}  ineligible valued {r[3]:>5}  queued {r[4]:>4}")


if __name__ == "__main__":
    main()
