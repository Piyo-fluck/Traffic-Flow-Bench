"""Task 2 EDA 04: refine the onset/ongoing rules and test them the way validation/private will use them.

Two evaluation sets per condition:
  test  : candidates whose day falls in Jan-Feb 2031; priors fitted on Jun-Dec 2030 only (temporal holdout,
          like train -> Mar/Apr)
  offic.: the 40 official train windows of that condition; priors leave-one-day-out over all train days
Onset candidates are earliest-onset origins (t2_eda03); ongoing candidates are every-30-min origins
whose history is ongoing (t2_eda02).

    py analysis/t2_eda04.py
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
from t2_eda02 import T2_PANELS, build, candidates, iou  # noqa: E402
from t2_eda03 import F, earliest_onset, labels  # noqa: E402
from tfb.data import RELEASE, load  # noqa: E402
from tfb.t1eval import FIT_END, family_mean  # noqa: E402

SMOOTHS = (0, 1, 2, 3)
GROUPS = ("wt2", "dow7", "all")


def smooth(x: np.ndarray, w: int) -> np.ndarray:
    return x if w == 0 else sum(np.roll(x, k, axis=1) for k in range(-w, w + 1)) / (2 * w + 1)


def gids(z: dict, mode: str) -> np.ndarray:
    wd = z["weekday"].astype(int)
    return {"wt2": (wd >= 5).astype(int), "dow7": wd, "all": np.zeros_like(wd)}[mode]


def prior_at(xs: np.ndarray, gid: np.ndarray, use: np.ndarray, d: np.ndarray, t: np.ndarray, loo: bool) -> np.ndarray:
    """Group mean over `use` days at (d, t) -> (N, S); leave-one-day-out when loo."""
    out = np.zeros((len(d), xs.shape[2]), np.float32)
    for g in np.unique(gid):
        m = use & (gid == g)
        sel = gid[d] == g
        if not sel.any():
            continue
        tot, n = xs[m].sum(0), m.sum()
        vals = tot[t[sel]]
        if loo:
            own = m[d[sel]]
            vals = vals - xs[d[sel], t[sel]] * own[:, None]
            out[sel] = vals / np.maximum(n - own, 1)[:, None]
        else:
            out[sel] = vals / max(n, 1)
    return out


def topk_mask(p: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros_like(p, bool)
    if k > 0:
        np.put_along_axis(out, np.argsort(-p, axis=1)[:, :k], True, axis=1)
    return out


def official(z: dict, panel: str, cond: str) -> tuple[np.ndarray, np.ndarray]:
    W = pd.read_csv(RELEASE / "task2" / panel / "train" / "window_index.csv")
    W = W[W.condition == cond]
    d_of = {d: i for i, d in enumerate(z["dates"])}
    s = W.forecast_origin.astype(str)
    return (W.date.astype(str).map(d_of).to_numpy(),
            (s.str.slice(11, 13).astype(int) * 12 + s.str.slice(14, 16).astype(int) // 5).to_numpy())


def main() -> None:
    on_res: dict[tuple, dict[str, dict[str, float]]] = {}
    og_res: dict[tuple, dict[str, dict[str, float]]] = {}
    step_stats = np.zeros((F, 3))
    for panel in T2_PANELS:
        z = load(panel, "train")
        q, el = labels(z)
        fitd = z["dates"] < FIT_END
        eo = earliest_onset(q)
        onset_cell = np.zeros_like(q)
        dd, tt = np.nonzero(eo)
        onset_cell[dd, tt + F] = q[dd, tt + F]
        base = {"P_q": q.astype(np.float32), "P_on": onset_cell.astype(np.float32)}
        sm = {(n, w): smooth(x, w) for n, x in base.items() for w in SMOOTHS}

        # ---------------- onset ----------------
        test = ~fitd[dd]
        od, ot = official(z, panel, "queue_onset")
        on_sets = {"test": (dd[test], tt[test], False), "offic": (od, ot, True)}
        for sname, (d, t0, loo) in on_sets.items():
            truth, elig = q[d, t0 + F], el[d, t0 + F]
            use = np.ones_like(fitd) if loo else fitd
            for g, w in itertools.product(GROUPS, SMOOTHS):
                gid = gids(z, g)
                pq_ = prior_at(sm[("P_q", w)], gid, use, d, t0 + F, loo)
                pon = prior_at(sm[("P_on", w)], gid, use, d, t0 + F, loo)
                for a in (0.0, 0.25, 0.5, 1.0, None):  # score = P_on + a*P_q ; None = P_q only
                    p = pq_ if a is None else pon + a * pq_
                    for k in (2, 3, 4, 5):
                        pred = topk_mask(p, k) & elig
                        inter, union = (pred & truth).sum(1), (pred | truth).sum(1)
                        v = np.where(union == 0, 1.0, inter / np.maximum(union, 1)).mean()
                        on_res.setdefault((g, w, "P_q" if a is None else f"P_on+{a}P_q", k), {}).setdefault(sname, {})[panel] = v

        # ---------------- ongoing ----------------
        c = candidates(z)
        c = {kk: v[c["ongoing"]] for kk, v in c.items()}
        ctest = {kk: v[~fitd[c["d"]]] for kk, v in c.items()}
        od, ot = official(z, panel, "queue_ongoing")
        co = build(z, od, ot)
        for sname, cs, loo in (("test", ctest, False), ("offic", co, True)):
            use = np.ones_like(fitd) if loo else fitd
            pon = prior_at(sm[("P_on", 2)], gids(z, "wt2"), use, cs["d"], cs["t0"] + F, loo)
            vh = np.where(cs["ehist"], cs["vhist"], np.nan)
            for cut, last in itertools.product((0.7, 0.75, 0.8, 0.85, 0.9), (1, 2, 3, 4, 6)):
                with np.errstate(invalid="ignore"):
                    link = np.nan_to_num(np.nanmin(vh[:, -last:], axis=1), nan=9) < cut
                pred = np.repeat(link[:, None], F, axis=1)
                for k_add in (0, 1, 2):
                    p2 = pred.copy()
                    if k_add:
                        p2[:, F - 1] |= topk_mask(np.where(link, -1, pon), k_add)
                    og_res.setdefault((cut, last, k_add), {}).setdefault(sname, {})[panel] = iou(p2, cs).mean()
                if sname == "test" and cut == 0.8 and last == 3:
                    p = pred & cs["efut"]
                    y = cs["fut"]
                    step_stats += np.stack([(p & y).sum((0, 2)), (p & ~y).sum((0, 2)), (~p & y).sum((0, 2))], 1)
        print(f"{panel}: onset test {int(test.sum())} / offic {len(od)}; ongoing test {len(ctest['d'])}", flush=True)

    def table(res, title, n=12):
        rows = [(k, family_mean(v["test"]), family_mean(v["offic"])) for k, v in res.items()]
        rows.sort(key=lambda r: -(r[1] + r[2]))
        print(f"\n== {title}: top {n} by test+official ==")
        for k, a, b in rows[:n]:
            print(f"  {str(k):<40} test {a:.4f}  official {b:.4f}")
        return rows

    on_rows = table(on_res, "onset (group, smooth, score, k)")
    ref = on_res[("wt2", 2, "P_on+0.0P_q", 3)]
    print(f"  current rule (wt2, 2, P_on, 3): test {family_mean(ref['test']):.4f} official {family_mean(ref['offic']):.4f}")
    og_rows = table(og_res, "ongoing (cut, last steps, extra onset links at T+30)")
    ref = og_res[(0.8, 3, 0)]
    print(f"  current rule (0.8, 3, 0): test {family_mean(ref['test']):.4f} official {family_mean(ref['offic']):.4f}")
    print("\nongoing rule (0.8, 3) on test candidates, per horizon step: TP / FP / FN")
    for h in range(F):
        tp, fp, fn = step_stats[h]
        print(f"  T+{5 * (h + 1):>2}: TP {tp:>8.0f}  FP {fp:>7.0f}  FN {fn:>7.0f}  IoU-like {tp / max(tp + fp + fn, 1):.3f}")


if __name__ == "__main__":
    main()
