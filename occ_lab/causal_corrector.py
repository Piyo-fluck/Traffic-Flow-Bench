"""Fit the residual corrector on the scored month's OWN EARLIER DAYS (causal, as idea 9b does for the onset CNN).

Note 87 measured that the corrector is strongly data-recency bound: fitted on the 2031-01/02 tail it gives
+0.0044 / +0.0045 family IoU on March / April, fitted on the adjacent month it gives +0.0117 / +0.0079. The rule
allows, for a window whose origin is on day d, every released observation at or before that origin - including
the scored month's own earlier days, whose queue labels are in the released masked layer.

So for every day d of a scored month the corrector is refitted on the windows of days < d of that month (plus,
optionally, the tail), and only day d is predicted with it. Nothing after a window's origin is ever used.

    tail            candidate AI: the 2031-01/02 tail only, one model for the month   (the reference)
    past            the month's days < d only, refitted per day
    tail+past       both, concatenated
    tail+past w3    both, the month's own days weighted 3x

Early days have little history, so `--min-cells` falls back to the tail-only model until the month's own pool is
big enough. Scored on the same pseudo windows as every other local test, day by day.

    python occ_lab/causal_corrector.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta"))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

import common as C  # noqa: E402
from logit_update import IDX  # noqa: E402
from residual_cov import CACHE, PARAMS, load_env  # noqa: E402
from residual_diag import NAMES, TAU  # noqa: E402
from residual_oof import panel_data as scored_panel  # noqa: E402

SETTINGS = [(e, g) for e in (0.50, 0.75, 1.00) for g in (0.0, 0.05, 0.10)]
VARIANTS = ("tail", "past", "tail+past", "tail+past w3")


def train(params, X, y, p0, w=None):
    return lgb.train(dict(PARAMS, num_threads=32),
                     lgb.Dataset(X, label=y - p0, weight=w, feature_name=NAMES, free_raw_data=False),
                     num_boost_round=params)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-cells", type=int, default=20_000)
    ap.add_argument("--max-day", type=int, default=0, help="only score days up to this day of the month "
                                                           "(the official ongoing windows sit on days 1-17)")
    ap.add_argument("--rounds", type=int, default=400)
    args = ap.parse_args()

    # ------------------------------------------------------------------ the tail (AI's training set)
    Xs, ys, ps = [], [], []
    for p in C.T2_PANELS:
        lab = np.load(CACHE / f"cov_labels_{p}.npz")
        f, base, _ = load_env("obs", p)
        m = lab["e"]
        Xs.append(f[m][:, IDX]); ys.append(lab["y"][m].astype(np.float32)); ps.append(base[m])
    Xt, yt, pt = np.concatenate(Xs), np.concatenate(ys), np.concatenate(ps)
    del Xs, ys, ps
    m_tail = train(args.rounds, Xt, yt, pt)
    print(f"tail corrector: {len(yt):,} cells", flush=True)

    for split in C.SPLITS:
        data, days = {}, None
        for p in C.T2_PANELS:
            d = scored_panel(p, split)
            fc = CACHE / f"cov_scored_{split}_{p}.npz"
            if d is None or not fc.exists():
                continue
            f, y, e, base = d
            z = np.load(fc)
            data[p] = (f, y, e, base, z["date"].astype(str))
            days = np.union1d(days if days is not None else [], np.unique(z["date"].astype(str)))
        print(f"\n=== {split}: {len(days)} days, {len(data)} panels ===", flush=True)
        fam_b, fam = {}, {}
        n_fallback = 0
        if args.max_day:
            days = np.array([d for d in days if int(d[-2:]) <= args.max_day])
        BUCKETS = [("day 1", 1, 1), ("days 2-3", 2, 3), ("days 4-7", 4, 7), ("days 8-14", 8, 14),
                   ("days 15+", 15, 31)]
        for di, day in enumerate(days):
            sel_past = {p: data[p][4] < day for p in data}
            sel_now = {p: data[p][4] == day for p in data}
            if not any(sel_now[p].any() for p in data):
                continue
            Xp, yp, pp = [], [], []
            for p in data:
                f, y, e, base, _ = data[p]
                m = e & sel_past[p][:, None, None]
                if m.any():
                    Xp.append(f[m]); yp.append(y[m].astype(np.float32)); pp.append(base[m])
            n_past = sum(len(a) for a in yp)
            models = {"tail": m_tail}
            if n_past >= args.min_cells:
                Xp, yp, pp = np.concatenate(Xp), np.concatenate(yp), np.concatenate(pp)
                models["past"] = train(args.rounds, Xp, yp, pp)
                Xb = np.concatenate([Xt, Xp]); yb = np.concatenate([yt, yp]); pb = np.concatenate([pt, pp])
                models["tail+past"] = train(args.rounds, Xb, yb, pb)
                w = np.r_[np.ones(len(yt), np.float32), np.full(len(yp), 3.0, np.float32)]
                models["tail+past w3"] = train(args.rounds, Xb, yb, pb, w)
                del Xb, yb, pb, Xp, yp, pp
            else:
                n_fallback += 1
                for v in VARIANTS[1:]:
                    models[v] = m_tail
            for p in data:
                f, y, e, base, _ = data[p]
                w = sel_now[p]
                if not w.any():
                    continue
                yy, ee, bb, ff = y[w], e[w], base[w], f[w]
                fam_b.setdefault(p, []).append(C.iou(bb >= TAU, yy, ee))
                fam_b.setdefault((p, int(day[-2:])), []).append(fam_b[p][-1])
                for v in VARIANTS:
                    rh = np.zeros(bb.shape, np.float32)
                    rh[ee] = models[v].predict(ff[ee]).astype(np.float32)
                    for eta, g in SETTINGS:
                        q = np.clip(bb + eta * np.where(np.abs(rh) >= g, rh, 0.0), 0, 1)
                        s = C.iou(q >= TAU, yy, ee)
                        fam.setdefault((v, eta, g), {}).setdefault(p, []).append(s)
                        fam[(v, eta, g)].setdefault((p, int(day[-2:])), []).append(s)
            if di % 5 == 0:
                print(f"  {day}: {n_past:,} past cells", flush=True)
        def agg(d, lo=None, hi=None):
            """Family mean over all days, or over the days in [lo, hi] when a bucket is asked for."""
            out = {}
            for k, v in d.items():
                if lo is None:
                    if not isinstance(k, tuple):
                        out.setdefault(k, []).extend(v)
                elif isinstance(k, tuple) and lo <= k[1] <= hi:
                    out.setdefault(k[0], []).extend(v)
            if not out:
                return float("nan")
            return C.family_mean({k: np.concatenate(v).mean() for k, v in out.items() if v})

        b = agg(fam_b)
        print(f"  blend {b:.4f}; {n_fallback} of {len(days)} days fell back to the tail model")
        for v in VARIANTS:
            row = {(eta, g): agg(fam[(v, eta, g)]) - b for eta, g in SETTINGS}
            (be, bg), best = max(row.items(), key=lambda kv: kv[1])
            print(f"    {v:<14} AI setting {row[(0.75, 0.10)]:+.4f}   best {best:+.4f} at {be:.2f}/{bg:.2f}",
                  flush=True)
            parts = []
            for nm, lo, hi in BUCKETS:
                bb_ = agg(fam_b, lo, hi)
                if np.isfinite(bb_):
                    parts.append(f"{nm} {agg(fam[(v, 0.75, 0.10)], lo, hi) - bb_:+.4f}")
            print(f"      by day (AI setting): " + "  ".join(parts), flush=True)


if __name__ == "__main__":
    main()
