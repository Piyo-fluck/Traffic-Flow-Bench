"""Is the onset model limited by how many events it sees, or by how much the observables say?

The onset half is fitted on first-of-morning windows only - `t2_cnn.train_windows` takes `ctx.eo` and
`train_occ.windows` takes `earliest_onset(ctx.q)` - which is 1,722 windows over the whole train period, against
24,087 free-to-queued transitions that exist in the same data. The case for reformulating onset as a discrete
time hazard rests on that ratio being the binding constraint.

It might not be. Two disjoint feature sets both landed on 0.76 top-1 (`occ_lab/onset_bottleneck_rerank.py`), and
that is equally consistent with the observables simply not saying more. The cheap discriminator is the learning
curve: train the same model on a fraction of the fit-day windows and read it on the held-out days. A curve still
climbing at 100% says events are the constraint and the hazard reformulation is worth building; a flat curve
says it is not.

Everything is fitted on days before FIT_END and read on the days after. The prior is rebuilt from the fit days
only, so the curve is not riding on the train-run location memory; `--no-prior` drops it entirely.

    python occ_lab/onset_learning_curve.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta"))
sys.path.insert(0, str(LAB.parent / "analysis"))
sys.path.insert(0, str(LAB.parent / "scripts"))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

import common as C  # noqa: E402
from t2_expected_iou import select  # noqa: E402
from t2_first_onset_eval import first_onsets  # noqa: E402
from t2_m1_predict import onset_X  # noqa: E402
from tfb.t1eval import FIT_END  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.125, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--reps", type=int, default=3, help="subsample draws per fraction")
    ap.add_argument("--tol", type=int, default=2, help="a hit is within this many links of a true onset cell")
    args = ap.parse_args()
    X, Y, EV, KEY, PANEL, ELG = [], [], [], [], [], []
    t0_ = time.time()
    for i, panel in enumerate(C.T2_PANELS):
        ctx = C.Ctx(panel, i)
        ctx.fitd = np.ones_like(ctx.fitd)
        z = ctx.z
        w = first_onsets(ctx.vf, z, truth=True, blank=None)
        ev = z["dates"][w["d"]] >= FIT_END
        fitd = z["dates"] < FIT_END
        tabs = {n: np.stack([xs[(ctx.g == g) & fitd].mean(0) for g in range(int(ctx.g.max()) + 1)])
                for n, xs in (("q", ctx.xs_q), ("on", ctx.xs_on))}
        pri = lambda n, key, t: tabs[n][key[:, 1], np.clip(t, 0, tabs[n].shape[1] - 1)]  # noqa: E731
        gg = (z["weekday"][w["d"]] >= 5).astype(int)
        names, XW = onset_X(ctx, w["v"], w["qf"], w["ok"], w["t0"], gg, pri, np.stack([w["d"], gg], 1))
        X.append(XW)
        Y.append(w["cells"].astype(np.float32))
        ELG.append(ctx.el[w["d"], w["t0"] + C.F])
        EV.append(ev)
        KEY.append(np.arange(len(w["d"])) + i * 10 ** 5)
        PANEL.append(np.full(len(w["d"]), i))
        print(f"  {panel}: {len(w['d'])} windows, {XW.shape[-1]} features "
              f"({time.time() - t0_:.0f}s)", flush=True)
        ctx.z = ctx.q = ctx.el = None
    NAMES = names
    pri_cols = [k for k, n in enumerate(NAMES) if n.startswith("pq") or n.startswith("pon")]
    keep_np = [k for k in range(len(NAMES)) if k not in pri_cols]

    def top1(P, Yv, tol):
        hit = []
        for j in range(len(P)):
            t = np.flatnonzero(Yv[j])
            if not len(t):
                continue
            hit.append(bool(np.abs(t - int(np.argmax(P[j]))).min() <= tol))
        return float(np.mean(hit)), len(hit)

    def onset_iou(P, Yv, El):
        """The production decision: the expected-IoU ratio rule at T+30, scored inside eligibility."""
        out = []
        for j in range(len(P)):
            m = El[j]
            if not m.any() or not Yv[j][m].any():
                continue
            pr = select(P[j][m], "ratio", 1.0)
            t = Yv[j][m] > 0.5
            out.append(int((pr & t).sum()) / max(int((pr | t).sum()), 1))
        return float(np.mean(out)), len(out)

    par = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 60,
           "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1,
           "num_threads": 8}
    print(f"\ntop-1 bottleneck within +-{args.tol} links, held-out days, mean of {args.reps} subsamples\n")
    print(f"{'fit windows':>12}{'frac':>7}{'top1 pri':>10}{'top1 nopri':>12}"
          f"{'IoU pri':>10}{'IoU nopri':>11}")
    for f in args.fracs:
        accs, ious = {True: [], False: []}, {True: [], False: []}
        n_fit = 0
        for r in range(args.reps if f < 1.0 else 1):
            rng = np.random.default_rng(100 * r + 7)
            for use_pri in (True, False):
                cols = list(range(len(NAMES))) if use_pri else keep_np
                xs, ys = [], []
                for Xp, Yp, evp in zip(X, Y, EV):
                    idx = np.flatnonzero(~evp)
                    if f < 1.0:
                        idx = rng.choice(idx, max(int(round(len(idx) * f)), 2), replace=False)
                    xs.append(Xp[idx][..., cols].reshape(-1, len(cols)))
                    ys.append(Yp[idx].ravel())
                n_fit = sum(len(a) for a in ys) // X[0].shape[1]
                m_ = lgb.train(par, lgb.Dataset(np.concatenate(xs), label=np.concatenate(ys)),
                               num_boost_round=args.rounds)
                h, n, hi_, ni_ = [], 0, [], 0
                for Xp, Yp, evp, Ep in zip(X, Y, EV, ELG):
                    idx = np.flatnonzero(evp)
                    if not len(idx):
                        continue
                    P = m_.predict(Xp[idx][..., cols].reshape(-1, len(cols))).reshape(len(idx), -1)
                    a, k = top1(P, Yp[idx], args.tol)
                    h.append(a * k)
                    n += k
                    b, k2 = onset_iou(P, Yp[idx], Ep[idx])
                    hi_.append(b * k2)
                    ni_ += k2
                accs[use_pri].append(sum(h) / max(n, 1))
                ious[use_pri].append(sum(hi_) / max(ni_, 1))
        print(f"{n_fit:>12,}{f:>7.3f}{np.mean(accs[True]):>10.4f}{np.mean(accs[False]):>12.4f}"
              f"{np.mean(ious[True]):>10.4f}{np.mean(ious[False]):>11.4f}", flush=True)


if __name__ == "__main__":
    main()
