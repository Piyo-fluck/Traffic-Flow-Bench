"""The user's first-priority Task 2 item: a run-specific online residual adapter on top of the global corrector.

The global corrector learned conditional competence - which member to trust where - from the train tail, and it
works (+0.0081 ongoing family on AR's blend). Candidate AR then showed that direct queue supervision from the
same run's earlier days is strong enough to carry 0.60 of the ongoing blend. Conditional competence is itself a
property of the run, so that supervision should reach the corrector layer too, not only the prediction layer.

    p_global = clip(p0 + eta * rhat_global)                    unchanged, the shipped corrector
    delta_d  = X beta_d                                        a small linear adapter, fitted per day
    p_final  = clip(p_global + lam_d * delta_d)

beta_d is fitted on cells from windows on STRICTLY EARLIER days of the same run, with labels from the released
layer, by ridge toward beta = 0 - so the shrinkage centre is "use the global corrector unchanged", which is what
"the global corrector as a prior" means here. Item 5 settled that the shipped corrector is the right prior: a
version refitted on AR's own p0 along the adaptation trajectory scored 0.8487 against its 0.8505.

lam_d ramps with the number of past days available, so day 1 of a month applies nothing.

Features, all available at inference:

    1, rhat_global, p_PS - p_base, p_tfm - p_base, horizon, position, |p_global - 0.5|

Walking the days in order makes the reported number out of sample by construction.

    python occ_lab/online_adapter.py --tag s20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta"))
sys.path.insert(0, str(LAB.parent / "analysis"))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

import common as C  # noqa: E402
from residual_diag import NAMES, TAU, lg  # noqa: E402

CACHE = LAB / "cache"
W_PS, ETA, GATE = 0.60, 0.75, 0.10
MEMBERS = ("gs1", "tfm8", "cnn3", "occ_cnn", "occ_tfm", "stid")
COLS = ["1", "rhat", "ps - old", "tfm8 - base", "horizon", "position", "|pg - 0.5|"]
RICH = ["gs1 - base", "stid - base", "occ_cnn - base", "member sd", "ps", "qT", "qT * horizon"]


def adapter_cols(D, base, pg, rh, rich: bool = False):
    """The adapter's design matrix, the one thing the fit and `scripts/t2_residual_predict.py` must share."""
    N, Fh, S = base.shape
    hor = np.broadcast_to(np.arange(Fh, dtype=np.float32)[None, :, None] / max(Fh - 1, 1), base.shape)
    pos = np.broadcast_to((np.arange(S, dtype=np.float32) / max(S - 1, 1))[None, None, :], base.shape)
    cols = [np.ones_like(base), rh, D["ps"] - D["old"], D["P"]["tfm8"] - base,
            hor, pos, np.abs(pg - 0.5)]
    if rich:
        mm = np.stack([D["P"][k] for k in MEMBERS])
        qT = D.get("qT")
        q = np.broadcast_to(qT[:, None, :], base.shape).astype(np.float32) if qT is not None \
            else np.zeros_like(base)
        cols += [D["P"]["gs1"] - base, D["P"]["stid"] - base, D["P"]["occ_cnn"] - base,
                 mm.std(0), D["ps"], q, q * hor]
    return np.stack(cols, -1)


def mean_of(z, prefix):
    ks = [k for k in z.files if k.startswith(prefix)]
    return np.mean([z[k].astype(np.float32) for k in ks], 0) if ks else None


def global_features(P, base):
    N, Fh, S = base.shape
    hor = np.broadcast_to(np.arange(Fh, dtype=np.float32)[None, :, None], base.shape)
    pos = np.broadcast_to((np.arange(S, dtype=np.float32) / max(S - 1, 1))[None, None, :], base.shape)
    members = np.stack([P[k] for k in ("gs1", "tfm8", "cnn3", "occ_cnn", "occ_tfm", "stid")])
    return np.stack([base, lg(base), np.abs(base - 0.5), members.std(0), members.max(0) - members.min(0),
                     P["gs1"], P["tfm8"], P["cnn3"], P["occ_cnn"], P["occ_tfm"], P["stid"],
                     lg(P["stid"]) - lg(base), lg(P["occ_cnn"]) - lg(base), hor, pos], -1)


def load_split(i, panel, tag, w_ps: float = W_PS):
    ctx, stat, Wd = C.windows(i, panel)
    out = {}
    for s in C.SPLITS:
        og, ref = Wd[s]["og"], Wd[s]["ref"]
        fm, fp = CACHE / f"ogmem_{s}_{panel}.npz", CACHE / f"pseudo_{tag}_{s}_{panel}.npz"
        if not len(og["d"]) or not (fm.exists() and fp.exists()):
            continue
        mem = np.load(fm)
        P = {"gs1": ref["og_gs1"], "tfm8": ref["og_tfm8"], "cnn3": ref["og_cnn3"],
             "occ_cnn": mem["occ_cnn"], "occ_tfm": mem["occ_tfm"], "stid": mem["stid"]}
        ps = np.load(fp)["og"].astype(np.float32)
        old = 0.2 * P["gs1"] + 0.6 * P["tfm8"] + 0.2 * P["cnn3"]
        vq = og["ok"] & (np.nan_to_num(og["v"], nan=1e9) <= 0.6 * ctx.vf)
        idx = np.where(og["ok"][:, ::-1], np.arange(og["ok"].shape[1])[::-1][None, :, None], -1).max(1)
        S = vq.shape[2]
        qT = np.zeros((vq.shape[0], S), bool)
        ok = idx >= 0
        jj = np.broadcast_to(np.arange(vq.shape[0])[:, None], idx.shape)
        ss = np.broadcast_to(np.arange(S), idx.shape)
        qT[ok] = vq[jj[ok], idx[ok], ss[ok]]                 # queued at T: the class of note 118's two thresholds
        out[s] = dict(P=P, ps=ps, old=old, base=(1 - w_ps) * old + w_ps * ps,
                      y=og["fut"], e=og["efut"], d=og["d"], qT=qT,
                      dates=np.asarray(Wd[s]["z"]["dates"]).astype(str), n_days=len(Wd[s]["z"]["dates"]))
    ctx.z = None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="s20")
    ap.add_argument("--lams", type=float, nargs="+", default=[1e5, 1e4, 1e3, 1e2])
    ap.add_argument("--steps", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    ap.add_argument("--ramp", type=int, default=5, help="lam_d = min(1, past_days / ramp)")
    ap.add_argument("--rich", action="store_true",
                    help="a wider adapter: per-member disagreements, the member spread, and the queued-at-T "
                         "class, which carries a systematic +0.09 calibration bias the adapter can then learn "
                         "per run (it subsumes the two-threshold result)")
    ap.add_argument("--early", type=int, default=4,
                    help="also report the subset with at most this many earlier days, which is the regime the "
                         "OFFICIAL windows are in: analysis/t2_run_supervision_budget.py puts 85% of them on a "
                         "day with at least one earlier pool window, and the median is 3 to 38 windows")
    args = ap.parse_args()
    gbm = lgb.Booster(model_file=str(LAB / "models" / "residual_lgb.txt"))
    fam, fam_e = {}, {}
    for i, panel in enumerate(C.T2_PANELS):
        sp = load_split(i, panel, args.tag)
        for s, D in sp.items():
            qT = D.get("qT")
            base, y, e, dd = D["base"], D["y"], D["e"], D["d"]
            F = global_features(D["P"], base)
            rh = gbm.predict(F.reshape(-1, F.shape[-1])).reshape(base.shape).astype(np.float32)
            pg = np.clip(base + ETA * np.where(np.abs(rh) >= GATE, rh, 0.0), 0, 1)
            _ = qT
            X = adapter_cols(D, base, pg, rh, args.rich).astype(np.float64)
            t = (y.astype(np.float64) - pg)                      # what the global corrector still leaves
            days = np.unique(dd)
            early = np.isin(dd, days[1:args.early + 1])
            fam.setdefault((s, "global"), {})[panel] = C.iou(pg >= TAU, y, e).mean()
            fam_e.setdefault((s, "global"), {})[panel] = C.iou(pg >= TAU, y, e)[early].mean()
            for lam in args.lams:
                for st in args.steps:
                    pf = pg.copy()
                    for k, day in enumerate(days):
                        past = np.isin(dd, days[:k])
                        cur = dd == day
                        if k == 0 or not past.any():
                            continue
                        mp = past[:, None, None] & e
                        Xp, tp = X[mp], t[mp]
                        if len(tp) < 500:
                            continue
                        G = Xp.T @ Xp + lam * np.eye(X.shape[-1])
                        beta = np.linalg.solve(G, Xp.T @ tp)
                        lam_d = min(1.0, k / args.ramp) * st
                        mc = cur[:, None, None] & e
                        pf[mc] = np.clip(pg[mc] + lam_d * (X[mc] @ beta), 0, 1)
                    v_ = C.iou(pf >= TAU, y, e)
                    fam.setdefault((s, f"lam {lam:g} step {st}"), {})[panel] = v_.mean()
                    fam_e.setdefault((s, f"lam {lam:g} step {st}"), {})[panel] = v_[early].mean()
        print(f"  {panel}", flush=True)

    names = ["global"] + [f"lam {l:g} step {s}" for l in args.lams for s in args.steps]
    print(f"\nongoing family IoU after the threshold; adapter fitted on strictly earlier days of the same run")
    print(f"ramp {args.ramp} days, so lam_d = min(1, past_days/{args.ramp}) * step\n")
    print(f"{'setting':<22}" + "".join(f"{s:>13}" for s in C.SPLITS) + f"{'mean':>10}{'vs global':>11}")
    g0 = None
    for nm in names:
        if (C.SPLITS[0], nm) not in fam:
            continue
        v = [C.family_mean(fam[(s, nm)]) for s in C.SPLITS]
        m = float(np.mean(v))
        if g0 is None:
            g0 = m
        print(f"{nm:<22}" + "".join(f"{x:>13.4f}" for x in v) + f"{m:>10.4f}{m - g0:>+11.4f}")

    print(f"\nthe same, restricted to windows with at most {args.early} earlier days - the official windows' "
          f"own regime\n")
    print(f"{'setting':<22}" + "".join(f"{s:>13}" for s in C.SPLITS) + f"{'mean':>10}{'vs global':>11}")
    g0 = None
    for nm in names:
        if (C.SPLITS[0], nm) not in fam_e:
            continue
        v = [C.family_mean(fam_e[(s, nm)]) for s in C.SPLITS]
        m = float(np.mean(v))
        if g0 is None:
            g0 = m
        print(f"{nm:<22}" + "".join(f"{x:>13.4f}" for x in v) + f"{m:>10.4f}{m - g0:>+11.4f}")


if __name__ == "__main__":
    main()
