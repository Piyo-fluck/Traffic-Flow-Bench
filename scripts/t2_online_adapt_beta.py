"""Fit the run-specific online residual adapter's per-day coefficients, for production to apply.

Item 1 of the user's Task 2 plan, measured in `occ_lab/online_adapter.py`: the global corrector learned
conditional competence - which member to trust where - from the train tail, and candidate AR showed the same
run's earlier days carry enough direct queue supervision to hold 0.60 of the ongoing blend. Conditional
competence is itself a property of the run, so that supervision should reach the corrector layer too.

    p_global = clip(p0 + eta * rhat)                     the shipped corrector, unchanged
    p_final  = clip(p_global + lam_d * X beta_d)

beta_d is a ridge toward zero, so the shrinkage centre is "use the global corrector unchanged", and lam_d ramps
with the number of earlier days available - the first day of a run applies nothing. Measured +0.0049 on the
ongoing family on the whole month. The OFFICIAL windows are not in that regime: they have one to four earlier
days, not ten to twenty (note 120), and re-measured there the gain is +0.0014 at the tuned ramp of 5 days.
Shortening the ramp so lam_d is up from the first available day recovers it to +0.0030, on an interior plateau
over lam 30-100, step 0.75-1.0 and ramp 1-3; ramp 2, lam 30, step 0.75 sits inside on all three axes.

The fit runs on the RELEASE-derived pseudo ongoing windows of the scored month (`common.windows`, which excludes
the official blanks), whose labels are released observations - so nothing here needs a scored horizon. Only the
coefficients cross into production: `scripts/t2_residual_predict.py --adapt-dir` applies them to the official
windows' own features.

Writes beta_<split>_<panel>.npz with beta (D, K), lam (D,), dates (D,) and the column names.

    .venv/Scripts/python.exe scripts/t2_online_adapt_beta.py --out-dir work/t2_adapt_beta --tag s20
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "occ_lab"))
sys.path.insert(0, str(ROOT / "tta_lab" / "tta"))
sys.path.insert(0, str(ROOT / "analysis"))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

import common as C  # noqa: E402
from online_adapter import COLS, adapter_cols, global_features, load_split  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--tag", default="s20")
    ap.add_argument("--corrector", type=Path, default=ROOT / "occ_lab" / "models" / "residual_lgb.txt")
    ap.add_argument("--lam", type=float, default=30.0)
    ap.add_argument("--step", type=float, default=0.75)
    ap.add_argument("--ramp", type=int, default=2)
    ap.add_argument("--w-pseudo", type=float, default=0.60)
    ap.add_argument("--eta", type=float, default=0.75)
    ap.add_argument("--gate", type=float, default=0.10)
    ap.add_argument("--min-cells", type=int, default=500)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    gbm = lgb.Booster(model_file=str(args.corrector))
    print(f"adapter: lam {args.lam:g}, step {args.step}, ramp {args.ramp} days, {len(COLS)} columns; "
          f"base w_pseudo {args.w_pseudo}, corrector eta {args.eta} gate {args.gate}", flush=True)
    t0_ = time.time()
    for i, panel in enumerate(C.T2_PANELS):
        sp = load_split(i, panel, args.tag, w_ps=args.w_pseudo)
        for s, D in sp.items():
            base, y, e, dd = D["base"], D["y"], D["e"], D["d"]
            F = global_features(D["P"], base)
            rh = gbm.predict(F.reshape(-1, F.shape[-1])).reshape(base.shape).astype(np.float32)
            pg = np.clip(base + args.eta * np.where(np.abs(rh) >= args.gate, rh, 0.0), 0, 1)
            X = adapter_cols(D, base, pg, rh).astype(np.float64)
            t = y.astype(np.float64) - pg
            days = np.unique(dd)
            nd = int(D["n_days"])
            beta = np.zeros((nd, X.shape[-1]))
            lam_d = np.zeros(nd)
            for day in range(nd):
                past = np.isin(dd, days[days < day])
                n_past = int((days < day).sum())
                if not past.any() or n_past == 0:
                    continue
                mp = past[:, None, None] & e
                Xp, tp = X[mp], t[mp]
                if len(tp) < args.min_cells:
                    continue
                G = Xp.T @ Xp + args.lam * np.eye(X.shape[-1])
                beta[day] = np.linalg.solve(G, Xp.T @ tp)
                lam_d[day] = min(1.0, n_past / args.ramp) * args.step
            f = args.out_dir / f"beta_{s}_{panel}.npz"
            np.savez(f, beta=beta, lam=lam_d, dates=D["dates"], cols=np.array(COLS),
                     lam_ridge=args.lam, step=args.step, ramp=args.ramp,
                     w_pseudo=args.w_pseudo, eta=args.eta, gate=args.gate)
            print(f"[{s} {panel}] {len(days)} pool days of {nd}, {int((lam_d > 0).sum())} days get an adapter, "
                  f"|beta| median {np.median(np.abs(beta[lam_d > 0])):.2e} ({time.time() - t0_:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
