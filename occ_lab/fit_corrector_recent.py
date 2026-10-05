"""One corrector per scored month, trained on everything released before that month, recent days weighted up.

Every measurement of the residual corrector says the same thing: the closer the training days are to the target,
the larger the gain (tail -> March / April +0.0044 / +0.0045; adjacent month +0.0117 / +0.0079; January ->
February +0.017 .. +0.025, notes 86-87). Note 91 showed that using the scored month's OWN earlier days fails
where it matters, because the official ongoing windows sit on days 1-6, when that pool is empty or tiny.

What is available without that problem:

    April (private)      the 2031-01/02 tail + ALL of March, March weighted 3x
                         (every March observation precedes every April origin)
    March (validation)   the tail only, February weighted 3x
                         (nothing between February and March exists)

    python occ_lab/fit_corrector_recent.py
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
from residual_diag import NAMES  # noqa: E402
from residual_oof import panel_data as scored_panel  # noqa: E402

W_RECENT = 3.0


def tail_parts():
    """The 2031-01/02 windows, with the month of each cell so February can be weighted up."""
    X, y, p0, feb = [], [], [], []
    for p in C.T2_PANELS:
        lab = np.load(CACHE / f"cov_labels_{p}.npz")
        f, base, _ = load_env("obs", p)
        m = lab["e"]
        X.append(f[m][:, IDX]); y.append(lab["y"][m].astype(np.float32)); p0.append(base[m])
        is_feb = np.char.startswith(lab["date"].astype(str), "2031-02")
        feb.append(np.broadcast_to(is_feb[:, None, None], lab["e"].shape)[m])
    return np.concatenate(X), np.concatenate(y), np.concatenate(p0), np.concatenate(feb)


def month_parts(split):
    X, y, p0 = [], [], []
    for p in C.T2_PANELS:
        d = scored_panel(p, split)
        if d is None:
            continue
        f, yy, e, base = d
        X.append(f[e]); y.append(yy[e].astype(np.float32)); p0.append(base[e])
    return np.concatenate(X), np.concatenate(y), np.concatenate(p0)


def train(X, y, p0, w, out):
    gbm = lgb.train(dict(PARAMS, num_threads=32),
                    lgb.Dataset(X, label=y - p0, weight=w, feature_name=NAMES, free_raw_data=False),
                    num_boost_round=400)
    gbm.save_model(str(out))
    print(f"  saved {out.name}: {len(y):,} cells, mean weight {w.mean():.2f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=LAB / "models")
    args = ap.parse_args()
    Xt, yt, pt, feb = tail_parts()
    print(f"tail: {len(yt):,} cells, {feb.mean() * 100:.0f}% of them February", flush=True)

    # March: the tail with February weighted up
    train(Xt, yt, pt, np.where(feb, W_RECENT, 1.0).astype(np.float32),
          args.out_dir / "residual_lgb_recent_validation.txt")

    # April: the tail plus all of March, March weighted up
    Xm, ym, pm = month_parts("validation")
    print(f"March: {len(ym):,} cells", flush=True)
    train(np.concatenate([Xt, Xm]), np.concatenate([yt, ym]), np.concatenate([pt, pm]),
          np.r_[np.ones(len(yt), np.float32), np.full(len(ym), W_RECENT, np.float32)],
          args.out_dir / "residual_lgb_recent_private.txt")


if __name__ == "__main__":
    main()
