"""Idea 3 (T4C 2022 winner: impute the full state first, then forecast) - how much is there to win?

The members are trained on the training run's TRUE histories (every eligible cell observed) but the official window
histories are only ~76 % observed. On the training run's early-stopping windows (Dec 2030, never trained on) the
cnn3 members are scored with
    full        the true history, every eligible cell (the training condition; an oracle for imputation)
    masked      the official windows' missingness applied (each ES window gets the observed-mask of a random
                official window of the same panel, both splits, shifted in time as is)
    interp      masked, then each link's missing slots filled by linear interpolation in time (edges: nearest),
                links with no observation filled from the +-1 / +-2 neighbours' values
Ongoing family IoU (tau 0.5) and onset (ratio rule).

    CUDA_VISIBLE_DEVICES=0 .venv/bin/python occ_lab/fill_diag.py
"""
from __future__ import annotations

import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
import longctx as LC  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

C, cnn = LC.C, LC.cnn
H, F, DEV = LC.H, LC.F, LC.DEV
sys.path.insert(0, str(LAB.parent / "scripts"))
from t2_m1_predict import hist_from_release  # noqa: E402
from tfb.data import RELEASE  # noqa: E402

CKPTS = ["t2_cnn_lov", "t2_cnn_lov_s1", "t2_cnn_lov_s2"]


def load_net(tag, cin):
    net = cnn.T2Net(cin, 64).to(DEV)                  # cnn3 are width 64 (longctx sets WIDTH 192)
    net.load_state_dict(torch.load(LAB.parent / "work" / "models" / f"{tag}.pt", map_location=DEV), strict=False)
    return net.eval()


def interp_fill(v, ok):
    """(N, H, S) values, observed mask -> filled values and 'filled' mask (in time, then space)."""
    out = v.copy()
    N, Hh, S = v.shape
    t = np.arange(Hh)
    have = ok.copy()
    for n in range(N):
        for s in range(S):
            o = ok[n, :, s]
            if o.all() or not o.any():
                continue
            out[n, ~o, s] = np.interp(t[~o], t[o], v[n, o, s])
            have[n, :, s] = True
    for n in range(N):                                # links never observed: nearest observed neighbour's series
        miss = ~have[n].any(0)
        for s in np.flatnonzero(miss):
            for r in (1, -1, 2, -2):
                k = s + r
                if 0 <= k < S and have[n, :, k].any():
                    out[n, :, s] = out[n, :, k]; have[n, :, s] = True
                    break
    return out, have


def main() -> None:
    rng = np.random.default_rng(0)
    parts = LC.train_panels()
    res = {}
    for p in parts:
        ctx, stat = p["ctx"], p["stat"]
        # official missingness masks of this panel
        masks = []
        for split in C.SPLITS:
            base = RELEASE / "task2" / p["panel"] / split
            Wi = pd.read_csv(base / "window_index.csv")
            Hh = pd.read_parquet(base / "window_history.parquet")
            _, _, ok, _, _ = hist_from_release(ctx, Hh, Wi)
            masks.append(ok)
        masks = np.concatenate(masks)
        n_on = len(ctx.eo[0])
        es = np.flatnonzero(p["es"])
        for kind, sel in (("onset", es[es < n_on]), ("ongoing", es[es >= n_on])):
            if not len(sel):
                continue
            d, t0 = p["d"][sel], p["t0"][sel]
            hi = t0[:, None] + np.arange(-H, 0)
            v, qf, okf = p["v"][d[:, None], hi], p["qf"][d[:, None], hi], p["ok"][d[:, None], hi]
            mk = masks[rng.integers(len(masks), size=len(sel))] & okf
            fi = t0[:, None] + np.arange(1, F + 1)
            y, m = p["q"][d[:, None], fi], p["el"][d[:, None], fi]
            vi, hv = interp_fill(np.where(mk, v, np.nan), mk)
            qi, _ = interp_fill(np.where(mk, qf, np.nan), mk)
            X = {"full": (v, qf, okf), "masked": (v, qf, mk), "interp": (vi, qi, hv & okf)}
            for nm, (vv, qq, oo) in X.items():
                x = cnn.make_x(ctx, stat, np.nan_to_num(vv), np.nan_to_num(qq), oo, t0, p["wk"][d] >= 5)
                pr = np.mean([cnn.predict(load_net(t, x.shape[1]), x) for t in CKPTS], 0)
                if kind == "ongoing":
                    sc = C.iou(pr >= 0.5, y, m)
                else:
                    pred = np.zeros(y.shape, bool)
                    pred[:, F - 1] = np.stack([C.select(q, "ratio", 1.0) for q in pr[:, F - 1]])
                    sc = C.iou(pred, y & m, m)
                res.setdefault((kind, nm), {})[p["panel"]] = sc.mean()
            print(f"  {p['panel']} {kind}: observed share full {okf.mean():.3f} masked {mk.mean():.3f}  " +
                  "  ".join(f"{nm} {res[(kind, nm)][p['panel']]:.4f}" for nm in X), flush=True)
    for kind in ("ongoing", "onset"):
        print(f"{kind}: " + "  ".join(f"{nm} {C.family_mean(res[(kind, nm)]):.4f}" for nm in ("full", "masked", "interp")))


if __name__ == "__main__":
    main()
