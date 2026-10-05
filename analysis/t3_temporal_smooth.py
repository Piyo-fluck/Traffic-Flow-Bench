"""Post-hoc temporal smoothing of the submitted state: trade a little S_state for S_LWR.

The scoring weights are   total = 0.35 S_state + 0.05 S_FD + 0.10 S_LWR,  so one point of S_LWR is worth 0.29
points of S_state: a trade pays whenever it buys more than 3.5 times what it costs.

S_LWR's residual is |(e[t+1] - e[t]) L|, so errors that are correlated in time cancel. Pulling each target cell
towards its temporal neighbours IN THE SUBMITTED STATE does two things at once: where a neighbour is visible the
prediction is anchored to the truth, and where it is not the errors become correlated.

    x_new[t] = (1 - w) x_pred[t] + w * mean(x_submitted[t-1], x_submitted[t+1])      at target cells only

Nothing here uses anything but the released layer and our own predictions.

    py analysis/t3_temporal_smooth.py [tag]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tfb.data import PANELS, load  # noqa: E402
from tfb.metrics import state_score  # noqa: E402
from tfb.physics import fd_params, physics_scores, ramp_ok  # noqa: E402
from tfb.t1eval import family_mean  # noqa: E402

PRED = ROOT / "work" / "preds"
TAG = sys.argv[1] if len(sys.argv) > 1 else "t1_m3A"
WS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5)


def smooth(pred, sub, tgt, w, valid):
    """One pass: pull target cells towards the mean of their VALID temporal neighbours in the submitted state."""
    nb = np.zeros_like(pred)
    cnt = np.zeros_like(pred)
    sv = np.where(valid, sub, 0.0)
    nb[:, 1:] += sv[:, :-1]; cnt[:, 1:] += valid[:, :-1]
    nb[:, :-1] += sv[:, 1:]; cnt[:, :-1] += valid[:, 1:]
    has = cnt > 0
    nb = np.where(has, nb / np.maximum(cnt, 1), pred)
    return np.where(tgt & has, (1 - w) * pred + w * nb, pred)


def main() -> None:
    out = {}
    for panel in PANELS:
        f = PRED / f"{TAG}_{panel}_test.npz"
        if not f.exists():
            continue
        p = np.load(f)
        z = load(panel, "train")
        days = p["days"]
        prm, ok = fd_params(panel, z["link_ids"]), ramp_ok(panel, "train", z)
        ts, tq = z["true_speed"], z["true_flow"]
        tgt = z["target"] & np.isfinite(ts) & np.isfinite(tq)
        sel = np.zeros(len(z["dates"]), bool); sel[days] = True
        pv = np.where(np.isfinite(ts), ts, 0.0).copy(); pq = np.where(np.isfinite(tq), tq, 0.0).copy()
        pv[days] = np.where(tgt[days], p["pv"], np.nan_to_num(ts[days]))
        pq[days] = np.where(tgt[days], p["pq"], np.nan_to_num(tq[days]))
        for w in WS:
            valid = z["elig"] & np.isfinite(ts) & np.isfinite(tq)
            sv = smooth(pv, pv, tgt & sel[:, None, None], w, valid)
            sq = smooth(pq, pq, tgt & sel[:, None, None], w, valid)
            ph = physics_scores(z, sv, sq, prm, ok, days=sel)
            st = state_score(z, sv, sq, cells=sel[:, None, None])
            out.setdefault(w, {})[panel] = (ph["mean"]["S_LWR"], ph["mean"]["S_FD"], ph["mean"]["S_phys"],
                                            st["mean"]["S"])
        print(f"  {panel}", flush=True)
    print(f"\n{'w':>6} {'S_LWR':>8} {'S_FD':>8} {'S_phys':>8} {'S_state':>9} {'0.35S+0.15P':>12} {'vs w=0':>9}")
    base = None
    for w in WS:
        v = [family_mean({p: out[w][p][i] for p in out[w]}) for i in range(4)]
        tot = 0.35 * v[3] + 0.15 * v[2]
        base = tot if base is None else base
        print(f"{w:6.2f} {v[0]:8.4f} {v[1]:8.4f} {v[2]:8.4f} {v[3]:9.4f} {tot:12.5f} {tot - base:+9.5f}")


if __name__ == "__main__":
    main()
