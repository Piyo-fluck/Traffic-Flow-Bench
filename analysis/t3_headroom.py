"""Task 3: how much of the S_LWR error is reducible at all, and where?

S_LWR's residual at a target transition is the density error, and the truth itself carries observation noise
(about 0.9 km/h and 18 veh/h/lane; note 01), so part of the error is a noise floor no method can remove. Oracle
variants on the train holdout (saved `m3all` predictions):
    current    the CNN prediction
    sm3        every target cell = the truth smoothed over 3 time slots (a near-noise-free truth)
    cong_sm3   sm3 only at congested target cells (truth v <= 0.6 vf); CNN elsewhere
    free_sm3   sm3 only at free-flow target cells; CNN elsewhere
sm3 approximates the best possible predictor, so S_LWR(sm3) - S_LWR(current) bounds the reducible part and
the congested/free split shows where it lies. S_state is shown for scale (its noise ceiling is about 0.96).

    py analysis/t3_headroom.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tfb.data import PANELS, load  # noqa: E402
from tfb.features import profile, shift, truth_valid  # noqa: E402
from tfb.metrics import state_score  # noqa: E402
from tfb.physics import fd_params, physics_scores, ramp_ok  # noqa: E402
from tfb.t1eval import family_mean, split_days  # noqa: E402

TAG = "m3all"


def smooth3(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    x = np.where(valid, x, np.nan)
    with np.errstate(all="ignore"):
        return np.nanmean(np.stack([shift(x, 1, 1), x, shift(x, -1, 1)]), 0)


def main() -> None:
    out: dict[str, dict[str, tuple[float, float]]] = {}
    for panel in PANELS:
        f = ROOT / "work" / "preds" / f"t1_{TAG}_{panel}_test.npz"
        if not f.exists():
            continue
        z = load(panel, "train")
        fit, _ = split_days(z)
        p = np.load(f)
        valid = truth_valid(z)
        pv = profile(z["true_speed"], valid, z["weekday"], fit)
        pq = profile(z["true_flow"], valid, z["weekday"], fit)
        pv[p["days"]], pq[p["days"]] = p["pv"], p["pq"]
        days = np.zeros(len(z["dates"]), bool)
        days[p["days"]] = True
        prm, ok = fd_params(panel, z["link_ids"]), ramp_ok(panel, "train", z)
        tgt = z["target"]
        ts, tq = z["true_speed"], z["true_flow"]
        use = z["elig"] & np.isfinite(ts) & np.isfinite(tq)
        sv, sq = smooth3(ts, use), smooth3(tq, use)
        sv, sq = np.where(np.isfinite(sv), sv, pv), np.where(np.isfinite(sq), sq, pq)
        cong = np.nan_to_num(ts, nan=1e9) <= 0.6 * z["free_speed"]
        variants = {
            "current": (pv, pq),
            "sm3": (np.where(tgt, sv, pv), np.where(tgt, sq, pq)),
            "cong_sm3": (np.where(tgt & cong, sv, pv), np.where(tgt & cong, sq, pq)),
            "free_sm3": (np.where(tgt & ~cong, sv, pv), np.where(tgt & ~cong, sq, pq)),
        }
        line = []
        for name, (a, b) in variants.items():
            s = state_score(z, a, b, cells=days[:, None, None])["mean"]["S"]
            l = physics_scores(z, a, b, prm, ok, days=days)["mean"]["S_LWR"]
            out.setdefault(name, {})[panel] = (s, l)
            line.append(f"{name} {s:.4f}/{l:.4f}")
        print(f"{panel:<11} S_state/S_LWR~  " + "  ".join(line), flush=True)
    print("\n== family mean (train holdout) ==")
    for name, d in out.items():
        s = family_mean({k: v[0] for k, v in d.items()})
        l = family_mean({k: v[1] for k, v in d.items()})
        print(f"  {name:<9} S_state {s:.4f}  S_LWR~ {l:.4f}")


if __name__ == "__main__":
    main()
