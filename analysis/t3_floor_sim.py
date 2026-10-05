"""Task 3: what is the best S_LWR any method can reach? (a proper floor, replacing note 23's smoothed-truth oracle)

S_LWR compares 5-min changes of the submitted density with those of the noisy observed truth (the organizer rhs is
projected onto the released observations). A perfect answer, the exact noisy values, scores S_physics about 0.96
(README). A method can at best recover the underlying (noise-free) state. Its error against the noisy truth is
then exactly the observation noise, which cannot be predicted if it is independent per cell.

Note 23 used the truth smoothed over 3 slots as that best predictor. That oracle is biased (it even scores below
the CNN on S_state), so it does not establish a floor. Here the best predictor is simulated honestly instead: at
target cells, truth + fresh noise of the measured size, so its error has the same distribution as an ideal
predictor's error.
  noise size   robust sigma from the second difference in time (note 01), per panel: speed separately for free
               flow and congestion (v <= 0.6 vf), flow per lane
  variants     current CNN (`m3all`); floor = truth + noise(sigma); floor at 0.5 sigma and 1.5 sigma (sensitivity)
Reported: S_state and S_LWR~ (proxy, train holdout days), family mean.

    py analysis/t3_floor_sim.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tfb.data import PANELS, load  # noqa: E402
from tfb.features import profile, truth_valid  # noqa: E402
from tfb.metrics import state_score  # noqa: E402
from tfb.physics import fd_params, physics_scores, ramp_ok  # noqa: E402
from tfb.t1eval import family_mean, split_days  # noqa: E402

TAG = "m3all"


def robust_sigma(x: np.ndarray, ok: np.ndarray, sel: np.ndarray) -> float:
    d2 = x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]
    m = ok[:, 2:] & ok[:, 1:-1] & ok[:, :-2] & sel[:, 1:-1]
    v = d2[m]
    return float(1.4826 * np.median(np.abs(v - np.median(v))) / np.sqrt(6))


def main() -> None:
    rng = np.random.default_rng(0)
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
        ts, tq, lanes = z["true_speed"], z["true_flow"], z["lanes"]
        use = z["elig"] & np.isfinite(ts) & np.isfinite(tq)
        cong = np.nan_to_num(ts, nan=1e9) <= 0.6 * z["free_speed"]
        s_vf = robust_sigma(ts, use, ~cong)
        s_vc = robust_sigma(ts, use, cong)
        s_q = robust_sigma(tq / lanes, use, np.ones_like(use))
        variants = {"current": (pv, pq)}
        for mult in (1.0, 0.5, 1.5):
            sv = np.where(cong, s_vc, s_vf) * mult
            nv = ts + rng.normal(0, 1, ts.shape) * sv
            nq = tq + rng.normal(0, 1, tq.shape) * s_q * mult * lanes
            variants[f"floor x{mult}"] = (np.where(np.isfinite(nv), np.maximum(nv, 1.0), pv),
                                          np.where(np.isfinite(nq), np.maximum(nq, 0.0), pq))
        line = []
        for name, (a, b) in variants.items():
            s = state_score(z, a, b, cells=days[:, None, None])["mean"]["S"]
            l = physics_scores(z, a, b, prm, ok, days=days)["mean"]["S_LWR"]
            out.setdefault(name, {})[panel] = (s, l)
            line.append(f"{name} {s:.4f}/{l:.4f}")
        print(f"{panel:<11} sigma v free {s_vf:.2f} cong {s_vc:.2f} q/lane {s_q:.1f} | S_state/S_LWR~  " + "  ".join(line),
              flush=True)
    print("\n== family mean (train holdout) ==")
    for name, d in out.items():
        s = family_mean({k: v[0] for k, v in d.items()})
        l = family_mean({k: v[1] for k, v in d.items()})
        print(f"  {name:<12} S_state {s:.4f}  S_LWR~ {l:.4f}")


if __name__ == "__main__":
    main()
