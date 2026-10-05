"""Task 1/3 probe: does blending each target prediction with its time neighbours raise S_LWR without hurting S_state?

S_LWR punishes the time difference of density errors at consecutive cells
(note 04, F17), and ensembling helped it mostly by removing independent noise
(note 07). Post-process: at target cells only,
    new_t = (1 - 2w) * p_t + w * x_{t-1} + w * x_{t+1}
where x is the released observation at visible cells and the prediction
elsewhere. Applied to speed and flow separately. Scored on the saved
test-day predictions of a method.

    py analysis/t1_post_smooth.py m3all D12_I5_N D7_I210_W
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tfb.data import load  # noqa: E402
from tfb.features import profile, truth_valid  # noqa: E402
from tfb.metrics import state_score  # noqa: E402
from tfb.physics import fd_params, physics_scores, ramp_ok  # noqa: E402
from tfb.t1eval import family_mean, split_days  # noqa: E402

WS = (0.0, 0.1, 0.2, 0.3)


def blend(pred: np.ndarray, obs: np.ndarray, target: np.ndarray, w: float) -> np.ndarray:
    if w == 0:
        return pred
    x = np.where(np.isfinite(obs), obs, pred)
    prev = np.concatenate([x[:, :1], x[:, :-1]], axis=1)
    nxt = np.concatenate([x[:, 1:], x[:, -1:]], axis=1)
    return np.where(target, (1 - 2 * w) * pred + w * prev + w * nxt, pred)


def main() -> None:
    tag, *panels = sys.argv[1:]
    res = {w: ({}, {}, {}) for w in WS}
    for panel in panels:
        z = load(panel, "train")
        fit, test = split_days(z)
        p = np.load(ROOT / "work" / "preds" / f"t1_{tag}_{panel}_test.npz")
        valid = truth_valid(z)
        pv = profile(z["true_speed"], valid, z["weekday"], fit)
        pq = profile(z["true_flow"], valid, z["weekday"], fit)
        pv[p["days"]], pq[p["days"]] = p["pv"], p["pq"]
        mask = np.zeros(len(z["dates"]), bool)
        mask[p["days"]] = True
        prm, ok = fd_params(panel, z["link_ids"]), ramp_ok(panel, "train", z)
        tgt = z["target"]
        for w in WS:
            bv, bq = blend(pv, z["obs_speed"], tgt, w), blend(pq, z["obs_flow"], tgt, w)
            s = state_score(z, bv, bq, cells=mask[:, None, None])["mean"]["S"]
            f = physics_scores(z, bv, bq, prm, ok, days=mask)["mean"]
            res[w][0][panel], res[w][1][panel], res[w][2][panel] = s, f["S_LWR"], f["S_phys"]
            print(f"{panel:<11} w={w:.1f}  S_state {s:.4f}  S_LWR~ {f['S_LWR']:.4f}  S_phys~ {f['S_phys']:.4f}", flush=True)
    print(f"\n[{tag}] family mean over {len(panels)} panels")
    for w, (s, l, ph) in res.items():
        a, b, c = family_mean(s), family_mean(l), family_mean(ph)
        print(f"  w={w:.1f}  S_state {a:.4f}  S_LWR~ {b:.4f}  S_phys~ {c:.4f}  0.35*S_state+0.15*S_phys~ {0.35 * a + 0.15 * c:.4f}")


if __name__ == "__main__":
    main()
