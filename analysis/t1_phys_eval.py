"""Score saved test-day predictions of one method on Task 1 and the Task 3 proxy, per panel and overall.

    py analysis/t1_phys_eval.py m3all            # every panel with a saved prediction
    py analysis/t1_phys_eval.py m3 D12_I5_N D7_I210_W
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


def main() -> None:
    tag, *panels = sys.argv[1:]
    panels = panels or [p for p in PANELS if (ROOT / "work" / "preds" / f"t1_{tag}_{p}_test.npz").exists()]
    st, ph, lw = {}, {}, {}
    print(f"{'panel':<11} {'S_state':>8} {'S_FD':>7} {'S_LWR~':>7} {'S_phys~':>8}   S_LWR~ R1/R2/R3")
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
        s = state_score(z, pv, pq, cells=mask[:, None, None])["mean"]["S"]
        f = physics_scores(z, pv, pq, fd_params(panel, z["link_ids"]), ramp_ok(panel, "train", z), days=mask)
        st[panel], ph[panel], lw[panel] = s, f["mean"]["S_phys"], f["mean"]["S_LWR"]
        print(f"{panel:<11} {s:>8.4f} {f['mean']['S_FD']:>7.4f} {f['mean']['S_LWR']:>7.4f} {f['mean']['S_phys']:>8.4f}   "
              + "/".join(f"{f[r]['S_LWR']:.3f}" for r in ("R1", "R2", "R3")), flush=True)
    s, f, l = family_mean(st), family_mean(ph), family_mean(lw)
    print(f"\n[{tag}] family mean: S_state {s:.4f} | S_LWR~ {l:.4f} | S_phys~ {f:.4f} | "
          f"0.35*S_state + 0.15*S_phys~ = {0.35 * s + 0.15 * f:.4f}")


if __name__ == "__main__":
    main()
