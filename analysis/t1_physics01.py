"""Task 3 EDA 01: what does the physics score make of each Task 1 reconstruction?

Scores, on the holdout test days (Jan-Feb 2031): the fit-day profile baseline,
the time-anomaly interpolation, the saved M2 / M3 predictions, and the truth
itself. S_FD is exact; S_LWR is the proxy from tfb/physics.py.

    py analysis/t1_physics01.py D12_I5_N D7_I210_W
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tfb.data import load  # noqa: E402
from tfb.features import interp_along, profile, truth_valid  # noqa: E402
from tfb.metrics import state_score  # noqa: E402
from tfb.physics import fd_params, fmt_phys, physics_scores, ramp_ok  # noqa: E402
from tfb.t1eval import split_days  # noqa: E402


def main() -> None:
    for panel in sys.argv[1:]:
        z = load(panel, "train")
        fit, test = split_days(z)
        prm = fd_params(panel, z["link_ids"])
        ok = ramp_ok(panel, "train", z)
        valid = truth_valid(z)
        base_v = profile(z["true_speed"], valid, z["weekday"], fit)
        base_q = profile(z["true_flow"], valid, z["weekday"], fit)
        vis = np.isfinite(z["obs_speed"]) & np.isfinite(z["obs_flow"])
        tv = base_v + np.nan_to_num(interp_along(z["obs_speed"] - base_v, vis, 1)[0])
        tq = base_q + np.nan_to_num(interp_along(z["obs_flow"] - base_q, vis, 1)[0])
        cands = {"baseline profile": (base_v, base_q), "time-anomaly interp": (tv, tq),
                 "truth (perfect)": (z["true_speed"], z["true_flow"])}
        for tag in ("m2", "m3"):
            f = ROOT / "work" / "preds" / f"t1_{tag}_{panel}_test.npz"
            if f.exists():
                p = np.load(f)
                pv, pq = base_v.copy(), base_q.copy()
                pv[p["days"]], pq[p["days"]] = p["pv"], p["pq"]
                cands[tag.upper()] = (pv, pq)
        print(f"\n== {panel}: ramp-rule pass share {ok.mean():.3f} ==")
        for name, (pv, pq) in cands.items():
            st = state_score(z, pv, pq, cells=test[:, None, None])["mean"]["S"]
            ph = physics_scores(z, pv, pq, prm, ok, days=test)
            print(f"  {name:<20} S_state {st:.4f} | {fmt_phys(ph)} | transitions touching a target {ph['mean']['touched']:.2f}")
        # How big are density errors against the conservation signal?
        pv, pq = cands.get("M3", cands["time-anomaly interp"])
        m = z["target"] & test[:, None, None] & np.isfinite(z["true_speed"])
        k_t = z["true_flow"] / np.maximum(z["true_speed"], 1)
        k_p = pq / np.maximum(pv, 1)
        rel = np.abs(k_p - k_t)[m] / np.maximum(k_t[m], 1e-6)
        dn = np.abs(np.diff(k_t * prm["length"], axis=1))
        nn = (k_t * prm["length"])[:, 1:]
        sig = dn[z["elig"][:, 1:]] / np.maximum(nn[z["elig"][:, 1:]], 1e-6)
        print(f"  density error at targets (best model): median {np.median(rel):.4f}, mean {rel.mean():.4f} | "
              f"conservation signal |dN|/N: median {np.nanmedian(sig):.4f}, mean {np.nanmean(sig):.4f}")


if __name__ == "__main__":
    main()
