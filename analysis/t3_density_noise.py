"""Task 3: the S_LWR floor depends on the noise of DENSITY k = q/v. Is it what independent speed and flow noise implies?

t3_floor_sim put the best reachable S_LWR~ at about 0.684 (the CNN has 0.649), assuming independent noise on speed
and flow. S_LWR only sees k = q/v. If the observation noise on q and v is correlated, the noise on k is smaller or
larger than assumed, and the floor moves a lot (0.5 sigma → 0.84, 1.5 sigma → 0.53). Per panel (train, eligible,
free flow = v > 0.6 vf):
    sig_logk_meas   robust sigma of the noise on log k, from its second difference in time
    sig_logk_indep  what independent speed and flow noise would give: sqrt(sig_logq^2 + sig_logv^2)
    corr_d2         correlation of the second differences of log q and log v (noise correlation)
and the floor recomputed with density noise at the measured size: at target cells v' = v_true,
q' = k_true * exp(sig_logk * N(0, 1)) * v_true (free and congested cells with their own sigma).

    py analysis/t3_density_noise.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tfb.data import PANELS, load  # noqa: E402
from tfb.features import profile, truth_valid  # noqa: E402
from tfb.physics import fd_params, physics_scores, ramp_ok  # noqa: E402
from tfb.t1eval import family_mean, split_days  # noqa: E402

TAG = "m3all"


def d2(x):
    return x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]


def rsig(v):
    v = v[np.isfinite(v)]
    return float(1.4826 * np.median(np.abs(v - np.median(v))) / np.sqrt(6))


def main() -> None:
    rng = np.random.default_rng(1)
    rows, floors = {}, {}
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
        ts, tq = z["true_speed"], z["true_flow"]
        use = z["elig"] & np.isfinite(ts) & np.isfinite(tq) & (ts > 1) & (tq > 1)
        lv, lq = np.log(np.where(use, ts, np.nan)), np.log(np.where(use, tq, np.nan))
        lk = lq - lv
        cong = np.nan_to_num(ts, nan=1e9) <= 0.6 * z["free_speed"]
        m3 = use[:, 2:] & use[:, 1:-1] & use[:, :-2]
        fr, cg = m3 & ~cong[:, 1:-1], m3 & cong[:, 1:-1]
        dv, dq, dk = d2(lv), d2(lq), d2(lk)
        s_v, s_q, s_k = rsig(dv[fr]), rsig(dq[fr]), rsig(dk[fr])
        s_kc = rsig(dk[cg]) if cg.sum() > 100 else s_k
        a, b = dv[fr], dq[fr]
        lo_a, hi_a = np.percentile(a, [1, 99]); lo_b, hi_b = np.percentile(b, [1, 99])
        keep = (a > lo_a) & (a < hi_a) & (b > lo_b) & (b < hi_b)
        corr = float(np.corrcoef(a[keep], b[keep])[0, 1])
        rows[panel] = dict(sig_logk_meas=s_k, sig_logk_indep=float(np.hypot(s_v, s_q)), corr_d2=corr, sig_logk_cong=s_kc)
        k_true = tq / np.maximum(ts, 1.0)
        sig = np.where(cong, s_kc, s_k)
        k_new = k_true * np.exp(sig * rng.normal(0, 1, k_true.shape))
        v2 = np.where(np.isfinite(ts), ts, pv)
        q2 = np.where(np.isfinite(k_new), k_new * v2, pq)
        l_cur = physics_scores(z, pv, pq, prm, ok, days=days)["mean"]["S_LWR"]
        l_flr = physics_scores(z, v2, q2, prm, ok, days=days)["mean"]["S_LWR"]
        floors[panel] = (l_cur, l_flr)
        r = rows[panel]
        print(f"{panel:<11} sig log k: measured {r['sig_logk_meas']:.4f} vs independent {r['sig_logk_indep']:.4f} "
              f"(corr d2 log q / log v {r['corr_d2']:+.2f}; congested {r['sig_logk_cong']:.4f}) | "
              f"S_LWR~ current {l_cur:.4f}, density-noise floor {l_flr:.4f}", flush=True)
    print("\n== family mean (train) ==")
    for k in ("sig_logk_meas", "sig_logk_indep", "corr_d2", "sig_logk_cong"):
        print(f"  {k:<15} {family_mean({p: r[k] for p, r in rows.items()}):.4f}")
    print(f"  S_LWR~ current {family_mean({p: v[0] for p, v in floors.items()}):.4f}  "
          f"density-noise floor {family_mean({p: v[1] for p, v in floors.items()}):.4f}")


if __name__ == "__main__":
    main()
