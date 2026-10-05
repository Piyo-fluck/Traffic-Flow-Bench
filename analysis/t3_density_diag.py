"""Task 3: where the S_LWR error comes from, and can a density-targeted projection reduce it without hurting S_state?

The official S_LWR residual is |dN_state - rhs| with rhs = the organizer boundary flows projected onto the
released observations. So, as far as the score goes, rhs = the truth's own dN. The state is the truth except at
Task 1 targets, where the submission enters through k = q/v and N = k*L. S_LWR is therefore an error on
*density* at target cells, relative to sum |dN_true| over every transition. The CNN is trained on speed and flow
MSE, and q_hat / v_hat is not the best density estimate (it blows up in congestion).

On the train holdout (saved `m3all` predictions, 10 panels), this script reports:
  1. Decomposition of the residual by transition type (edge: one target end; interior: both ends are targets)
     and by the truth's congestion at the target end.
  2. Density estimates at target cells:
       qv     q_hat / v_hat (current)
       ti     linear interpolation in time of the observed density k_obs = q/v across the gap (same link, same day)
       blend  w(g) * ti + (1 - w(g)) * qv, with w by gap distance g (slots to the nearest visible cell)
  3. Projection: for a target density k*, choose v minimising a (v - v_hat)^2 + b (k* v - q_hat)^2 (a, b = the
     S_state weights per lane), and set q = k* v. Reports S_state and S_LWR~ (proxy) for each variant.

    py analysis/t3_density_diag.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tfb.data import PANELS, load  # noqa: E402
from tfb.features import interp_along, neighbors_along, profile, truth_valid  # noqa: E402
from tfb.metrics import FLOW_NORM, FLOW_W, SPEED_NORM, SPEED_W, state_score  # noqa: E402
from tfb.physics import fd_params, physics_scores, ramp_ok  # noqa: E402
from tfb.t1eval import family_mean, split_days  # noqa: E402

TAG = "m3all"
GAP_BINS = [(1, 1), (2, 3), (4, 12), (13, 10_000)]
W_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


def project(v_hat, q_hat, k_star, lanes):
    a = SPEED_W / SPEED_NORM ** 2
    b = FLOW_W / (FLOW_NORM * lanes) ** 2
    v = (a * v_hat + b * k_star * q_hat) / (a + b * k_star ** 2)
    v = np.clip(v, 1.0, None)
    return v, k_star * v


def main() -> None:
    res_tab = {}
    scores: dict[str, dict[str, tuple[float, float]]] = {}
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
        prm = fd_params(panel, z["link_ids"])
        ok = ramp_ok(panel, "train", z)
        tgt = z["target"]
        lanes, L = z["lanes"], prm["length"]

        # observed density and gap distance (visible = eligible, finite observation, not a target)
        ov, oq = z["obs_speed"], z["obs_flow"]
        vis = ~tgt & np.isfinite(ov) & np.isfinite(oq)
        k_obs = np.where(vis, oq / np.maximum(ov, 1.0), np.nan)
        k_ti, _ = interp_along(k_obs, vis, axis=1)
        pos = np.broadcast_to(np.arange(z["obs_speed"].shape[1])[None, :, None], vis.shape).astype(float)
        _, _, dp, dn = neighbors_along(np.where(vis, pos, np.nan), vis, axis=1)
        gap = np.minimum(np.nan_to_num(dp, nan=1e4), np.nan_to_num(dn, nan=1e4))
        k_qv = pq / np.maximum(pv, 1.0)
        k_ti = np.where(np.isfinite(k_ti), k_ti, k_qv)

        # 1. decomposition of the current residual (holdout days only)
        ts, tq = z["true_speed"], z["true_flow"]
        use = z["elig"] & np.isfinite(ts) & np.isfinite(tq)
        k_true = tq / np.maximum(ts, 1.0)
        sv, sq = np.where(tgt, pv, ts), np.where(tgt, pq, tq)
        k_st = sq / np.maximum(sv, 1.0)
        trans = use[:, :-1] & use[:, 1:] & ok[:, :-1] & days[:, None, None]
        dn_st, dn_tr = (k_st[:, 1:] - k_st[:, :-1]) * L, (k_true[:, 1:] - k_true[:, :-1]) * L
        r = np.where(trans, np.abs(dn_st - dn_tr), 0.0)  # NaN outside the scored transitions must not leak into sums
        denom = np.where(trans, np.abs(dn_tr), 0.0).sum()
        interior = tgt[:, :-1] & tgt[:, 1:]
        edge = tgt[:, :-1] ^ tgt[:, 1:]
        tv_end = np.where(tgt[:, 1:], ts[:, 1:], ts[:, :-1]) / z["free_speed"]
        cong = np.nan_to_num(tv_end, nan=1) <= 0.6
        res_tab[panel] = dict(E=r.sum() / denom, edge=(r * edge).sum() / denom, interior=(r * interior).sum() / denom,
                              cong=(r * cong).sum() / denom, share_cong_cells=float((tgt & days[:, None, None] & use &
                                                                                  (ts <= 0.6 * z["free_speed"])).sum() /
                                                                                 max((tgt & days[:, None, None] & use).sum(), 1)))

        if "decomp" in sys.argv[1:]:  # decomposition only (fast)
            t = res_tab[panel]
            print(f"{panel:<11} E {t['E']:.3f} (edge {t['edge']:.3f}, interior {t['interior']:.3f}; congested ends "
                  f"{t['cong']:.3f}; congested target cells {t['share_cong_cells']:.3f})", flush=True)
            continue

        # 2-3. density variants + projection
        def run(name, k_star):
            v2, q2 = project(pv, pq, k_star, lanes)
            v2, q2 = np.where(tgt, v2, pv), np.where(tgt, q2, pq)
            s = state_score(z, v2, q2, cells=days[:, None, None])["mean"]["S"]
            l = physics_scores(z, v2, q2, prm, ok, days=days)["mean"]["S_LWR"]
            scores.setdefault(name, {})[panel] = (s, l)

        s0 = state_score(z, pv, pq, cells=days[:, None, None])["mean"]["S"]
        l0 = physics_scores(z, pv, pq, prm, ok, days=days)["mean"]["S_LWR"]
        scores.setdefault("current", {})[panel] = (s0, l0)
        run("proj_qv", k_qv)
        run("proj_ti", k_ti)
        for lo, hi in GAP_BINS:  # one gap bin at a time uses weight w; the other bins keep qv
            for w in W_GRID[1:]:  # w = 0 is identical to the current prediction
                wmap = np.where((gap >= lo) & (gap <= hi), w, np.nan)
                kb = np.where(np.isfinite(wmap), wmap * k_ti + (1 - wmap) * k_qv, k_qv)
                run(f"bin{lo}-{hi}_w{w}", kb)
        t = res_tab[panel]
        print(f"{panel:<11} E {t['E']:.3f} (edge {t['edge']:.3f}, interior {t['interior']:.3f}; congested ends "
              f"{t['cong']:.3f}; congested target cells {t['share_cong_cells']:.3f}) | current S_state {s0:.4f} "
              f"S_LWR~ {l0:.4f} | proj_ti S_LWR~ {scores['proj_ti'][panel][1]:.4f} S_state {scores['proj_ti'][panel][0]:.4f}",
              flush=True)

    print("\n== family mean (train holdout) ==")
    for key in ("E", "edge", "interior", "cong", "share_cong_cells"):
        print(f"  {key:<17} {family_mean({p: t[key] for p, t in res_tab.items()}):.4f}")
    if not scores:
        return
    for name in ("current", "proj_qv", "proj_ti"):
        s = family_mean({p: v[0] for p, v in scores[name].items()})
        l = family_mean({p: v[1] for p, v in scores[name].items()})
        print(f"  {name:<14} S_state {s:.4f}  S_LWR~ {l:.4f}  (0.35 S_state + 0.10 S_LWR~ = {0.35 * s + 0.10 * l:.4f})")
    print("  blend: change in S_LWR~ / S_state when only one gap bin uses weight w (others keep qv)")
    base_s = family_mean({p: v[0] for p, v in scores["current"].items()})
    base_l = family_mean({p: v[1] for p, v in scores["current"].items()})
    for lo, hi in GAP_BINS:
        row = []
        for w in W_GRID[1:]:  # w = 0 is the current prediction (not rerun)
            d = scores[f"bin{lo}-{hi}_w{w}"]
            row.append(f"w{w}: {family_mean({p: v[1] for p, v in d.items()}) - base_l:+.4f}/"
                       f"{family_mean({p: v[0] for p, v in d.items()}) - base_s:+.4f}")
        print(f"    gap {lo}-{hi}: " + "  ".join(row))


if __name__ == "__main__":
    main()
