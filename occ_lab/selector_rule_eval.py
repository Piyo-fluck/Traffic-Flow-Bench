"""Use the earliest-origin selection of queue_ongoing windows in the decision: does it raise AU's ongoing IoU?

The organizers keep the EARLIEST origin that passes the conditions (confirmed for onset; replicated for ongoing on
train: 30/40 windows exact with >= 2 queued history slots, persistence IoU <= 0.9, 360-min spacing, onset first).
So when the origin T - 5 passed every visible condition, it failed on its horizon: persistence from T - 5 had
IoU > 0.9 over T .. T+25. That fixes our first five steps near the state last seen before T - 5.

On the official-like ongoing pool (labels from the released layer), a window is "rule" when T - 5 had >= 2 queued
history slots, coverage >= 0.7, 07-21 h, and persistence IoU(T - 5) > 0.9. There, AU's answer is changed on steps
T+5 .. T+25 towards the T - 5 persistence state P5:
    blend a      p' = (1 - a) p + a P5 on steps 1-5, a in 0.3 / 0.5 / 0.7 / 1.0 (1.0 = P5 itself)
    fill         p' = max(p, P5) on steps 1-5 (the persisted cells are kept queued)
Scored on all ongoing windows (the rule windows changed, the rest as AU) and on the rule windows alone, and on
days 1-5 (where the official windows are).

    .venv/Scripts/python.exe occ_lab/selector_rule_eval.py
"""
from __future__ import annotations

import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta"))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

import common as C  # noqa: E402
from online_adapter import ETA, GATE, global_features, load_split  # noqa: E402
from refit_eval import fam_lab  # noqa: E402

H, F = C.H, C.F


def last_state(q, m, d, t):
    hq, hm = q[d, t - H:t], m[d, t - H:t]
    idx = np.where(hm[::-1], np.arange(H)[::-1][:, None], -1).max(0)
    last = np.zeros(hq.shape[1], bool)
    k = idx >= 0
    last[k] = hq[idx[k], np.arange(hq.shape[1])[k]]
    return last


def pers_iou(q, m, d, t):
    last = last_state(q, m, d, t)
    fut, fm = q[d, t + 1:t + 1 + F], m[d, t + 1:t + 1 + F]
    p = np.broadcast_to(last[None], fut.shape) & fm
    y = fut & fm
    u = (p | y).sum()
    return 1.0 if u == 0 else (p & y).sum() / u


def main() -> None:
    gbm = lgb.Booster(model_file=str(LAB / "models" / "residual_lgb.txt"))

    def corr(base, P):
        Fe = global_features(P, base)
        r = gbm.predict(Fe.reshape(-1, 15)).reshape(base.shape).astype(np.float32)
        return np.clip(base + ETA * np.where(np.abs(r) >= GATE, r, 0.0), 0, 1)

    fam, n_rule = {}, {}
    for i, panel in enumerate(C.T2_PANELS):
        ctx, stat, W = C.windows(i, panel)
        for s, D in load_split(i, panel, "ssl_w192ssl_avg4").items():
            og, z = W[s]["og"], W[s]["z"]
            ov = z["obs_speed"]
            ok = z["elig"] & np.isfinite(ov) & np.isfinite(z["obs_flow"])
            fin = np.isfinite(ov)
            q = fin & (np.nan_to_num(ov, nan=1e9) <= 0.6 * ctx.vf)
            tad = fam_lab("tfm", s, panel)
            P, ps, y, e, d = D["P"], D["ps"], D["y"], D["e"], D["d"]
            assert (d == og["d"]).all()
            au = corr(0.4 * (0.2 * P["gs1"] + 0.6 * tad + 0.2 * P["cnn3"]) + 0.6 * ps, P)
            rule = np.zeros(len(d), bool)
            P5 = np.zeros(au.shape, np.float32)
            for n, (dd, t0) in enumerate(zip(og["d"], og["t0"])):
                t = int(t0) - 1
                if t <= H or not (7 <= t / 12 < 21):
                    continue
                hq = q[dd, t - H:t] & fin[dd, t - H:t]
                if hq.any(1).sum() < 2 or z["elig"][dd, t - H:t].mean() < 0.7:
                    continue
                if pers_iou(q, fin, dd, t) > 0.9:
                    rule[n] = True
                    P5[n] = last_state(q, fin, dd, t)[None, :].astype(np.float32)
            V = {"AU": au}
            for a in (0.3, 0.5, 0.7, 1.0):
                x = au.copy(); x[rule, :F - 1] = (1 - a) * au[rule, :F - 1] + a * P5[rule, :F - 1]
                V[f"blend {a}"] = x
            x = au.copy(); x[rule, :F - 1] = np.maximum(au[rule, :F - 1], P5[rule, :F - 1]); V["fill"] = x
            for nm, pv in V.items():
                sc = C.iou(pv >= 0.5, y, e)
                for sub, msk in (("all", np.ones(len(sc), bool)), ("rule windows", rule), ("days 1-5", d <= 4),
                                 ("rule, days 1-5", rule & (d <= 4))):
                    if msk.any():
                        fam.setdefault((s, nm, sub), {})[panel] = sc[msk].mean()
            n_rule[(s, panel)] = (int(rule.sum()), len(rule), int((rule & (d <= 4)).sum()), int((d <= 4).sum()))
        print(f"  {panel}", flush=True)
    for s in C.SPLITS:
        r = np.array([v for (ss, _), v in n_rule.items() if ss == s]).sum(0)
        print(f"{s}: rule windows {r[0]} of {r[1]} ({r[0] / r[1]:.0%}); days 1-5: {r[2]} of {r[3]}")
    for sub in ("all", "days 1-5", "rule windows", "rule, days 1-5"):
        print(f"\n=== ongoing family IoU, {sub} ===")
        ref = {s: C.family_mean(fam[(s, "AU", sub)]) for s in C.SPLITS if (s, "AU", sub) in fam}
        for nm in ("AU", "blend 0.3", "blend 0.5", "blend 0.7", "blend 1.0", "fill"):
            v = {s: C.family_mean(fam[(s, nm, sub)]) for s in ref}
            print(f"  {nm:<10}" + "".join(f"  {s[:4]} {v[s]:.4f} ({v[s] - ref[s]:+.4f})" for s in ref))


if __name__ == "__main__":
    main()
