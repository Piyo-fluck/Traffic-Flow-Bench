"""Proof test (measurement only, NOT used for any submission): do horizon ramp flows raise S_queue?

Three predictors on the same windows, scored with local labels on train:
  rules   : our legitimate rules v2 (history T-60..T-5 + recurrence priors)
  +ramps  : rules plus links whose attached on-ramp flow at the horizon step falls below
            RATIO x its weekday-type/tod mean, with gaps of <= GAP stations between such links filled
  placebo : identical to +ramps, but the ramp flows come from ANOTHER day of the same weekday type
            (same slots). If only +ramps gains, the gain is the same-time horizon information.
Onset windows use the flags at T+30 only (fallback: rules); ongoing windows add the flags at every step.

Window sets: the official train windows (5 per condition and panel), and candidate windows over
all train days (earliest-onset origins; ongoing origins every 30 min).

    py analysis/t2_ramp_leak_proof.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "scripts"))
from t2_eda02 import T2_PANELS, build, candidates  # noqa: E402
from t2_eda03 import F, earliest_onset, labels  # noqa: E402
from t2_eda04 import official  # noqa: E402
from t2_predict_rules import group_of, train_priors  # noqa: E402
from tfb.data import load  # noqa: E402
from tfb.features import neighbors_along  # noqa: E402
from tfb.physics import load_ramps  # noqa: E402
from tfb.t1eval import family_mean  # noqa: E402

RATIO, GAP = 0.2, 6


def ramp_flags(z: dict, r: dict) -> np.ndarray:
    """(D,T,S) on-ramp collapse flags, with gaps of <= GAP stations between flags filled."""
    wt = z["weekday"] >= 5
    flow, valid = r["on_q"], r["on_v"]
    prof = np.zeros_like(flow)
    for w in (False, True):
        m = valid[wt == w]
        prof[wt == w] = np.where(m, flow[wt == w], 0).sum(0) / np.maximum(m.sum(0), 1)
    flag = valid & (prof > 20) & (flow < RATIO * prof)
    S = flag.shape[2]
    pos = np.broadcast_to(np.arange(S, dtype=np.float32), flag.shape)
    _, _, dp, dn = neighbors_along(np.where(flag, pos, np.nan), flag, axis=2)
    return flag | (np.isfinite(dp) & np.isfinite(dn) & (dp + dn <= GAP))


def placebo_days(z: dict, rng: np.random.Generator) -> np.ndarray:
    """For each day, another day of the same weekday type."""
    wt = z["weekday"] >= 5
    out = np.arange(len(wt))
    for w in (False, True):
        idx = np.flatnonzero(wt == w)
        perm = rng.permutation(idx)
        while np.any(perm == idx):
            perm = rng.permutation(idx)
        out[idx] = perm
    return out


def iou(pred: np.ndarray, truth: np.ndarray, elig: np.ndarray) -> np.ndarray:
    p = pred & elig
    inter, union = (p & truth).sum((1, 2)), (p | truth).sum((1, 2))
    return np.where(union == 0, 1.0, inter / np.maximum(union, 1))


def predictors(z, c, cond, flags, dmap, score):
    d, t0 = c["d"], c["t0"]
    N, S = len(d), z["lanes"].shape[0]
    steps = t0[:, None] + np.arange(1, F + 1)
    out = {}
    for name, dd in (("+ramps", d), ("placebo", dmap[d])):
        out[name] = flags[dd[:, None], steps]  # (N, F, S)
    if cond == "queue_onset":
        g = group_of(z["weekday"][d], "wt2")
        top = np.zeros((N, S), bool)
        np.put_along_axis(top, np.argsort(-score[g, t0 + F], axis=1)[:, :3], True, axis=1)
        rules = np.zeros((N, F, S), bool)
        rules[:, F - 1] = top
        res = {"rules": rules}
        for name in ("+ramps", "placebo"):
            fl = out[name][:, F - 1]
            p = np.zeros((N, F, S), bool)
            p[:, F - 1] = np.where(fl.any(1, keepdims=True), fl, top)
            res[name] = p
    else:
        vh = np.where(c["ehist"], c["vhist"], np.nan)
        with np.errstate(invalid="ignore"):
            link = np.nan_to_num(np.nanmin(vh[:, -3:], axis=1), nan=9) < 0.8
        rules = np.repeat(link[:, None], F, axis=1)
        res = {"rules": rules, "+ramps": rules | out["+ramps"], "placebo": rules | out["placebo"]}
    return res


def main() -> None:
    rng = np.random.default_rng(0)
    res: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for panel in T2_PANELS:
        z = load(panel, "train")
        q, el = labels(z)
        flags = ramp_flags(z, load_ramps(panel, "train", z))
        dmap = placebo_days(z, rng)
        pr = train_priors(z, 1, "wt2")
        score = pr["P_on"] + 0.5 * pr["P_q"]
        eo_d, eo_t = np.nonzero(earliest_onset(q))
        cand = candidates(z)
        sets = {
            ("official", "queue_onset"): build(z, *official(z, panel, "queue_onset")),
            ("official", "queue_ongoing"): build(z, *official(z, panel, "queue_ongoing")),
            ("candidates", "queue_onset"): build(z, eo_d, eo_t),
            ("candidates", "queue_ongoing"): {k: v[cand["ongoing"]] for k, v in cand.items()},
        }
        for (sname, cond), c in sets.items():
            for name, p in predictors(z, c, cond, flags, dmap, score).items():
                res.setdefault(sname, {}).setdefault(name, {}).setdefault(cond, {})[panel] = iou(p, c["fut"], c["efut"]).mean()
        print(f"{panel}: official 5+5, candidates onset {len(eo_d)} / ongoing {int(cand['ongoing'].sum())}", flush=True)

    for sname, r in res.items():
        print(f"\n== {sname} windows: IoU (family mean over 8 panels) ==")
        print(f"  {'predictor':<9} {'onset':>7} {'ongoing':>8} {'S_queue':>8} {'dS_queue':>9} {'d total score':>14}")
        base = None
        for name in ("rules", "+ramps", "placebo"):
            on, og = family_mean(r[name]["queue_onset"]), family_mean(r[name]["queue_ongoing"])
            s = (on + og) / 2
            base = s if base is None else base
            print(f"  {name:<9} {on:>7.3f} {og:>8.3f} {s:>8.3f} {s - base:>+9.3f} {0.30 * (s - base):>+14.3f}")
        print("  per panel S_queue (rules -> +ramps / placebo):")
        for p in T2_PANELS:
            sq = {n: (r[n]["queue_onset"][p] + r[n]["queue_ongoing"][p]) / 2 for n in ("rules", "+ramps", "placebo")}
            print(f"    {p:<11} {sq['rules']:.3f} -> {sq['+ramps']:.3f} / {sq['placebo']:.3f}")


if __name__ == "__main__":
    main()
