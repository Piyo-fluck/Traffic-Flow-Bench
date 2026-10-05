"""Task 2 H15: which queue cut did the organizer use? And how much would a mismatch cost us?

The official train windows were selected with the organizer's labels:
  - onset windows: no ongoing queue in the history, and the first queued horizon step is T+30;
  - ongoing windows: an ongoing queue in the history.
So the organizer's cut should make that structure hold most cleanly on the official windows.
Candidate cuts (speed <= cut):
  links   : 0.6 x links.csv free speed (ours)
  flat105 : 0.6 x 105 km/h = 63 km/h (the lwr_mainline_topology free_flow_speed)
  f0.55 / f0.65 : 0.55 / 0.65 x links free speed
For each cut: the share of official onset windows with a clean structure, and the persistence and
rules-v2 IoU when the labels use that cut but our predictions keep the "links" cut.

    py analysis/t2_label_cut_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "scripts"))
from t2_eda02 import T2_PANELS  # noqa: E402
from t2_eda03 import F, H  # noqa: E402
from t2_eda04 import official  # noqa: E402
from t2_predict_rules import group_of, train_priors  # noqa: E402
from tfb.data import load  # noqa: E402
from tfb.t1eval import family_mean  # noqa: E402


def cuts(vf: np.ndarray) -> dict[str, np.ndarray]:
    return {"links": 0.6 * vf, "flat105": np.full_like(vf, 63.0), "f0.55": 0.55 * vf, "f0.65": 0.65 * vf}


def iou(pred, truth, elig):
    p = pred & elig
    inter, union = (p & truth).sum((1, 2)), (p | truth).sum((1, 2))
    return np.where(union == 0, 1.0, inter / np.maximum(union, 1))


def main() -> None:
    struct, pers, rules = {}, {}, {}
    for panel in T2_PANELS:
        z = load(panel, "train")
        ts = z["true_speed"]
        el = z["elig"] & np.isfinite(ts)
        vf = z["free_speed"]
        score = (lambda pr: pr["P_on"] + 0.5 * pr["P_q"])(train_priors(z, 1, "wt2"))
        qo = (np.nan_to_num(ts, nan=1e9) <= 0.6 * vf) & el  # our cut: used for the predictions
        for cond in ("queue_onset", "queue_ongoing"):
            d, t0 = official(z, panel, cond)
            hi = t0[:, None] + np.arange(-H, 0)
            fi = t0[:, None] + np.arange(1, F + 1)
            e = el[d[:, None], fi]
            # predictions (fixed: our cut, history only)
            if cond == "queue_onset":
                g = group_of(z["weekday"][d], "wt2")
                top = np.zeros((len(d), len(vf)), bool)
                np.put_along_axis(top, np.argsort(-score[g, t0 + F], 1)[:, :3], True, 1)
                p_rules = np.zeros((len(d), F, len(vf)), bool)
                p_rules[:, F - 1] = top
            else:
                vh = np.where(el[d[:, None], hi], ts[d[:, None], hi] / vf, np.nan)
                with np.errstate(invalid="ignore", all="ignore"):
                    link = np.nan_to_num(np.nanmin(vh[:, -3:], 1), nan=9) < 0.8
                p_rules = np.repeat(link[:, None], F, 1)
            p_pers = np.repeat(qo[d, t0 - 1][:, None], F, 1)
            for name, c in cuts(vf).items():
                q = (np.nan_to_num(ts, nan=1e9) <= c) & el
                hist, fut = q[d[:, None], hi], q[d[:, None], fi]
                ongoing = (hist.sum((1, 2)) >= 2) & (hist.sum(1).max(1) >= 2)
                if cond == "queue_onset":
                    clean = ~ongoing & ~fut[:, :F - 1].any((1, 2)) & fut[:, F - 1].any(1)
                else:
                    clean = ongoing & fut.any((1, 2))
                struct.setdefault(name, {}).setdefault(cond, {})[panel] = clean.mean()
                pers.setdefault(name, {}).setdefault(cond, {})[panel] = iou(p_pers, fut, e).mean()
                rules.setdefault(name, {}).setdefault(cond, {})[panel] = iou(p_rules, fut, e).mean()
        print(f"{panel} done", flush=True)
    print("\nshare of official windows whose structure holds under each label cut (family mean)")
    for name in struct:
        print(f"  {name:<8} onset {family_mean(struct[name]['queue_onset']):.3f}   ongoing {family_mean(struct[name]['queue_ongoing']):.3f}")
    print("\nIoU of fixed predictions (made with our cut) against labels built with each cut")
    for name in struct:
        po, pg = family_mean(pers[name]["queue_onset"]), family_mean(pers[name]["queue_ongoing"])
        ro, rg = family_mean(rules[name]["queue_onset"]), family_mean(rules[name]["queue_ongoing"])
        print(f"  {name:<8} persistence S_queue {(po + pg) / 2:.3f} (on {po:.3f} og {pg:.3f}) | "
              f"rules S_queue {(ro + rg) / 2:.3f} (on {ro:.3f} og {rg:.3f})")


if __name__ == "__main__":
    main()
