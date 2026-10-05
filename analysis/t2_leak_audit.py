"""Task 2 audit (measurement only, NOT used for any submission): how much would the
information that remains visible around a window reveal?

On validation/private the masked layer blanks T+5..T+90 but shows the slot T
itself and everything from T+95 on, at the normal Task 1 visibility. On
train, where the truth is released, simulate that visibility with the train
masked layer and score, on the official train windows:

  rules       : our submitted rules v2 (history T-60..T-5 + train priors)          <- legitimate
  +slot T     : also treat links seen slow at T (masked layer) as queued (ongoing)  <- outside the contract
  +post-buffer: links seen queued at T+95..T+120 (masked layer) define the onset links
                at T+30 and extend ongoing queues                                    <- future information

    py analysis/t2_leak_audit.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "scripts"))
from t2_eda02 import T2_PANELS, build  # noqa: E402
from t2_eda03 import F, labels  # noqa: E402
from t2_eda04 import official  # noqa: E402
from t2_predict_rules import group_of, train_priors  # noqa: E402
from tfb.data import load  # noqa: E402
from tfb.t1eval import family_mean  # noqa: E402

POST = np.arange(19, 25)  # T+95 .. T+120


def iou_rows(pred: np.ndarray, truth: np.ndarray, elig: np.ndarray) -> np.ndarray:
    p = pred & elig
    inter, union = (p & truth).sum((1, 2)), (p | truth).sum((1, 2))
    return np.where(union == 0, 1.0, inter / np.maximum(union, 1))


def main() -> None:
    res: dict[str, dict[str, dict[str, float]]] = {}
    for panel in T2_PANELS:
        z = load(panel, "train")
        q, el = labels(z)
        vf = z["free_speed"]
        ov = z["obs_speed"]  # train masked layer = released visibility pattern (Task 1 mask applied)
        vis_q = np.nan_to_num(ov, nan=1e9) <= 0.6 * vf
        vis_slow = np.nan_to_num(ov, nan=1e9) < 0.8 * vf
        score = (lambda pr: pr["P_on"] + 0.5 * pr["P_q"])(train_priors(z, 1, "wt2"))  # all train days: optimistic
        for cond in ("queue_onset", "queue_ongoing"):
            d, t0 = official(z, panel, cond)
            c = build(z, d, t0)
            truth, elig = c["fut"], c["efut"]
            N, S = len(d), q.shape[2]
            post_q = np.zeros((N, S), bool)
            for j in POST:
                tt = np.minimum(t0 + j, q.shape[1] - 1)
                post_q |= vis_q[d, tt]
            preds = {}
            if cond == "queue_onset":
                g = group_of(z["weekday"][d], "wt2")
                sc = score[g, t0 + F]
                top = np.zeros((N, S), bool)
                np.put_along_axis(top, np.argsort(-sc, axis=1)[:, :3], True, axis=1)
                base = np.zeros((N, F, S), bool)
                base[:, F - 1] = top
                preds["rules"] = base
                preds["+slot T"] = base
                pb = np.zeros((N, F, S), bool)
                pb[:, F - 1] = np.where(post_q.any(1, keepdims=True), post_q, top)
                preds["+post-buffer"] = pb
            else:
                vh = np.where(c["ehist"], c["vhist"], np.nan)
                with np.errstate(invalid="ignore"):
                    link = np.nan_to_num(np.nanmin(vh[:, -3:], axis=1), nan=9) < 0.8
                preds["rules"] = np.repeat(link[:, None], F, axis=1)
                link_t = link | vis_slow[d, t0]
                preds["+slot T"] = np.repeat(link_t[:, None], F, axis=1)
                preds["+post-buffer"] = np.repeat((link_t & post_q)[:, None], F, axis=1) | np.repeat((link_t | post_q)[:, None], F, axis=1) & (np.arange(F) >= 3)[None, :, None]
            for name, p in preds.items():
                res.setdefault(name, {}).setdefault(cond, {})[panel] = iou_rows(p, truth, elig).mean()
        print(f"{panel} done", flush=True)
    print("\nIoU on official train windows (family mean over 8 panels); S_queue = mean of the two conditions")
    for name, r in res.items():
        on, og = family_mean(r["queue_onset"]), family_mean(r["queue_ongoing"])
        print(f"  {name:<13} onset {on:.3f}  ongoing {og:.3f}  S_queue {(on + og) / 2:.3f}")


if __name__ == "__main__":
    main()
