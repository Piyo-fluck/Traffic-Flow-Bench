"""Chronos-2 (Amazon, 2025; Apache 2.0) as a zero-shot Task 2 forecaster.

Why: every Task 2 member we have is trained on the train run, and the run shift of the scored months is where the
onset score is lost (note 108: H 0.924 on the train tail -> 0.79 on the scored months). A pretrained forecaster
has never seen the train run at all.

Per window (panel, day d, origin t0) the whole corridor is one multivariate series: every link's speed / vf from
midnight of day d (optionally --prev-days earlier days of the same split) up to slot t0 - 1. Unobserved or
ineligible cells are NaN. Nothing at or after T is read (the online contract: the history ends at T-5, the slot T
observation is excluded as in production). Chronos-2's quantile forecast for T+5..T+30 is turned into
P(v / vf <= 0.6) by interpolating the predictive CDF between its quantile levels.

Scored on the official-like windows of both scored months (tta_lab common.windows), alone and in the production
blends. Probabilities are cached to fm_lab/cache/chronos_<tag>_<split>_<panel>.npz.

    .venv/Scripts/python.exe fm_lab/chronos_t2.py --tag d0 --prev-days 0
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

LAB = Path(__file__).resolve().parent
PROJ = LAB.parent
sys.path.insert(0, str(PROJ / "tta_lab" / "tta"))
import common as C  # noqa: E402

QL = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,
      0.99]
CUT = 0.6


def p_below(q: np.ndarray, cut: float = CUT) -> np.ndarray:
    """q (..., n_levels) predictive quantiles of v/vf -> P(v/vf <= cut), linear CDF interpolation, flat tails."""
    lv = np.asarray(QL, np.float32)
    qs = np.sort(q, -1)
    below = (qs <= cut).sum(-1)                          # number of quantile values at or below the cut
    out = np.empty(q.shape[:-1], np.float32)
    lo = below == 0
    hi = below == len(lv)
    out[lo] = 0.0
    out[hi] = 1.0
    mid = ~(lo | hi)
    k = below[mid] - 1
    qa = np.take_along_axis(qs[mid], k[:, None], -1)[:, 0]
    qb = np.take_along_axis(qs[mid], (k + 1)[:, None], -1)[:, 0]
    w = np.clip((cut - qa) / np.maximum(qb - qa, 1e-6), 0, 1)
    out[mid] = lv[k] + w * (lv[k + 1] - lv[k])
    return out


def contexts(z, vf, d, t0, prev_days):
    """(S, L) v/vf series for each window, NaN where unseen; L varies (list)."""
    v = z["obs_speed"]
    ok = z["elig"] & np.isfinite(v)
    r = np.where(ok, v / vf, np.nan).astype(np.float32)          # (days, T, S)
    out = []
    for dd, tt in zip(d, t0):
        parts = [r[k] for k in range(max(0, dd - prev_days), dd)] + [r[dd, :tt]]
        out.append(np.concatenate(parts, 0).T.copy())            # (S, L)
    return out


def forecast(pipe, ctx, bs):
    q, _ = pipe.predict_quantiles(ctx, prediction_length=C.F, quantile_levels=QL, batch_size=bs)
    return [p_below(x.float().cpu().numpy()) for x in q]           # each (S, F)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="d0")
    ap.add_argument("--prev-days", type=int, default=0)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--panels", nargs="*", default=None)
    ap.add_argument("--model", default="amazon/chronos-2", help="hub id or a fine-tuned checkpoint directory")
    args = ap.parse_args()
    from chronos import Chronos2Pipeline
    pipe = Chronos2Pipeline.from_pretrained(args.model, device_map="cuda")
    (LAB / "cache").mkdir(exist_ok=True)
    t_start = time.time()
    res = {}
    for i, panel in enumerate(C.T2_PANELS):
        if args.panels and panel not in args.panels:
            continue
        ctx_, stat, W = C.windows(i, panel)
        vf = np.asarray(ctx_.vf, np.float32)
        for s in C.SPLITS:
            w = W[s]
            on, og, z = w["on"], w["og"], w["z"]
            p_on = np.zeros((len(on["d"]), ctx_.S), np.float32)
            p_og = np.zeros(og["fut"].shape, np.float32)
            with torch.no_grad():
                if len(on["d"]):
                    p_on = np.stack(forecast(pipe, contexts(z, vf, on["d"], on["t0"], args.prev_days), args.bs))[:, :, -1]
                if len(og["d"]):
                    p_og = np.stack(forecast(pipe, contexts(z, vf, og["d"], og["t0"], args.prev_days), args.bs)
                                    ).transpose(0, 2, 1)
            np.savez(LAB / "cache" / f"chronos_{args.tag}_{s}_{panel}.npz", on=p_on, og=p_og.astype(np.float16))
            ref = w["ref"]
            H, ad = ref["on_H"], ref["on_ad9b"]
            old = 0.2 * ref["og_gs1"] + 0.6 * ref["og_tfm8"] + 0.2 * ref["og_cnn3"]
            r = res.setdefault(s, {})
            if len(on["d"]):
                for nm, p in (("onset chronos", p_on), ("onset ad9b", ad), ("onset H", H),
                              ("onset 0.7H+0.3ad9b", 0.7 * H + 0.3 * ad),
                              ("onset 0.6H+0.25ad9b+0.15chr", 0.6 * H + 0.25 * ad + 0.15 * p_on)):
                    r.setdefault(nm, {})[panel] = C.onset_iou(p, on).mean()
            if len(og["d"]):
                for nm, p in (("ongoing chronos", p_og), ("ongoing old(12)", old),
                              ("ongoing 0.85old+0.15chr", 0.85 * old + 0.15 * p_og),
                              ("ongoing 0.7old+0.3chr", 0.7 * old + 0.3 * p_og)):
                    r.setdefault(nm, {})[panel] = C.ongoing_iou(p, og, 0.5).mean()
            print(f"  {panel} {s}: onset {len(on['d'])} ongoing {len(og['d'])} windows ({time.time() - t_start:.0f}s)",
                  flush=True)
    for s in C.SPLITS:
        print(f"\n=== {s} (family mean, tau 0.50 / ratio rule) ===")
        for nm, d in res.get(s, {}).items():
            print(f"  {nm:<32} {C.family_mean(d):.4f}")


if __name__ == "__main__":
    main()
