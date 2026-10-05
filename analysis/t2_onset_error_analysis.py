"""Task 2 onset on first-morning-onset windows: where do H's onset predictions go wrong, location or count?

Uses the saved H onset models (work/sub_t2robust_bag5, 5 bags) with the production scoring priors (train table).
Per event (the first morning queue; days < 14), with p = the mean bag probability over links at T+30:
    rank      the rank of the best-ranked true onset cell (1 = the most probable link)
    top1_hit  the most probable link is a true cell;  top1_near: within +-1 link of one
    n_true / k  the number of true onset cells vs the production count k = round(1.2 sum p)
    iou_prod  production IoU;  iou_kopt  IoU when the top-n_true links are taken (perfect count, oracle)
It also tests a legitimate combination with past days: score = p * (1 + beta * loc_past), where loc_past is the
normalised frequency (+-1 link smoothing) of first-onset cells on this month's earlier days (days < d only),
for beta in 0.5 / 1 / 2 (the same count k).

    py analysis/t2_onset_error_analysis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "scripts"))
from t2_eda02 import T2_PANELS  # noqa: E402
from t2_eda03 import F  # noqa: E402
from t2_first_onset_eval import first_onsets, smooth_links  # noqa: E402
from t2_m1_models import Ctx, onset_decide  # noqa: E402
from t2_m1_predict import onset_X, table_prior  # noqa: E402
from t2_pseudo_eval import iou  # noqa: E402
from tfb.data import load  # noqa: E402
from tfb.t1eval import family_mean  # noqa: E402

EVAL_DAYS = 14
BETAS = (0.5, 1.0, 2.0)


def fmean(d):
    d = {k: v for k, v in d.items() if np.isfinite(v)}
    return family_mean(d) if d else float("nan")


def main() -> None:
    ms = [lgb.Booster(model_file=str(f)) for f in sorted((ROOT / "work" / "sub_t2robust_bag5").glob("t2_onset*.txt"))]
    stats: dict = {}
    for i, panel in enumerate(T2_PANELS):
        ctx = Ctx(panel, i)
        ctx.fitd = np.ones_like(ctx.fitd)
        for split in ("validation", "private"):
            z = load(panel, split)
            ev = first_onsets(ctx.vf, z)
            sel = ev["d"] < EVAL_DAYS
            if not sel.any():
                continue
            grp = (z["weekday"][ev["d"]] >= 5).astype(int)
            key = np.stack([ev["d"], grp], 1)
            _, X = onset_X(ctx, ev["v"], ev["qf"], ev["ok"], ev["t0"], grp, table_prior(ctx), key)
            p = np.mean([m.predict(X.reshape(-1, X.shape[-1])) for m in ms], 0).reshape(X.shape[:2])
            n, S = p.shape
            true = ev["fut"][:, -1] & ev["efut"][:, -1]
            order = np.argsort(-p, 1)
            rank = np.array([np.flatnonzero(true[j][order[j]])[0] + 1 if true[j].any() else np.nan for j in range(n)])
            top1 = order[:, 0]
            near = np.array([true[j, max(0, top1[j] - 1):top1[j] + 2].any() for j in range(n)])
            k = np.maximum(1, np.round(1.2 * p.sum(1))).astype(int)
            pr = np.zeros((n, F, S), bool)
            pr[:, F - 1] = onset_decide(p, ("sum", 1.2))
            iou_prod = iou(pr, ev["fut"], ev["efut"])
            pk = np.zeros((n, F, S), bool)
            for j in range(n):
                pk[j, F - 1, order[j, :max(1, true[j].sum())]] = True
            iou_kopt = iou(pk, ev["fut"], ev["efut"])
            cells = smooth_links(ev["cells"].astype(float))
            past = np.array([cells[ev["d"] < d].sum(0) for d in ev["d"]])
            past = past / np.maximum(past.max(1, keepdims=True), 1e-9)
            comb = {}
            for b in BETAS:
                sc = p * (1 + b * past)
                pc = np.zeros((n, F, S), bool)
                oc = np.argsort(-sc, 1)
                for j in range(n):
                    pc[j, F - 1, oc[j, :k[j]]] = True
                comb[b] = iou(pc, ev["fut"], ev["efut"])
            r = stats.setdefault(split, {})
            for name, val in (("rank_med", np.nanmedian(rank[sel])), ("rank<=3", np.mean(rank[sel] <= 3)),
                              ("rank<=10", np.mean(rank[sel] <= 10)), ("top1_hit", np.mean(rank[sel] == 1)),
                              ("top1_near", near[sel].mean()), ("n_true", true[sel].sum(1).mean()), ("k", k[sel].mean()),
                              ("iou_prod", iou_prod[sel].mean()), ("iou_kopt", iou_kopt[sel].mean()),
                              *[(f"iou_past_b{b}", comb[b][sel].mean()) for b in BETAS]):
                r.setdefault(name, {})[panel] = float(val)
            print(f"[{panel} {split}] events {sel.sum()} | rank med {np.nanmedian(rank[sel]):.0f}, top1 hit {np.mean(rank[sel] == 1):.2f}, "
                  f"near {near[sel].mean():.2f} | n_true {true[sel].sum(1).mean():.1f} vs k {k[sel].mean():.1f} | "
                  f"IoU prod {iou_prod[sel].mean():.3f}, perfect-count {iou_kopt[sel].mean():.3f}, "
                  + ", ".join(f"past b{b} {comb[b][sel].mean():.3f}" for b in BETAS), flush=True)
    for split, r in stats.items():
        print(f"\n== {split} (days < {EVAL_DAYS}), family mean ==")
        print("  " + "  ".join(f"{k} {fmean(v):.3f}" for k, v in r.items()))


if __name__ == "__main__":
    main()
