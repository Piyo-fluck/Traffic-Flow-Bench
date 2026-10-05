"""Ongoing: a cell-level stacker for the cells where the three models disagree (note 80).

Note 80 measured where the ongoing blend's remaining error sits: the three members agree on 97.1% of the eligible
cells and are 99.3% right there, while on the 2.9% they dispute the blend is only 70% right. Deciding those cells
perfectly is worth +0.077 / +0.083 ongoing family (+0.012 total); fixing a tenth of the errors is worth +0.001.

This script learns that decision instead of taking the fixed 0.2/0.6/0.2 average at tau 0.50. The training labels
come from the scored month's OWN EARLIER DAYS, which the organizers confirmed (2026-09-20) may be used, so for a
window on day d the stacker is fitted on the disputed cells of days < d (expanding window, per split, panels
pooled). No new inference: the member probabilities are cached for every day.

Features per disputed cell: the three probabilities and their mean / spread / min / max, the horizon step, the
state of the cell's link at the end of the history (speed ratio, flow over capacity, queued flag), how far the
link is from the nearest link queued at T, how many of its neighbours the blend predicts queued, and the link's
position in the corridor.

    .venv/Scripts/python.exe analysis/t2_disputed_stack.py [--min-days 5] [--rounds 200]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tta_lab" / "tta"))
import common as C  # noqa: E402

TAU = {"gs1": 0.55, "tfm8": 0.5, "cnn3": 0.6}
BLEND_W, BLEND_TAU = (0.2, 0.6, 0.2), 0.5
NAMES = ["p_gs1", "p_tfm8", "p_cnn3", "p_mean", "p_std", "p_min", "p_max", "horizon", "v_last", "qcap_last",
         "queued_T", "dist_queue", "nb_pred", "pos", "panel"]


def build(panel_id, og, ctx, P):
    """Per-cell features, truth, eligibility, dispute mask and the blend prediction, all (N, F, S)."""
    bl = BLEND_W[0] * P["gs1"] + BLEND_W[1] * P["tfm8"] + BLEND_W[2] * P["cnn3"]
    pred = bl >= BLEND_TAU
    pr = {n: P[n] >= TAU[n] for n in P}
    dis = ~((pr["gs1"] == pr["tfm8"]) & (pr["tfm8"] == pr["cnn3"]))
    N, F, S = bl.shape
    stack = np.stack([P["gs1"], P["tfm8"], P["cnn3"]])
    with np.errstate(all="ignore"):
        v_last = np.nan_to_num(og["v"][:, -1] / ctx.vf, nan=0.0)
        q_last = np.nan_to_num(og["qf"][:, -1] / ctx.cap, nan=0.0)
    queued_T = (og["ok"][:, -1] & (np.nan_to_num(og["v"][:, -1], nan=1e9) <= 0.6 * ctx.vf))
    idx = np.arange(S)
    dist = np.full((N, S), S, np.float32)
    for j in range(N):
        q = np.flatnonzero(queued_T[j])
        if len(q):
            dist[j] = np.abs(idx[:, None] - q[None, :]).min(1)
    nb = (np.roll(pred, 1, -1).astype(np.float32) + np.roll(pred, -1, -1)) / 2
    f = [stack[0], stack[1], stack[2], stack.mean(0), stack.std(0), stack.min(0), stack.max(0),
         np.broadcast_to(np.arange(1, F + 1, dtype=np.float32)[None, :, None], bl.shape),
         np.broadcast_to(v_last[:, None, :], bl.shape), np.broadcast_to(q_last[:, None, :], bl.shape),
         np.broadcast_to(queued_T[:, None, :].astype(np.float32), bl.shape),
         np.broadcast_to(dist[:, None, :], bl.shape), nb,
         np.broadcast_to((idx / max(S - 1, 1)).astype(np.float32)[None, None, :], bl.shape),
         np.full(bl.shape, panel_id, np.float32)]
    return np.stack(f, -1).astype(np.float32), pred, dis


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-days", type=int, default=5, help="days of history before the stacker replaces the blend")
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--leaves", type=int, default=31)
    args = ap.parse_args()
    data = {s: [] for s in C.SPLITS}
    for i, panel in enumerate(C.T2_PANELS):
        ctx, stat, W = C.windows(i, panel)
        for s in C.SPLITS:
            og, ref = W[s]["og"], W[s]["ref"]
            if not len(og["d"]):
                continue
            P = {"gs1": ref["og_gs1"], "tfm8": ref["og_tfm8"], "cnn3": ref["og_cnn3"]}
            X, pred, dis = build(i, og, ctx, P)
            data[s].append(dict(panel=panel, d=og["d"], X=X, pred=pred, dis=dis & og["efut"],
                                y=og["fut"], e=og["efut"]))
        print(f"  built {panel}", flush=True)

    for s in C.SPLITS:
        D = data[s]
        days = np.unique(np.concatenate([p["d"] for p in D]))
        out = {p["panel"]: p["pred"].copy() for p in D}
        acc = {"blend": [0, 0], "stack": [0, 0]}
        for d in days:
            past = [(p, p["d"] < d) for p in D]
            if sum(int(m.sum()) for _, m in past) < args.min_days * 8:
                continue
            Xtr = np.concatenate([p["X"][m][p["dis"][m]] for p, m in past])
            ytr = np.concatenate([p["y"][m][p["dis"][m]] for p, m in past]).astype(np.float32)
            if len(ytr) < 2000 or ytr.min() == ytr.max():
                continue
            mdl = lgb.train(dict(objective="binary", learning_rate=0.05, num_leaves=args.leaves, verbose=-1,
                                 feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1),
                            lgb.Dataset(Xtr, ytr, feature_name=NAMES), args.rounds)
            for p in D:
                m = p["d"] == d
                if not m.any():
                    continue
                sel = p["dis"][m]
                if not sel.any():
                    continue
                q = mdl.predict(p["X"][m][sel]) >= 0.5
                blk = out[p["panel"]][m]
                blk[sel] = q
                out[p["panel"]][m] = blk
                acc["stack"][0] += int((q == p["y"][m][sel]).sum()); acc["stack"][1] += int(sel.sum())
                acc["blend"][0] += int((p["pred"][m][sel] == p["y"][m][sel]).sum()); acc["blend"][1] += int(sel.sum())
        fam_b, fam_s, wb, ws = {}, {}, [], []
        for p in D:
            ib, is_ = C.iou(p["pred"], p["y"], p["e"]), C.iou(out[p["panel"]], p["y"], p["e"])
            fam_b[p["panel"]], fam_s[p["panel"]] = ib.mean(), is_.mean()
            wb.append(ib); ws.append(is_)
        b, st = C.family_mean(fam_b), C.family_mean(fam_s)
        print(f"\n=== ongoing {s} ===")
        print(f"  disputed-cell accuracy: blend {acc['blend'][0] / max(acc['blend'][1], 1):.3f} -> "
              f"stacker {acc['stack'][0] / max(acc['stack'][1], 1):.3f}  ({acc['stack'][1]:,} cells decided)")
        import numpy as _np
        print(f"  family IoU: blend {b:.4f} -> stacker {st:.4f}  ({st - b:+.4f} family, total {0.15 * (st - b):+.5f})")
        print(f"  window delta {C.ci(_np.concatenate(ws) - _np.concatenate(wb))}")


if __name__ == "__main__":
    main()
