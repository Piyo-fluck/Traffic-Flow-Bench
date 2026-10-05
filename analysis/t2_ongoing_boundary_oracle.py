"""Task 2 ONGOING: the ceiling of the "queue object / boundary regression" formulation (the user's track ①).

Notes 52/55: 70% of the ongoing errors are missed cells, the largest bucket is queue GROWTH, the last two horizon
steps carry 43-45% of the loss, and perfect prediction of the cells within +-1 link of the current queue edge is
worth +0.011 total. The user's proposal is to stop predicting per-cell occupancy and predict the two ENDS of each
queue instead (tail/head displacement per horizon step, as an ordinal target).

Before building that model, measure what the formulation can reach at best:

    reference      R's production ensemble (0.25 gs1 + 0.50 tfm x4 + 0.25 cnn x3, tau 0.45)
    ORACLE-BOUND   every queue visible at T is carried forward with its two ends placed EXACTLY where the truth
                   puts them (each base run matched to the truth run it overlaps most; a run that clears predicts
                   nothing). Queues that appear where nothing was queued at T are NOT predicted, because boundary
                   regression on the existing objects cannot invent them.
    unreachable    the share of true queue cells that sit in runs overlapping no run visible at T - the part of
                   the truth the formulation can never reach.

Also reported, to size the target of an ordinal model: the distribution of tail and head displacement in links per
horizon step, and how often a run clears entirely.

    .venv/Scripts/python.exe analysis/t2_ongoing_boundary_oracle.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "scripts"))
from t2_eda02 import T2_PANELS  # noqa: E402
from t2_eda03 import F  # noqa: E402
from t2_first_onset_eval import official_blank  # noqa: E402
from t2_m1_models import Ctx  # noqa: E402
from t2_official_like_eval import ongoing_official  # noqa: E402
from t2_onset_geometry import runs  # noqa: E402
from t2_ongoing_shockwave import nearest_true  # noqa: E402
from t2_pseudo_eval import iou  # noqa: E402
from tfb.data import CACHE, load  # noqa: E402

CNN3 = ["t2_cnn_lov", "t2_cnn_lov_s1", "t2_cnn_lov_s2"]
TFM4 = ["t2_cnn_lov_tfm_s3", "t2_cnn_lov_tfm_s4", "t2_cnn_lov_tfm_s5", "t2_cnn_lov_tfm_s6"]
TAU_R = 0.45
boot = np.random.default_rng(83)


def ci(delta: np.ndarray, n_boot: int = 2000) -> str:
    idx = boot.integers(0, len(delta), (n_boot, len(delta)))
    m = delta[idx].mean(1)
    return f"{delta.mean():+.4f} [{np.percentile(m, 2.5):+.4f}, {np.percentile(m, 97.5):+.4f}]"


def last_state(vis_q, ok):
    has = ok.any(1)
    idx = ok.shape[1] - 1 - np.argmax(ok[:, ::-1], axis=1)
    return np.take_along_axis(vis_q, idx[:, None], axis=1)[:, 0] & has


def group_mean(tags, split, panel, key="og_p"):
    out = []
    for t in tags:
        f = CACHE / f"t2cnn_{t}_{split}_{panel}.npz"
        if not f.exists():
            return None
        out.append(np.load(f)[key])
    return np.mean(out, 0).astype(np.float32)


def main() -> None:
    t0 = time.time()
    acc: dict = {}
    disp: dict = {"tail": [[] for _ in range(F)], "head": [[] for _ in range(F)], "clear": [0] * F, "runs": [0] * F}
    for i, panel in enumerate(T2_PANELS):
        ctx = Ctx(panel, i)
        ctx.fitd = np.ones_like(ctx.fitd)
        for split in ("validation", "private"):
            z = load(panel, split)
            og, _ = ongoing_official(ctx, z, official_blank(panel, split, z))
            n = len(og["d"])
            if not n:
                continue
            c = np.load(CACHE / f"t2probs_{split}_{panel}.npz")
            assert (c["og_d"] == og["d"]).all()
            g = np.load(CACHE / f"t2gs_{split}_{panel}.npz")
            cnn3, tfm4 = group_mean(CNN3, split, panel), group_mean(TFM4, split, panel)
            pR = 0.25 * g["og_p1"] + 0.50 * tfm4 + 0.25 * cnn3
            truth, elig = og["fut"], og["efut"]
            vis_q = og["ok"] & (np.nan_to_num(og["v"], nan=1e9) <= 0.6 * ctx.vf)
            base = last_state(vis_q, og["ok"])                         # (n, S) queue at T
            # --- matching-free checks: persistence (must be BELOW any correct oracle) and the distance of the
            # true cells from the queue visible at T (independent of how runs are matched)
            r0 = acc.setdefault(split, {})
            r0.setdefault("pers", []).append(iou(np.broadcast_to(base[:, None], truth.shape), truth, elig))
            db, _ = nearest_true(base)                                  # (n, S) links to the nearest queue at T
            db3 = np.broadcast_to(db[:, None], truth.shape)
            for lo, hi_ in ((0, 0), (1, 2), (3, 4), (5, 99)):
                r0[f"td{lo}"] = r0.get(f"td{lo}", 0) + int((truth & (db3 >= lo) & (db3 <= hi_)).sum())
            # ORACLE without run matching: predict every TRUE cell within k links of the queue visible at T and
            # nothing further away. No false positives, so this is a proper ceiling for boundary-focused modelling
            # (the run-matching version of this oracle was wrong: it scored below persistence, which is impossible).
            for k in (0, 2, 4):
                ok_k = truth & (db3 <= k)
                r0.setdefault(f"OB{k}", []).append(iou(ok_k, truth, elig))
                for h in range(F):
                    r0.setdefault(f"OB{k}_{h}", []).append(
                        iou(ok_k[:, h:h + 1], truth[:, h:h + 1], elig[:, h:h + 1]))
            pred_or = np.zeros(truth.shape, bool)
            unreach = 0
            tot_true = 0
            for j in range(n):
                br = runs(base[j])
                for h in range(F):
                    tr = runs(truth[j, h])
                    used = set()
                    for s, e in br:
                        best, bov = None, 0
                        for k, (ts, te) in enumerate(tr):
                            ov = min(e, te) - max(s, ts) + 1
                            if ov > bov:
                                best, bov = k, ov
                        if best is not None:
                            ts, te = tr[best]
                            pred_or[j, h, ts:te + 1] = True
                            used.add(best)
                            disp["tail"][h].append(ts - s)
                            disp["head"][h].append(te - e)
                        else:
                            disp["clear"][h] += 1
                        disp["runs"][h] += 1
                    unreach += sum(te - ts + 1 for k, (ts, te) in enumerate(tr) if k not in used)
                    tot_true += int(truth[j, h].sum())
            r = acc.setdefault(split, {})
            r.setdefault("R", []).append(iou(pR >= TAU_R, truth, elig))
            r.setdefault("oracle", []).append(iou(pred_or, truth, elig))
            for h in range(F):
                r.setdefault(f"R{h}", []).append(iou((pR >= TAU_R)[:, h:h + 1], truth[:, h:h + 1], elig[:, h:h + 1]))
                r.setdefault(f"O{h}", []).append(iou(pred_or[:, h:h + 1], truth[:, h:h + 1], elig[:, h:h + 1]))
            r["unreach"] = r.get("unreach", 0) + unreach
            r["tot_true"] = r.get("tot_true", 0) + tot_true
        print(f"[{panel}] done ({time.time() - t0:.0f}s)", flush=True)
    for split, r in acc.items():
        R, O = np.concatenate(r["R"]), np.concatenate(r["oracle"])
        print(f"\n=== {split}: {len(R)} official-like ongoing windows ===")
        print(f"  R (production ensemble, tau {TAU_R})   IoU {R.mean():.4f}")
        for k in (0, 2, 4):
            V = np.concatenate(r[f"OB{k}"])
            print(f"  ORACLE true cells within {k} links of the T queue   IoU {V.mean():.4f}  delta {ci(V - R)}  "
                  f"-> total {0.15 * (V.mean() - R.mean()):+.4f}")
            print("      per step  " + "  ".join(
                f"+{5 * (h + 1)}: {np.concatenate(r[f'OB{k}_{h}']).mean():.3f}" for h in range(F)))
        print(f"  (run-matching oracle, KNOWN BUGGY - scores below persistence: {O.mean():.4f})")
        P = np.concatenate(r["pers"])
        print(f"  persistence (freeze the T queue)           IoU {P.mean():.4f}   "
              f"[sanity: a correct boundary oracle cannot be below this]")
        tot = max(sum(r[f"td{lo}"] for lo in (0, 1, 3, 5)), 1)
        print("  where the true queue cells sit relative to the queue at T: " + "  ".join(
            f"d{lo}: {r[f'td{lo}'] / tot:.3f}" for lo in (0, 1, 3, 5)))
        print(f"  true cells unreachable by boundary tracking (new queues): {r['unreach'] / max(r['tot_true'], 1):.3f}")
        print("  per horizon step  " + "  ".join(
            f"+{5 * (h + 1)}: R {np.concatenate(r[f'R{h}']).mean():.3f} / O {np.concatenate(r[f'O{h}']).mean():.3f}"
            for h in range(F)))
    print("\n=== displacement of the ends, in links (what an ordinal model must predict) ===")
    for h in range(F):
        t_, hd = np.array(disp["tail"][h]), np.array(disp["head"][h])
        print(f"  +{5 * (h + 1):2d}min  runs {disp['runs'][h]:6d}  cleared {disp['clear'][h] / max(disp['runs'][h], 1):.3f} | "
              f"tail median {np.median(t_):+.0f} p10/p90 {np.percentile(t_, 10):+.0f}/{np.percentile(t_, 90):+.0f} "
              f"|move|>0 {np.mean(t_ != 0):.2f} | head median {np.median(hd):+.0f} "
              f"p10/p90 {np.percentile(hd, 10):+.0f}/{np.percentile(hd, 90):+.0f} |move|>0 {np.mean(hd != 0):.2f}")
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
