"""Task 2 onset, diagnostic for the gating / stacking proposal (no training, cached probabilities only).

M's onset probability p = 0.7 * H(LightGBM) + 0.3 * CNN x3 on the blank-aware official-like windows. Note 44 showed
that 62-74% of the onset loss is the ~12% of windows that miss the bottleneck entirely (IoU = 0). Run 1 (note 51)
answered the proposal's premises; this run adds what run 1 got wrong or left open:
  - N(B) = B +- 2 covered only 74% of the true cells, so the "true bottleneck" oracles were truncated. Radii 2/4/6.
  - the confusion itself: how far the chosen bottleneck is from the true one, and whether the true one is a rare
    bottleneck in train while the chosen one is the usual one (the "confidently wrong" picture from run 1).
  - a LEGITIMATE use of the bottleneck-conditioned template (run 1 only had the oracle version): take M's OWN top1
    bottleneck and use the train template P(link in the T+30 set | bottleneck) as a shape prior on M's probability.
  - the re-ranking headroom: the best candidate among M's top-3 (an upper bound for any stacker that only re-ranks).

    .venv/Scripts/python.exe analysis/t2_onset_gate_diag.py
"""
from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "scripts"))
from t2_eda02 import T2_PANELS  # noqa: E402
from t2_eda03 import F  # noqa: E402
from t2_expected_iou import select  # noqa: E402
from t2_first_onset_eval import first_onsets, official_blank  # noqa: E402
from t2_m1_models import Ctx  # noqa: E402
from tfb.data import CACHE, load  # noqa: E402

TAGS = ["t2_cnn_lov", "t2_cnn_lov_s1", "t2_cnn_lov_s2"]
N_CAND, SEP = 10, 3
RADS = (2, 4, 6)                     # neighbourhood radii tried for N(B)
RAD_MAIN = 4                         # the radius used for the bottleneck assignment and the confusion analysis
ALPHAS = np.round(np.arange(0, 1.01, 0.1), 2)
BETAS = (0.5, 1.0, 2.0, 3.0, 5.0)    # template shape-prior strength, multiplicative
WMIX = (0.1, 0.2, 0.4)               # template shape-prior strength, additive mixture
W_M = 0.7
boot = np.random.default_rng(11)


def peaks(counts: np.ndarray, k: int, sep: int) -> np.ndarray:
    c, out = counts.astype(float).copy(), []
    while len(out) < k and np.nanmax(c) > 0:
        s = int(np.argmax(c))
        out.append(s)
        c[max(0, s - sep):s + sep + 1] = 0
    return np.array(sorted(out), int)


def nb_matrix(cands: np.ndarray, S: int, rad: int) -> np.ndarray:
    M = np.zeros((len(cands), S), bool)
    for i, b in enumerate(cands):
        M[i, max(0, b - rad):b + rad + 1] = True
    return M


def voronoi(cands: np.ndarray, S: int) -> np.ndarray:
    """(C, S) bool: every link belongs to exactly one candidate, the nearest one. Radius masks overlap when the
    candidates are closer than 2*rad, which makes "which bottleneck" ambiguous; this partition does not."""
    lab = np.abs(np.arange(S)[None] - cands[:, None]).argmin(0)
    return np.stack([lab == c for c in range(len(cands))])


def iou1(pred: np.ndarray, truth: np.ndarray, elig: np.ndarray) -> float:
    p = pred & elig
    u = (p | truth).sum()
    return float((p & truth).sum() / u) if u else 1.0


def sel(score: np.ndarray) -> np.ndarray:
    return select(np.asarray(score, float), "ratio", 1.0)


def ci(delta: np.ndarray, n_boot: int = 2000) -> str:
    if not len(delta):
        return "n/a"
    idx = boot.integers(0, len(delta), (n_boot, len(delta)))
    m = delta[idx].mean(1)
    return f"{delta.mean():+.4f} [{np.percentile(m, 2.5):+.4f}, {np.percentile(m, 97.5):+.4f}]"


def main() -> None:
    t0_all = time.time()
    rows: dict = {}
    for i, panel in enumerate(T2_PANELS):
        ctx = Ctx(panel, i)
        ctx.fitd = np.ones_like(ctx.fitd)
        S = ctx.S
        tr = first_onsets(ctx.vf, ctx.z, truth=True)
        cands = peaks(tr["cells"].sum(0), N_CAND, SEP)
        NBr = {r: nb_matrix(cands, S, r) for r in RADS}
        NB = voronoi(cands, S)          # the assignment and the candidate scores use the non-overlapping partition

        def bneck_of(cells, M=NB):
            cov = cells.astype(int) @ M.T
            return np.where(cov.max(1) > 0, cov.argmax(1), -1), cov
        b_tr, _ = bneck_of(tr["cells"])
        tr_freq = np.array([(b_tr == c).mean() for c in range(len(cands))])      # how usual each bottleneck is
        tpl = np.zeros((len(cands), S), np.float32)
        for c in range(len(cands)):
            m = b_tr == c
            tpl[c] = tr["cells"][m].mean(0) if m.sum() >= 5 else NB[c].astype(np.float32)
        tpl_n = tpl / np.maximum(tpl.max(1, keepdims=True), 1e-6)
        for split in ("validation", "private"):
            z = load(panel, split)
            on = first_onsets(ctx.vf, z, blank=official_blank(panel, split, z))
            n = len(on["d"])
            if not n:
                continue
            cache = np.load(CACHE / f"t2probs_{split}_{panel}.npz")
            assert (cache["on_d"] == on["d"]).all()
            pL = cache["on_p"]
            pC = np.mean([np.load(CACHE / f"t2cnn_{t}_{split}_{panel}.npz")["on_p"] for t in TAGS], 0)
            pM = W_M * pL + (1 - W_M) * pC
            truth, elig = on["fut"][:, F - 1], on["efut"][:, F - 1]
            b_true, cov = bneck_of(truth)
            for j in range(n):
                t_j, e_j = truth[j], elig[j]
                bt = b_true[j]
                sM = pM[j] @ NB.T
                oM, oL, oC = np.argsort(-sM), np.argsort(-(pL[j] @ NB.T)), np.argsort(-(pC[j] @ NB.T))
                pred = sel(pM[j])
                iou_m = iou1(pred, t_j, e_j)
                r = dict(panel=panel, iou=iou_m, n_true=int(t_j.sum()), b_true=int(bt),
                         top1=int(oM[0]), top2=int(oM[1]) if len(oM) > 1 else -1,
                         margin=float(sM[oM[0]] - sM[oM[1]]) if len(oM) > 1 else np.nan,
                         hit=bool(oM[0] == bt), hit2=bool(len(oM) > 1 and oM[1] == bt),
                         hit3=bool(bt in oM[:3]), agree=bool(oL[0] == oC[0]),
                         dist=int(abs(cands[oM[0]] - cands[bt])) if bt >= 0 else -1,
                         freq_true=float(tr_freq[bt]) if bt >= 0 else np.nan, freq_top1=float(tr_freq[oM[0]]),
                         ious_a=[iou1(sel(a * pL[j] + (1 - a) * pC[j]), t_j, e_j) for a in ALPHAS])
                r["covV"] = float((t_j & NB[bt]).sum() / max(t_j.sum(), 1)) if bt >= 0 else np.nan
                r["orV"] = iou1(sel(np.where(NB[bt], pM[j], 0.0)), t_j, e_j) if bt >= 0 else np.nan
                r["top3V"] = max(iou1(sel(np.where(NB[c], pM[j], 0.0)), t_j, e_j) for c in oM[:3])
                for rad in RADS:                                    # coverage and the oracles, per radius
                    Mr = NBr[rad]
                    r[f"cov{rad}"] = float((t_j & Mr[bt]).sum() / max(t_j.sum(), 1)) if bt >= 0 else np.nan
                    r[f"or{rad}"] = iou1(sel(np.where(Mr[bt], pM[j], 0.0)), t_j, e_j) if bt >= 0 else np.nan
                    r[f"top3{rad}"] = max(iou1(sel(np.where(Mr[c], pM[j], 0.0)), t_j, e_j) for c in oM[:3])
                r["tpl_or"] = iou1(sel(tpl[bt]), t_j, e_j) if bt >= 0 else np.nan
                r["tpl_top1"] = iou1(sel(tpl[oM[0]]), t_j, e_j)      # legit: M's own bottleneck, train shape
                for b in BETAS:                                      # legit: train shape as a prior on M
                    r[f"tplp{b}"] = iou1(sel(pM[j] * (1 + b * tpl_n[oM[0]])), t_j, e_j)
                for w in WMIX:
                    r[f"tplm{w}"] = iou1(sel((1 - w) * pM[j] + w * tpl[oM[0]]), t_j, e_j)
                rows.setdefault(split, []).append(r)
            print(f"[{panel} {split}] windows {n}, {len(cands)} candidates ({time.time() - t0_all:.0f}s)", flush=True)
    for split, rr in rows.items():
        g = lambda k: np.array([r[k] for r in rr])                   # noqa: E731
        iou_m, hit, marg = g("iou"), g("hit"), g("margin")
        zero, exact = iou_m == 0, iou_m == 1
        part = ~zero & ~exact
        A = np.array([r["ious_a"] for r in rr])
        print(f"\n=== {split}: {len(rr)} official-like onset windows, M onset IoU {iou_m.mean():.3f} ===")
        print(f"  true cells covered by the assigned bottleneck: voronoi {np.nanmean(g('covV')):.3f}  " +
              "  ".join(f"+-{r} {np.nanmean(g(f'cov{r}')):.3f}" for r in RADS))
        print(f"  bottleneck top1 correct (voronoi): {hit.mean():.3f}; true is top2 {g('hit2').mean():.3f}; "
              f"true in top3 {g('hit3').mean():.3f}")
        print("  --- 1) margin by outcome (top1 - top2 candidate score) ---")
        for name, m in (("exact (IoU=1)", exact), ("partial", part), ("miss (IoU=0)", zero)):
            if m.any():
                print(f"    {name:<14} n {m.sum():3d} ({m.mean():.2f})  margin med {np.nanmedian(marg[m]):.3f}  "
                      f"top1 correct {hit[m].mean():.2f}")
        print("  --- 2) the confusion on the missed windows ---")
        for name, m in (("miss", zero), ("exact", exact)):
            if m.any():
                d = g("dist")[m]
                print(f"    {name:<6} |chosen - true| links: med {np.median(d[d >= 0]):.0f}, <=4 {np.mean(d[d >= 0] <= 4):.2f}; "
                      f"train frequency of the TRUE bottleneck {np.nanmean(g('freq_true')[m]):.3f} vs of the CHOSEN one "
                      f"{np.nanmean(g('freq_top1')[m]):.3f}; true in top3 {g('hit3')[m].mean():.2f}")
        cc = Counter((r["panel"], r["top1"], r["b_true"]) for r in rr if r["iou"] == 0 and r["top1"] != r["b_true"])
        print("    most frequent (panel, chosen, true) confusions: " +
              ", ".join(f"{p.split('_', 1)[1]} {a}->{b} x{c}" for (p, a, b), c in cc.most_common(6)))
        print("  --- 3) the gate premise: LGBM vs CNN agreement ---")
        ag = g("agree")
        print(f"    agree {ag.mean():.2f}: IoU {iou_m[ag].mean():.3f}, miss {zero[ag].mean():.2f} | disagree: "
              f"IoU {iou_m[~ag].mean():.3f}, miss {zero[~ag].mean():.2f}; disagreement holds {zero[~ag].sum()}/{zero.sum()} of the misses")
        print("  --- 4) oracles and legitimate shape priors (paired delta vs M, 95% bootstrap CI) ---")
        keys = [("ORACLE true bottleneck, M's shape (voronoi)", "orV")]
        keys += [(f"ORACLE true bottleneck, M's shape (+-{r})", f"or{r}") for r in RADS]
        keys += [("ORACLE best of M's top-3 (voronoi)", "top3V")]
        keys += [(f"ORACLE best of M's top-3 (+-{r})", f"top3{r}") for r in RADS]
        keys += [("ORACLE true bottleneck, train template", "tpl_or"), ("M's top1 bottleneck, train template", "tpl_top1")]
        keys += [(f"M x (1 + {b} * template[top1])", f"tplp{b}") for b in BETAS]
        keys += [(f"(1 - {w}) * M + {w} * template[top1]", f"tplm{w}") for w in WMIX]
        for name, key in keys:
            v = g(key).astype(float)
            m = np.isfinite(v)
            print(f"    {name:<44} IoU {np.nanmean(v):.3f}  delta {ci(v[m] - iou_m[m])}")
        print("  --- 5) blend weight alpha ---")
        print("    fixed:  " + "  ".join(f"{a:.1f} {A[:, k].mean():.3f}" for k, a in enumerate(ALPHAS)))
        best = A.max(1)
        print(f"    ORACLE per-window alpha: IoU {best.mean():.3f}  delta {ci(best - iou_m)}; "
              f"alpha 0.7 optimal in {(A[:, 7] == best).mean():.2f} of windows")
    print(f"total {time.time() - t0_all:.0f}s")


if __name__ == "__main__":
    main()
