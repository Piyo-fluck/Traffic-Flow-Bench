"""Task 2 onset: geometry of the true T+30 queue sets, and where M's onset loss comes from (the user's
first-breakdown / competing-risk proposal, step 1: can the "breakdown origin -> queue extent" hypothesis survive?).

Events are first-morning onsets (first queue of the day at T+30, history free, 04-14 h), as in the official windows:
    train       : t2_first_onset_eval.first_onsets on train truth
    val / priv  : the blank-aware official-like onset windows of work/cache/t2probs_* (exactly M's evaluation set)
Per event: cells queued at T+30, the number of contiguous runs along the link order (strict, and merging 1-link
gaps), and the share of cells in the largest run. Per panel: how stable the run ends are across days (top-1 / top-3
share), and whether the scored runs' ends fall on train's top-5 ends (+-1 link).
Oracles: predict only the largest run; predict k links from either end of the largest run.
M's onset loss (0.7 H + 0.3 CNN x3, expected-IoU ratio rule): split of the mean (1 - IoU) into windows with IoU = 0
(wrong place) and partial windows (right place, wrong extent).

    .venv/Scripts/python.exe analysis/t2_onset_geometry.py
"""
from __future__ import annotations

import sys
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
from t2_first_onset_eval import first_onsets  # noqa: E402
from t2_m1_models import Ctx  # noqa: E402
from tfb.data import CACHE  # noqa: E402

TAGS = ["t2_cnn_lov", "t2_cnn_lov_s1", "t2_cnn_lov_s2"]


def runs(c: np.ndarray, gap: int = 0) -> list[list[int]]:
    idx = np.flatnonzero(c)
    if not len(idx):
        return []
    out = [[int(idx[0]), int(idx[0])]]
    for i in idx[1:]:
        if i - out[-1][1] <= gap + 1:
            out[-1][1] = int(i)
        else:
            out.append([int(i), int(i)])
    return out


def describe(cells: np.ndarray) -> dict:
    n, r0n, r1n, share, lo, hi, orc = [], [], [], [], [], [], {k: [] for k in (1, 2, 3, 4)}
    orc_lo = {k: [] for k in (1, 2, 3, 4)}
    for c in cells:
        r = runs(c)
        if not r:
            continue
        s, e = max(r, key=lambda x: x[1] - x[0])
        cnt = int(c.sum())
        n.append(cnt); r0n.append(len(r)); r1n.append(len(runs(c, 1))); share.append((e - s + 1) / cnt)
        lo.append(s); hi.append(e)
        for k in orc:  # k links from the high end / low end of the largest run
            ph = np.zeros_like(c); ph[max(0, e - k + 1):e + 1] = True
            pl = np.zeros_like(c); pl[s:s + k] = True
            orc[k].append((ph & c).sum() / (ph | c).sum())
            orc_lo[k].append((pl & c).sum() / (pl | c).sum())
    return dict(n=np.array(n), r0=np.array(r0n), r1=np.array(r1n), share=np.array(share), lo=np.array(lo),
                hi=np.array(hi), orc={k: np.mean(v) for k, v in orc.items()}, orc_lo={k: np.mean(v) for k, v in orc_lo.items()})


def stability(ends: np.ndarray) -> tuple[float, float]:
    c = Counter(ends.tolist())
    tot = max(len(ends), 1)
    top = [v for _, v in c.most_common(3)]
    return top[0] / tot if top else np.nan, sum(top) / tot if top else np.nan


def near_top(ends: np.ndarray, ref: np.ndarray, k: int = 5) -> float:
    top = [e for e, _ in Counter(ref.tolist()).most_common(k)]
    return float(np.mean([min(abs(e - t) for t in top) <= 1 for e in ends])) if len(ends) and top else np.nan


def main() -> None:
    pooled: dict = {}
    loss: dict = {}
    print("panel split | events | cells mean / median | runs=1 (strict, gap<=1) | largest-run share | "
          "end stability top1/top3 (low end, high end) | ends near train top-5 (low, high)")
    for i, panel in enumerate(T2_PANELS):
        ctx = Ctx(panel, i)
        tr = first_onsets(ctx.vf, ctx.z, truth=True)
        sets = {"train": tr["cells"]}
        for split in ("validation", "private"):
            c = np.load(CACHE / f"t2probs_{split}_{panel}.npz")
            if not len(c["on_d"]):
                continue
            truth = c["on_fut"][:, F - 1]
            sets[split] = truth
            cn = np.mean([np.load(CACHE / f"t2cnn_{t}_{split}_{panel}.npz")["on_p"] for t in TAGS], 0)
            p = 0.7 * c["on_p"] + 0.3 * cn
            pr = np.stack([select(q, "ratio", 1.0) for q in p]) & c["on_efut"][:, F - 1]  # official: ineligible cells drop out
            inter, union = (pr & truth).sum(1), (pr | truth).sum(1)
            iou = np.where(union > 0, inter / np.maximum(union, 1), 1.0)
            loss.setdefault(split, []).append(np.stack([iou, pr.sum(1), truth.sum(1), (pr & ~truth).sum(1) == 0,
                                                        (truth & ~pr).sum(1) == 0], 1).astype(float))
        dtr = describe(sets["train"])
        for name, cells in sets.items():
            d = describe(cells)
            s_lo, s_hi = stability(d["lo"]), stability(d["hi"])
            nt = (near_top(d["lo"], dtr["lo"]), near_top(d["hi"], dtr["hi"])) if name != "train" else (np.nan, np.nan)
            print(f"{panel:<10} {name:<10} | {len(d['n']):4d} | {d['n'].mean():.2f} / {np.median(d['n']):.0f} | "
                  f"{np.mean(d['r0'] == 1):.2f}, {np.mean(d['r1'] == 1):.2f} | {d['share'].mean():.2f} | "
                  f"{s_lo[0]:.2f}/{s_lo[1]:.2f}, {s_hi[0]:.2f}/{s_hi[1]:.2f} | {nt[0]:.2f}, {nt[1]:.2f}", flush=True)
            pooled.setdefault(name, []).append(d)
    print("\n== pooled over panels ==")
    for name, ds in pooled.items():
        n = np.concatenate([d["n"] for d in ds]); r0 = np.concatenate([d["r0"] for d in ds])
        r1 = np.concatenate([d["r1"] for d in ds]); sh = np.concatenate([d["share"] for d in ds])
        print(f"{name:<10} events {len(n)}; cells: " + ", ".join(f"{k}:{np.mean(n == k):.2f}" for k in (1, 2, 3, 4, 5))
              + f", >5:{np.mean(n > 5):.2f}; runs=1 strict {np.mean(r0 == 1):.2f}, gap<=1 {np.mean(r1 == 1):.2f}; "
              f"runs>=3 {np.mean(r0 >= 3):.2f}; largest-run share {sh.mean():.2f}")
        print(f"           oracle IoU, largest run only: {sh.mean():.3f}; k links from the high end: "
              + ", ".join(f"k{k} {np.mean([d['orc'][k] for d in ds]):.3f}" for k in (1, 2, 3, 4))
              + "; from the low end: " + ", ".join(f"k{k} {np.mean([d['orc_lo'][k] for d in ds]):.3f}" for k in (1, 2, 3, 4)))
    print("\n== M's onset loss (0.7 H + 0.3 CNN x3, ratio rule), all official-like days, pooled windows ==")
    for split, parts in loss.items():
        a = np.concatenate(parts)
        iou, npr, ntr, sub, sup = a[:, 0], a[:, 1], a[:, 2], a[:, 3] > 0, a[:, 4] > 0
        miss, exact = iou == 0, iou == 1
        part = ~miss & ~exact
        tot = max((1 - iou).sum(), 1e-9)
        print(f"{split:<10} windows {len(iou)}; mean IoU {iou.mean():.3f}; exact {exact.mean():.2f}; IoU=0 {miss.mean():.2f}; "
              f"partial {part.mean():.2f}; loss share: IoU=0 {(1 - iou)[miss].sum() / tot:.2f}, partial {(1 - iou)[part].sum() / tot:.2f}")
        print(f"           cells predicted / true: mean {npr.mean():.2f} / {ntr.mean():.2f}; partial windows: "
              f"pred inside truth (too few) {np.mean(sub[part]):.2f}, truth inside pred (too many) {np.mean(sup[part]):.2f}, "
              f"both wrong {np.mean(~sub[part] & ~sup[part]):.2f}; loss of the 'both wrong' partials "
              f"{(1 - iou)[part & ~sub & ~sup].sum() / tot:.2f}")


if __name__ == "__main__":
    main()
