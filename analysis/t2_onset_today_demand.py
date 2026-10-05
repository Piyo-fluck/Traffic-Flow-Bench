"""Onset diagnostic: does TODAY's demand level, measured before T, point at the bottleneck that breaks?

The organizers clarified (2026-09-20) that a Task 2 window may use any released data timestamped at or before its
origin T, including earlier hours of the same day. Our models see only the 12 history slots (T-60...T-5), so the
morning build-up of the day was never used.

Per official-like onset window and per link, from the released (masked) layer of the same day only:
    f3h, f6h      today's mean flow over [T-3h, T-1h) and [T-6h, T-1h), divided by the weekday x tod profile of
                  the same slots (work/cache/<panel>_t1tables.npz) - "how hot is this link today"
    r3h           the same ratio for the attached on-ramp flow (ramp_states, before T)
    load3h        f3h x (profile flow / capacity) - today's ratio scaled by how loaded the link usually is
    qcap          today's mean flow over [T-3h, T-1h) divided by capacity - the plain margin, for reference
Each score ranks the candidates of the window (the NMS peaks of candidate T's onset map, separation 3).

Reported, exactly like the occupancy / ramp diagnostic of note 73: the share of windows where the score picks the
true bottleneck among T's top 3, split into the windows T misses (IoU 0) and the rest. A usable signal has to beat
the chance level (about 0.35-0.5) on BOTH groups, not only on the missed ones.

    .venv/Scripts/python.exe analysis/t2_onset_today_demand.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tta_lab" / "tta"))
sys.path.insert(0, str(ROOT / "mm_lab" / "mm"))
sys.path.insert(0, str(ROOT / "analysis"))
import common as C  # noqa: E402
from diag_123 import hits, nms_peaks, rank_of_truth  # noqa: E402
from tfb.data import CACHE  # noqa: E402
from tfb.physics import load_ramps  # noqa: E402

SEP, NCAND = 3, 10


def today_scores(panel, z, ctx, d, t0, ramps):
    """{name: (N, S)} from the same day before the history only."""
    with np.load(CACHE / f"{panel}_t1tables.npz") as t:
        tq = t["tq"]
    q = z["obs_flow"]
    on_q = np.where(ramps["on_v"], ramps["on_q"], np.nan) * ramps["on_att"]
    out = {}
    for name, hours in (("f3h", 3), ("f6h", 6)):
        lo = np.maximum(t0 - 12 * hours, 0)
        num, den, rnum, rden = [], [], [], []
        for j, (dd, a, b) in enumerate(zip(d, lo, t0 - 12)):
            sl = slice(a, max(b, a + 1))
            with np.errstate(all="ignore"):
                num.append(np.nanmean(q[dd, sl], 0))
                den.append(np.nanmean(tq[z["weekday"][dd], sl], 0))
                rnum.append(np.nanmean(on_q[dd, sl], 0))
                rden.append(np.nanmean(np.where(np.isfinite(on_q[:, sl]), on_q[:, sl], np.nan)[
                    z["weekday"] == z["weekday"][dd]], (0, 1)))
        num, den = np.array(num), np.array(den)
        out[name] = np.nan_to_num(num / np.maximum(den, 1.0), nan=1.0)
        if hours == 3:
            out["r3h"] = np.nan_to_num(np.array(rnum) / np.maximum(np.array(rden), 1.0), nan=1.0)
            out["qcap"] = np.nan_to_num(num / np.maximum(ctx.cap[None, :], 1.0), nan=0.0)
            out["load3h"] = out["f3h"] * np.nan_to_num(den / np.maximum(ctx.cap[None, :], 1.0), nan=0.0)
            # controls that use NO same-day data: the profile level, and the history-only margin
            out["prof_qcap"] = np.nan_to_num(den / np.maximum(ctx.cap[None, :], 1.0), nan=0.0)
    return out


def main() -> None:
    acc, ranks = {}, {}
    for i, panel in enumerate(C.T2_PANELS):
        ctx, stat, W = C.windows(i, panel)
        for split in C.SPLITS:
            w, z, ref = W[split]["on"], W[split]["z"], W[split]["ref"]
            if not len(w["d"]):
                continue
            ramps = load_ramps(panel, split, z)
            sc = today_scores(panel, z, ctx, w["d"], w["t0"], ramps)
            with np.errstate(all="ignore"):   # history-only margin: the 12 slots the models already see
                sc["hist_qcap"] = np.nan_to_num(np.nanmean(np.where(w["ok"], w["qf"], np.nan), 1)
                                                / np.maximum(ctx.cap[None, :], 1.0), nan=0.0)
            pT = C.W_ON * ref["on_H"] + (1 - C.W_ON) * ref["on_ad9b"]
            iT = C.onset_iou(pT, w)
            for j in range(len(w["d"])):
                el, tr = w["efut"][j, C.F - 1], w["cells"][j]
                c3 = [c for c in nms_peaks(pT[j], el, 3, SEP)]
                if len(c3) < 2 or not any(hits(c, tr, SEP) for c in c3):
                    continue
                miss = iT[j] == 0
                pick = c3[0]
                acc.setdefault((miss, "T"), []).append(hits(pick, tr, SEP))
                for name, v in sc.items():
                    val = [np.nan_to_num(v[j])[max(0, c - 1):c + 2].max() for c in c3]
                    acc.setdefault((miss, name), []).append(hits(c3[int(np.argmax(val))], tr, SEP))
                    ranks.setdefault((miss, name), []).append(rank_of_truth(np.nan_to_num(v[j]), el, tr, NCAND, SEP))
                ranks.setdefault((miss, "T"), []).append(rank_of_truth(pT[j], el, tr, NCAND, SEP))
        print(f"  {panel}", flush=True)
    names = ["T", "f3h", "f6h", "r3h", "qcap", "load3h", "prof_qcap", "hist_qcap"]
    for miss in (True, False):
        n = len(acc[(miss, "T")])
        print(f"\n=== {'T-missed (IoU 0)' if miss else 'other'} windows, truth in T top-3, n={n} ===")
        for name in names:
            a = np.mean(acc[(miss, name)])
            r = np.array(ranks[(miss, name)], float)
            print(f"  {name:<8} pick-correct among top-3 {a:.2f}   own ranking: top1 {np.mean(r <= 1):.2f} "
                  f"top3 {np.mean(r <= 3):.2f}")


if __name__ == "__main__":
    main()
