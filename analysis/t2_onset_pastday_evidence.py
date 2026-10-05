"""Task 2 onset: do the rare bottlenecks we miss show up in the SAME RUN on earlier days? (the user's top diagnostic)

Note 51: about 12% of the official-like onset windows miss the bottleneck entirely, and on those days the true
bottleneck is a median 26 links away and RARE in train (train frequency 0.25-0.28 vs 0.42-0.45 for the one M
picks). M is confidently wrong, so no confidence- or disagreement-based routing can catch it.

The question here: is the run's own past visible data enough to tell that this rare bottleneck is "active" in THIS
simulation run? Not "did the first morning queue happen there yesterday" (note 28 showed the official onsets of
earlier days are blanked), but "does this location show congestion at all on the days before the window's day".
Allowed data: the scored month's released layer on strictly earlier days (user decision 2026-09-14).

Measured per candidate bottleneck B (Voronoi cell of the train first-onset peaks), from days < d only:
    ev_q      slots with speed <= 0.60 vf (queued)            ev_slow   slots with speed <= 0.85 vf
    ev_days   days with any queued slot                       ev_qc     max flow / capacity
    ev_dur    longest run of queued slots in a day (mean over days)
and reported separately for the TRUE bottleneck and the one M CHOSE, split by whether M missed the window.

Also reported: how many past days each official-like window actually has (the official windows sit on days 1-7,
so many have none), and whether ranking the candidates by past-day evidence would put the true bottleneck above
M's choice.

    .venv/Scripts/python.exe analysis/t2_onset_pastday_evidence.py
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
from t2_first_onset_eval import first_onsets, official_blank  # noqa: E402
from t2_m1_models import Ctx  # noqa: E402
from t2_onset_gate_diag import N_CAND, SEP, iou1, peaks, sel, voronoi  # noqa: E402
from tfb.data import CACHE, load  # noqa: E402

TAGS = ["t2_cnn_lov", "t2_cnn_lov_s1", "t2_cnn_lov_s2"]
W_M = 0.7
LO, HI = 48, 168          # 04:00-14:00, the morning band the onset windows live in
EV = ["ev_q", "ev_slow", "ev_days", "ev_qc", "ev_dur"]


def evidence(z, ctx, NB, d_max: int) -> np.ndarray:
    """(C, len(EV)) evidence per candidate from days < d_max of the released layer (observed cells only)."""
    C = NB.shape[0]
    out = np.zeros((C, len(EV)), np.float32)
    if d_max <= 0:
        return out
    v = z["obs_speed"][:d_max, LO:HI]
    q = z["obs_flow"][:d_max, LO:HI]
    ok = z["elig"][:d_max, LO:HI] & np.isfinite(v)
    qd = ok & (np.nan_to_num(v, nan=1e9) <= 0.60 * ctx.vf)
    sl = ok & (np.nan_to_num(v, nan=1e9) <= 0.85 * ctx.vf)
    with np.errstate(all="ignore"):
        qc = np.where(ok & np.isfinite(q), q / np.maximum(ctx.cap, 1.0), np.nan)
    for c in range(C):
        m = NB[c]
        if not m.any():
            continue
        cell_q = qd[:, :, m]                                   # (D, T, S_c)
        per_day = cell_q.any(2)                                # (D, T) any link of the cell queued
        dur = 0.0
        for day in range(per_day.shape[0]):
            run = best = 0
            for t in per_day[day]:
                run = run + 1 if t else 0
                best = max(best, run)
            dur += best
        out[c] = [cell_q.sum() / max(d_max, 1), sl[:, :, m].sum() / max(d_max, 1),
                  per_day.any(1).sum(), np.nan_to_num(np.nanmax(qc[:, :, m]) if np.isfinite(qc[:, :, m]).any() else 0.0),
                  dur / max(d_max, 1)]
    return out


def main() -> None:
    t0 = time.time()
    rows: list = []
    for i, panel in enumerate(T2_PANELS):
        ctx = Ctx(panel, i)
        ctx.fitd = np.ones_like(ctx.fitd)
        tr = first_onsets(ctx.vf, ctx.z, truth=True)
        cands = peaks(tr["cells"].sum(0), N_CAND, SEP)
        NB = voronoi(cands, ctx.S)
        b_tr = np.where((tr["cells"].astype(int) @ NB.T).max(1) > 0, (tr["cells"].astype(int) @ NB.T).argmax(1), -1)
        tr_freq = np.array([(b_tr == c).mean() for c in range(len(cands))], np.float32)
        for split in ("validation", "private"):
            z = load(panel, split)
            on = first_onsets(ctx.vf, z, blank=official_blank(panel, split, z))
            n = len(on["d"])
            if not n:
                continue
            c = np.load(CACHE / f"t2probs_{split}_{panel}.npz")
            assert (c["on_d"] == on["d"]).all()
            cn = np.mean([np.load(CACHE / f"t2cnn_{t}_{split}_{panel}.npz")["on_p"] for t in TAGS], 0)
            pM = W_M * c["on_p"] + (1 - W_M) * cn
            truth, elig = on["fut"][:, F - 1], on["efut"][:, F - 1]
            cov = truth.astype(int) @ NB.T
            b_true = np.where(cov.max(1) > 0, cov.argmax(1), -1)
            ev_cache: dict = {}
            for j in range(n):
                d = int(on["d"][j])
                if d not in ev_cache:
                    ev_cache[d] = evidence(z, ctx, NB, d)       # days strictly before d
                E = ev_cache[d]
                s = pM[j] @ NB.T
                top1 = int(np.argmax(s))
                bt = int(b_true[j])
                iou = iou1(sel(pM[j]), truth[j], elig[j])
                rk = int((E[:, 0] > E[bt, 0]).sum()) + 1 if bt >= 0 else -1      # rank of the true B by past queued slots
                rk1 = int((E[:, 0] > E[top1, 0]).sum()) + 1
                rows.append(dict(panel=panel, split=split, d=d, past_days=d, iou=iou, hit=top1 == bt,
                                 n_cand=len(cands), b_true=bt, top1=top1,
                                 tf_true=float(tr_freq[bt]) if bt >= 0 else np.nan, tf_top1=float(tr_freq[top1]),
                                 **{f"t_{k}": float(E[bt, m]) for m, k in enumerate(EV)} if bt >= 0 else {},
                                 **{f"c_{k}": float(E[top1, m]) for m, k in enumerate(EV)},
                                 rank_true=rk, rank_top1=rk1))
        print(f"[{panel}] done ({time.time() - t0:.0f}s)", flush=True)
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "work" / "t2_onset_pastday_evidence.csv", index=False)
    miss = df.iou == 0
    print(f"\n=== {len(df)} official-like onset windows, {int(miss.sum())} missed (IoU = 0) ===")
    print("  past days available per window: " + "  ".join(
        f"{k}: {int((df.past_days == k).sum())}" for k in sorted(df.past_days.unique())[:10]))
    print(f"  windows with NO past day (day 0): {int((df.past_days == 0).sum())} "
          f"({(df.past_days == 0).mean():.2f}); missed among them {int(((df.past_days == 0) & miss).sum())}")
    ok_past = df.past_days > 0
    print("\n  --- was the TRUE bottleneck congested on an earlier day of the same run? ---")
    for name, m in (("missed windows", miss & ok_past), ("correct windows", ~miss & ok_past)):
        if not m.any():
            continue
        sub = df[m]
        print(f"    {name:<16} n {len(sub):3d} | true B: any queued day {np.mean(sub.t_ev_days > 0):.2f}, "
              f"mean queued slots/day {sub.t_ev_q.mean():.2f}, slow slots/day {sub.t_ev_slow.mean():.2f}, "
              f"max q/c {sub.t_ev_qc.mean():.2f} | chosen B: any queued day {np.mean(sub.c_ev_days > 0):.2f}, "
              f"queued slots/day {sub.c_ev_q.mean():.2f}")
    print("\n  --- would past-day evidence rank the true bottleneck above M's choice? ---")
    for name, m in (("missed windows", miss & ok_past), ("correct windows", ~miss & ok_past)):
        if not m.any():
            continue
        sub = df[m]
        print(f"    {name:<16} rank of true B by past queued slots: median {sub.rank_true.median():.0f} "
              f"of {sub.n_cand.mean():.0f} candidates; top-1 {np.mean(sub.rank_true == 1):.2f}, "
              f"top-3 {np.mean(sub.rank_true <= 3):.2f} | evidence(true) > evidence(chosen): "
              f"{np.mean(sub.t_ev_q > sub.c_ev_q):.2f}")
    print(f"\n  train frequency, missed windows: true {df.loc[miss, 'tf_true'].mean():.3f} vs chosen "
          f"{df.loc[miss, 'tf_top1'].mean():.3f} (the note-51 result, as a check)")
    print(f"  saved work/t2_onset_pastday_evidence.csv; total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
