"""Task 2 onset: does the Task 1 model's RECONSTRUCTION RESIDUAL / ENSEMBLE SPREAD rank the rare bottleneck?

The user's ideas (2) and (3). Notes 59/60 both failed the same way: absolute demand and absolute past activity are
highest at the HABITUAL bottleneck, so ranking by them reproduces M's error. Residual and spread are the opposite
kind of quantity - "today this place does not behave the way the traffic model expects" - and Task 2 has never
seen them.

Task 1 reconstructs the whole scored month from the released layer, and its predictions are cached per model:
    work/sub_m3ttt      the production reconstruction (test-time trained ensemble, used by L/M/R)
    work/sub_m3v1, work/sub_m3v1_s1, work/sub_m3w128   the three individual CNNs -> ensemble spread
The residual at an OBSERVED cell is not zero (mean |dv| 1.5-2.7 km/h, 50-60% of cells above 1 km/h): the network
smooths rather than copies, so the cached arrays are usable without re-running inference.

Per candidate bottleneck (the Voronoi cells of notes 51/59/60), from the window history only:
    r_v3 / r_v12    mean signed speed residual (obs - Task1) over the last 3 / 12 slots  (negative = slower
                    than the traffic model expects, i.e. an unexplained slowdown)
    a_v3            mean |speed residual|                 r_q3   mean signed flow residual
    sd_v3           the residual's standard deviation across the last 6 slots
    sig_v3          the Task 1 ensemble spread                 mix = |r_v3| * sig_v3
Each score ranks the candidates; reported exactly like notes 59/60 so the numbers are comparable.

    .venv/Scripts/python.exe analysis/t2_onset_t1residual.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

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
MODELS = ["m3v1", "m3v1_s1", "m3w128"]
W_M, H = 0.7, 12
# sign: +1 ranks the candidates by the largest value, -1 by the smallest (a more negative residual = more anomalous)
SCORES = {"r_v3": -1, "r_v12": -1, "a_v3": +1, "r_q3": +1, "sd_v3": +1, "sig_v3": +1, "mix": +1}


def cell_mean(x, ok, NB):
    """(n, S) values and validity -> (n, C) mean over each candidate cell."""
    out = np.zeros((x.shape[0], NB.shape[0]), np.float32)
    for c, m in enumerate(NB):
        v, o = x[:, m], ok[:, m]
        with np.errstate(all="ignore"):
            out[:, c] = np.where(o.sum(1) > 0, (np.where(o, v, 0.0)).sum(1) / np.maximum(o.sum(1), 1), np.nan)
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
            d, t0w = on["d"], on["t0"]
            hi = t0w[:, None] + np.arange(-H, 0)
            obs_v, obs_q = z["obs_speed"][d[:, None], hi], z["obs_flow"][d[:, None], hi]
            ok = z["elig"][d[:, None], hi] & np.isfinite(obs_v) & np.isfinite(obs_q)
            ttt = np.load(ROOT / "work" / "sub_m3ttt" / f"pred_{split}_{panel}.npz")
            pv, pq = ttt["pv"][d[:, None], hi], ttt["pq"][d[:, None], hi]
            spread = np.std(np.stack([np.load(ROOT / "work" / f"sub_{m}" / f"pred_{split}_{panel}.npz")["pv"][d[:, None], hi]
                                      for m in MODELS]), 0)
            rv, rq = obs_v - pv, obs_q - pq
            feats = {
                "r_v3": cell_mean(np.nanmean(np.where(ok[:, -3:], rv[:, -3:], np.nan), 1), ok[:, -3:].any(1), NB),
                "r_v12": cell_mean(np.nanmean(np.where(ok, rv, np.nan), 1), ok.any(1), NB),
                "a_v3": cell_mean(np.nanmean(np.where(ok[:, -3:], np.abs(rv[:, -3:]), np.nan), 1), ok[:, -3:].any(1), NB),
                "r_q3": cell_mean(np.nanmean(np.where(ok[:, -3:], rq[:, -3:], np.nan), 1), ok[:, -3:].any(1), NB),
                "sd_v3": cell_mean(np.nanstd(np.where(ok[:, -6:], rv[:, -6:], np.nan), 1), ok[:, -6:].any(1), NB),
                "sig_v3": cell_mean(np.nanmean(np.where(ok[:, -3:], spread[:, -3:], np.nan), 1), ok[:, -3:].any(1), NB),
            }
            feats["mix"] = np.abs(feats["r_v3"]) * feats["sig_v3"]
            for j in range(n):
                bt = int(b_true[j])
                r = dict(panel=panel, split=split, iou=iou1(sel(pM[j]), truth[j], elig[j]), b_true=bt,
                         m_top1=int(np.argmax(pM[j] @ NB.T)), n_cand=len(cands))
                for k, sgn in SCORES.items():
                    v = np.nan_to_num(feats[k][j] * sgn, nan=-1e9)
                    r[f"rank_{k}"] = int((v > v[bt]).sum()) + 1 if bt >= 0 else -1
                rows.append(r)
        print(f"[{panel}] done ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "work" / "t2_onset_t1residual.csv", index=False)
    miss = df.iou == 0
    print(f"\n=== {len(df)} official-like onset windows, {int(miss.sum())} missed (IoU = 0) ===")
    print(f"  M's own bottleneck top-1: all {np.mean(df.m_top1 == df.b_true):.2f}, "
          f"missed {np.mean(df.loc[miss, 'm_top1'] == df.loc[miss, 'b_true']):.2f}")
    print("  (note 60 for comparison: local flow 0.35, path load 0.42, ORACLE future demand 0.45 on missed)")
    print(f"\n  {'score':<8} {'all: top1':>10} {'top3':>6} {'med':>5} | {'missed: top1':>13} {'top3':>6} {'med':>5}")
    for k in SCORES:
        r, rm = df[f"rank_{k}"], df.loc[miss, f"rank_{k}"]
        print(f"  {k:<8} {np.mean(r == 1):>10.2f} {np.mean(r <= 3):>6.2f} {r.median():>5.0f} | "
              f"{np.mean(rm == 1):>13.2f} {np.mean(rm <= 3):>6.2f} {rm.median():>5.0f}")
    print(f"\n  saved work/t2_onset_t1residual.csv; total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
