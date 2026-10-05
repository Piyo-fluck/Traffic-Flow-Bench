"""Task 2 onset: ideas 6, 7 and 8 of the user in one shared frame - candidate-level learning without the location prior.

The diagnosis behind all three (notes 51, 59, 60, 63): M picks the habitual bottleneck with high confidence and the
true one is a rare, distant candidate. Every score that ranks candidates by an absolute level (demand, past-day
activity) reproduces that error, because the habitual bottleneck leads on those too.

    idea 8  the features live in a CANDIDATE-CENTERED coordinate system: offsets -8..+8 links around the candidate,
            plus its local structure, and NO panel id, NO absolute position, NO candidate identity. A candidate is
            described the same way in every panel, so examples transfer across corridors.
    idea 7  a class-balanced model: every candidate class carries the same total weight, so the train frequency of
            a bottleneck cannot be learned from the label distribution.
    idea 6  a likelihood-ratio model: pairs (true candidate, other candidate) of the same window, sampled equally
            per class, trained on the feature DIFFERENCE. This learns P(X | B_i) / P(X | not B_i) rather than the
            posterior, which is where the prior sneaks in.

Rows: one per (window, candidate). Labels: 1 for the candidate whose Voronoi cell covers most of the T+30 truth.
Training windows come from the train split (the earliest-onset windows, as H uses); scoring uses the blank-aware
official-like onset windows, and the report is the same as notes 59/60/63 so the numbers are comparable:
top-1 / top-3 of the true bottleneck over all windows and over the 40 windows M misses.

    .venv/Scripts/python.exe analysis/t2_onset_candidate_models.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "scripts"))
from t2_eda03 import F  # noqa: E402
from t2_eda02 import T2_PANELS  # noqa: E402
from t2_first_onset_eval import first_onsets, official_blank  # noqa: E402
from t2_m1_models import Ctx, hist_from_truth  # noqa: E402
from t2_onset_gate_diag import N_CAND, SEP, iou1, peaks, sel, voronoi  # noqa: E402
from tfb.data import CACHE, load  # noqa: E402
from tfb.physics import load_ramps  # noqa: E402

TAGS = ["t2_cnn_lov", "t2_cnn_lov_s1", "t2_cnn_lov_s2"]
MODELS = ["m3v1", "m3v1_s1", "m3w128"]
OFFS = (-8, -6, -4, -2, -1, 0, 1, 2, 4, 6, 8)
W_M, H, MAX_WIN = 0.7, 12, 1500
PAR = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=40, feature_fraction=0.9,
           bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=4)
rng = np.random.default_rng(17)


def feature_names():
    n = []
    for o in OFFS:
        n += [f"vr{o:+d}", f"qc{o:+d}", f"kc{o:+d}", f"ln{o:+d}", f"on{o:+d}"]
    return n + ["vr_cell", "qc_cell", "t1_a", "t1_sig", "t1_mix", "hour", "wknd", "cell_len", "n_cand"]


def build_rows(ctx, panel, split, d, t0, v, qf, ok, cands, NB):
    """(n * C, K) features, (n * C,) candidate index, window index."""
    S = ctx.S
    ramps = load_ramps(panel, "train", ctx.z)
    on_att = ramps["on_att"].astype(np.float32)
    lanes = np.asarray(ctx.lanes, np.float32)
    with np.errstate(all="ignore"):
        vr = np.where(ok, v / ctx.vf, np.nan)
        qc = np.where(ok, qf / ctx.cap, np.nan)
        kc = np.where(ok, qf / np.maximum(v, 1.0) / ctx.kcrit, np.nan)
        vr3 = np.nanmin(vr[:, -3:], 1)
        qc3 = np.nanmean(qc[:, -3:], 1)
        kc3 = np.nanmean(kc[:, -3:], 1)
    pv = [np.load(ROOT / "work" / f"sub_{m}" / f"pred_{split}_{panel}.npz")["pv"][d[:, None], t0[:, None] + np.arange(-H, 0)]
          for m in MODELS]
    PV = np.stack(pv)
    sig = PV.std(0)
    rv = v - PV.mean(0)
    with np.errstate(all="ignore"):
        a_v = np.nanmean(np.where(ok[:, -3:], np.abs(rv[:, -3:]), np.nan), 1)
        s_v = np.nanmean(np.where(ok[:, -3:], sig[:, -3:], np.nan), 1)
    mix = a_v * s_v
    n, C = len(d), len(cands)
    rows = np.zeros((n, C, len(feature_names())), np.float32)
    for c, centre in enumerate(cands):
        k = 0
        for o in OFFS:
            j = int(np.clip(centre + o, 0, S - 1))
            for arr in (vr3, qc3, kc3):
                rows[:, c, k] = np.nan_to_num(arr[:, j], nan=-1.0); k += 1
            rows[:, c, k] = lanes[j]; k += 1
            rows[:, c, k] = on_att[j]; k += 1
        m = NB[c]
        with np.errstate(all="ignore"):
            rows[:, c, k] = np.nan_to_num(np.nanmean(vr3[:, m], 1), nan=-1.0); k += 1
            rows[:, c, k] = np.nan_to_num(np.nanmean(qc3[:, m], 1), nan=-1.0); k += 1
            rows[:, c, k] = np.nan_to_num(np.nanmean(a_v[:, m], 1), nan=0.0); k += 1
            rows[:, c, k] = np.nan_to_num(np.nanmean(s_v[:, m], 1), nan=0.0); k += 1
            rows[:, c, k] = np.nan_to_num(np.nanmean(mix[:, m], 1), nan=0.0); k += 1
        rows[:, c, k] = t0 / 12.0; k += 1
        rows[:, c, k] = 0.0; k += 1          # weekday flag, filled by the caller
        rows[:, c, k] = m.sum(); k += 1
        rows[:, c, k] = C
    return rows


def main() -> None:
    t0_all = time.time()
    tr_X, tr_y, tr_cls, tr_win = [], [], [], []
    scored: dict = {}
    wcount = 0
    for i, panel in enumerate(T2_PANELS):
        ctx = Ctx(panel, i)
        ctx.fitd = np.ones_like(ctx.fitd)
        z = ctx.z
        tr = first_onsets(ctx.vf, z, truth=True)
        cands = peaks(tr["cells"].sum(0), N_CAND, SEP)
        NB = voronoi(cands, ctx.S)
        d, t0 = ctx.eo
        if len(d) > MAX_WIN:
            s = np.sort(rng.choice(len(d), MAX_WIN, replace=False))
            d, t0 = d[s], t0[s]
        v, qf, ok = hist_from_truth(ctx, d, t0)
        R = build_rows(ctx, panel, "train", d, t0, v, qf, ok, cands, NB)
        R[:, :, -3] = (z["weekday"][d] >= 5).astype(np.float32)[:, None]
        truth = ctx.q[d, t0 + F]
        cov = truth.astype(int) @ NB.T
        b = np.where(cov.max(1) > 0, cov.argmax(1), -1)
        keep = b >= 0
        C = len(cands)
        tr_X.append(R[keep].reshape(-1, R.shape[-1]))
        lab = np.zeros((keep.sum(), C), np.float32)
        lab[np.arange(keep.sum()), b[keep]] = 1.0
        tr_y.append(lab.reshape(-1))
        tr_cls.append(np.broadcast_to(b[keep][:, None], (keep.sum(), C)).reshape(-1))
        tr_win.append((np.arange(keep.sum()) + wcount)[:, None].repeat(C, 1).reshape(-1))
        wcount += int(keep.sum())
        for split in ("validation", "private"):
            zz = load(panel, split)
            on = first_onsets(ctx.vf, zz, blank=official_blank(panel, split, zz))
            if not len(on["d"]):
                continue
            c = np.load(CACHE / f"t2probs_{split}_{panel}.npz")
            cn = np.mean([np.load(CACHE / f"t2cnn_{t}_{split}_{panel}.npz")["on_p"] for t in TAGS], 0)
            pM = W_M * c["on_p"] + (1 - W_M) * cn
            Rs = build_rows(ctx, panel, split, on["d"], on["t0"], on["v"], on["qf"], on["ok"], cands, NB)
            Rs[:, :, -3] = (zz["weekday"][on["d"]] >= 5).astype(np.float32)[:, None]
            t30, e30 = on["fut"][:, F - 1], on["efut"][:, F - 1]
            cov = t30.astype(int) @ NB.T
            bt = np.where(cov.max(1) > 0, cov.argmax(1), -1)
            iou_m = np.array([iou1(sel(pM[j]), t30[j], e30[j]) for j in range(len(on["d"]))])
            scored.setdefault(split, []).append(dict(panel=panel, X=Rs, b=bt, iou=iou_m,
                                                     m_rank=(pM @ NB.T)))
        print(f"[{panel}] train windows {int(keep.sum())}, candidates {C} ({time.time() - t0_all:.0f}s)", flush=True)
    X = np.concatenate(tr_X); y = np.concatenate(tr_y); cls = np.concatenate(tr_cls); win = np.concatenate(tr_win)
    names = feature_names()
    print(f"train rows {len(y):,}, positives {y.mean():.3f} ({time.time() - t0_all:.0f}s)", flush=True)

    # --- idea 7: class-balanced weights, so the frequency of a bottleneck carries no information
    w = np.ones(len(y), np.float32)
    for c in np.unique(cls):
        m = cls == c
        w[m] = 1.0 / max(m.sum(), 1)
    w *= len(y) / w.sum()
    m_bal = lgb.train(dict(PAR, seed=0), lgb.Dataset(X, y, weight=w, feature_name=names), 500)
    m_plain = lgb.train(dict(PAR, seed=0), lgb.Dataset(X, y, feature_name=names), 500)

    # --- idea 6: likelihood ratio from balanced pairs (true vs other, same window)
    pos = np.flatnonzero(y == 1)
    pairs_a, pairs_b = [], []
    for p in pos:
        wsame = np.flatnonzero(win == win[p])
        others = wsame[y[wsame] == 0]
        if len(others):
            pick = rng.choice(others, size=min(3, len(others)), replace=False)
            for q in pick:
                pairs_a.append(p); pairs_b.append(q)
    pa, pb = np.array(pairs_a), np.array(pairs_b)
    D = np.concatenate([X[pa] - X[pb], X[pb] - X[pa]])
    dy = np.concatenate([np.ones(len(pa)), np.zeros(len(pb))])
    dw = np.concatenate([1.0 / np.maximum(np.bincount(cls[pa], minlength=N_CAND + 1)[cls[pa]], 1)] * 2).astype(np.float32)
    m_pair = lgb.train(dict(PAR, seed=0), lgb.Dataset(D, dy, weight=dw, feature_name=names), 500)
    print(f"pairs {len(pa):,} ({time.time() - t0_all:.0f}s)", flush=True)

    out = []
    for split, items in scored.items():
        for it in items:
            Xs = it["X"]
            n, C, K = Xs.shape
            flat = Xs.reshape(-1, K)
            sc = {"balanced": m_bal.predict(flat).reshape(n, C),
                  "plain": m_plain.predict(flat).reshape(n, C),
                  "M": it["m_rank"]}
            pr = np.zeros((n, C))
            for j in range(n):                       # likelihood ratio: score each candidate against the others
                for c in range(C):
                    diff = Xs[j, c][None] - Xs[j]
                    pr[j, c] = m_pair.predict(diff).mean()
            sc["pairwise"] = pr
            for j in range(n):
                if it["b"][j] < 0:
                    continue
                row = dict(split=split, panel=it["panel"], iou=it["iou"][j])
                for k, v in sc.items():
                    row[f"rank_{k}"] = int((v[j] > v[j][it["b"][j]]).sum()) + 1
                out.append(row)
    df = pd.DataFrame(out)
    df.to_csv(ROOT / "work" / "t2_onset_candidate_models.csv", index=False)
    miss = df.iou == 0
    print(f"\n=== {len(df)} official-like onset windows, {int(miss.sum())} missed (IoU = 0) ===")
    print("  (notes 59/60/63: past-day 0.17, path load 0.42, ORACLE demand 0.45, |residual| x spread 0.35 on the misses)")
    print(f"\n  {'model':<10} {'all: top1':>10} {'top3':>6} | {'missed: top1':>13} {'top3':>6}")
    for k in ("M", "plain", "balanced", "pairwise"):
        r, rm = df[f"rank_{k}"], df.loc[miss, f"rank_{k}"]
        print(f"  {k:<10} {np.mean(r == 1):>10.2f} {np.mean(r <= 3):>6.2f} | {np.mean(rm == 1):>13.2f} {np.mean(rm <= 3):>6.2f}")
    imp = m_bal.feature_importance("gain")
    print("\n  balanced model, top features: "
          + ", ".join(f"{names[j]} {imp[j] / imp.sum():.3f}" for j in np.argsort(-imp)[:8]))
    print(f"  saved work/t2_onset_candidate_models.csv; total {time.time() - t0_all:.0f}s")


if __name__ == "__main__":
    main()
