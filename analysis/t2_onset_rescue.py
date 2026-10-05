"""Task 2 onset: re-rank M top-3 candidates with the likelihood-ratio score of note 65 (the rescue rule).

Note 65: on the 40 official-like windows M misses, M holds the true bottleneck in its top 3 candidates 88% of the
time, and the likelihood-ratio model of idea 6 ranks the true one first on 30% of them (against M 0.05). Note 51
showed that swapping top-1 and top-2 blindly costs −0.016…−0.024, so the rescue has to fire selectively.

Rule: for each window, take M top-K candidates by probability mass in their Voronoi cell. Score them with the
balanced and the pairwise models. If the winner is not M top-1 and its margin over M top-1 exceeds `m`, multiply
M probability inside the winner cell by (1 + beta) and leave the rest of the window untouched. The expected-IoU
ratio rule then makes the selection as usual.

Reported: onset IoU on the blank-aware official-like windows, paired against M with a bootstrap CI, for a grid of
(score, margin, beta), plus how often the rule fires and how often it fires on a window M actually missed.

    .venv/Scripts/python.exe analysis/t2_onset_rescue.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "scripts"))
from t2_eda02 import T2_PANELS  # noqa: E402
from t2_eda03 import F  # noqa: E402
from t2_first_onset_eval import first_onsets, official_blank  # noqa: E402
from t2_m1_models import Ctx, hist_from_truth  # noqa: E402
from t2_onset_candidate_models import MAX_WIN, PAR, build_rows, feature_names  # noqa: E402
from t2_onset_gate_diag import N_CAND, SEP, iou1, peaks, sel, voronoi  # noqa: E402
from tfb.data import CACHE, load  # noqa: E402

TAGS = ["t2_cnn_lov", "t2_cnn_lov_s1", "t2_cnn_lov_s2"]
W_M, TOPK = 0.7, 3
MARGINS = (0.0, 0.05, 0.10, 0.20)
BETAS = (0.5, 1.0, 2.0)
rng = np.random.default_rng(17)
boot = np.random.default_rng(23)


def ci(delta):
    idx = boot.integers(0, len(delta), (2000, len(delta)))
    m = delta[idx].mean(1)
    return f"{delta.mean():+.4f} [{np.percentile(m, 2.5):+.4f}, {np.percentile(m, 97.5):+.4f}]"


def main() -> None:
    t0_all = time.time()
    tr_X, tr_y, tr_cls, tr_win, wcount = [], [], [], [], 0
    keep_ctx = []
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
        cov = ctx.q[d, t0 + F].astype(int) @ NB.T
        b = np.where(cov.max(1) > 0, cov.argmax(1), -1)
        m = b >= 0
        C = len(cands)
        tr_X.append(R[m].reshape(-1, R.shape[-1]))
        lab = np.zeros((m.sum(), C), np.float32)
        lab[np.arange(m.sum()), b[m]] = 1.0
        tr_y.append(lab.reshape(-1))
        tr_cls.append(np.broadcast_to(b[m][:, None], (m.sum(), C)).reshape(-1))
        tr_win.append((np.arange(m.sum()) + wcount)[:, None].repeat(C, 1).reshape(-1))
        wcount += int(m.sum())
        keep_ctx.append((ctx, cands, NB))
        print(f"[{panel}] train windows {int(m.sum())} ({time.time() - t0_all:.0f}s)", flush=True)
    X = np.concatenate(tr_X); y = np.concatenate(tr_y)
    cls = np.concatenate(tr_cls); win = np.concatenate(tr_win)
    names = feature_names()
    w = np.ones(len(y), np.float32)
    for c in np.unique(cls):
        sel_c = cls == c
        w[sel_c] = 1.0 / max(sel_c.sum(), 1)
    w *= len(y) / w.sum()
    m_bal = lgb.train(dict(PAR, seed=0), lgb.Dataset(X, y, weight=w, feature_name=names), 500)
    pos = np.flatnonzero(y == 1)
    pa, pb = [], []
    for p in pos:
        same = np.flatnonzero(win == win[p])
        others = same[y[same] == 0]
        if len(others):
            for q in rng.choice(others, size=min(3, len(others)), replace=False):
                pa.append(p); pb.append(q)
    pa, pb = np.array(pa), np.array(pb)
    D = np.concatenate([X[pa] - X[pb], X[pb] - X[pa]])
    dy = np.concatenate([np.ones(len(pa)), np.zeros(len(pb))])
    m_pair = lgb.train(dict(PAR, seed=0), lgb.Dataset(D, dy, feature_name=names), 500)
    print(f"models trained ({time.time() - t0_all:.0f}s)", flush=True)

    res: dict = {}
    fired: dict = {}
    for (ctx, cands, NB) in keep_ctx:
        for split in ("validation", "private"):
            z = load(ctx.panel, split)
            on = first_onsets(ctx.vf, z, blank=official_blank(ctx.panel, split, z))
            n = len(on["d"])
            if not n:
                continue
            c = np.load(CACHE / f"t2probs_{split}_{ctx.panel}.npz")
            cn = np.mean([np.load(CACHE / f"t2cnn_{t}_{split}_{ctx.panel}.npz")["on_p"] for t in TAGS], 0)
            pM = W_M * c["on_p"] + (1 - W_M) * cn
            Rs = build_rows(ctx, ctx.panel, split, on["d"], on["t0"], on["v"], on["qf"], on["ok"], cands, NB)
            Rs[:, :, -3] = (z["weekday"][on["d"]] >= 5).astype(np.float32)[:, None]
            truth, elig = on["fut"][:, F - 1], on["efut"][:, F - 1]
            cscore = pM @ NB.T
            C = len(cands)
            sb = m_bal.predict(Rs.reshape(-1, Rs.shape[-1])).reshape(n, C)
            sp = np.zeros((n, C))
            for j in range(n):
                for k in range(C):
                    sp[j, k] = m_pair.predict(Rs[j, k][None] - Rs[j]).mean()
            for j in range(n):
                base_iou = iou1(sel(pM[j]), truth[j], elig[j])
                res.setdefault(("M", 0.0, 0.0), []).append(base_iou)
                order = np.argsort(-cscore[j])[:TOPK]
                for sname, S in (("bal", sb), ("pair", sp)):
                    win_c = order[np.argmax(S[j][order])]
                    marg = S[j][win_c] - S[j][order[0]]
                    for mg in MARGINS:
                        for beta in BETAS:
                            p = pM[j]
                            hit = win_c != order[0] and marg > mg
                            if hit:
                                p = p * (1 + beta * NB[win_c])
                            res.setdefault((sname, mg, beta), []).append(iou1(sel(p), truth[j], elig[j]))
                            if beta == BETAS[0]:
                                f = fired.setdefault((sname, mg), [0, 0, 0])
                                f[0] += int(hit); f[1] += 1; f[2] += int(hit and base_iou == 0)
        print(f"  scored {ctx.panel} ({time.time() - t0_all:.0f}s)", flush=True)
    base = np.array(res[("M", 0.0, 0.0)])
    print(f"\n=== onset IoU on {len(base)} official-like windows, M = {base.mean():.4f} ===")
    for sname in ("bal", "pair"):
        for mg in MARGINS:
            f = fired[(sname, mg)]
            cells = []
            for beta in BETAS:
                v = np.array(res[(sname, mg, beta)])
                cells.append(f"beta {beta}: {v.mean():.4f} {ci(v - base)}")
            print(f"  {sname} margin {mg:.2f}  fires {f[0]}/{f[1]} ({f[0] / max(f[1], 1):.2f}), "
                  f"of which on missed windows {f[2]}  | " + " | ".join(cells), flush=True)
    print(f"total {time.time() - t0_all:.0f}s")


if __name__ == "__main__":
    main()
