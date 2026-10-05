"""Multimodality diagnostics 1-3 (see ../EXPERIMENTS_MM.md). No new model is trained.

One pass over the official-like onset windows replays idea 9b (rec_s20, 30 AdamW steps at 1e-4, predict-then-adapt,
the same code as tta_lab run_adapt.py). For every window, using the weights that would have predicted it, it records:
  base / adapted CNN probability at T+30                  -> diagnostic 2 (adaptation delta)
  base / adapted trunk embeddings                         -> diagnostic 1 (neighbour multimodality)
  counterfactual flow bumps on the top candidates         -> diagnostic 3 (breakdown margin)
Train-month onset windows of the same panel are embedded with the unadapted trunk (the train run is its own run).

Bottleneck bookkeeping (all per window, link axis runs upstream -> downstream):
  truth set      the queued eligible links at T+30 (on["cells"])
  candidates     greedy peaks (NMS, separation --sep) of the candidate-T onset map 0.7 H + 0.3 ad9b
  hit(i)         some truth link lies within --sep of candidate i
  rank(score)    rank of the first NMS peak of `score` that hits the truth (inf when none of the top --ncand does)
  missed window  candidate T's selected set has IoU 0 with the truth (also reported: IoU < 0.5)

    python mm/diag_123.py --name diag1 [--panels ...] [--k 20 50 100] [--eps 0.25 0.5 1 2 4 8]
Outputs: mm_lab/out/<name>/per_window.csv (one row per window, every diagnostic) and summary.txt.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta"))
import common as C  # noqa: E402

F, H = C.F, C.H
OUT = LAB / "out"


# ------------------------------------------------------------------------------------------------ bottleneck helpers
def nms_peaks(score, elig, k, sep):
    s = np.where(elig, score, -np.inf).astype(float)
    out = []
    for _ in range(k):
        j = int(np.argmax(s))
        if not np.isfinite(s[j]):
            break
        out.append(j)
        s[max(0, j - sep):j + sep + 1] = -np.inf
    return np.array(out, int)


def hits(c, truth, sep):
    t = np.flatnonzero(truth)
    return len(t) > 0 and np.abs(t - c).min() <= sep


def rank_of_truth(score, elig, truth, k, sep):
    for r, c in enumerate(nms_peaks(score, elig, k, sep), 1):
        if hits(c, truth, sep):
            return r
    return np.inf


def centre(truth):
    """Centre of the largest contiguous run of truth links (the window's bottleneck id), or -1."""
    t = np.flatnonzero(truth)
    if not len(t):
        return -1
    runs = np.split(t, np.flatnonzero(np.diff(t) > 1) + 1)
    r = max(runs, key=len)
    return int(r[len(r) // 2])


def logit(p):
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


# ------------------------------------------------------------------------------------------------ model helpers
@torch.no_grad()
def embed(net, x, pool=8):
    """(N, c * ceil(S/pool)) L2-normalised trunk embedding (the collapse output, pooled along links)."""
    out = []
    for j in range(0, len(x), 64):
        xb = C.to_dev(x[j:j + 64])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = net.collapse(net.blocks(net.inp(xb))).float()[:, :, 0]        # (B, c, S)
        h = torch.nn.functional.avg_pool1d(h, pool, ceil_mode=True).flatten(1)
        out.append(torch.nn.functional.normalize(h, dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 1), np.float32)


def p30(net, x):
    return C.cnn.predict(net, x, None)[:, F - 1]


def counterfactual(net, ctx, stat, w, j, z, cands, eps_grid, up, hist_slots, p_H):
    """eps*_i per candidate: the smallest relative flow bump (on the `up` links upstream of i, incl. i, over the last
    `hist_slots` history slots) for which candidate i becomes the top NMS peak. Both the CNN alone and the T-blend
    (0.7 H fixed + 0.3 bumped CNN) are measured. Also returns the margin gain at the largest eps."""
    S = ctx.S
    v, qf, ok, t0 = w["v"][j:j + 1], w["qf"][j:j + 1], w["ok"][j:j + 1], w["t0"][j:j + 1]
    wk = z["weekday"][w["d"][j:j + 1]] >= 5
    xs, meta = [], []
    for ci_, c in enumerate(cands):
        lo = max(0, c - up)
        for e in eps_grid:
            q2 = qf.copy()
            q2[:, -hist_slots:, lo:c + 1] *= (1 + e / 100)
            xs.append(C.cnn.make_x(ctx, stat, v, q2, ok, t0, wk)[0])
            meta.append((ci_, e))
    x0 = C.cnn.make_x(ctx, stat, v, qf, ok, t0, wk)
    p0 = p30(net, x0)[0]
    P = p30(net, np.stack(xs))
    elig = w["efut"][j, F - 1]
    res = {}
    for kind in ("cnn", "blend"):
        es = np.full(len(cands), np.inf)
        gain = np.zeros(len(cands))
        for (ci_, e), p in zip(meta, P):
            sc = p if kind == "cnn" else 0.7 * p_H + 0.3 * p
            base = p0 if kind == "cnn" else 0.7 * p_H + 0.3 * p0
            c = cands[ci_]
            top = nms_peaks(sc, elig, 1, 3)
            if len(top) and abs(top[0] - c) <= 3 and e < es[ci_]:
                es[ci_] = e
            if e == max(eps_grid):
                own = lambda s: s[max(0, c - 3):c + 4].max()  # noqa: E731
                gain[ci_] = own(sc) - own(base)
        res[kind] = (es, gain)
    return res


# ------------------------------------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--ckpt", default=str(C.MODELS / "t2_cnn_lov_rec_s20.pt"))
    ap.add_argument("--panels", nargs="*", default=None)
    ap.add_argument("--sep", type=int, default=3)
    ap.add_argument("--ncand", type=int, default=10)
    ap.add_argument("--k", type=int, nargs="+", default=[20, 50, 100])
    ap.add_argument("--eps", type=float, nargs="+", default=[0.25, 0.5, 1, 2, 4, 8])
    ap.add_argument("--up", type=int, default=4, help="links upstream of the candidate that get the bump")
    ap.add_argument("--slots", type=int, default=3, help="last history slots bumped (3 = 15 min)")
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    out = OUT / args.name
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    rows = []
    corr = {}                  # (split, panel) -> (pT, delta~ maps, windows) for the direct correction
    t_all = time.time()
    for i, panel in enumerate(C.T2_PANELS):
        if args.panels and panel not in args.panels:
            continue
        ctx, stat, W = C.windows(i, panel, need_train=True)
        # train-month onset windows, unadapted trunk
        base_net, _ = C.build_net(args.ckpt, "full")
        dtr, ttr = ctx.eo
        hi = ttr[:, None] + np.arange(-H, 0)
        z0 = ctx.z
        xtr = C.cnn.make_x(ctx, stat, z0["true_speed"][dtr[:, None], hi], z0["true_flow"][dtr[:, None], hi],
                           ctx.el[dtr[:, None], hi], ttr, z0["weekday"][dtr] >= 5)
        Etr = embed(base_net, xtr)
        ctr = np.array([centre(c) for c in ctx.q[dtr, ttr + F]])
        del xtr
        for split in C.SPLITS:
            w, z, ref = W[split]["on"], W[split]["z"], W[split]["ref"]
            if not len(w["d"]):
                continue
            net, params = C.build_net(args.ckpt, "full")
            pT = C.W_ON * ref["on_H"] + (1 - C.W_ON) * ref["on_ad9b"]
            iouT = C.onset_iou(pT, w)
            DT = np.zeros_like(pT)
            for day in sorted(set(w["d"].tolist())):
                m = np.flatnonzero(w["d"] == day)
                x = C.make_x(ctx, stat, w, m, z)
                pa, pb = p30(net, x), p30(base_net, x)
                ea, eb = embed(net, x), embed(base_net, x)
                for jj, j in enumerate(m):
                    truth, elig = w["cells"][j], w["efut"][j, F - 1]
                    cands = nms_peaks(pT[j], elig, args.ncand, args.sep)
                    r = dict(panel=panel, split=split, d=int(day), t0=int(w["t0"][j]), iouT=float(iouT[j]),
                             n_true=int(truth.sum()), true_in_cands=int(any(hits(c, truth, args.sep) for c in cands)),
                             rank_T=rank_of_truth(pT[j], elig, truth, args.ncand, args.sep),
                             rank_H=rank_of_truth(ref["on_H"][j], elig, truth, args.ncand, args.sep),
                             rank_adapt=rank_of_truth(pa[jj], elig, truth, args.ncand, args.sep),
                             rank_base=rank_of_truth(pb[jj], elig, truth, args.ncand, args.sep))
                    # ---- 2: adaptation delta
                    dl = logit(pa[jj]) - logit(pb[jj])
                    dt = dl - np.median(dl[elig]) if elig.any() else dl
                    DT[j] = dt
                    r["rank_delta"] = rank_of_truth(dt, elig, truth, args.ncand, args.sep)
                    r["rank_absdelta"] = rank_of_truth(np.abs(dt), elig, truth, args.ncand, args.sep)
                    # ---- 1: neighbour multimodality (same panel, train month)
                    cq = centre(truth)
                    cT = nms_peaks(pT[j], elig, 1, args.sep)
                    for tag, e in (("ad", ea[jj]), ("bs", eb[jj])):
                        sim = Etr @ e
                        order = np.argsort(-sim)
                        for k in args.k:
                            nb = ctr[order[:k]]
                            nb = nb[nb >= 0]
                            if not len(nb):
                                continue
                            share = lambda c: float(np.mean(np.abs(nb - c) <= args.sep))  # noqa: E731
                            modes = sorted({int(c) for c in nb}, key=lambda c: -share(c))
                            top_modes = []
                            for c in modes:                  # merge modes closer than sep
                                if all(abs(c - t) > args.sep for t in top_modes):
                                    top_modes.append(c)
                            shares = [share(c) for c in top_modes]
                            r[f"nb{tag}{k}_top_share"] = shares[0]
                            r[f"nb{tag}{k}_true_share"] = share(cq) if cq >= 0 else np.nan
                            r[f"nb{tag}{k}_Tpick_share"] = share(cT[0]) if len(cT) else np.nan
                            rk = [q for q, c in enumerate(top_modes, 1) if cq >= 0 and abs(c - cq) <= args.sep]
                            r[f"nb{tag}{k}_true_mode_rank"] = rk[0] if rk else np.inf
                            pr = np.array(shares) / max(sum(shares), 1e-9)
                            r[f"nb{tag}{k}_entropy"] = float(-(pr * np.log(pr + 1e-12)).sum())
                    # ---- 3: counterfactual breakdown margin (adapted weights, as T uses them)
                    if len(cands):
                        cf = counterfactual(net, ctx, stat, w, j, z, cands, args.eps, args.up, args.slots,
                                            ref["on_H"][j])
                        for kind, (es, gain) in cf.items():
                            order_c = np.lexsort((-gain, es))            # smallest eps*, then largest gain
                            hit_r = [q for q, ci_ in enumerate(order_c, 1) if hits(cands[ci_], truth, args.sep)]
                            r[f"cf_{kind}_rank"] = hit_r[0] if hit_r else np.inf
                            ht = [ci_ for ci_ in range(len(cands)) if hits(cands[ci_], truth, args.sep)]
                            r[f"cf_{kind}_eps_true"] = float(es[ht[0]]) if ht else np.nan
                            r[f"cf_{kind}_eps_Tpick"] = float(es[0])
                    rows.append(r)
                C.adapt(net, params, (lambda day=day: C.past_batch(ctx, stat, z, np.arange(day), 16, rng)), 30, 1e-4)
            corr[(split, panel)] = (pT, DT, w)
            print(f"[{panel} {split}] {len(w['d'])} windows ({time.time() - t_all:.0f}s)", flush=True)
        ctx.z = ctx.q = ctx.el = None
    df = pd.DataFrame(rows)
    df.to_csv(out / "per_window.csv", index=False)
    summarize(df, args, out)
    correction(corr, out)


LAMBDAS = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)


def correction(corr, out):
    """logit p_new = logit p_T + lambda * delta~ ; lambda picked on one month, applied to the other."""
    fam = {}
    for (split, panel), (pT, DT, w) in corr.items():
        for lam in LAMBDAS:
            p = 1 / (1 + np.exp(-(logit(pT) + lam * DT)))
            fam.setdefault((split, lam), {})[panel] = C.onset_iou(p, w).mean()
    lines = ["\n=== direct correction logit p_T + lambda * delta~ (onset family) ==="]
    best = {}
    for split in C.SPLITS:
        row = {lam: C.family_mean(fam[(split, lam)]) for lam in LAMBDAS if (split, lam) in fam}
        if row:
            lines.append(f"  {split}: " + "  ".join(f"l={l:g} {v:.4f} ({v - row[0.0]:+.4f})" for l, v in row.items()))
            best[split] = max(row, key=row.get)
    for split in C.SPLITS:
        other = "private" if split == "validation" else "validation"
        if other in best and (split, best[other]) in fam:
            lam = best[other]
            lines.append(f"  cross-split: {split} with lambda {lam:g} (chosen on {other}): "
                         f"{C.family_mean(fam[(split, lam)]) - C.family_mean(fam[(split, 0.0)]):+.4f}")
    txt = "\n".join(lines)
    print(txt)
    with open(out / "summary.txt", "a", encoding="utf-8") as f:
        f.write(txt + "\n")


def summarize(df, args, out):
    lines = []
    subsets = {"all": df, "missed (IoU_T = 0)": df[df.iouT == 0], "weak (IoU_T < 0.5)": df[df.iouT < 0.5],
               "hit (IoU_T >= 0.5)": df[df.iouT >= 0.5]}
    for split in ("both",) + C.SPLITS:
        for sname, s0 in subsets.items():
            s = s0 if split == "both" else s0[s0.split == split]
            if not len(s):
                continue
            lines.append(f"\n=== {split} | {sname} | {len(s)} windows ===")
            lines.append(f"  truth among the top-{args.ncand} candidates of T: {s.true_in_cands.mean():.2f}")
            for col in ["rank_T", "rank_H", "rank_base", "rank_adapt", "rank_delta", "rank_absdelta",
                        "cf_cnn_rank", "cf_blend_rank"]:
                if col in s:
                    v = s[col].to_numpy(float)
                    lines.append(f"  {col:<16} top1 {np.mean(v <= 1):.2f}  top3 {np.mean(v <= 3):.2f}  "
                                 f"median {np.median(v):.1f}")
            for col in ["cf_cnn_eps_true", "cf_cnn_eps_Tpick", "cf_blend_eps_true", "cf_blend_eps_Tpick"]:
                if col in s:
                    v = s[col].to_numpy(float)
                    lines.append(f"  {col:<18} finite {np.mean(np.isfinite(v)):.2f}  median(finite) "
                                 f"{np.nanmedian(np.where(np.isfinite(v), v, np.nan)) if np.isfinite(v).any() else float('nan'):.2f}%")
            for tag in ("ad", "bs"):
                for k in args.k:
                    p = f"nb{tag}{k}_"
                    if p + "top_share" not in s:
                        continue
                    ts, tr = s[p + "top_share"], s[p + "true_share"]
                    rk = s[p + "true_mode_rank"].to_numpy(float)
                    lines.append(f"  kNN {'adapted' if tag == 'ad' else 'base   '} k={k:<3} top-mode share {ts.mean():.2f}"
                                 f"  true share {tr.mean():.2f} (=0: {np.mean(tr == 0):.2f}, >=0.1: {np.mean(tr >= 0.1):.2f})"
                                 f"  true mode rank1 {np.mean(rk <= 1):.2f} rank<=3 {np.mean(rk <= 3):.2f}"
                                 f"  T-pick share {s[p + 'Tpick_share'].mean():.2f}  entropy {s[p + 'entropy'].mean():.2f}")
    txt = "\n".join(lines)
    print(txt)
    (out / "summary.txt").write_text(txt + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
