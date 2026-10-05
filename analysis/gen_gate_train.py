"""Note 169: learn WHEN run-shift evidence should override the model's onset pick, on generated runs, and test it on
the scored months' lab windows.

Candidates per window: the ctrl CNN's top-5 sites (> 1 link apart), no adaptation on both sides (generated windows:
occ_lab/gen_onset_infer.py; lab: leap ctrl none caches). Label: the candidate is within +-1 of a link queued at T+30.
Features per candidate: CNN evidence (p, p - p_max, p / p_max, rank) and run-shift evidence (today's upstream entry
anomaly over 3 entries, the run-level entry ratio over 5 entries from the run's earlier days - both relative to the
train weekday profile - and each minus its max over the candidates). A LightGBM LambdaRank is fitted on the generated
windows only (all panels pooled); variants: CNN-only features, + run-shift features, + window-shuffled run-shift
features (placebo). Lab evaluation: top-1 hit of the reranked candidates, fixed / broken vs the CNN's pick.

    .venv/Scripts/python.exe analysis/gen_gate_train.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "analysis")); sys.path.insert(0, str(ROOT / "tta_lab" / "tta"))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

import common as C  # noqa: E402
from dlsim_gen2 import Gen  # noqa: E402
from entry_anomaly_robust import anomaly  # noqa: E402
from run_entry_ratio import run_ratio  # noqa: E402
from tfb.data import load  # noqa: E402

K = 5


def cands_of(p, e):
    pp = np.where(e, p, -1).copy(); out = []
    while len(out) < K and pp.max() > -1:
        c = int(np.argmax(pp)); out.append(c); pp[max(0, c - 1):c + 2] = -1
    return out


def up_mean(g, rank, vals, c, k):
    ups = [i for i, b in enumerate(g.ent) if b <= rank[c]][-k:]
    return float(np.mean(vals[ups])) if ups else 1.0


def window_rows(g, rank, p, y, e, today, run):
    cs = cands_of(p, e)
    nt = y.copy(); nt[1:] |= y[:-1]; nt[:-1] |= y[1:]
    lab = np.array([int(nt[c]) for c in cs])
    pc = np.array([p[c] for c in cs])
    t = np.array([up_mean(g, rank, today, c, 3) for c in cs])
    r = np.array([up_mean(g, rank, run, c, 5) for c in cs])
    base = np.stack([pc, pc - pc.max(), pc / max(pc.max(), 1e-9), np.arange(len(cs))], 1)
    shift = np.stack([t, r, t - t.max(), r - r.max()], 1)
    return base, shift, lab


def collect_generated():
    B, S, Y, G = [], [], [], []
    for i, panel in enumerate(C.T2_PANELS):
        zt = load(panel, "train")
        g = Gen(panel, zt); g.fit_profiles(zt)
        rank = np.full(g.S, -1); rank[g.order] = np.arange(g.n)
        for f in sorted((ROOT / "work" / __import__("os").environ.get("GEN_ONSET_DIR", "genonset")).glob(f"genonset_{panel}_r*.npz")):
            c = np.load(f)
            run = np.load(ROOT / "work" / __import__("os").environ.get("GEN_RUN_DIR", "genruns_ctm_hb") / f"{f.stem[len('genonset_'):]}.npz")
            zg = dict(obs_flow=run["true_flow"], obs_speed=run["true_speed"], weekday=run["weekday"])
            for n in range(len(c["d"])):
                y, e = c["y"][n] & c["e"][n], c["e"][n]
                if not y.any():
                    continue
                d, t0 = int(c["d"][n]), int(c["t0"][n])
                today = anomaly(g, zg["obs_flow"][d], zg["obs_speed"][d], t0, int(zg["weekday"][d]), 3)
                rr = run_ratio(g, zg, d, g.P)
                if rr is None:
                    continue
                b, s, lab = window_rows(g, rank, c["prob"][n], y, e, today, rr)
                if not lab.any():
                    continue
                B.append(b); S.append(s); Y.append(lab); G.append(len(lab))
        print(f"  generated {panel}: {len(G)} windows so far", flush=True)
    return B, S, Y, G


def collect_lab():
    out = []
    for i, panel in enumerate(C.T2_PANELS):
        zt = load(panel, "train")
        g = Gen(panel, zt); g.fit_profiles(zt)
        rank = np.full(g.S, -1); rank[g.order] = np.arange(g.n)
        ctx, stat, W = C.windows(i, panel)
        for s in C.SPLITS:
            on, z = W[s]["on"], W[s]["z"]
            p = np.mean([np.load(ROOT / "work" / "leap_ctrl_none" / f"fam_leap_ctrl_s{k}_none_{s}_{panel}.npz")["on"] for k in (20, 21)], 0)
            for n in range(len(p)):
                y, e = on["fut"][n, -1] & on["efut"][n, -1], on["efut"][n, -1]
                if not y.any():
                    continue
                d, t0 = int(on["d"][n]), int(on["t0"][n])
                today = anomaly(g, z["obs_flow"][d], z["obs_speed"][d], t0, int(z["weekday"][d]), 3)
                rr = run_ratio(g, z, d, g.P)
                if rr is None:
                    rr = np.ones(len(g.ent))
                b, sh, lab = window_rows(g, rank, p[n], y, e, today, rr)
                out.append((b, sh, lab))
    return out


def main():
    warnings.simplefilter("ignore")
    B, S, Y, G = collect_generated()
    tgt = float(__import__("os").environ.get("MATCH_CNN", "0"))
    if tgt > 0:
        # match the CNN's reliability to the real one (note 169): on generated windows the CNN's pick is right far
        # less often (0.4) than on real ones (0.86), so a ranker learns to override it. Resample windows (by group)
        # so the pick is right on a share tgt of them - the ranker then only learns what the run shift adds.
        rng0 = np.random.default_rng(1)
        hit = np.array([y[0] == 1 for y in Y]); ih, im = np.flatnonzero(hit), np.flatnonzero(~hit)
        n_m = int(round(len(ih) * (1 - tgt) / tgt))
        keep = np.concatenate([ih, rng0.choice(im, min(n_m, len(im)), replace=False)]) if n_m < len(im) else             np.concatenate([rng0.choice(ih, int(round(len(im) * tgt / (1 - tgt))), replace=True), im])
        B, S, Y, G = [B[i] for i in keep], [S[i] for i in keep], [Y[i] for i in keep], [G[i] for i in keep]
        print(f"  resampled to CNN-pick share {np.mean([y[0] == 1 for y in Y]):.3f}: {len(G)} windows", flush=True)
    lab_rows = collect_lab()
    print(f"generated windows {len(G)} (truth among the top-{K}: all), lab windows {len(lab_rows)}")
    rng = np.random.default_rng(0)
    par = {"objective": "lambdarank", "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 50,
           "feature_fraction": 0.9, "verbose": -1, "num_threads": 8, "label_gain": [0, 1]}
    for name in ("CNN only", "CNN + run shift", "CNN + shuffled run shift"):
        top, fixed, broken = [], 0, 0
        for seed in range(3):
            Xs = []
            perm = rng.permutation(len(S))
            for j in range(len(B)):
                if name == "CNN only":
                    Xs.append(B[j])
                else:
                    sh = S[perm[j]] if name.startswith("CNN + shuffled") else S[j]
                    if len(sh) != len(B[j]):
                        sh = np.resize(sh, (len(B[j]), sh.shape[1]))
                    Xs.append(np.concatenate([B[j], sh], 1))
            m = lgb.train({**par, "seed": seed}, lgb.Dataset(np.concatenate(Xs), np.concatenate(Y), group=np.array(G)), 300)
            t_, f_, b_ = [], 0, 0
            for b, sh, lab in lab_rows:
                X = b if name == "CNN only" else np.concatenate([b, sh], 1)
                pick = int(np.argmax(m.predict(X)))
                t_.append(lab[pick]); f_ += int(lab[pick] == 1 and lab[0] == 0); b_ += int(lab[pick] == 0 and lab[0] == 1)
            top.append(np.mean(t_)); fixed += f_; broken += b_
        cnn = np.mean([lab[0] for _, _, lab in lab_rows])
        print(f"  {name:<26} lab top-1 {np.mean(top):.4f} (CNN pick {cnn:.4f}) | per seed fixed {fixed / 3:.1f}, broken {broken / 3:.1f}")


if __name__ == "__main__":
    main()
