"""Does the residual corrector survive a change of history coverage, and does training across coverages help?

Material: `cache_cov.py` - the same 2031-01/02 held-out windows re-inferred under five history availabilities
(obs = masked layer, about 0.52; c50 / c60 / c70; full = true layer, about the official 0.76).

Correctors (all unweighted L2 on r = y - p0, 400 trees, the recipe of candidate AI):
    AI         fitted on obs only, the 15 features of AI                      (what is on the LB now)
    full       fitted on full only, + per-window coverage
    all        fitted on all five environments, + coverage
    all-nopos  as `all` without the link-position feature                     (is position a shortcut?)

Tests:
  1. time split inside the tail - fit on the January windows, score the February windows of EVERY environment.
     The `full` column is the one that resembles the official windows.
  2. the two scored months on the usual pseudo windows (about 0.52 covered), fitted on the whole tail.

    python occ_lab/residual_cov.py
"""
from __future__ import annotations

import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta"))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

import common as C  # noqa: E402
from residual_diag import NAMES, TAU, lg  # noqa: E402
from residual_oof import panel_data as scored_panel  # noqa: E402

CACHE = LAB / "cache"
ENVS = ("obs", "c50", "c60", "c70", "full")
SETTINGS = [(eta, g) for eta in (0.50, 0.75, 1.00) for g in (0.0, 0.10)]
PARAMS = {"objective": "l2", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 200,
          "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "num_threads": 16}


def feats(P, cov_w):
    base = 0.2 * P["gs1"] + 0.6 * P["tfm8"] + 0.2 * P["cnn3"]
    N, Fh, S = base.shape
    hor = np.broadcast_to(np.arange(Fh, dtype=np.float32)[None, :, None], base.shape)
    pos = np.broadcast_to((np.arange(S, dtype=np.float32) / max(S - 1, 1))[None, None, :], base.shape)
    cov = np.broadcast_to(cov_w[:, None, None].astype(np.float32), base.shape)
    members = np.stack([P[k] for k in ("gs1", "tfm8", "cnn3", "occ_cnn", "occ_tfm", "stid")])
    f = np.stack([base, lg(base), np.abs(base - 0.5), members.std(0), members.max(0) - members.min(0),
                  P["gs1"], P["tfm8"], P["cnn3"], P["occ_cnn"], P["occ_tfm"], P["stid"],
                  lg(P["stid"]) - lg(base), lg(P["occ_cnn"]) - lg(base), hor, pos, cov], -1)
    return f, base


FNAMES = NAMES + ["coverage"]
VARIANTS = {"AI": [n for n in NAMES], "full": FNAMES, "all": FNAMES,
            "all-nopos": [n for n in FNAMES if n != "position"]}
TRAIN_ENVS = {"AI": ("obs",), "full": ("full",), "all": ENVS, "all-nopos": ENVS}


def load_env(env, panel):
    z = np.load(CACHE / f"cov_{env}_{panel}.npz")
    P = {k: z[k].astype(np.float32) for k in ("gs1", "tfm8", "cnn3", "occ_cnn", "occ_tfm", "stid")}
    return feats(P, z["cov"]) + (float(z["cov"].mean()),)


def fit(X, y, p0, cols):
    idx = [FNAMES.index(c) for c in cols]
    return lgb.train(PARAMS, lgb.Dataset(X[:, idx], label=y - p0, feature_name=cols, free_raw_data=False),
                     num_boost_round=400), idx


def score(gbm, idx, f, base, y, e):
    rh = np.zeros(base.shape, np.float32)
    rh[e] = gbm.predict(f[e][:, idx]).astype(np.float32)
    out = {}
    for eta, g in SETTINGS:
        p = np.clip(base + eta * np.where(np.abs(rh) >= g, rh, 0.0), 0, 1)
        out[(eta, g)] = C.iou(p >= TAU, y, e)
    return out


def main() -> None:
    lab = {p: np.load(CACHE / f"cov_labels_{p}.npz") for p in C.T2_PANELS}
    env = {(v, p): load_env(v, p) for v in ENVS for p in C.T2_PANELS}
    print("mean coverage by environment: " + "  ".join(
        f"{v} {np.mean([env[(v, p)][2] for p in C.T2_PANELS]):.3f}" for v in ENVS), flush=True)
    jan = {p: np.char.startswith(lab[p]["date"].astype(str), "2031-01") for p in C.T2_PANELS}

    # ---------------------------------------------------------------- 1. January -> February, every environment
    print("\n=== 1. fit on 2031-01, score 2031-02 (family IoU gain over the frozen blend) ===")
    for name, cols in VARIANTS.items():
        Xs, ys, ps = [], [], []
        for v in TRAIN_ENVS[name]:
            for p in C.T2_PANELS:
                f, base, _ = env[(v, p)]
                m = lab[p]["e"] & jan[p][:, None, None]
                Xs.append(f[m]); ys.append(lab[p]["y"][m].astype(np.float32)); ps.append(base[m])
        gbm, idx = fit(np.concatenate(Xs), np.concatenate(ys), np.concatenate(ps), cols)
        row = {}
        for v in ENVS:
            fam_b, fam = {}, {}
            for p in C.T2_PANELS:
                f, base, _ = env[(v, p)]
                w = ~jan[p]
                y, e = lab[p]["y"][w], lab[p]["e"][w]
                fam_b[p] = C.iou(base[w] >= TAU, y, e).mean()
                for k, s in score(gbm, idx, f[w], base[w], y, e).items():
                    fam.setdefault(k, {})[p] = s.mean()
            b = C.family_mean(fam_b)
            row[v] = (b, {k: C.family_mean(d) - b for k, d in fam.items()})
        print(f"  {name:<10} " + "  ".join(f"{v} {row[v][1][(0.75, 0.10)]:+.4f}" for v in ENVS)
              + "   (eta 0.75, gate 0.10)", flush=True)
        best = {v: max(row[v][1].items(), key=lambda kv: kv[1]) for v in ENVS}
        print(f"  {'':<10} best: " + "  ".join(f"{v} {best[v][1]:+.4f}@{best[v][0][0]:.2f}/{best[v][0][1]:.2f}"
                                               for v in ENVS), flush=True)
        if name == "AI":
            print(f"  {'':<10} blend: " + "  ".join(f"{v} {row[v][0]:.4f}" for v in ENVS), flush=True)

    # ---------------------------------------------------------------- 2. whole tail -> the scored months
    print("\n=== 2. fit on the whole tail, score the two months (pseudo windows, ~0.52 covered) ===")
    scored = {}
    for s in C.SPLITS:
        for p in C.T2_PANELS:
            d = scored_panel(p, s)
            fc = CACHE / f"cov_scored_{s}_{p}.npz"
            if d is None or not fc.exists():
                continue
            f15, y, e, base = d
            cw = np.load(fc)["cov"].astype(np.float32)
            f = np.concatenate([f15, np.broadcast_to(cw[:, None, None, None], f15.shape[:3] + (1,))], -1)
            scored[(s, p)] = (f, y, e, base)
    if not scored:
        print("  (no cov_scored_* cache - run with --scored-cov first)")
        return
    for name, cols in VARIANTS.items():
        Xs, ys, ps = [], [], []
        for v in TRAIN_ENVS[name]:
            for p in C.T2_PANELS:
                f, base, _ = env[(v, p)]
                m = lab[p]["e"]
                Xs.append(f[m]); ys.append(lab[p]["y"][m].astype(np.float32)); ps.append(base[m])
        gbm, idx = fit(np.concatenate(Xs), np.concatenate(ys), np.concatenate(ps), cols)
        out = []
        for s in C.SPLITS:
            fam_b, fam = {}, {}
            for p in C.T2_PANELS:
                if (s, p) not in scored:
                    continue
                f, y, e, base = scored[(s, p)]
                fam_b[p] = C.iou(base >= TAU, y, e).mean()
                for k, sc in score(gbm, idx, f, base, y, e).items():
                    fam.setdefault(k, {})[p] = sc.mean()
            b = C.family_mean(fam_b)
            g = {k: C.family_mean(d) - b for k, d in fam.items()}
            out.append(f"{s} {g[(0.75, 0.10)]:+.4f} (best {max(g.values()):+.4f})")
        print(f"  {name:<10} " + "   ".join(out), flush=True)


if __name__ == "__main__":
    if "--scored-cov" in sys.argv:
        for i, panel in enumerate(C.T2_PANELS):
            ctx, stat, W = C.windows(i, panel)
            for s in C.SPLITS:
                og = W[s]["og"]
                if len(og["d"]):
                    np.savez(CACHE / f"cov_scored_{s}_{panel}.npz", cov=og["ok"].mean((1, 2)).astype(np.float32),
                             d=og["d"].astype(np.int32), t0=og["t0"].astype(np.int32),
                             date=W[s]["z"]["dates"][og["d"]].astype(str))
            print(f"  {panel} coverage cached", flush=True)
    else:
        main()
