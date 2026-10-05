"""Stability-selected corrector (the user's proposal after note 89).

AI's corrector is kept, but a correction is allowed only where correctors fitted on DIFFERENT TIME BLOCKS of the
held-out tail reproduce it. The question is not how confident a model is, but whether the correction survives a
change of training days - AJ and AK (notes 87, 88) looked better locally and lost on the LB, which points at
variance in WHICH cells get corrected.

Correctors, all AI's recipe (15 features, the tail's masked-layer windows, L2 on y - p0, 400 trees):
    AI        the whole tail
    lobo8     8 contiguous day blocks, leave one block out -> 8 correctors (each sees 7/8 of the days)
    half6     4 contiguous two-week blocks, leave two out -> 6 correctors (each sees half, as in the original
              stability selection of Meinshausen and Buhlmann)

Filters on the cells AI corrects (eta 0.75, gate 0.10 throughout):
    sign   keep AI's correction where >= k of the block correctors give r_hat the same sign as AI
    flip   keep it where >= k of them, applied alone with the same eta / gate, flip the same cell the same way
    median replace AI's r_hat by the median of the block correctors and keep it where the sign agreement is >= k
           (the user's literal formulation)

Headline, per month: the share of AI's corrected cells whose corrected decision is right, all of them vs the kept
ones; how many are kept; rescue / spoil; family IoU gain over the frozen blend.

    python occ_lab/stability.py
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta"))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

import common as C  # noqa: E402
from logit_update import IDX  # noqa: E402
from residual_cov import CACHE, PARAMS, load_env  # noqa: E402
from residual_diag import NAMES, TAU  # noqa: E402
from residual_oof import panel_data as scored_panel  # noqa: E402

ETA, GATE = 0.75, 0.10
MODELS = LAB / "models"


def step(r):
    return ETA * np.where(np.abs(r) >= GATE, r, 0.0)


def fit(train, day_ok):
    Xs, ys, ps = [], [], []
    for p, (f, y, e, base, day) in train.items():
        m = e & day_ok(day)[:, None, None]
        Xs.append(f[m]); ys.append(y[m].astype(np.float32)); ps.append(base[m])
    X, y, p0 = np.concatenate(Xs), np.concatenate(ys), np.concatenate(ps)
    return lgb.train(dict(PARAMS, num_threads=32), lgb.Dataset(X, label=y - p0, feature_name=NAMES,
                                                                free_raw_data=False), num_boost_round=400)


def main() -> None:
    train = {}
    for p in C.T2_PANELS:
        lab = np.load(CACHE / f"cov_labels_{p}.npz")
        f, base, _ = load_env("obs", p)
        train[p] = (f[..., IDX], lab["y"], lab["e"], base, lab["date"].astype(str))
    days = np.unique(np.concatenate([t[4] for t in train.values()]))
    print(f"{len(days)} held-out days, {days[0]} .. {days[-1]}", flush=True)
    b8 = np.array_split(days, 8)
    b4 = np.array_split(days, 4)
    schemes = {"lobo8": [set(days) - set(b) for b in b8],
               "half6": [set(np.concatenate([b4[i], b4[j]])) for i, j in combinations(range(4), 2)]}

    ai = fit(train, lambda d: np.ones(len(d), bool))
    ai.save_model(str(MODELS / "stab_ai.txt"))
    models = {}
    for name, keep_sets in schemes.items():
        models[name] = []
        for k, ks in enumerate(keep_sets):
            ks = np.array(sorted(ks))
            models[name].append(fit(train, lambda d, ks=ks: np.isin(d, ks)))
            models[name][-1].save_model(str(MODELS / f"stab_{name}_{k}.txt"))
        print(f"  fitted {name}: {len(keep_sets)} correctors, each on {len(keep_sets[0])} of {len(days)} days",
              flush=True)

    for s in C.SPLITS:
        print(f"\n=== {s} ===", flush=True)
        fam_b, fam, stats = {}, {}, {}
        for p in C.T2_PANELS:
            d = scored_panel(p, s)
            if d is None:
                continue
            f, y, e, base = d
            fe = f[e]
            b = base[e]
            yy = y[e]
            r_ai = ai.predict(fe).astype(np.float32)
            p_ai = np.clip(b + step(r_ai), 0, 1)
            dec_b, dec_ai = b >= TAU, p_ai >= TAU
            corrected = dec_ai != dec_b
            fam_b[p] = C.iou(base >= TAU, y, e).mean()
            variants = {"AI": dec_ai}
            for name, ms in models.items():
                R = np.stack([m.predict(fe).astype(np.float32) for m in ms])          # (K, n)
                K = len(ms)
                sign_agree = (np.sign(R) == np.sign(r_ai)[None]).sum(0)
                flip_agree = ((np.clip(b[None] + step(R), 0, 1) >= TAU) == dec_ai[None]).sum(0)
                med = np.median(R, 0)
                med_agree = (np.sign(R) == np.sign(med)[None]).sum(0)
                p_med = np.clip(b + step(med), 0, 1) >= TAU
                for k in range(K - 2, K + 1):
                    variants[f"{name} sign>={k}"] = np.where(sign_agree >= k, dec_ai, dec_b)
                    variants[f"{name} flip>={k}"] = np.where(flip_agree >= k, dec_ai, dec_b)
                    variants[f"{name} median>={k}"] = np.where(med_agree >= k, p_med, dec_b)
            for n, dec in variants.items():
                full = base >= TAU
                full[e] = dec
                fam.setdefault(n, {})[p] = C.iou(full, y, e).mean()
                ch = dec != dec_b
                right = dec == yy
                a = stats.setdefault(n, [0, 0, 0, 0, 0, 0, 0])
                a[0] += int(ch.sum()); a[1] += int((ch & right).sum())
                a[2] += int((ch & corrected).sum())
                bw = dec_b != yy
                a[3] += int((bw & right).sum()); a[4] += int(bw.sum())
                a[5] += int((~bw & ~right).sum()); a[6] += int((~bw).sum())
        b0 = C.family_mean(fam_b)
        a_ai = stats["AI"]
        print(f"  blend {b0:.4f}; AI corrects {a_ai[0]:,} cells, right on {a_ai[1] / a_ai[0]:.3f} of them")
        print(f"  {'variant':<22} {'IoU gain':>9} {'changed':>8} {'kept of AI':>10} {'precision':>9} "
              f"{'rescue':>7} {'spoil':>7}")
        for n in variants:
            a = stats[n]
            print(f"  {n:<22} {C.family_mean(fam[n]) - b0:+9.4f} {a[0]:8,} {a[2] / max(a_ai[0], 1):10.2f} "
                  f"{a[1] / max(a[0], 1):9.3f} {a[3] / max(a[4], 1):7.3f} {a[5] / max(a[6], 1):7.4f}", flush=True)


if __name__ == "__main__":
    main()
