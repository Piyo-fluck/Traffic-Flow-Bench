"""Cache the shockwave-v2 ongoing probabilities so the model can enter the note-56 ensemble pool.

Note 54 v2 gains +0.005...+0.008 inside M's OLD blend, but note 56's ensemble (2*tfm_more + cnn3 + gs1 at tau 0.45)
does not even use the production LightGBM, so the only question that matters is whether v2 still adds anything
inside the new mix. Answering it needs v2's probabilities as a pool source; the v2 experiment did not save its
models, so this retrains them (production I configuration + shockwave v1 + v2 blocks, 2 seeds, trained on the train
run only) and writes, per panel and split:

    work/cache/t2cnn_t2_sw2_<split>_<panel>.npz   with og_p (N, F, S) and og_d (N,)

which is exactly the layout `t2_hill_climb.load_group` reads. Onset is not cached (this model is ongoing-only);
`load_group` skips sources that lack the requested key.

    .venv/Scripts/python.exe analysis/t2_cache_sw2_probs.py
    then: T2_EXTRA_TAG=t2_sw2 .venv/Scripts/python.exe analysis/t2_ongoing_consensus.py
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
from t2_demand_stress import load_tq  # noqa: E402
from t2_eda02 import T2_PANELS, candidates  # noqa: E402
from t2_eda03 import F  # noqa: E402
from t2_first_onset_eval import official_blank  # noqa: E402
from t2_m1_models import Ctx, hist_from_truth  # noqa: E402
from t2_official_like_eval import ongoing_official  # noqa: E402
from t2_ongoing_shockwave import SW  # noqa: E402
from t2_ongoing_shockwave2 import CHUNK, LEAVES, MAX_OG, ROUNDS, S2, SEEDS, build  # noqa: E402
from t2_shift_remedies import PARAMS  # noqa: E402
from tfb.data import CACHE, load  # noqa: E402
from tfb.physics import fd_params  # noqa: E402

TAG = "t2_sw2"
rng = np.random.default_rng(5)


def main() -> None:
    t0 = time.time()
    parts, ctxs, tqs, lens, names = [], [], {}, {}, None
    for i, panel in enumerate(T2_PANELS):
        ctx = Ctx(panel, i)
        ctx.fitd = np.ones_like(ctx.fitd)
        z = ctx.z
        tqs[panel], lens[panel] = load_tq(panel), fd_params(panel, z["link_ids"])["length"].astype(np.float32)
        c = candidates(z)
        d, t0w = c["d"][c["ongoing"]], c["t0"][c["ongoing"]]
        xs, ys = [], []
        for j in range(0, len(d), CHUNK):
            dd, tt = d[j:j + CHUNK], t0w[j:j + CHUNK]
            v, qf, ok = hist_from_truth(ctx, dd, tt)
            nm, Xh, E1, E2, base = build(ctx, lens[panel], tqs[panel], v, qf, ok, tt, ctx.g[dd], dd, z["weekday"][dd])
            names = nm + SW + S2
            X = np.concatenate([Xh, E1, E2], -1)
            y = ctx.q[dd[:, None], tt[:, None] + np.arange(1, F + 1)]
            e = ctx.el[dd[:, None], tt[:, None] + np.arange(1, F + 1)]
            near = np.minimum(base["dist_slow_up"], base["dist_slow_dn"]) <= 10
            keep = np.flatnonzero(e & (np.broadcast_to(near[:, None, :], y.shape) | y | (rng.random(y.shape) < 0.02)))
            xs.append(X.reshape(-1, X.shape[-1])[keep].astype(np.float32))
            ys.append(y.reshape(-1)[keep])
        X, y = np.concatenate(xs), np.concatenate(ys)
        if len(y) > MAX_OG:
            k = rng.choice(len(y), MAX_OG, replace=False)
            X, y = X[k], y[k]
        parts.append((X, y))
        ctx.z = {k: z[k] for k in ("link_ids", "weekday", "dates")}
        ctx.q = ctx.el = None
        ctxs.append(ctx)
        print(f"[{panel}] {len(y)} rows ({time.time() - t0:.0f}s)", flush=True)
    X_all = np.concatenate([p[0] for p in parts])
    y_all = np.concatenate([p[1] for p in parts])
    del parts
    models = [lgb.train(dict(PARAMS, num_leaves=LEAVES, seed=sd), lgb.Dataset(X_all, y_all, feature_name=names), ROUNDS)
              for sd in SEEDS]
    del X_all, y_all
    print(f"trained {len(models)} seeds ({time.time() - t0:.0f}s)", flush=True)
    for ctx in ctxs:
        for split in ("validation", "private"):
            z = load(ctx.panel, split)
            og, _ = ongoing_official(ctx, z, official_blank(ctx.panel, split, z))
            n = len(og["d"])
            if not n:
                continue
            ps = []
            for j in range(0, n, CHUNK):
                sl = slice(j, j + CHUNK)
                grp = (z["weekday"][og["d"][sl]] >= 5).astype(int)
                key = np.stack([og["d"][sl], grp], 1)
                _, Xh, E1, E2, _ = build(ctx, lens[ctx.panel], tqs[ctx.panel], og["v"][sl], og["qf"][sl], og["ok"][sl],
                                         og["t0"][sl], grp, key, z["weekday"][og["d"][sl]].astype(int))
                XX = np.concatenate([Xh, E1, E2], -1)
                p = np.mean([m.predict(XX.reshape(-1, XX.shape[-1])) for m in models], 0).reshape(XX.shape[:3])
                ps.append(p.astype(np.float32))
            out = CACHE / f"t2cnn_{TAG}_{split}_{ctx.panel}.npz"
            np.savez_compressed(out, og_p=np.concatenate(ps), og_d=og["d"])
            print(f"  wrote {out.name} ({time.time() - t0:.0f}s)", flush=True)
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
