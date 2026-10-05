"""Train the disputed-cell stacker (note 81) on TRAIN-month windows, whose histories match the official ones.

Note 76 measured that the official Task 2 histories are 76% observed while our scored-month pseudo windows are
only 52% observed, because the pseudo windows inherit the Task 1 target masks. The train month's TRUE layer has
exactly the official coverage (median 0.764 over eligible cells), so windows built from it - the same windows the
production models are trained on - are the right training distribution for a stacker that will run on official
windows.

Per panel: sample ongoing candidate windows of the train month, build their histories from the true layer, compute
the three ongoing members on them (CNN x3 and transformer x8 from their checkpoints, gs1 from the global-state
LightGBM), keep the cells where the three disagree and fit one LightGBM on all panels.

    .venv/Scripts/python.exe scripts/t2_stack_train_month.py --out work/models_stack/stack_train.txt \
        --cnn <cnn x3> --tfm <tfm s3..s10>
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tta_lab" / "tta"))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "scripts"))
import common as C  # noqa: E402
from t2_disputed_stack import NAMES, build  # noqa: E402
from t2_eda02 import candidates  # noqa: E402
from t2_gstate import G_NAMES, gstate  # noqa: E402
from t2_m1_predict import feats_og, zero_prior  # noqa: E402

C.cnn.REC = 0.0
H, F = C.H, C.F


def load_nets(paths, arch):
    out = []
    for p in paths:
        sd = torch.load(p, map_location=C.DEV)
        cin = sd["gru.weight_ih_l0"].shape[1] if arch == "tfm" else sd["inp.weight"].shape[1]
        net = (C.cnn.T2Tfm if arch == "tfm" else C.cnn.T2Net)(cin, C.cnn.WIDTH)
        net.load_state_dict(sd)
        out.append(net.to(C.DEV).eval())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cnn", type=Path, nargs="+", required=True)
    ap.add_argument("--tfm", type=Path, nargs="+", required=True)
    ap.add_argument("--gs-models", type=Path, default=ROOT / "work" / "models_gstate")
    ap.add_argument("--per-panel", type=int, default=800, help="train windows sampled per panel")
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--leaves", type=int, default=63)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout-from", default="", help="date: train on earlier days, evaluate on these (honest "
                                                      "check at the official coverage, same run)")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    nets_cnn, nets_tfm = load_nets(args.cnn, "cnn"), load_nets(args.tfm, "tfm")
    m_og = lgb.Booster(model_file=str(args.gs_models / "ongoing_g1.txt"))
    t0 = time.time()
    Xs, ys, ho = [], [], []
    for i, panel in enumerate(C.T2_PANELS):
        ctx = C.Ctx(panel, i)
        ctx.fitd = np.ones_like(ctx.fitd)
        ctx.loff = 0
        stat = C.cnn.static_channels(ctx, C.load_ramps(panel, "train", ctx.z))
        z = ctx.z
        c = candidates(z)
        d, t0w = c["d"][c["ongoing"]], c["t0"][c["ongoing"]]
        if len(d) > args.per_panel:
            sel = np.sort(rng.choice(len(d), args.per_panel, replace=False))
            d, t0w = d[sel], t0w[sel]
        hi = t0w[:, None] + np.arange(-H, 0)
        fi = t0w[:, None] + np.arange(1, F + 1)
        v, qf = z["true_speed"][d[:, None], hi], z["true_flow"][d[:, None], hi]
        ok = ctx.el[d[:, None], hi] & np.isfinite(v) & np.isfinite(qf)
        grp = ctx.g[d]
        x = C.cnn.make_x(ctx, stat, v, qf, ok, t0w, z["weekday"][d] >= 5)
        p_cnn = np.mean([C.cnn.predict(n, x) for n in nets_cnn], 0)
        p_tfm = np.mean([C.cnn.predict(n, x) for n in nets_tfm], 0)
        pz = np.load(args.gs_models / f"pca_{panel}.npz")
        key = np.stack([d, grp], 1).astype(int)
        _, Xh, _ = feats_og(ctx, True, v, qf, ok, t0w, grp, zero_prior(ctx), key, z["weekday"][d].astype(int))
        g = np.broadcast_to(gstate(ctx, (pz["mu"], pz["vt"]), v, qf, ok)[:, None, None],
                            Xh.shape[:3] + (len(G_NAMES),))
        Xg = np.concatenate([Xh, g], -1)
        p_gs = m_og.predict(Xg.reshape(-1, Xg.shape[-1])).reshape(Xh.shape[:3]).astype(np.float32)
        X, pred, dis = build(i, dict(v=v, qf=qf, ok=ok), ctx, {"gs1": p_gs, "tfm8": p_tfm, "cnn3": p_cnn})
        y, e = ctx.q[d[:, None], fi], ctx.el[d[:, None], fi]
        m = dis & e
        late = (z["dates"][d] >= args.holdout_from) if args.holdout_from else np.zeros(len(d), bool)
        Xs.append((X[~late & True][m[~late]], X[late][m[late]]))
        ys.append((y[~late][m[~late]].astype(np.float32), y[late][m[late]].astype(np.float32)))
        if args.holdout_from:
            ho.append(dict(panel=panel, X=X[late], pred=pred[late], dis=m[late], y=y[late], e=e[late]))
        print(f"[{panel}] {len(d)} windows, coverage {ok.mean():.3f}, disputed cells {int(m.sum()):,} "
              f"({time.time() - t0:.0f}s)", flush=True)
        ctx.z = ctx.q = ctx.el = None
    X, y = np.concatenate([a for a, _ in Xs]), np.concatenate([a for a, _ in ys])
    print(f"training on {len(y):,} disputed cells, {y.mean():.3f} queued", flush=True)
    mdl = lgb.train(dict(objective="binary", learning_rate=0.05, num_leaves=args.leaves, verbose=-1,
                         feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1),
                    lgb.Dataset(X, y, feature_name=NAMES), args.rounds)
    mdl.save_model(str(args.out))
    if ho:
        fam_b, fam_s, wb, ws, ok_b, ok_s, n = {}, {}, [], [], 0, 0, 0
        for h in ho:
            out = h["pred"].copy()
            if h["dis"].any():
                out[h["dis"]] = mdl.predict(h["X"][h["dis"]]) >= 0.5
                ok_b += int((h["pred"][h["dis"]] == h["y"][h["dis"]]).sum())
                ok_s += int((out[h["dis"]] == h["y"][h["dis"]]).sum())
                n += int(h["dis"].sum())
            ib, is_ = C.iou(h["pred"], h["y"], h["e"]), C.iou(out, h["y"], h["e"])
            fam_b[h["panel"]], fam_s[h["panel"]] = ib.mean(), is_.mean()
            wb.append(ib); ws.append(is_)
        for thr in (0.30, 0.35, 0.40, 0.45, 0.50):
            fb, fs, nb, ns, nn = {}, {}, 0, 0, 0
            for h in ho:
                o2 = h["pred"].copy()
                if h["dis"].any():
                    o2[h["dis"]] = mdl.predict(h["X"][h["dis"]]) >= thr
                    nb += int((h["pred"][h["dis"]] == h["y"][h["dis"]]).sum())
                    ns += int((o2[h["dis"]] == h["y"][h["dis"]]).sum()); nn += int(h["dis"].sum())
                fb[h["panel"]] = C.iou(h["pred"], h["y"], h["e"]).mean()
                fs[h["panel"]] = C.iou(o2, h["y"], h["e"]).mean()
            print(f"  thr {thr:.2f}: accuracy {ns / max(nn, 1):.3f}, family {C.family_mean(fs) - C.family_mean(fb):+.4f}, "
                  f"queued cells {sum(int((h['pred'].copy().__setitem__(h['dis'], mdl.predict(h['X'][h['dis']]) >= thr) or h['pred']).sum()) for h in ho if h['dis'].any())}",
                  flush=True)
        b, st = C.family_mean(fam_b), C.family_mean(fam_s)
        print(f"HOLDOUT (train days >= {args.holdout_from}, official coverage): disputed accuracy "
              f"{ok_b / max(n, 1):.3f} -> {ok_s / max(n, 1):.3f} ({n:,} cells); family {b:.4f} -> {st:.4f} "
              f"({st - b:+.4f}, total {0.15 * (st - b):+.5f}); window {C.ci(np.concatenate(ws) - np.concatenate(wb))}")
    imp = mdl.feature_importance("gain")
    print("saved " + str(args.out) + "; top features: " +
          ", ".join(f"{NAMES[j]} {imp[j] / imp.sum():.2f}" for j in np.argsort(-imp)[:6]))


if __name__ == "__main__":
    main()
