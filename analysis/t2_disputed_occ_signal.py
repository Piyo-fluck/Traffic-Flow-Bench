"""Is there any signal left in OCCUPANCY for the ongoing cells the three members dispute?

Note 80: the members dispute 2.9% of the cells and the blend is 70-73% right there. The blend probability is
close to calibrated on those cells (p 0.50 -> 0.44-0.46 truly queued), so no threshold or monotone recalibration
can improve them - only new information can. Occupancy is the one released mainline channel Task 2 never uses
(the CNN/transformer inputs are speed, flow, the observed flag and the queued flag), and density is what decides
whether a queue front grows or discharges.

This measures the residual signal, it does not build a model: on the train month (true-layer histories, the
official coverage), for the disputed cells, a logistic regression on logit(p) alone is compared with one that also
sees occupancy features of the cell's link:
    occ_last, occ_mean3, d_occ (last minus the mean of the first three slots), occ over the weekday x tod profile,
    and the same for the two neighbouring links (upstream / downstream).
Reported: log-loss and AUC of both fits, 5-fold by panel (fit on 7 panels, score the eighth), plus the accuracy
at 0.5 for reference.

    .venv/Scripts/python.exe analysis/t2_disputed_occ_signal.py --cnn ... --tfm ...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "tta_lab" / "tta"))
import common as C  # noqa: E402
from t2_disputed_rules import _load, TAU  # noqa: E402
from t2_disputed_stack import BLEND_W  # noqa: E402
from t2_eda02 import candidates  # noqa: E402
from t2_gstate import G_NAMES, gstate  # noqa: E402
from t2_m1_predict import feats_og, zero_prior  # noqa: E402

C.cnn.REC = 0.0
H, F = C.H, C.F


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnn", type=Path, nargs="+", required=True)
    ap.add_argument("--tfm", type=Path, nargs="+", required=True)
    ap.add_argument("--gs-models", type=Path, default=ROOT / "work" / "models_gstate")
    ap.add_argument("--per-panel", type=int, default=600)
    args = ap.parse_args()
    nets_cnn = [_load(p, "cnn") for p in args.cnn]
    nets_tfm = [_load(p, "tfm") for p in args.tfm]
    m_og = lgb.Booster(model_file=str(args.gs_models / "ongoing_g1.txt"))
    rng = np.random.default_rng(0)
    rows = []
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
        hi, fi = t0w[:, None] + np.arange(-H, 0), t0w[:, None] + np.arange(1, F + 1)
        v, qf = z["true_speed"][d[:, None], hi], z["true_flow"][d[:, None], hi]
        occ = z["true_occ"][d[:, None], hi]
        ok = ctx.el[d[:, None], hi] & np.isfinite(v) & np.isfinite(qf)
        x = C.cnn.make_x(ctx, stat, v, qf, ok, t0w, z["weekday"][d] >= 5)
        P = {"cnn3": np.mean([C.cnn.predict(n, x) for n in nets_cnn], 0),
             "tfm8": np.mean([C.cnn.predict(n, x) for n in nets_tfm], 0)}
        pz = np.load(args.gs_models / f"pca_{panel}.npz")
        grp = ctx.g[d]
        _, Xh, _ = feats_og(ctx, True, v, qf, ok, t0w, grp, zero_prior(ctx), np.stack([d, grp], 1).astype(int),
                            z["weekday"][d].astype(int))
        g = np.broadcast_to(gstate(ctx, (pz["mu"], pz["vt"]), v, qf, ok)[:, None, None], Xh.shape[:3] + (len(G_NAMES),))
        Xg = np.concatenate([Xh, g], -1)
        P["gs1"] = m_og.predict(Xg.reshape(-1, Xg.shape[-1])).reshape(Xh.shape[:3]).astype(np.float32)
        bl = BLEND_W[0] * P["gs1"] + BLEND_W[1] * P["tfm8"] + BLEND_W[2] * P["cnn3"]
        dis = ~(((P["gs1"] >= TAU["gs1"]) == (P["tfm8"] >= TAU["tfm8"])) &
                ((P["tfm8"] >= TAU["tfm8"]) == (P["cnn3"] >= TAU["cnn3"]))) & ctx.el[d[:, None], fi]
        y = ctx.q[d[:, None], fi]
        with np.errstate(all="ignore"):
            oc = np.where(ok, occ, np.nan)
            o_last = np.nan_to_num(oc[:, -1], nan=0.0)
            o_m3 = np.nan_to_num(np.nanmean(oc[:, -3:], 1), nan=0.0)
            o_first = np.nan_to_num(np.nanmean(oc[:, :3], 1), nan=0.0)
            prof = np.nan_to_num(np.nanmean(np.where(ctx.el, z["true_occ"], np.nan), (0, 1)), nan=0.05)
        feats = [o_last, o_m3, o_m3 - o_first, o_last / np.maximum(prof[None, :], 1e-3),
                 np.roll(o_last, 1, -1), np.roll(o_last, -1, -1), np.roll(o_m3 - o_first, 1, -1)]
        Fo = np.stack([np.broadcast_to(f[:, None, :], bl.shape) for f in feats], -1)
        lg = np.log(np.clip(bl, 1e-5, 1 - 1e-5) / (1 - np.clip(bl, 1e-5, 1 - 1e-5)))
        rows.append(dict(panel=panel, p=lg[dis], occ=Fo[dis], y=y[dis].astype(int)))
        print(f"  {panel}: {int(dis.sum()):,} disputed cells", flush=True)
        ctx.z = ctx.q = ctx.el = None

    res = {"p only": [], "p + occupancy": []}
    for k, hold in enumerate(rows):
        tr = [r for r in rows if r is not hold]
        Xtr_p = np.concatenate([r["p"] for r in tr])[:, None]
        Xte_p = hold["p"][:, None]
        Xtr_o = np.concatenate([np.c_[r["p"], r["occ"]] for r in tr])
        Xte_o = np.c_[hold["p"], hold["occ"]]
        ytr, yte = np.concatenate([r["y"] for r in tr]), hold["y"]
        for name, Xtr, Xte in (("p only", Xtr_p, Xte_p), ("p + occupancy", Xtr_o, Xte_o)):
            m = lgb.train(dict(objective="binary", learning_rate=0.05, num_leaves=15, verbose=-1,
                               feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1),
                          lgb.Dataset(Xtr, ytr.astype(float)), 150)
            pr = np.clip(m.predict(Xte), 1e-6, 1 - 1e-6)
            ll = -float(np.mean(yte * np.log(pr) + (1 - yte) * np.log(1 - pr)))
            o = np.argsort(pr); r = np.empty(len(pr)); r[o] = np.arange(len(pr))
            n1, n0 = yte.sum(), len(yte) - yte.sum()
            auc = float((r[yte == 1].sum() - n1 * (n1 - 1) / 2) / max(n1 * n0, 1))
            res[name].append((ll, auc, float(((pr >= 0.5) == yte).mean())))
    print("\n=== leave-one-panel-out on the disputed cells of the train month ===")
    for name, v in res.items():
        a = np.array(v)
        print(f"  {name:<14} log-loss {a[:, 0].mean():.4f}  AUC {a[:, 1].mean():.4f}  accuracy@0.5 {a[:, 2].mean():.4f}")


if __name__ == "__main__":
    main()
