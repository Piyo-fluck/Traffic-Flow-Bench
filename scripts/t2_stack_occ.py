"""Re-decide the disputed ongoing cells with OCCUPANCY, keeping the number of queued cells unchanged.

Why occupancy: the three ongoing members see speed, flow, the observed flag and the queued flag, never occupancy.
On the cells they dispute (note 80) the blend probability is calibrated (note: p 0.50 -> 0.44-0.46 truly queued),
so no threshold or monotone recalibration can help - but occupancy carries residual signal there. Measured on the
train month, leave-one-panel-out over 38,000 disputed cells: AUC 0.772 -> 0.785, log-loss 0.5685 -> 0.5437
(`analysis/t2_disputed_occ_signal.py`).

Why the rate constraint: candidate AE re-decided the same cells with an accuracy-trained stacker at 0.5 and lost
0.0012 on the LB. It had cut the queued cells (the upload went from 10,627 to 10,342 queued rows) because the
disputed cells are only 44-47% queued, so maximising accuracy deletes queue. IoU punishes that: 70% of the
ongoing error is already missed queue (note 52). Here the stacker's threshold is set per split so that the number
of queued cells among the disputed ones equals the blend's - the decision moves WHICH cells are queued, not how
many.

Training: train-month ongoing windows with true-layer histories, whose coverage (0.764) matches the official
windows (note 76). Applied to the official windows, writing the same pred_<split>_<panel>.npz that
scripts/t2_cnn_blend_predict.py --og-pred-dir reads.

    .venv/Scripts/python.exe scripts/t2_stack_occ.py --out-dir work/sub_t2stackocc --cnn ... --tfm ... \
        --gs-dir work/sub_t2gs
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tta_lab" / "tta"))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "scripts"))
import common as C  # noqa: E402
from t2_disputed_stack import BLEND_TAU, BLEND_W, TAU  # noqa: E402
from t2_eda02 import candidates  # noqa: E402
from t2_gstate import G_NAMES, gstate  # noqa: E402
from t2_m1_predict import feats_og, hist_from_release, slot, zero_prior  # noqa: E402
from tfb.data import RELEASE  # noqa: E402

C.cnn.REC = 0.0
H, F = C.H, C.F
NAMES = ["logit_p", "p_gs1", "p_tfm8", "p_cnn3", "p_std", "horizon", "v_last", "qcap_last", "queued_T",
         "dist_queue", "occ_last", "occ_m3", "d_occ", "occ_rel", "occ_up", "occ_dn", "d_occ_up"]


def _load(p, arch):
    sd = torch.load(p, map_location=C.DEV)
    cin = sd["gru.weight_ih_l0"].shape[1] if arch == "tfm" else sd["inp.weight"].shape[1]
    net = (C.cnn.T2Tfm if arch == "tfm" else C.cnn.T2Net)(cin, C.cnn.WIDTH)
    net.load_state_dict(sd)
    return net.to(C.DEV).eval()


def occ_from_release(ctx, H_df, W, t0):
    """(N, H, S) occupancy of the official window histories, aligned like hist_from_release."""
    s_of = {l: i for i, l in enumerate(ctx.z["link_ids"])}
    out = np.full((len(W), H, ctx.S), np.nan, np.float32)
    w_of = {w: i for i, w in enumerate(W.window_id)}
    wi = H_df.window_id.map(w_of).to_numpy()
    hi = slot(H_df.timestamp) - t0[wi] + H
    si = H_df.link_id.astype(str).map(s_of).to_numpy()
    good = (hi >= 0) & (hi < H)
    out[wi[good], hi[good], si[good]] = H_df.occupancy.to_numpy(float)[good]
    return out


def features(ctx, P, v, qf, ok, occ, occ_prof):
    """(N, F, S, len(NAMES)) plus the blend probability and the dispute mask."""
    bl = BLEND_W[0] * P["gs1"] + BLEND_W[1] * P["tfm8"] + BLEND_W[2] * P["cnn3"]
    dis = ~(((P["gs1"] >= TAU["gs1"]) == (P["tfm8"] >= TAU["tfm8"])) &
            ((P["tfm8"] >= TAU["tfm8"]) == (P["cnn3"] >= TAU["cnn3"])))
    with np.errstate(all="ignore"):
        oc = np.where(ok, occ, np.nan)
        o_last = np.nan_to_num(oc[:, -1], nan=0.0)
        o_m3 = np.nan_to_num(np.nanmean(oc[:, -3:], 1), nan=0.0)
        d_occ = o_m3 - np.nan_to_num(np.nanmean(oc[:, :3], 1), nan=0.0)
        v_last = np.nan_to_num(v[:, -1] / ctx.vf, nan=0.0)
        q_last = np.nan_to_num(qf[:, -1] / ctx.cap, nan=0.0)
    queued_T = ok[:, -1] & (np.nan_to_num(v[:, -1], nan=1e9) <= 0.6 * ctx.vf)
    idx = np.arange(ctx.S)
    dist = np.full(queued_T.shape, ctx.S, np.float32)
    for j in range(len(queued_T)):
        q = np.flatnonzero(queued_T[j])
        if len(q):
            dist[j] = np.abs(idx[:, None] - q[None, :]).min(1)
    lg = np.log(np.clip(bl, 1e-5, 1 - 1e-5) / (1 - np.clip(bl, 1e-5, 1 - 1e-5)))
    per_link = [o_last, o_m3, d_occ, o_last / np.maximum(occ_prof[None, :], 1e-3),
                np.roll(o_last, 1, -1), np.roll(o_last, -1, -1), np.roll(d_occ, 1, -1)]
    bc = lambda a: np.broadcast_to(a[:, None, :], bl.shape)  # noqa: E731
    f = [lg, P["gs1"], P["tfm8"], P["cnn3"], np.std(np.stack([P[n] for n in P]), 0),
         np.broadcast_to(np.arange(1, F + 1, dtype=np.float32)[None, :, None], bl.shape),
         bc(v_last), bc(q_last), bc(queued_T.astype(np.float32)), bc(dist)] + [bc(a) for a in per_link]
    return np.stack(f, -1).astype(np.float32), bl, dis


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--cnn", type=Path, nargs="+", required=True)
    ap.add_argument("--tfm", type=Path, nargs="+", required=True)
    ap.add_argument("--gs-dir", type=Path, required=True)
    ap.add_argument("--gs-models", type=Path, default=ROOT / "work" / "models_gstate")
    ap.add_argument("--per-panel", type=int, default=800)
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--leaves", type=int, default=31)
    ap.add_argument("--free-rate", action="store_true", help="drop the rate constraint and decide at 0.5")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    nets_cnn, nets_tfm = [_load(p, "cnn") for p in args.cnn], [_load(p, "tfm") for p in args.tfm]
    m_og = lgb.Booster(model_file=str(args.gs_models / "ongoing_g1.txt"))
    rng = np.random.default_rng(0)
    t0 = time.time()
    Xtr, ytr, official = [], [], []
    for i, panel in enumerate(C.T2_PANELS):
        ctx = C.Ctx(panel, i)
        ctx.fitd = np.ones_like(ctx.fitd)
        ctx.loff = 0
        stat = C.cnn.static_channels(ctx, C.load_ramps(panel, "train", ctx.z))
        z = ctx.z
        with np.errstate(all="ignore"):
            occ_prof = np.nan_to_num(np.nanmean(np.where(ctx.el, z["true_occ"], np.nan), (0, 1)), nan=0.05)

        def members(v, qf, ok, t0w, weekend, d_idx, grp, wd):
            x = C.cnn.make_x(ctx, stat, v, qf, ok, t0w, weekend)
            P = {"cnn3": np.mean([C.cnn.predict(n, x) for n in nets_cnn], 0),
                 "tfm8": np.mean([C.cnn.predict(n, x) for n in nets_tfm], 0)}
            pz = np.load(args.gs_models / f"pca_{panel}.npz")
            _, Xh, _ = feats_og(ctx, True, v, qf, ok, t0w, grp, zero_prior(ctx),
                                np.stack([d_idx, grp], 1).astype(int), wd.astype(int))
            g = np.broadcast_to(gstate(ctx, (pz["mu"], pz["vt"]), v, qf, ok)[:, None, None],
                                Xh.shape[:3] + (len(G_NAMES),))
            Xg = np.concatenate([Xh, g], -1)
            P["gs1"] = m_og.predict(Xg.reshape(-1, Xg.shape[-1])).reshape(Xh.shape[:3]).astype(np.float32)
            return P

        # ---- training rows: train-month windows, true-layer histories (official coverage)
        c = candidates(z)
        d, t0w = c["d"][c["ongoing"]], c["t0"][c["ongoing"]]
        if len(d) > args.per_panel:
            sel = np.sort(rng.choice(len(d), args.per_panel, replace=False))
            d, t0w = d[sel], t0w[sel]
        hi, fi = t0w[:, None] + np.arange(-H, 0), t0w[:, None] + np.arange(1, F + 1)
        v, qf = z["true_speed"][d[:, None], hi], z["true_flow"][d[:, None], hi]
        occ = z["true_occ"][d[:, None], hi]
        ok = ctx.el[d[:, None], hi] & np.isfinite(v) & np.isfinite(qf)
        P = members(v, qf, ok, t0w, z["weekday"][d] >= 5, d, ctx.g[d], z["weekday"][d])
        X, bl, dis = features(ctx, P, v, qf, ok, occ, occ_prof)
        m = dis & ctx.el[d[:, None], fi]
        Xtr.append(X[m])
        ytr.append(ctx.q[d[:, None], fi][m].astype(np.float32))
        # ---- official windows
        for s in C.SPLITS:
            base = RELEASE / "task2" / panel / s
            Wi = pd.read_csv(base / "window_index.csv")
            Hh = pd.read_parquet(base / "window_history.parquet")
            v2, qf2, ok2, t02, grp2 = hist_from_release(ctx, Hh, Wi)
            occ2 = occ_from_release(ctx, Hh, Wi, t02)
            d2 = np.zeros(len(Wi), int)          # only used for the prior key, which is zeroed anyway
            wd2 = np.array([pd.Timestamp(str(x)).weekday() for x in Wi.date])
            P2 = members(v2, qf2, ok2, t02, grp2.astype(bool), d2, grp2, wd2)
            gs = np.load(args.gs_dir / f"prob_{s}_{panel}.npz")
            assert (gs["window_id"] == Wi.window_id.to_numpy().astype(str)).all()
            P2["gs1"] = gs["prob"].astype(np.float32)          # the production gs1 probabilities
            X2, bl2, dis2 = features(ctx, P2, v2, qf2, ok2, occ2, occ_prof)
            official.append(dict(panel=panel, split=s, X=X2, bl=bl2, dis=dis2,
                                 is_on=(Wi.condition == "queue_onset").to_numpy(),
                                 ids=Wi.window_id.to_numpy().astype(str)))
        print(f"[{panel}] train disputed {int(m.sum()):,}; official prepared ({time.time() - t0:.0f}s)", flush=True)
        ctx.z = ctx.q = ctx.el = None

    X, y = np.concatenate(Xtr), np.concatenate(ytr)
    print(f"training on {len(y):,} disputed cells, {y.mean():.3f} queued", flush=True)
    mdl = lgb.train(dict(objective="binary", learning_rate=0.05, num_leaves=args.leaves, verbose=-1,
                         feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1),
                    lgb.Dataset(X, y, feature_name=NAMES), args.rounds)
    imp = mdl.feature_importance("gain")
    print("top features: " + ", ".join(f"{NAMES[j]} {imp[j] / imp.sum():.2f}" for j in np.argsort(-imp)[:6]),
          flush=True)

    for s in C.SPLITS:                                   # one threshold per split keeps the queued count fixed
        sel = [o for o in official if o["split"] == s]
        p_all = np.concatenate([mdl.predict(o["X"][o["dis"] & ~o["is_on"][:, None, None]]) for o in sel])
        n_pos = sum(int(((o["bl"] >= BLEND_TAU) & o["dis"] & ~o["is_on"][:, None, None]).sum()) for o in sel)
        thr = 0.5 if args.free_rate else float(np.sort(p_all)[::-1][min(n_pos, len(p_all) - 1)])
        print(f"[{s}] disputed cells {len(p_all):,}, blend keeps {n_pos} queued -> threshold {thr:.3f}", flush=True)
        for o in sel:
            out = o["bl"] >= BLEND_TAU
            m = o["dis"] & ~o["is_on"][:, None, None]
            if m.any():
                out[m] = mdl.predict(o["X"][m]) >= thr
            ch = int((out[m] != (o["bl"] >= BLEND_TAU)[m]).sum())
            np.savez_compressed(args.out_dir / f"pred_{s}_{o['panel']}.npz", pred=out, window_id=o["ids"])
            print(f"  [{o['panel']} {s}] disputed {int(m.sum())}, changed {ch}", flush=True)


if __name__ == "__main__":
    main()
