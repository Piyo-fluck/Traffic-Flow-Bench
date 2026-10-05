"""Coverage environments for the residual corrector (the user's step 1 after note 86).

The corrector of candidate AI was fitted on train-tail windows whose histories come from the MASKED layer, about
52% observed, and it is applied to official windows that are about 76% observed (note 76). It reads |p0 - 0.5| and
member disagreement, which move with coverage, so the mismatch hits it directly.

The train split also carries the TRUE layer, whose eligible cells are observed at about the official level. So the
same held-out windows (2031-01/02, the ones `cache_oof.py` uses, same selection) are re-inferred under several
history availabilities:

    obs     the masked layer as released (what AI was fitted on)
    full    the true layer on every eligible cell (about the official 0.76)
    cXX     between the two: for a share lambda of the (window, link) pairs the link keeps its released mask,
            the rest see the true layer; lambda is set per panel so the mean coverage is XX%. Whole links switch,
            so the gaps keep the temporal structure of the real masks instead of becoming iid holes.

Labels stay the masked-layer queues on eligible cells, exactly as in every other local evaluation, so the
environments differ in their inputs only. Coverage here is the share of the 12 x S history cells that are
eligible and observed (note 76's definition).

Writes occ_lab/cache/cov_<env>_<panel>.npz: every member's probabilities, per-window coverage, the day index and
date of each window; the labels are in cov_labels_<panel>.npz.

    CUDA_VISIBLE_DEVICES=0 python occ_lab/cache_cov.py --env c50
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ["T2CNN_ARCH"] = "cnn"
LAB = Path(__file__).resolve().parent
PROJ = LAB.parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(PROJ / "tta_lab" / "tta"))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import common as C  # noqa: E402
from cache_oof import BLOCK, GSTATE, MODELS, REMOTE, load_net  # noqa: E402
from t2_gstate import G_NAMES, gstate  # noqa: E402
from t2_m1_predict import feats_og, zero_prior  # noqa: E402
from t2_official_like_eval import ongoing_official  # noqa: E402
from tfb.t1eval import FIT_END  # noqa: E402
from train_occ import occ_channels  # noqa: E402
from train_stid import STID, predict as stid_predict  # noqa: E402

CACHE = LAB / "cache"
ENVS = {"obs": None, "c50": 0.50, "c60": 0.60, "c70": 0.70, "full": 1.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, choices=list(ENVS))
    ap.add_argument("--max-windows", type=int, default=4000)
    args = ap.parse_args()
    C.cnn.REC = 0.0
    CACHE.mkdir(parents=True, exist_ok=True)
    groups = {"cnn3": [load_net(REMOTE / f"t2_cnn_lov{s}.pt") for s in ("", "_s1", "_s2")],
              "tfm8": [load_net(REMOTE / f"t2_cnn_lov_tfm_s{s}.pt") for s in range(3, 11)],
              "occ_cnn": [load_net(MODELS / f"occ_cnn_s{s}.pt") for s in range(4)],
              "occ_tfm": [load_net(MODELS / f"occ_tfm_s{s}.pt") for s in range(4)]}
    stids = []
    for s in range(4):
        d = torch.load(MODELS / f"stid_s{s}.pt", map_location=C.DEV)
        net = STID(d["cin"], d["n_links"], d["n_slots"]).to(C.DEV)
        net.load_state_dict(d["state"])
        stids.append(net.eval())
    m_og = lgb.Booster(model_file=str(GSTATE / "ongoing_g1.txt"))
    profs = np.load(MODELS / "occ_profiles.npz")
    t_start, loff = time.time(), 0
    for i, panel in enumerate(C.T2_PANELS):
        rng = np.random.default_rng(0)                  # the exact window selection of cache_oof.py
        ctx = C.Ctx(panel, i)
        ctx.loff = 0
        pca = np.load(GSTATE / f"pca_{panel}.npz")
        ctx.pca = (pca["mu"], pca["vt"])
        stat = C.cnn.static_channels(ctx, C.load_ramps(panel, "train", ctx.z))
        lid = np.arange(ctx.S) + loff
        loff += ctx.S
        z = ctx.z
        og, _ = ongoing_official(ctx, z, None)
        keep = np.flatnonzero(z["dates"][og["d"]] >= FIT_END)
        if len(keep) > args.max_windows:
            keep = np.sort(rng.choice(keep, args.max_windows, replace=False))
        og = {k: (v[keep] if isinstance(v, np.ndarray) and len(v) == len(og["d"]) else v) for k, v in og.items()}
        n = len(og["d"])
        hi = og["t0"][:, None] + np.arange(-C.H, 0)
        dd = og["d"][:, None]
        if args.env == "obs":
            v, qf, ok = og["v"], og["qf"], og["ok"]
            occ = z["obs_occ"][dd, hi]
            lam = None
        else:
            el = ctx.el[dd, hi]                                    # eligible and observed in the true layer
            obs_ok = og["ok"] & el
            c_el, c_obs = el.mean(), obs_ok.mean()
            target = ENVS[args.env]
            lam = 0.0 if target >= c_el else float(np.clip((c_el - target) / max(c_el - c_obs, 1e-6), 0, 1))
            sel = np.random.default_rng(1000 + i).random((n, ctx.S)) < lam      # links that keep the real mask
            ok = np.where(sel[:, None, :], obs_ok, el)
            v, qf = z["true_speed"][dd, hi], z["true_flow"][dd, hi]
            occ = z["true_occ"][dd, hi]
        cov = ok.mean((1, 2)).astype(np.float32)
        w = dict(v=v, qf=qf, ok=ok, t0=og["t0"], d=og["d"])
        dow = z["weekday"][og["d"]].astype(np.int64)
        grp = (dow >= 5).astype(int)
        P = {k: np.zeros(og["fut"].shape, np.float16) for k in list(groups) + ["stid", "gs1"]}
        for a in range(0, n, BLOCK):
            m = np.arange(a, min(a + BLOCK, n))
            x = C.make_x(ctx, stat, w, m, z)
            xo = np.concatenate([x, occ_channels(occ[m], ok[m], profs[panel])], 1)
            for k, nets in groups.items():
                P[k][m] = np.mean([C.cnn.predict(net, xo if k.startswith("occ") else x) for net in nets],
                                  0).astype(np.float16)
            P["stid"][m] = np.mean([stid_predict(net, xo, lid, og["t0"][m].astype(np.int64), dow[m])
                                    for net in stids], 0).astype(np.float16)
            key = np.stack([og["d"][m], grp[m]], 1)
            _, Xh, _ = feats_og(ctx, True, v[m], qf[m], ok[m], og["t0"][m], grp[m], zero_prior(ctx), key, dow[m])
            gf = np.broadcast_to(gstate(ctx, ctx.pca, v[m], qf[m], ok[m])[:, None, None],
                                 Xh.shape[:3] + (len(G_NAMES),))
            Xo = np.concatenate([Xh, gf], -1)
            P["gs1"][m] = m_og.predict(Xo.reshape(-1, Xo.shape[-1])).reshape(Xh.shape[:3]).astype(np.float16)
        np.savez(CACHE / f"cov_{args.env}_{panel}.npz", cov=cov, **P)
        if args.env == "obs":
            np.savez(CACHE / f"cov_labels_{panel}.npz", y=og["fut"], e=og["efut"], d=og["d"], t0=og["t0"],
                     date=z["dates"][og["d"]].astype(str))
        print(f"  {args.env} {panel}: {n:,} windows, coverage {cov.mean():.3f}"
              + ("" if lam is None else f" (lambda {lam:.2f})") + f" ({time.time() - t_start:.0f}s)", flush=True)
        ctx.z = ctx.q = ctx.el = None


if __name__ == "__main__":
    main()
