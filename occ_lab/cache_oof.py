"""Out-of-fold material for the residual corrector: the train period's held-out tail.

The production members are fitted on train days before 2031-01-01 (`tfb.t1eval.FIT_END`; the LightGBM through
`ctx.fitd`, the neural nets through their `es` split, which only ever selects a checkpoint). That leaves
2031-01/02 as 59 days the members never fitted, earlier than both scored months - so a corrector fitted here is
out-of-fold for the members AND admissible for March and April alike, which a corrector fitted on a scored month
is not.

Writes occ_lab/cache/og_oof_<panel>.npz and stid_oof_<panel>.npz in the layout of cache_occ.py / cache_stid.py.

    python occ_lab/cache_oof.py --max-windows 4000
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
from t2_gstate import G_NAMES, gstate  # noqa: E402
from t2_m1_predict import feats_og, zero_prior  # noqa: E402
from t2_official_like_eval import ongoing_official  # noqa: E402
from tfb.t1eval import FIT_END  # noqa: E402
from train_occ import occ_channels  # noqa: E402
from train_stid import STID, predict as stid_predict  # noqa: E402

CACHE = LAB / "cache"
MODELS = LAB / "models"
GSTATE = PROJ / "work" / "models_gstate"
REMOTE = PROJ / "work" / ("models_remote" if (PROJ / "work" / "models_remote").exists() else "models")
BLOCK = 256


def load_net(p):
    sd = torch.load(p, map_location=C.DEV)
    tfm = "gru.weight_ih_l0" in sd
    cin = sd["gru.weight_ih_l0"].shape[1] if tfm else sd["inp.weight"].shape[1]
    net = (C.cnn.T2Tfm if tfm else C.cnn.T2Net)(cin, C.cnn.WIDTH)
    net.load_state_dict(sd)
    return net.to(C.DEV).eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-windows", type=int, default=4000)
    args = ap.parse_args()
    C.cnn.REC = 0.0
    CACHE.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    groups = {"cnn3": [load_net(REMOTE / f"t2_cnn_lov{s}.pt") for s in ("", "_s1", "_s2")],
              "tfm8": [load_net(REMOTE / f"t2_cnn_lov_tfm_s{s}.pt") for s in range(3, 11)],
              "occ_cnn": [load_net(MODELS / f"occ_cnn_s{s}.pt") for s in range(4)],
              "occ_tfm": [load_net(MODELS / f"occ_tfm_s{s}.pt") for s in range(4)]}
    stids = {}
    for s in range(4):
        d = torch.load(MODELS / f"stid_s{s}.pt", map_location=C.DEV)
        net = STID(d["cin"], d["n_links"], d["n_slots"]).to(C.DEV)
        net.load_state_dict(d["state"])
        stids[f"stid_s{s}"] = net.eval()
    m_og = lgb.Booster(model_file=str(GSTATE / "ongoing_g1.txt"))
    profs = np.load(MODELS / "occ_profiles.npz")
    t_start, loff = time.time(), 0
    for i, panel in enumerate(C.T2_PANELS):
        ctx = C.Ctx(panel, i)
        ctx.loff = 0
        pca = np.load(GSTATE / f"pca_{panel}.npz")
        ctx.pca = (pca["mu"], pca["vt"])
        stat = C.cnn.static_channels(ctx, C.load_ramps(panel, "train", ctx.z))
        lid = np.arange(ctx.S) + loff
        loff += ctx.S
        z = ctx.z
        og, _ = ongoing_official(ctx, z, None)
        keep = np.flatnonzero(z["dates"][og["d"]] >= FIT_END)          # the held-out tail only
        if len(keep) > args.max_windows:
            keep = np.sort(rng.choice(keep, args.max_windows, replace=False))
        og = {k: (v[keep] if isinstance(v, np.ndarray) and len(v) == len(og["d"]) else v) for k, v in og.items()}
        n = len(og["d"])
        dow = z["weekday"][og["d"]].astype(np.int64)
        grp = (dow >= 5).astype(int)
        P = {k: np.zeros(og["fut"].shape, np.float16) for k in list(groups) + list(stids) + ["gs1"]}
        for a in range(0, n, BLOCK):
            m = np.arange(a, min(a + BLOCK, n))
            x = C.make_x(ctx, stat, og, m, z)
            hi = og["t0"][m][:, None] + np.arange(-C.H, 0)
            xo = np.concatenate([x, occ_channels(z["obs_occ"][og["d"][m][:, None], hi], og["ok"][m],
                                                 profs[panel])], 1)
            for k, nets in groups.items():
                P[k][m] = np.mean([C.cnn.predict(net, xo if k.startswith("occ") else x) for net in nets],
                                  0).astype(np.float16)
            for k, net in stids.items():
                P[k][m] = stid_predict(net, xo, lid, og["t0"][m].astype(np.int64), dow[m]).astype(np.float16)
            key = np.stack([og["d"][m], grp[m]], 1)
            _, Xh, _ = feats_og(ctx, True, og["v"][m], og["qf"][m], og["ok"][m], og["t0"][m], grp[m],
                                zero_prior(ctx), key, dow[m])
            gf = np.broadcast_to(gstate(ctx, ctx.pca, og["v"][m], og["qf"][m], og["ok"][m])[:, None, None],
                                 Xh.shape[:3] + (len(G_NAMES),))
            Xo = np.concatenate([Xh, gf], -1)
            P["gs1"][m] = m_og.predict(Xo.reshape(-1, Xo.shape[-1])).reshape(Xh.shape[:3]).astype(np.float16)
        np.savez(CACHE / f"og_oof_{panel}.npz", y=og["fut"], e=og["efut"],
                 **{k: P[k] for k in ("gs1", "tfm8", "cnn3")},
                 **{k: P[k] for k in P if k.startswith("occ")})
        np.savez(CACHE / f"stid_oof_{panel}.npz", **{k: P[k] for k in stids})
        print(f"  {panel}: {n:,} held-out windows, {int(og['efut'].sum()):,} eligible cells "
              f"({time.time() - t_start:.0f}s)", flush=True)
        ctx.z = ctx.q = ctx.el = None


if __name__ == "__main__":
    main()
