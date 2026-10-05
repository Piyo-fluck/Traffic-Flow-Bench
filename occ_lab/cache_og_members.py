"""The three corrector features we do not have locally: occ_cnn, occ_tfm and the ongoing STID, on the local
ongoing windows. Without them `ongoing_pseudo_sweep.py` has to substitute cnn3/tfm8/ad9b, which distorts the
disagreement features the residual corrector reads (note 86).

Writes occ_lab/cache/ogmem_<split>_<panel>.npz.

    python occ_lab/cache_og_members.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ["T2CNN_ARCH"] = "cnn"
os.environ.setdefault("T2CNN_REC", "0.3")
LAB = Path(__file__).resolve().parent
PROJ = LAB.parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(PROJ / "tta_lab" / "tta"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import common as C  # noqa: E402
from train_occ import occ_channels  # noqa: E402
from train_stid import STID, predict as stid_predict  # noqa: E402

CACHE = LAB / "cache"
REMOTE = PROJ / "work" / "models_remote"


def load_net(p, arch):
    sd = torch.load(p, map_location=C.DEV)
    cin = sd["gru.weight_ih_l0"].shape[1] if arch == "tfm" else sd["inp.weight"].shape[1]
    net = (C.cnn.T2Tfm if arch == "tfm" else C.cnn.T2Net)(cin, C.cnn.WIDTH)
    net.load_state_dict(sd)
    return net.to(C.DEV).eval()


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    C.cnn.REC = 0.0
    occ_cnn = [load_net(REMOTE / f"occ_cnn_s{s}.pt", "cnn") for s in range(4)]
    occ_tfm = [load_net(REMOTE / f"occ_tfm_s{s}.pt", "tfm") for s in range(4)]
    stids = []
    for s in range(4):
        d = torch.load(LAB / "models" / f"stid_s{s}.pt", map_location=C.DEV)
        net = STID(d["cin"], d["n_links"], d["n_slots"]).to(C.DEV)
        net.load_state_dict(d["state"])
        stids.append(net.eval())
    profs = np.load(REMOTE / "occ_profiles.npz")
    t0_, loff = time.time(), 0
    for i, panel in enumerate(C.T2_PANELS):
        ctx, stat, W = C.windows(i, panel)
        lid = np.arange(ctx.S) + loff
        loff += ctx.S
        for s in C.SPLITS:
            og, z = W[s]["og"], W[s]["z"]
            n = len(og["d"])
            if not n:
                continue
            x = C.make_x(ctx, stat, og, np.arange(n), z)
            hi = og["t0"][:, None] + np.arange(-C.H, 0)
            xo = np.concatenate([x, occ_channels(z["obs_occ"][og["d"][:, None], hi], og["ok"], profs[panel])], 1)
            dow = z["weekday"][og["d"]].astype(np.int64)
            out = dict(
                occ_cnn=np.mean([C.cnn.predict(net, xo) for net in occ_cnn], 0),
                occ_tfm=np.mean([C.cnn.predict(net, xo) for net in occ_tfm], 0),
                stid=np.mean([stid_predict(net, xo, lid, og["t0"].astype(np.int64), dow) for net in stids], 0),
            )
            np.savez(CACHE / f"ogmem_{s}_{panel}.npz", **{k: v.astype(np.float32) for k, v in out.items()})
        ctx.z = None
        print(f"  {panel} ({time.time() - t0_:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
