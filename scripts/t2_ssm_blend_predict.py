"""Ongoing decisions with the spatial-SSM as a fourth member, for the official windows (note 83).

ongoing p = (1 - w) * (0.20 gs1 + 0.60 tfm8 + 0.20 cnn3) + w * mean(SSM seeds), thresholded at tau.
Writes pred_<split>_<panel>.npz for scripts/t2_cnn_blend_predict.py --og-pred-dir; onset rows stay False.

    .venv/Scripts/python.exe scripts/t2_ssm_blend_predict.py --out-dir work/sub_t2ssm --w 0.10 \
        --cnn <cnn x3> --tfm <tfm s3..s10> --ssm <ssm seeds> --gs-dir work/sub_t2gs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tta_lab" / "tta"))
sys.path.insert(0, str(ROOT / "dcrnn_lab"))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "scripts"))
import common as C  # noqa: E402
from spatial_ssm import SpatialSSM  # noqa: E402
from t2_m1_predict import hist_from_release  # noqa: E402
from tfb.data import RELEASE  # noqa: E402

C.cnn.REC = 0.0


def load(p, arch):
    sd = torch.load(p, map_location=C.DEV)
    if arch == "ssm":
        net = SpatialSSM(sd["embed.weight"].shape[1] // C.H, C.cnn.WIDTH,
                         sum(1 for k in sd if k.endswith(".to_a.weight")))
    else:
        cin = sd["gru.weight_ih_l0"].shape[1] if arch == "tfm" else sd["inp.weight"].shape[1]
        net = (C.cnn.T2Tfm if arch == "tfm" else C.cnn.T2Net)(cin, C.cnn.WIDTH)
    net.load_state_dict(sd)
    return net.to(C.DEV).eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--cnn", type=Path, nargs="+", required=True)
    ap.add_argument("--tfm", type=Path, nargs="+", required=True)
    ap.add_argument("--ssm", type=Path, nargs="+", required=True)
    ap.add_argument("--gs-dir", type=Path, required=True)
    ap.add_argument("--w", type=float, default=0.10)
    ap.add_argument("--tau", type=float, default=0.50)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    nets = {k: [load(p, k) for p in v] for k, v in (("cnn", args.cnn), ("tfm", args.tfm), ("ssm", args.ssm))}
    for i, panel in enumerate(C.T2_PANELS):
        ctx, stat, W = C.windows(i, panel)
        for s in C.SPLITS:
            base = RELEASE / "task2" / panel / s
            Wi = pd.read_csv(base / "window_index.csv")
            Hh = pd.read_parquet(base / "window_history.parquet")
            v, qf, ok, t0, grp = hist_from_release(ctx, Hh, Wi)
            x = C.cnn.make_x(ctx, stat, v, qf, ok, t0, grp.astype(bool))
            p = {k: np.mean([C.cnn.predict(n, x) for n in ns], 0) for k, ns in nets.items()}
            gs = np.load(args.gs_dir / f"prob_{s}_{panel}.npz")
            assert (gs["window_id"] == Wi.window_id.to_numpy().astype(str)).all()
            bl = 0.2 * gs["prob"].astype(np.float32) + 0.6 * p["tfm"] + 0.2 * p["cnn"]
            mix = (1 - args.w) * bl + args.w * p["ssm"]
            np.savez_compressed(args.out_dir / f"pred_{s}_{panel}.npz", pred=mix >= args.tau,
                                window_id=Wi.window_id.to_numpy().astype(str))
            ch = int(((mix >= args.tau) != (bl >= args.tau)).sum())
            print(f"[{panel} {s}] cells changed vs the 3-member blend: {ch}", flush=True)


if __name__ == "__main__":
    main()
