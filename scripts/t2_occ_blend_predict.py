"""Ongoing decisions for the official windows with the occupancy-fed members added (note 84).

    ongoing p = 0.20 gs1 + (0.60 - w_tfm) tfm8 + w_tfm occ_tfm + (0.20 - w_cnn) cnn3 + w_cnn occ_cnn,  then p >= tau

The occupancy models take three extra input channels (occupancy x10, occupancy over the link's weekday x tod mean
clipped to [0, 5], its change across the history x10), built here from the `occupancy` column of the official
window history with the window's own eligibility mask - the same treatment speed and flow get.

Writes ongoing_<split>.csv (window_id, timestamp, link_id, queue_pred) for the ongoing windows only, so the onset
rows of the candidate it patches stay untouched.

    .venv/Scripts/python.exe scripts/t2_occ_blend_predict.py --out-dir work/sub_t2occ \
        --cnn <cnn x3> --tfm <tfm s3..s10> --occ-cnn <occ_cnn s0..s3> --occ-tfm <occ_tfm s0..s3> \
        --profiles work/models_remote/occ_profiles.npz --gs-dir work/sub_t2gs --w-cnn 0.05 --w-tfm 0.05
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
sys.path.insert(0, str(ROOT / "occ_lab"))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "scripts"))
import common as C  # noqa: E402
from t2_m1_predict import hist_from_release  # noqa: E402
from t2_predict_rules import slot  # noqa: E402
from tfb.data import RELEASE  # noqa: E402
from train_occ import occ_channels  # noqa: E402

C.cnn.REC = 0.0


def load(p, arch):
    sd = torch.load(p, map_location=C.DEV)
    cin = sd["gru.weight_ih_l0"].shape[1] if arch == "tfm" else sd["inp.weight"].shape[1]
    net = (C.cnn.T2Tfm if arch == "tfm" else C.cnn.T2Net)(cin, C.cnn.WIDTH)
    net.load_state_dict(sd)
    return net.to(C.DEV).eval()


def occ_from_release(ctx, H_df, Wi, t0):
    """(N, H, S) occupancy of the official histories, NaN where the release has no value."""
    s_of = {l: i for i, l in enumerate(ctx.z["link_ids"])}
    o = np.full((len(Wi), C.H, ctx.S), np.nan, np.float32)
    w_of = {w: i for i, w in enumerate(Wi.window_id)}
    wi = H_df.window_id.map(w_of).to_numpy()
    hi = slot(H_df.timestamp) - t0[wi] + C.H
    si = H_df.link_id.astype(str).map(s_of).to_numpy()
    good = (hi >= 0) & (hi < C.H)
    o[wi[good], hi[good], si[good]] = H_df.occupancy.to_numpy(float)[good]
    return o


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--cnn", type=Path, nargs="+", required=True)
    ap.add_argument("--tfm", type=Path, nargs="+", required=True)
    ap.add_argument("--occ-cnn", type=Path, nargs="+", required=True)
    ap.add_argument("--occ-tfm", type=Path, nargs="+", required=True)
    ap.add_argument("--profiles", type=Path, required=True)
    ap.add_argument("--gs-dir", type=Path, required=True)
    ap.add_argument("--w-cnn", type=float, default=0.05, help="weight of occ_cnn, taken out of the 0.20 cnn3 slot")
    ap.add_argument("--w-tfm", type=float, default=0.05, help="weight of occ_tfm, taken out of the 0.60 tfm8 slot")
    ap.add_argument("--tau", type=float, default=0.50)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    nets = {k: [load(p, "tfm" if "tfm" in k else "cnn") for p in v] for k, v in
            (("cnn", args.cnn), ("tfm", args.tfm), ("occ_cnn", args.occ_cnn), ("occ_tfm", args.occ_tfm))}
    profs = np.load(args.profiles)
    print(f"gs1 0.20 | tfm8 {0.60 - args.w_tfm:.2f} + occ_tfm {args.w_tfm:.2f} | "
          f"cnn3 {0.20 - args.w_cnn:.2f} + occ_cnn {args.w_cnn:.2f}, tau {args.tau}", flush=True)
    for split in C.SPLITS:
        frames, changed, total = [], 0, 0
        for i, panel in enumerate(C.T2_PANELS):
            ctx = C.Ctx(panel, i)
            ctx.loff = 0
            stat = C.cnn.static_channels(ctx, C.load_ramps(panel, "train", ctx.z))
            base = RELEASE / "task2" / panel / split
            Wi = pd.read_csv(base / "window_index.csv")
            Hh = pd.read_parquet(base / "window_history.parquet")
            Q = pd.read_csv(base / "sample_submission_queue.csv", usecols=["window_id", "timestamp", "link_id"])
            v, qf, ok, t0, grp = hist_from_release(ctx, Hh, Wi)
            x = C.cnn.make_x(ctx, stat, v, qf, ok, t0, grp.astype(bool))
            xo = np.concatenate([x, occ_channels(occ_from_release(ctx, Hh, Wi, t0), ok, profs[panel])], 1)
            p = {k: np.mean([C.cnn.predict(n, xo if k.startswith("occ") else x) for n in ns], 0)
                 for k, ns in nets.items()}
            gs = np.load(args.gs_dir / f"prob_{split}_{panel}.npz")
            assert (gs["window_id"] == Wi.window_id.to_numpy().astype(str)).all(), "gs1 window order differs"
            bl = 0.2 * gs["prob"].astype(np.float32) + 0.6 * p["tfm"] + 0.2 * p["cnn"]
            mix = (0.2 * gs["prob"].astype(np.float32)
                   + (0.6 - args.w_tfm) * p["tfm"] + args.w_tfm * p["occ_tfm"]
                   + (0.2 - args.w_cnn) * p["cnn"] + args.w_cnn * p["occ_cnn"])
            og = (Wi.condition != "queue_onset").to_numpy()
            w_of = {w: k for k, w in enumerate(Wi.window_id)}
            s_of = {l: k for k, l in enumerate(ctx.z["link_ids"])}
            wi = Q.window_id.map(w_of).to_numpy()
            hi = np.asarray(slot(Q.timestamp)) - t0[wi] - 1
            si = Q.link_id.astype(str).map(s_of).to_numpy().astype(int)
            keep = og[wi]
            Q = Q[keep].copy()
            Q["queue_pred"] = (mix[wi[keep], hi[keep], si[keep]] >= args.tau).astype(int)
            ch = int(((mix[og] >= args.tau) != (bl[og] >= args.tau)).sum())
            changed += ch
            total += int(keep.sum())
            frames.append(Q)
            print(f"[{split} {panel}] ongoing rows {int(keep.sum()):,}, window cells changed vs the 3-member "
                  f"blend {ch}", flush=True)
            ctx.z = ctx.q = ctx.el = None
        out = pd.concat(frames, ignore_index=True)
        out.to_csv(args.out_dir / f"ongoing_{split}.csv", index=False)
        print(f"wrote {args.out_dir / f'ongoing_{split}.csv'}: {len(out):,} rows, share 1 = "
              f"{out.queue_pred.mean():.4f}, window cells changed {changed}", flush=True)


if __name__ == "__main__":
    main()
