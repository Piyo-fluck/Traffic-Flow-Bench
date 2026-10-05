"""Put the pretrained onset model into the production blend and see what the blend does.

Standalone on the scored months' official-like onset windows the pretrained model is worth +0.0563 over the same
architecture trained from scratch (`occ_lab/eval_onset_hazard.py`). That number cannot be spent directly: the
model occupies the 0.30 STID slot of

    p(T+30) = 0.35 H + 0.175 ad9b + 0.175 pseudo + 0.30 STID

and the blend already scores far above any single member. What matters is the blend.

Compared here, all with the production expected-IoU ratio rule at T+30 and released inputs:

    production          the four members as shipped (STID = onset_stid_s0..s3, 4 seeds)
    swap                the STID slot replaced by the pretrained model
    both                the STID slot replaced by the mean of the two
    a weight sweep      the STID slot's weight moved, taken out of H

    python occ_lab/onset_blend_hazard.py --haz occ_lab/models/onset_hazard_earliest_ftall_s0.pt
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("T2CNN_ARCH", "cnn")
os.environ.setdefault("T2CNN_REC", "0.3")
LAB = Path(__file__).resolve().parent
PROJ = LAB.parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(PROJ / "tta_lab" / "tta"))
sys.path.insert(0, str(PROJ / "analysis"))
sys.path.insert(0, str(PROJ / "scripts"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import common as C  # noqa: E402
from t2_expected_iou import select  # noqa: E402
from train_occ import occ_channels  # noqa: E402
from train_onset_stid import OnsetSTID, predict as stid_predict  # noqa: E402

CACHE = LAB / "cache"
MODELS = LAB / "models"
WB = (0.35, 0.175, 0.175, 0.30)


def iou_per_window(P, y, e):
    out = []
    for j in range(len(P)):
        m = e[j]
        if not m.any() or not y[j][m].any():
            continue
        pr = select(P[j][m], "ratio", 1.0)
        t = y[j][m]
        out.append(int((pr & t).sum()) / max(int((pr | t).sum()), 1))
    return float(np.mean(out)) if out else np.nan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--haz", type=Path, required=True)
    ap.add_argument("--tag", default="s20")
    ap.add_argument("--w-st", type=float, nargs="+", default=[0.20, 0.30, 0.40, 0.50])
    args = ap.parse_args()
    profs = np.load(MODELS / "occ_profiles.npz")
    d_ = torch.load(args.haz, map_location=C.DEV)
    haz = OnsetSTID(d_["cin"], d_["n_links"], d_["n_slots"], ident=d_.get("ident", "id")).to(C.DEV)
    haz.load_state_dict(d_["state"])
    haz.eval()
    prod = []
    for s in range(4):
        p = MODELS / f"onset_stid_s{s}.pt"
        if not p.exists():
            continue
        q = torch.load(p, map_location=C.DEV)
        n = OnsetSTID(q["cin"], q["n_links"], q["n_slots"], ident=q.get("ident", "id")).to(C.DEV)
        n.load_state_dict(q["state"])
        prod.append(n.eval())
    print(f"production STID seeds {len(prod)}, hazard {args.haz.name}", flush=True)

    fam, loff, t0_ = {}, 0, time.time()
    for i, panel in enumerate(C.T2_PANELS):
        ctx, stat, Wd = C.windows(i, panel)
        lid = np.arange(ctx.S) + loff
        loff += ctx.S
        for s in C.SPLITS:
            on, z, ref = Wd[s]["on"], Wd[s]["z"], Wd[s]["ref"]
            if not len(on["d"]):
                continue
            x = C.cnn.make_x(ctx, stat, on["v"], on["qf"], on["ok"], on["t0"], z["weekday"][on["d"]] >= 5)
            hi = on["t0"][:, None] + np.arange(-C.H, 0)
            xo = np.concatenate([x, occ_channels(z["obs_occ"][on["d"][:, None], hi], on["ok"],
                                                 profs[panel])], 1)
            tod = on["t0"].astype(np.int64)
            dow = z["weekday"][on["d"]].astype(np.int64)
            P_prod = np.mean([stid_predict(n, xo, lid, tod, dow) for n in prod], 0) if prod else None
            P_haz = stid_predict(haz, xo, lid, tod, dow)
            f = CACHE / f"pseudo_{args.tag}_{s}_{panel}.npz"
            ps = np.load(f)["on"].astype(np.float32) if f.exists() else ref["on_ad9b"]
            base3 = WB[0] * ref["on_H"] + WB[1] * ref["on_ad9b"] + WB[2] * ps
            y, e = on["fut"][:, C.F - 1], on["efut"][:, C.F - 1]
            for nm, st in (("production", P_prod), ("swap", P_haz),
                           ("both", 0.5 * P_prod + 0.5 * P_haz)):
                fam.setdefault((s, nm), {})[panel] = iou_per_window(base3 + WB[3] * st, y, e)
            for w in args.w_st:
                sc = (WB[0] + WB[3] - w) * ref["on_H"] + WB[1] * ref["on_ad9b"] + WB[2] * ps
                fam.setdefault((s, f"swap w_st {w:g}"), {})[panel] = iou_per_window(sc + w * P_haz, y, e)
                fam.setdefault((s, f"both w_st {w:g}"), {})[panel] = iou_per_window(
                    sc + w * (0.5 * P_prod + 0.5 * P_haz), y, e)
            fam.setdefault((s, "STID alone"), {})[panel] = iou_per_window(P_prod, y, e)
            fam.setdefault((s, "hazard alone"), {})[panel] = iou_per_window(P_haz, y, e)
        ctx.z = ctx.q = ctx.el = None
        print(f"  {panel} ({time.time() - t0_:.0f}s)", flush=True)

    names = (["STID alone", "hazard alone", "production", "swap", "both"]
             + [f"{a} w_st {w:g}" for w in args.w_st for a in ("swap", "both")])
    print(f"\nonset family IoU, official-like windows of the scored months, ratio rule c 1.0\n")
    print(f"{'setting':<18}" + "".join(f"{s:>13}" for s in C.SPLITS) + f"{'mean':>10}{'vs prod':>9}")
    ref_m = None
    for nm in names:
        if (C.SPLITS[0], nm) not in fam:
            continue
        v = [C.family_mean(fam[(s, nm)]) for s in C.SPLITS]
        m = float(np.mean(v))
        if nm == "production":
            ref_m = m
        d = f"{m - ref_m:>+9.4f}" if ref_m is not None else " " * 9
        print(f"{nm:<18}" + "".join(f"{x:>13.4f}" for x in v) + f"{m:>10.4f}{d}")


if __name__ == "__main__":
    main()
