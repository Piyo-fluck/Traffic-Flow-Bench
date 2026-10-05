"""Two questions about the pretrained onset model, and nothing else.

1. Is the gain on the held-out train days real? The reported half is only about 180 windows and the official
   config puts the per-window IoU standard deviation at 0.3752, so the unpaired error bar is wide. The windows
   are the same for every checkpoint, so the paired difference is the quantity with a usable error bar.

2. Does it hold on the SCORED months? Note 58's rule is that onset-side gains fail outright on the leaderboard,
   so a gain measured on the train tail means little until it reappears on the official-like onset windows of
   validation and private (`first_onsets` with the official blanks, the surface of note 51).

Inputs on the scored months are the RELEASED layer, which is what test time has - not the true layer the models
were trained on.

    python occ_lab/eval_onset_hazard.py --ckpts base=... ftall=... ftq=...
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("T2CNN_ARCH", "cnn")
LAB = Path(__file__).resolve().parent
PROJ = LAB.parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(PROJ / "analysis"))
sys.path.insert(0, str(PROJ / "scripts"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import t2_cnn as cnn  # noqa: E402
from t2_eda02 import T2_PANELS  # noqa: E402
from t2_eda03 import F, H  # noqa: E402
from t2_expected_iou import select  # noqa: E402
from t2_first_onset_eval import first_onsets, official_blank  # noqa: E402
from t2_m1_models import Ctx  # noqa: E402
from tfb.data import load  # noqa: E402
from tfb.physics import load_ramps  # noqa: E402
from tfb.t1eval import FIT_END, family_mean  # noqa: E402
from train_occ import occ_channels  # noqa: E402
from train_onset_hazard import index_windows  # noqa: E402
from train_onset_stid import OnsetSTID  # noqa: E402

MODELS = LAB / "models"


def predict(net, x, lid, t0, dow, bs=64):
    out = []
    with torch.no_grad():
        for a in range(0, len(x), bs):
            sl = slice(a, a + bs)
            n = len(x[sl])
            tt = lambda z, d: torch.from_numpy(np.ascontiguousarray(z)).to(cnn.DEV, d)  # noqa: E731
            lg = net(tt(x[sl], torch.float32), tt(np.broadcast_to(lid, (n, len(lid))).copy(), torch.long),
                     tt(t0[sl], torch.long), tt(dow[sl], torch.long))[:, 0]
            out.append(torch.sigmoid(lg).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(lid)), np.float32)


def iou_per_window(P, y, e):
    out = []
    for j in range(len(P)):
        m = e[j]
        if not m.any() or not y[j][m].any():
            out.append(np.nan)
            continue
        pr = select(P[j][m], "ratio", 1.0)
        t = y[j][m]
        out.append(int((pr & t).sum()) / max(int((pr | t).sum()), 1))
    return np.asarray(out, float)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True, help="name=path")
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--pair", nargs=2, default=["base", "ftq"])
    args = ap.parse_args()
    names, paths = [], []
    for kv in args.ckpts:
        k, v = kv.split("=", 1)
        names.append(k)
        paths.append(Path(v))
    nets = {}
    n_links = n_slots = 0
    for k, p in zip(names, paths):
        d = torch.load(p, map_location=cnn.DEV)
        n_links, n_slots = d["n_links"], d["n_slots"]
        net = OnsetSTID(d["cin"], n_links, n_slots, ident=d.get("ident", "id")).to(cnn.DEV)
        net.load_state_dict(d["state"])
        nets[k] = net.eval()
    print(f"{len(nets)} checkpoints, {n_links} links", flush=True)

    tail = {k: [] for k in names}
    scored = {k: {} for k in names}
    loff = 0
    t0_ = time.time()
    for i, panel in enumerate(T2_PANELS):
        ctx = Ctx(panel, i)
        ctx.loff = loff
        lid = np.arange(ctx.S) + loff
        loff += ctx.S
        stat = cnn.static_channels(ctx, load_ramps(panel, "train", ctx.z))
        prof = np.load(MODELS / "occ_profiles.npz")[panel]
        z = ctx.z

        # 1. the untouched half of the held-out train days
        de, te = index_windows(ctx, "earliest")
        keep = (z["dates"][de] >= FIT_END) & (de % 2 == 0)
        d, t0 = de[keep], te[keep]
        if len(d):
            hi = t0[:, None] + np.arange(-H, 0)
            ok = ctx.el[d[:, None], hi]
            x = cnn.make_x(ctx, stat, z["true_speed"][d[:, None], hi], z["true_flow"][d[:, None], hi], ok,
                           t0, z["weekday"][d] >= 5)
            x = np.concatenate([x, occ_channels(z["true_occ"][d[:, None], hi], ok, prof)], 1)
            y, e = ctx.q[d, t0 + F], ctx.el[d, t0 + F]
            for k in names:
                tail[k].append(iou_per_window(predict(nets[k], x, lid, t0.astype(np.int64),
                                                      z["weekday"][d].astype(np.int64)), y, e))

        # 2. the scored months, official-like onset windows, released inputs
        for split in ("validation", "private"):
            zs = load(panel, split)
            w = first_onsets(ctx.vf, zs, blank=official_blank(panel, split, zs))
            if not len(w["d"]):
                continue
            xs = cnn.make_x(ctx, stat, w["v"], w["qf"], w["ok"], w["t0"], zs["weekday"][w["d"]] >= 5)
            hi = w["t0"][:, None] + np.arange(-H, 0)
            xs = np.concatenate([xs, occ_channels(zs["obs_occ"][w["d"][:, None], hi], w["ok"], prof)], 1)
            yy = w["fut"][:, F - 1]
            ee = w["efut"][:, F - 1]
            for k in names:
                v = iou_per_window(predict(nets[k], xs, lid, w["t0"].astype(np.int64),
                                           zs["weekday"][w["d"]].astype(np.int64)), yy, ee)
                scored[k].setdefault(split, {})[panel] = float(np.nanmean(v))
        print(f"  {panel} ({time.time() - t0_:.0f}s)", flush=True)
        ctx.z = ctx.q = ctx.el = None

    T = {k: np.concatenate(v) for k, v in tail.items()}
    good = ~np.isnan(next(iter(T.values())))
    for k in T:
        good &= ~np.isnan(T[k])
    n = int(good.sum())
    print(f"\n1. held-out train days, untouched half: {n} windows\n")
    print(f"{'checkpoint':<10}{'onset IoU':>11}")
    for k in names:
        print(f"{k:<10}{T[k][good].mean():>11.4f}")
    a, b = args.pair
    if a in T and b in T:
        d_ = T[b][good] - T[a][good]
        rng = np.random.default_rng(0)
        bs = np.array([d_[rng.integers(n, size=n)].mean() for _ in range(args.boot)])
        lo, hi_ = np.percentile(bs, [2.5, 97.5])
        print(f"\n  paired {b} - {a}: {d_.mean():+.4f}  95% CI [{lo:+.4f}, {hi_:+.4f}]  "
              f"P(>0) {float((bs > 0).mean()):.3f}")
        print(f"  windows better {int((d_ > 0).sum())}, worse {int((d_ < 0).sum())}, "
              f"tied {int((d_ == 0).sum())}")

    print(f"\n2. scored months, official-like onset windows, released inputs\n")
    print(f"{'checkpoint':<10}{'validation':>12}{'private':>11}{'mean':>10}")
    for k in names:
        v = [family_mean(scored[k][s]) for s in ("validation", "private") if s in scored[k]]
        if v:
            print(f"{k:<10}" + "".join(f"{x:>12.4f}" for x in v) + f"{float(np.mean(v)):>10.4f}")


if __name__ == "__main__":
    main()
