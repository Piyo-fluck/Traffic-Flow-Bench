"""Cross-panel pooled causal adaptation: one network per scored run, adapted on the pseudo windows of ALL its panels.

The official windows sit early in the month (onset median day 3, ongoing median day 4; note 133), so a window has
0-4 earlier days of its run and the per-panel adaptation (note 106) sees very little data. All panels of a split
are one simulation run (note 17) and the corridors share their demand (+0.585, note 116). If the run's shift is
common to the corridors, pooling the pseudo windows of every panel gives the adaptation about 8x the data.

    none       the checkpoint as it is
    perpanel   note 106 as shipped: per panel and split, per day, sup+rec on that panel's pseudo windows, 30 steps
    pooled_N   per split, ONE network: per day, N steps of sup+rec, each batch drawn from a random panel's pseudo
               windows (days before d, and day d before the EARLIEST first origin of any panel that day, so every
               window predicted on day d sees only data at or before its own origin); then every panel's day-d
               windows are predicted

Scored by the ongoing and onset IoU of the single network, all windows and the early ones (day index <= 4, where
the official windows are).

    python occ_lab/pseudo_adapt_pooled.py --ckpt work/models/t2_cnn_lov_w192_rec_ssl30k_s20.pt --tag s20
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ["T2CNN_ARCH"] = "cnn"
os.environ.setdefault("T2CNN_REC", "0.3")
os.environ.setdefault("T2CNN_LOVASZ", "0.5")
os.environ.setdefault("T2CNN_WIDTH", "192")
LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta"))
import numpy as np  # noqa: E402

import common as C  # noqa: E402
import pseudo_adapt as PA  # noqa: E402

CACHE = LAB / "cache"


def predict_day(net, P, d, out):
    for pn, p in P.items():
        on, og, z = p["on"], p["og"], p["z"]
        io = np.flatnonzero(on["d"] == d)
        if len(io):
            out[pn][0][io] = C.cnn.predict(net, C.make_x(p["ctx"], p["stat"], on, io, z))[:, C.F - 1]
        ig = np.flatnonzero(og["d"] == d)
        if len(ig):
            out[pn][1][ig] = C.cnn.predict(net, C.make_x(p["ctx"], p["stat"], og, ig, z))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--modes", nargs="*", default=["none", "perpanel", "pooled_60", "pooled_240"])
    args = ap.parse_args()
    t_start = time.time()
    res = {}
    for s in C.SPLITS:
        P = {}
        for i, panel in enumerate(C.T2_PANELS):
            ctx, stat, W = C.windows(i, panel)
            w = W[s]
            if not (len(w["og"]["d"]) or len(w["on"]["d"])):
                continue
            P[panel] = dict(ctx=ctx, stat=stat, z=w["z"], on=w["on"], og=w["og"], pool=PA.pseudo_pool(ctx, w["z"]),
                            rng=np.random.default_rng(7 + i))
        dates = {pn: tuple(np.asarray(p["z"]["dates"]).astype(str)) for pn, p in P.items()}
        assert len(set(dates.values())) == 1, "panels of a split must share their days"
        days = sorted({int(d) for p in P.values() for d in np.concatenate([p["on"]["d"], p["og"]["d"]])})
        first = {}
        for pn, p in P.items():
            for d in days:
                f = min(p["on"]["t0"][p["on"]["d"] == d].min(initial=10 ** 6),
                        p["og"]["t0"][p["og"]["d"] == d].min(initial=10 ** 6))
                first[(pn, d)] = int(f)
        print(f"[{s}] {len(P)} panels, {len(days)} days ({time.time() - t_start:.0f}s)", flush=True)
        for mode in args.modes:
            out = {pn: (np.zeros((len(p["on"]["d"]), p["ctx"].S), np.float32),
                        np.zeros(p["og"]["fut"].shape, np.float32)) for pn, p in P.items()}
            if mode == "none":
                net, _ = C.build_net(args.ckpt, "full")
                for d in days:
                    predict_day(net, P, d, out)
            elif mode == "perpanel":
                for pn, p in P.items():
                    net, params = C.build_net(args.ckpt, "full")
                    for d in days:
                        fd = first[(pn, d)]
                        if fd >= 10 ** 6:
                            continue
                        past = np.arange(d)
                        pool_days = np.append(past, d) if fd > C.H + C.F + 2 else past
                        cuts = {int(k): (fd if int(k) == d else 10 ** 6) for k in pool_days}
                        if len(pool_days):
                            rng = p["rng"]
                            PA.adapt_sup(net, params,
                                         lambda: PA.sup_batch(p["ctx"], p["stat"], p["z"], p["pool"], pool_days, cuts,
                                                              args.bs, rng, "ongoing"),
                                         30, args.lr, 1.0, lambda: C.past_batch(p["ctx"], p["stat"], p["z"], past,
                                                                                args.bs, rng))
                        predict_day(net, {pn: p}, d, out)
            else:
                steps = int(mode.split("_")[1])
                net, params = C.build_net(args.ckpt, "full")
                rng = np.random.default_rng(11)
                names = list(P)
                for d in days:
                    fd = min(first[(pn, d)] for pn in P)
                    past = np.arange(d)
                    pool_days = np.append(past, d) if fd > C.H + C.F + 2 else past
                    cuts = {int(k): (fd if int(k) == d else 10 ** 6) for k in pool_days}
                    if len(pool_days):
                        def sup():
                            p = P[names[rng.integers(len(names))]]
                            return PA.sup_batch(p["ctx"], p["stat"], p["z"], p["pool"], pool_days, cuts, args.bs, rng,
                                                "ongoing")

                        def rec():
                            if not len(past):
                                return None
                            p = P[names[rng.integers(len(names))]]
                            return C.past_batch(p["ctx"], p["stat"], p["z"], past, args.bs, rng)
                        PA.adapt_sup(net, params, sup, steps, args.lr, 1.0, rec)
                    predict_day(net, P, d, out)
            for pn, p in P.items():
                p_on, p_og = out[pn]
                on, og = p["on"], p["og"]
                so, sg = C.onset_iou(p_on, on), C.ongoing_iou(p_og, og, 0.5)
                a = res.setdefault((s, mode), {})
                a[pn] = (so.mean() if len(so) else np.nan, sg.mean(),
                         sg[og["d"] <= 4].mean() if (og["d"] <= 4).any() else np.nan)
                np.savez(CACHE / f"pooled_{args.tag}_{mode}_{s}_{pn}.npz", on=p_on, og=p_og.astype(np.float16))
            print(f"  [{s}] {mode} done ({time.time() - t_start:.0f}s)", flush=True)
        del P
    for s in C.SPLITS:
        print(f"\n=== {s}: {args.tag} ===")
        print(f"{'adaptation':<12}{'onset':>9}{'ongoing':>10}{'ongoing d<=4':>14}")
        for mode in args.modes:
            d = res[(s, mode)]
            fm = lambda j: C.family_mean({k: v[j] for k, v in d.items() if np.isfinite(v[j])})  # noqa: E731
            print(f"{mode:<12}{fm(0):9.4f}{fm(1):10.4f}{fm(2):14.4f}")


if __name__ == "__main__":
    main()
