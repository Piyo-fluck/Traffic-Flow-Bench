"""A residual corrector for the onset half, the way note 86 built one for ongoing.

    p0      = 0.35 H + 0.35 ad9b + 0.30 STID          the blend candidate AN ships
    r       = y - p0                                   at T+30, on eligible links
    r_hat   = f(member probabilities, disagreement, |p0 - 0.5|, hour, link position)
    p_final = clip(p0 + eta * r_hat, 0, 1)             then the expected-IoU ratio rule (c = 1.0) as usual

Fitted on the tail's onset windows (`cache_onset_oof.py`: 2031-01/02, out of fold for every member), applied to
the two scored months. Two differences from the ongoing corrector matter and are reported:

  * the training set is 61k cells, not 3.9M - the corrector is deliberately small;
  * onset is decided by a top-k SET rule, so the correction changes how many links each window predicts, not
    only which ones. The mean change in set size is printed next to the IoU.

    python occ_lab/onset_residual.py
"""
from __future__ import annotations

import argparse
import os
import sys
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
from residual_diag import lg  # noqa: E402
from t2_expected_iou import select  # noqa: E402

CACHE = LAB / "cache"
MODELS = LAB / "models"
REMOTE = PROJ / "work" / ("models_remote" if (PROJ / "work" / "models_remote").exists() else "models")
W = (0.35, 0.35, 0.30)
NAMES = ["p0", "logit_p0", "margin", "member_std", "H", "ad9b", "stid", "onset30", "cnn3",
         "d_logit_stid", "d_logit_H", "hour", "position"]
ETAS = (0.10, 0.25, 0.50, 0.75, 1.00)
PARAMS = {"objective": "l2", "learning_rate": 0.03, "num_leaves": 15, "min_data_in_leaf": 500,
          "feature_fraction": 0.7, "bagging_fraction": 0.7, "bagging_freq": 1, "lambda_l2": 10.0,
          "verbose": -1, "num_threads": 16}


def feats(P, hour):
    p0 = W[0] * P["H"] + W[1] * P["ad9b"] + W[2] * P["stid"]
    S = p0.shape[1]
    pos = np.broadcast_to((np.arange(S, dtype=np.float32) / max(S - 1, 1))[None, :], p0.shape)
    hr = np.broadcast_to(hour[:, None].astype(np.float32), p0.shape)
    mem = np.stack([P["H"], P["ad9b"], P["stid"], P["onset30"], P["cnn3"]])
    f = np.stack([p0, lg(p0), np.abs(p0 - 0.5), mem.std(0), P["H"], P["ad9b"], P["stid"], P["onset30"],
                  P["cnn3"], lg(P["stid"]) - lg(p0), lg(P["H"]) - lg(p0), hr, pos], -1)
    return f.astype(np.float32), p0


def load_cnn(p):
    sd = torch.load(p, map_location=C.DEV)
    net = C.cnn.T2Net(sd["inp.weight"].shape[1], C.cnn.WIDTH)
    net.load_state_dict(sd)
    return net.to(C.DEV).eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=150)
    args = ap.parse_args()
    C.cnn.REC = 0.0
    Xs, ys, ps = [], [], []
    for panel in C.T2_PANELS:
        z = np.load(CACHE / f"onset_oof_{panel}.npz")
        P = {k: z[k] for k in ("H", "ad9b", "stid", "onset30", "cnn3")}
        f, p0 = feats(P, z["hour"])
        e = z["e"]
        Xs.append(f[e]); ys.append(z["y"][e].astype(np.float32)); ps.append(p0[e])
    X, y, p0 = np.concatenate(Xs), np.concatenate(ys), np.concatenate(ps)
    print(f"tail onset cells {len(y):,}, queued {y.mean() * 100:.2f}%, "
          f"blend |r| mean {np.abs(y - p0).mean():.4f}", flush=True)
    gbm = lgb.train(PARAMS, lgb.Dataset(X, label=y - p0, feature_name=NAMES, free_raw_data=False),
                    num_boost_round=args.rounds)
    imp = sorted(zip(gbm.feature_importance("gain"), NAMES), reverse=True)[:5]
    tot = sum(g for g, _ in imp) or 1
    print("top gain: " + ", ".join(f"{n} {g / tot * 100:.0f}%" for g, n in imp), flush=True)

    on30 = [load_cnn(MODELS / f"onset30_cnn_s{s}.pt") for s in range(4)]
    cnn3_nets = [load_cnn(REMOTE / f"t2_cnn_lov{s}.pt") for s in ("", "_s1", "_s2")]
    fam, ksz = {}, {}
    for i, panel in enumerate(C.T2_PANELS):
        ctx, stat, Wd = C.windows(i, panel)
        for s in C.SPLITS:
            on, ref, z = Wd[s]["on"], Wd[s]["ref"], Wd[s]["z"]
            fs = CACHE / f"onset_adapt_stid30_{s}_{panel}.npz"
            if not len(on["d"]) or not fs.exists():
                continue
            x = C.make_x(ctx, stat, on, np.arange(len(on["d"])), z)
            P = {"H": ref["on_H"], "ad9b": ref["on_ad9b"], "stid": np.load(fs)["prob"],
                 "onset30": np.mean([C.cnn.predict(n, x)[:, C.F - 1] for n in on30], 0),
                 "cnn3": ref["on_cnn3"]}
            f, p0 = feats(P, on["t0"] / 12)
            rh = gbm.predict(f.reshape(-1, f.shape[-1])).reshape(p0.shape).astype(np.float32)
            fam.setdefault((s, "AN blend"), {})[panel] = C.onset_iou(p0, on).mean()
            k0 = np.array([select(q, "ratio", 1.0).sum() for q in p0], float)
            ksz.setdefault((s, "AN blend"), []).append(k0)
            for eta in ETAS:
                p = np.clip(p0 + eta * rh, 0, 1)
                fam.setdefault((s, f"+corrector eta {eta:.2f}"), {})[panel] = C.onset_iou(p, on).mean()
                k = np.array([select(q, "ratio", 1.0).sum() for q in p], float)
                ksz.setdefault((s, f"+corrector eta {eta:.2f}"), []).append(k - k0)
        print(f"  {panel}", flush=True)
    for s in C.SPLITS:
        b = C.family_mean(fam[(s, "AN blend")])
        k0 = np.concatenate(ksz[(s, "AN blend")]).mean()
        print(f"\n=== onset {s}: AN blend {b:.4f}, mean predicted links per window {k0:.2f} ===")
        for (ss, n), d in fam.items():
            if ss == s and n != "AN blend":
                dk = np.concatenate(ksz[(s, n)])
                print(f"   {n:<22} {C.family_mean(d):.4f} ({C.family_mean(d) - b:+.4f})   "
                      f"set size {dk.mean():+.2f} links/window, |change| {np.abs(dk).mean():.2f}")


if __name__ == "__main__":
    main()
