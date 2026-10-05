"""Out-of-fold onset material for a residual corrector: the train period's held-out tail, T+30 only.

The same construction as `cache_oof.py` does for ongoing (note 86), for the onset half:

    windows   the tail's onset windows (2031-01/02, days no member was fitted on) - the day's first onset, the
              official condition, plus (with --extra) every other origin that satisfies the same selector, since
              one window a day is thin
    members   LightGBM H (the 5 production bags), the past-day-adapted CNN (idea 9b, adapted causally over the
              tail days exactly as production does), the onset STID, the single-horizon onset30, cnn3
    labels    the queue at T+30 on eligible links

Writes occ_lab/cache/onset_oof_<panel>.npz.

    python occ_lab/cache_onset_oof.py --extra
"""
from __future__ import annotations

import argparse
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
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import common as C  # noqa: E402
from t2_eda03 import H as HIST  # noqa: E402
from t2_first_onset_eval import first_onsets  # noqa: E402
from t2_m1_predict import feats_on, table_prior  # noqa: E402
from t2_pseudo_eval import pseudo_windows  # noqa: E402
from tfb.t1eval import FIT_END  # noqa: E402
from train_occ import occ_channels  # noqa: E402
from train_onset_stid import OnsetSTID, predict as stid_predict  # noqa: E402

CACHE = LAB / "cache"
MODELS = LAB / "models"
REMOTE = PROJ / "work" / ("models_remote" if (PROJ / "work" / "models_remote").exists() else "models")
BAGS = PROJ / "work" / "sub_t2robust_bag5"


def load_cnn(p):
    sd = torch.load(p, map_location=C.DEV)
    net = C.cnn.T2Net(sd["inp.weight"].shape[1], C.cnn.WIDTH)
    net.load_state_dict(sd)
    return net.to(C.DEV).eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", action="store_true", help="also take the non-first onset origins of each day")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--truth-hist", action="store_true",
                    help="note 108: build the histories from the TRUE layer restricted to eligibility, so the "
                         "coverage matches the official windows (0.764 vs 0.761) instead of inheriting the "
                         "Task 1 mask (0.52, note 76). Writes onset_hicov_<panel>.npz")
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    m_on = [lgb.Booster(model_file=str(f)) for f in sorted(BAGS.glob("t2_onset*.txt"))]
    stids = []
    for s in range(4):
        d = torch.load(MODELS / f"onset_stid_s{s}.pt", map_location=C.DEV)
        net = OnsetSTID(d["cin"], d["n_links"], d["n_slots"]).to(C.DEV)
        net.load_state_dict(d["state"])
        stids.append(net.eval())
    C.cnn.REC = 0.0
    on30 = [load_cnn(MODELS / f"onset30_cnn_s{s}.pt") for s in range(4)
            if (MODELS / f"onset30_cnn_s{s}.pt").exists()]        # not a production member; skip when absent
    cnn3 = [load_cnn(REMOTE / f"t2_cnn_lov{s}.pt") for s in ("", "_s1", "_s2")]
    C.cnn.REC = 0.3
    profs = np.load(MODELS / "occ_profiles.npz" if (MODELS / "occ_profiles.npz").exists() else REMOTE / "occ_profiles.npz")
    t0_ = time.time()
    loff = 0
    for i, panel in enumerate(C.T2_PANELS):
        ctx = C.Ctx(panel, i)
        ctx.fitd = np.ones_like(ctx.fitd)
        ctx.loff = 0
        stat = C.cnn.static_channels(ctx, C.load_ramps(panel, "train", ctx.z))
        lid = np.arange(ctx.S) + loff
        loff += ctx.S
        z = ctx.z
        KEYS = ("d", "t0", "v", "qf", "ok")              # all this needs; the labels come from ctx.q
        fo = first_onsets(ctx.vf, z, truth=args.truth_hist, blank=None)
        on = {k: fo[k] for k in KEYS}
        if args.extra:                                   # every other origin with the same structure
            w = pseudo_windows(ctx, z)
            sel = np.flatnonzero(w["onset"])
            same = set(zip(fo["d"].tolist(), fo["t0"].tolist()))
            sel = sel[[(d, t) not in same for d, t in zip(w["d"][sel], w["t0"][sel])]]
            on = {k: np.concatenate([on[k], w[k][sel]]) for k in KEYS}
        keep = np.flatnonzero(z["dates"][on["d"]] >= FIT_END)
        on = {k: (v[keep] if isinstance(v, np.ndarray) and len(v) == len(on["d"]) else v) for k, v in on.items()}
        n = len(on["d"])
        grp = (z["weekday"][on["d"]] >= 5).astype(int)
        dow = z["weekday"][on["d"]].astype(np.int64)
        key = np.stack([on["d"], grp], 1)
        _, X = feats_on(ctx, False, on["v"], on["qf"], on["ok"], on["t0"], grp, table_prior(ctx), key, dow)
        P = {"H": np.mean([m.predict(X.reshape(-1, X.shape[-1])) for m in m_on], 0).reshape(X.shape[:2])}
        hi = on["t0"][:, None] + np.arange(-HIST, 0)
        x = C.make_x(ctx, stat, on, np.arange(n), z)
        xo = np.concatenate([x, occ_channels(z["obs_occ"][on["d"][:, None], hi], on["ok"], profs[panel])], 1)
        P["stid"] = np.mean([stid_predict(net, xo, lid, on["t0"].astype(np.int64), dow) for net in stids], 0)
        if on30:
            P["onset30"] = np.mean([C.cnn.predict(net, x)[:, C.F - 1] for net in on30], 0)
        P["cnn3"] = np.mean([C.cnn.predict(net, x)[:, C.F - 1] for net in cnn3], 0)
        # idea 9b, causal over the tail days
        net, params = C.build_net(REMOTE / "t2_cnn_lov_rec_s20.pt", "full")
        rng = np.random.default_rng(100 + i)
        ad = np.zeros((n, ctx.S), np.float32)
        for day in np.unique(on["d"]):
            idx = np.flatnonzero(on["d"] == day)
            ad[idx] = C.cnn.predict(net, x[idx])[:, C.F - 1]
            if day:
                C.adapt(net, params, lambda: C.past_batch(ctx, stat, z, np.arange(day), 16, rng),
                        args.steps, args.lr)
        P["ad9b"] = ad
        fut = ctx.q[on["d"][:, None], (on["t0"] + C.F)[:, None]][:, 0]
        el = ctx.el[on["d"][:, None], (on["t0"] + C.F)[:, None]][:, 0]
        tag = "onset_hicov" if args.truth_hist else "onset_oof"
        cov = float(on["ok"].mean())
        np.savez(CACHE / f"{tag}_{panel}.npz", y=fut, e=el, hour=(on["t0"] / 12).astype(np.float32), cov=cov,
                 d=on["d"], **{k: v.astype(np.float32) for k, v in P.items()})
        print(f"  {panel}: {n:,} onset windows, {int(el.sum()):,} eligible links, "
              f"{float(fut[el].mean()) * 100:.2f}% queued, history coverage {cov:.3f} "
              f"({time.time() - t0_:.0f}s)", flush=True)
        ctx.z = ctx.q = ctx.el = None


if __name__ == "__main__":
    main()
