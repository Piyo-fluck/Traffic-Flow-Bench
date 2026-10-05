"""How much does the scored run's OWN past carry for ongoing, as analog (nearest-neighbour) forecasts?

The two changes that moved the leaderboard most recently both fed on examples of the scored run itself (p_PS:
supervised adaptation on earlier days; note 47's analog onset map). Both use them coarsely. Here, per official-like
ongoing window at (day d, origin T), the library is every ongoing-like pseudo window of the SAME split whose whole
horizon ends before T: earlier days at any time within +-2 h of T's time of day, and day d itself before T. Distance
= mean squared difference of v/vf over the 12 history slots x all links (cells unseen in either are skipped); the
k nearest give a similarity-weighted mean of their future queue masks (released layer, obs == true). No training,
nothing at or after T. Scored alone and blended into AT's ongoing p, on all windows and on the early days (<= 4
earlier days, the official regime).

    .venv/Scripts/python.exe occ_lab/analog_ongoing.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB.parent / "tta_lab" / "tta"))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

import common as C  # noqa: E402
from online_adapter import ETA, GATE, global_features, load_split  # noqa: E402

TAG = "ssl_w192ssl_avg4"
K, WIN = 20, 24                 # neighbours; +-2 h of time of day
WS = (0.1, 0.2, 0.3, 0.5)
H, F = C.H, C.F


def main() -> None:
    t_start = time.time()
    gbm = lgb.Booster(model_file=str(LAB / "models" / "residual_lgb.txt"))
    fam = {}
    for i, panel in enumerate(C.T2_PANELS):
        ctx, stat, W = C.windows(i, panel)
        vf = np.asarray(ctx.vf, np.float32)
        sp = load_split(i, panel, TAG)
        for s, D in sp.items():
            z, og = W[s]["z"], W[s]["og"]
            v = z["obs_speed"]
            ok = z["elig"] & np.isfinite(v)
            r = np.where(ok, v / vf, np.nan).astype(np.float32)             # (days, T, S)
            q = ok & (np.nan_to_num(v, nan=1e9) <= 0.6 * vf)
            nd, nt, S = r.shape
            # library: every origin whose history holds >= 2 queued cells, with its future mask and eligibility
            lib_d, lib_t = [], []
            for dd in range(nd):
                for tt in range(H + 1, nt - F):
                    if q[dd, tt - H:tt].sum() >= 2:
                        lib_d.append(dd); lib_t.append(tt)
            lib_d, lib_t = np.array(lib_d), np.array(lib_t)
            hist = np.stack([r[a, b - H:b] for a, b in zip(lib_d, lib_t)]).reshape(len(lib_d), -1)   # (L, H*S)
            fut = np.stack([q[a, b + 1:b + 1 + F] for a, b in zip(lib_d, lib_t)]).astype(np.float32)  # (L, F, S)
            fok = np.stack([ok[a, b + 1:b + 1 + F] for a, b in zip(lib_d, lib_t)])
            blend = D["base"]
            Fe = global_features(D["P"], blend)
            rh = gbm.predict(Fe.reshape(-1, Fe.shape[-1])).reshape(blend.shape).astype(np.float32)
            p = np.clip(blend + ETA * np.where(np.abs(rh) >= GATE, rh, 0.0), 0, 1)
            y, e, d = D["y"], D["e"], D["d"]
            pk = np.full(p.shape, np.nan, np.float32)
            for n in range(len(d)):
                dd, T0 = int(d[n]), int(og["t0"][n])
                cand = ((lib_d < dd) & (np.abs(lib_t - T0) <= WIN)) | ((lib_d == dd) & (lib_t + F < T0))
                idx = np.flatnonzero(cand)
                if len(idx) < 3:
                    continue
                qh = np.where(og["ok"][n], og["v"][n] / vf, np.nan).reshape(-1)
                Hc = hist[idx]
                diff = (Hc - qh[None]) ** 2
                m = np.isfinite(diff)
                dist = np.where(m, diff, 0).sum(1) / np.maximum(m.sum(1), 1)
                top = idx[np.argsort(dist)[:K]]
                dt = np.sort(dist)[:K]
                w = np.exp(-dt / max(np.median(dt), 1e-6))
                num = (fut[top] * fok[top] * w[:, None, None]).sum(0)
                den = (fok[top] * w[:, None, None]).sum(0)
                pk[n] = np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)
            has = np.isfinite(pk).all((1, 2)) | np.isfinite(pk).any((1, 2))
            pkf = np.where(np.isfinite(pk), pk, p)                        # fall back to p where no analog
            early = d <= 4
            for nm, qq in [("AT", p), ("analog alone", pkf)] + [(f"AT + analog w{w}", (1 - w) * p + w * pkf) for w in WS]:
                sc = C.iou(qq >= 0.5, y, e)
                fam.setdefault((s, nm, "all"), {})[panel] = sc.mean()
                if early.any():
                    fam.setdefault((s, nm, "early"), {})[panel] = sc[early].mean()
            fam.setdefault((s, "coverage", "all"), {})[panel] = float(has.mean())
        print(f"  {panel} ({time.time() - t_start:.0f}s)", flush=True)
    names = ["AT", "analog alone"] + [f"AT + analog w{w}" for w in WS]
    for sub in ("all", "early"):
        print(f"\n=== ongoing family IoU, {sub} windows ===")
        for nm in names:
            v = [C.family_mean(fam[(s, nm, sub)]) for s in C.SPLITS]
            print(f"  {nm:<18} " + "  ".join(f"{s} {x:.4f}" for s, x in zip(C.SPLITS, v)) + f"  mean {np.mean(v):.4f}")
    print("coverage (share of windows with analogs):", {s: round(C.family_mean(fam[(s, 'coverage', 'all')]), 3) for s in C.SPLITS})


if __name__ == "__main__":
    main()
