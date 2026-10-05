"""Task 2 onset: M's production onset rule scored on T1-filled (official-like) histories, and the analog map's increment
over M there (notes 46-47). The official window histories are ~76% observed like the training windows; the masked
pseudo windows only ~52%. Until now M was scored only on masked histories.

Same blank-aware official-like onset windows. For masked and filled histories, recomputed from the models:
    H    production LightGBM onset (work/sub_t2robust_bag5, 5 bags, table priors)
    CNN  the 3 production Lovasz CNN seeds (work/models_remote/t2_cnn_lov*.pt)
    M    0.7 H + 0.3 CNN, the expected-IoU ratio rule at T+30
    maps K40, K60 and 'xcov8 flow+speed' (analysis/t2_analog_tune.py), from the same histories
Variants: (1 - wm) * M + wm * map for wm in 0.2 .. 0.5; also H + map without the CNN.
Change vs M, val d1-14 | val all | priv d1-14 | priv all, family mean, panels better/worse.

    .venv/Scripts/python.exe analysis/t2_analog_mfilled.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "scripts"))
import t2_cnn as cnn  # noqa: E402
from t2_analog_prior import filled, rehist, train_profile  # noqa: E402
from t2_analog_tune import CONDS, REF, amap, build, fmean, score, state  # noqa: E402
from t2_eda02 import T2_PANELS  # noqa: E402
from t2_eda03 import F  # noqa: E402
from t2_first_onset_eval import first_onsets, official_blank  # noqa: E402
from t2_m1_models import Ctx, hist_from_truth  # noqa: E402
from t2_m1_predict import feats_on, table_prior  # noqa: E402
from tfb.data import load  # noqa: E402
from tfb.physics import load_ramps  # noqa: E402

MR = ROOT / "work" / "models_remote"
BASE = dict(REF, k=40)
MAPS = {"K40": {}, "K60": dict(k=60), "xcov8 f+s": dict(proj="xcov", npc=8, key="flowspeed")}
WM = (0.2, 0.3, 0.4, 0.5)


def main() -> None:
    t_start = time.time()
    m_on = [lgb.Booster(model_file=str(f)) for f in sorted((ROOT / "work" / "sub_t2robust_bag5").glob("t2_onset*.txt"))]
    nets = []
    for p in (MR / "t2_cnn_lov.pt", MR / "t2_cnn_lov_s1.pt", MR / "t2_cnn_lov_s2.pt"):
        n = cnn.T2Net(12, 64).to(cnn.DEV)
        n.load_state_dict(torch.load(p, map_location=cnn.DEV))
        n.eval()
        nets.append(n)
    res: dict = {}
    for i, panel in enumerate(T2_PANELS):
        ctx = Ctx(panel, i)
        ctx.fitd = np.ones_like(ctx.fitd)
        stat = cnn.static_channels(ctx, load_ramps(panel, "train", ctx.z))
        d, t0 = ctx.eo
        v, qf, ok = hist_from_truth(ctx, d, t0)
        y = ctx.q[d, t0 + F].astype(np.float32)
        grp = ctx.g[d].astype(int)
        prof = train_profile(ctx)
        pg = y.mean(0)
        raws = {k: state(ctx, prof, v, qf, ok, t0, grp, k) for k in ("flow6", "flowspeed")}
        kfs = {}
        for mname, change in MAPS.items():
            cfg = dict(BASE, **change)
            kfs[mname] = (cfg, build(cfg, raws[cfg["key"]], t0, cfg["npc"], lib_y=y),
                          dict(raw=raws[cfg["key"]], t0=t0, y=y, grp=grp))
        for split in ("validation", "private"):
            z = load(panel, split)
            on = first_onsets(ctx.vf, z, blank=official_blank(panel, split, z))
            if not len(on["d"]):
                continue
            qgrp = (z["weekday"][on["d"]] >= 5).astype(int)
            key = np.stack([on["d"], qgrp], 1)
            shrink = np.broadcast_to(pg, (len(on["d"]), ctx.S))
            for hname, (hv, hq, ho) in (("masked", (on["v"], on["qf"], on["ok"])),
                                         ("filled", rehist(filled(z, panel, split), on["d"], on["t0"]))):
                _, X = feats_on(ctx, False, hv, hq, ho, on["t0"], qgrp, table_prior(ctx), key, z["weekday"][on["d"]].astype(int))
                h = np.mean([m.predict(X.reshape(-1, X.shape[-1])) for m in m_on], 0).reshape(X.shape[:2])
                x = cnn.make_x(ctx, stat, hv, hq, ho, on["t0"], z["weekday"][on["d"]] >= 5)
                c = np.mean([cnn.predict(n, x) for n in nets], 0)[:, F - 1]
                m = 0.7 * h + 0.3 * c
                outs = {"M": m, "H alone": h}
                for mname, (cfg, kf, L) in kfs.items():
                    mp = amap(cfg, kf, L, state(ctx, prof, hv, hq, ho, on["t0"], qgrp, cfg["key"]), on["t0"], qgrp, shrink)
                    outs[f"map {mname} alone"] = mp
                    for wm in WM:
                        outs[f"M + {mname} wm{wm}"] = (1 - wm) * m + wm * mp
                    outs[f"H + {mname} wm0.4"] = 0.6 * h + 0.4 * mp
                for name, p in outs.items():
                    s = score(p, on["fut"], on["efut"])
                    for sp, dmax in CONDS:
                        if sp == split:
                            mm = on["d"] < dmax
                            res.setdefault((hname, name), {}).setdefault((sp, dmax), {})[panel] = s[mm].mean() if mm.any() else np.nan
        print(f"[{panel}] ({time.time() - t_start:.0f}s)", flush=True)
    for hname in ("masked", "filled"):
        base = res[(hname, "M")]
        print(f"\n== {hname} histories: onset IoU, val d1-14 | val all | priv d1-14 | priv all (delta vs M, panels +/-) ==")
        rows = []
        for (hn, name), r in res.items():
            if hn != hname:
                continue
            vals, dl, cnt = [], [], []
            for cnd in CONDS:
                vals.append(fmean(r[cnd])); dl.append(fmean(r[cnd]) - fmean(base[cnd]))
                diffs = [r[cnd][p] - base[cnd][p] for p in r[cnd] if np.isfinite(r[cnd][p]) and np.isfinite(base[cnd].get(p, np.nan))]
                cnt.append(f"{sum(x > 1e-9 for x in diffs)}/{sum(x < -1e-9 for x in diffs)}")
            rows.append((np.mean(dl), min(dl), name, vals, dl, cnt))
        for mean_d, min_d, name, vals, dl, cnt in sorted(rows, key=lambda x: -x[0]):
            print(f"  {name:<22} " + " | ".join(f"{v:.3f} ({x:+.3f} {c})" for v, x, c in zip(vals, dl, cnt)) + f"  mean {mean_d:+.4f} min {min_d:+.4f}")
    print(f"total {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
