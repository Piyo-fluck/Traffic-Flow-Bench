"""Task 2 onset: how much weight should the improved analog maps get (note 47, after sweep 2)? Caches only.

For the best maps of analysis/t2_analog_tune.py sweep 2, the map weight wm in (1 - wm) * model + wm * map is varied:
    M+map        model = M = 0.7 H + 0.3 CNN (production probabilities), masked histories
    base+map (f) model = the re-trained H (t2_analog_prior cache), T1-filled (official-like) histories
Reported as the change against the model alone (no map), val d1-14 | val all | priv d1-14 | priv all, family mean.

    .venv/Scripts/python.exe analysis/t2_analog_blendw.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "scripts"))
from t2_analog_prior import filled, rehist, train_profile  # noqa: E402
from t2_analog_tune import CONDS, REF, TAGS, amap, build, fmean, score, state  # noqa: E402
from t2_eda02 import T2_PANELS  # noqa: E402
from t2_eda03 import F  # noqa: E402
from t2_first_onset_eval import first_onsets, official_blank  # noqa: E402
from t2_m1_models import Ctx, hist_from_truth  # noqa: E402
from tfb.data import CACHE, load  # noqa: E402

BASE = dict(REF, k=40)
MAPS = {"K40": {}, "K60": dict(k=60), "xcov8 flow+speed": dict(proj="xcov", npc=8, key="flowspeed"),
        "xcov8 flow+speed K60": dict(proj="xcov", npc=8, key="flowspeed", k=60)}
WM = (0.2, 0.3, 0.4, 0.5)


def main() -> None:
    res: dict = {}
    for i, panel in enumerate(T2_PANELS):
        ctx = Ctx(panel, i)
        ctx.fitd = np.ones_like(ctx.fitd)
        d, t0 = ctx.eo
        v, qf, ok = hist_from_truth(ctx, d, t0)
        y = ctx.q[d, t0 + F].astype(np.float32)
        grp = ctx.g[d].astype(int)
        prof = train_profile(ctx)
        pg = y.mean(0)
        raws = {k: state(ctx, prof, v, qf, ok, t0, grp, k) for k in ("flow6", "flowspeed")}
        for split in ("validation", "private"):
            z = load(panel, split)
            on = first_onsets(ctx.vf, z, blank=official_blank(panel, split, z))
            if not len(on["d"]):
                continue
            c = np.load(CACHE / f"t2probs_{split}_{panel}.npz")
            assert (c["on_d"] == on["d"]).all()
            m_prob = 0.7 * c["on_p"] + 0.3 * np.mean([np.load(CACHE / f"t2cnn_{t}_{split}_{panel}.npz")["on_p"] for t in TAGS], 0)
            p_base_f = np.load(CACHE / f"t2analog_filled_{split}_{panel}.npz")["p_base"]
            qgrp = (z["weekday"][on["d"]] >= 5).astype(int)
            hist = {"masked": (on["v"], on["qf"], on["ok"]), "filled": rehist(filled(z, panel, split), on["d"], on["t0"])}
            shrink = np.broadcast_to(pg, (len(on["d"]), ctx.S))
            for metric, model, hname in (("M+map", m_prob, "masked"), ("base+map (f)", p_base_f, "filled")):
                outs = {"(model alone)": model}
                for mname, change in MAPS.items():
                    cfg = dict(BASE, **change)
                    kf = build(cfg, raws[cfg["key"]], t0, cfg["npc"], lib_y=y)
                    L = dict(raw=raws[cfg["key"]], t0=t0, y=y, grp=grp)
                    hv, hq, ho = hist[hname]
                    mp = amap(cfg, kf, L, state(ctx, prof, hv, hq, ho, on["t0"], qgrp, cfg["key"]), on["t0"], qgrp, shrink)
                    for wm in WM:
                        outs[f"{mname} wm{wm}"] = (1 - wm) * model + wm * mp
                for name, p in outs.items():
                    s = score(p, on["fut"], on["efut"])
                    for sp, dmax in CONDS:
                        if sp == split:
                            mm = on["d"] < dmax
                            res.setdefault((metric, name), {}).setdefault((sp, dmax), {})[panel] = s[mm].mean() if mm.any() else np.nan
        print(f"[{panel}]", flush=True)
    for metric in ("M+map", "base+map (f)"):
        base = res[(metric, "(model alone)")]
        print(f"\n== {metric}: val d1-14 | val all | priv d1-14 | priv all  (delta vs model alone) ==")
        rows = []
        for (mt, name), r in res.items():
            if mt != metric:
                continue
            vals = [fmean(r[c]) for c in CONDS]
            dl = [fmean(r[c]) - fmean(base[c]) for c in CONDS]
            rows.append((np.mean(dl), min(dl), name, vals, dl))
        for mean_d, min_d, name, vals, dl in sorted(rows, key=lambda x: -x[0]):
            print(f"  {name:<28} " + " | ".join(f"{v:.3f} ({x:+.3f})" for v, x in zip(vals, dl)) + f"  mean {mean_d:+.4f} min {min_d:+.4f}")


if __name__ == "__main__":
    main()
