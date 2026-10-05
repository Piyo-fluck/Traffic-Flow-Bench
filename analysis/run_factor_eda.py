"""Decompose run differences: per link, a demand proxy (mean free-flow flow at 06-20 h, v > 0.85 vf) and a
capacity proxy (p98 flow), scored run / train ratio; per ramp, mean flow ratio. Printed as quantiles and the
correlation of the ratio pattern along the corridor between the two scored runs."""
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tta_lab" / "tta"))
import numpy as np  # noqa: E402

import common as C  # noqa: E402
from tfb.data import load  # noqa: E402
from tfb.physics import load_ramps  # noqa: E402

warnings.simplefilter("ignore")


def stats(z, vf):
    v, q = z["obs_speed"][:, 72:240], z["obs_flow"][:, 72:240]
    ok = z["elig"][:, 72:240] & np.isfinite(v) & np.isfinite(q)
    ff = ok & (v > 0.85 * vf)
    dem = np.nansum(np.where(ff, q, 0), (0, 1)) / np.maximum(ff.sum((0, 1)), 1)
    cap = np.array([np.nanpercentile(q[:, :, j][ok[:, :, j]], 98) if ok[:, :, j].sum() > 50 else np.nan
                    for j in range(q.shape[2])])
    return dem, cap


for i, panel in enumerate(C.T2_PANELS):
    ctx = C.Ctx(panel, i)
    vf = np.asarray(ctx.vf, np.float32)
    S = {}
    for split in ("train", "validation", "private"):
        z = load(panel, split)
        dem, cap = stats(z, vf)
        r = load_ramps(panel, split, z)
        on = np.where(r["on_v"], r["on_q"], np.nan)[:, 72:240]
        off = np.where(r["off_v"], r["off_q"], np.nan)[:, 72:240]
        S[split] = (dem, cap, np.nanmean(on, (0, 1)), np.nanmean(off, (0, 1)))
    print(f"\n{panel}")
    for k, name in enumerate(("demand", "cap p98", "on-ramp", "off-ramp")):
        rs = []
        for s in ("validation", "private"):
            a, b = S[s][k], S["train"][k]
            r = a / b
            r = r[np.isfinite(r) & (b > 50)]
            rs.append(S[s][k] / S["train"][k])
            print(f"  {name:<9} {s[:3]}/train  q10 {np.quantile(r, .1):.3f}  q50 {np.quantile(r, .5):.3f}  "
                  f"q90 {np.quantile(r, .9):.3f}  n {len(r)}")
        m = np.isfinite(rs[0]) & np.isfinite(rs[1])
        if m.sum() > 3:
            print(f"  {name:<9} corr(val ratio, pri ratio) {np.corrcoef(rs[0][m], rs[1][m])[0, 1]:+.2f}")
