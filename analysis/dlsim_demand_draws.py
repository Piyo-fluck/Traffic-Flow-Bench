"""Note 169: how does a run's entry demand differ from train's? Per entry (OD-zone gap), the 06-20 h mean of the
free-flow-slot demand profile (dlsim_gen2.Gen.fit_profiles), scored run / train ratio: log-sd over entries, lag-1
autocorrelation of the log ratio along the corridor, correlation between the two scored runs, and the upstream
boundary ratio. Evaluation use of the scored months only (to size the generator's demand draws)."""
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "analysis")); sys.path.insert(0, str(ROOT / "tta_lab" / "tta"))
import numpy as np  # noqa: E402

import common as C  # noqa: E402
from dlsim_gen2 import Gen  # noqa: E402
from tfb.data import load  # noqa: E402

warnings.simplefilter("ignore")
for panel in C.T2_PANELS:
    zt = load(panel, "train")
    g = Gen(panel, zt)
    g.fit_profiles(zt)
    Pt, Ut = g.P[:5, 72:240].mean((0, 1)), g.U[:5, 72:240].mean()
    lr, ups = {}, {}
    for s in ("validation", "private"):
        zs = load(panel, s)
        g.fit_profiles(zs, vkey="obs_speed", qkey="obs_flow")
        Ps, Us = g.P[:5, 72:240].mean((0, 1)), g.U[:5, 72:240].mean()
        ok = (Pt > 50) & (Ps > 0)
        r = np.full(len(Pt), np.nan); r[ok] = np.log(Ps[ok] / Pt[ok])
        lr[s], ups[s] = r, Us / Ut
    out = []
    for s in ("validation", "private"):
        r = lr[s]; m = np.isfinite(r)
        a = r[m]
        ac = np.corrcoef(a[:-1], a[1:])[0, 1] if len(a) > 3 else np.nan
        out.append(f"{s[:3]}: log-ratio mean {np.nanmean(r):+.3f} sd {np.nanstd(r):.3f} lag1 {ac:+.2f} upstream x{ups[s]:.3f}")
    m = np.isfinite(lr["validation"]) & np.isfinite(lr["private"])
    print(f"{panel:<10} entries {len(Pt)} ({int(m.sum())} usable) | " + " | ".join(out) +
          f" | corr(val, pri) {np.corrcoef(lr['validation'][m], lr['private'][m])[0, 1]:+.2f}", flush=True)
