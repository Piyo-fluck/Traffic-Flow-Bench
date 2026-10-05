"""Does per-run supervision from EARLIER days of the scored month exist at all, for onset and for ongoing?

Both of the Task 2 items measured this week adapt on strictly earlier days of the same run. The lab surfaces
evaluated that on the release-derived pseudo windows themselves - fit on earlier pool days, score later pool days
- which is causal and out of sample, but says nothing about whether the OFFICIAL windows have any earlier pool
day behind them. This checks the thing production actually needs:

    for every official window, how many release-side pool windows lie on strictly earlier days of its month

`onset` uses the one-per-day first-morning events (`first_onsets` with the official blanks), the only pool the
FiLM gained on. `ongoing` uses `ongoing_official`, the pool the online residual adapter was fitted on. Both are
exactly what `common.windows` builds, so this is the same pool, not a stand-in.

    .venv/Scripts/python.exe analysis/t2_run_supervision_budget.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tta_lab" / "tta"))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import common as C  # noqa: E402
from t2_first_onset_eval import first_onsets, official_blank  # noqa: E402
from t2_official_like_eval import ongoing_official  # noqa: E402
from tfb.data import RELEASE, load  # noqa: E402


def main() -> None:
    print(f"{'panel':<11}{'split':<11}{'kind':<9}{'off w':>6}{'off days':>10}{'pool':>7}{'pool days':>11}"
          f"{'>=1 past w':>12}{'median past w':>15}")
    tot = {}
    for i, panel in enumerate(C.T2_PANELS):
        ctx = C.Ctx(panel, i)
        for split in C.SPLITS:
            z = load(panel, split)
            blank = official_blank(panel, split, z)
            pools = {"onset": first_onsets(ctx.vf, z, blank=blank)["d"],
                     "ongoing": ongoing_official(ctx, z, blank)[0]["d"]}
            Wi = pd.read_csv(RELEASE / "task2" / panel / split / "window_index.csv")
            d_of = {str(x): k for k, x in enumerate(z["dates"])}
            dd = np.array([d_of[str(x)] for x in Wi.date], int)
            on = (Wi.condition == "queue_onset").to_numpy()
            pools["onset/og pool"] = pools["ongoing"]     # can a BROAD pool reach the onset days at all?
            for kind, pd_ in pools.items():
                sel = dd[on if kind.startswith("onset") else ~on]
                n_past = np.array([int((pd_ < d).sum()) for d in sel])
                a, b = int((n_past >= 1).sum()), len(sel)
                print(f"{panel:<11}{split:<11}{kind:<9}{b:>6}{len(np.unique(sel)):>10}{len(pd_):>7}"
                      f"{len(np.unique(pd_)):>11}{a:>12}"
                      f"{int(np.median(n_past)) if b else 0:>15}")
                t = tot.setdefault(kind, [0, 0])
                t[0] += a
                t[1] += b
        ctx.z = ctx.q = ctx.el = None
    print()
    for kind, (a, b) in tot.items():
        print(f"{kind:<9}{a} of {b} official windows ({a / max(b, 1):.1%}) have at least one pool window "
              f"on a strictly earlier day of their own month")


if __name__ == "__main__":
    main()
