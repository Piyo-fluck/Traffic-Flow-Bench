"""How much release-side first-morning onset supervision an official onset window actually has behind it.

`occ_lab/onset_film.py` settled the supervision distribution: the FiLM must be fitted on FIRST-MORNING onset
events (one per day). Broadening the pool to every clean-history origin with a queue at T+30 costs -0.0288 on the
STID alone and -0.0028 on the blend, while the one-per-day pool gains +0.0145 / +0.0063 even with the day budget
capped at 6. So production needs the one-per-day pool - and that pool is thinned twice on the scored months:

  * a day that HOLDS an official onset window has its own first morning queue hidden, so it cannot be an event
  * a candidate whose history, slot T or horizon touches any official blank is dropped

which is exactly what `common.windows` builds as `on`. If the official onset windows occupy days 1-7 and those
days are the excluded ones, the budget could be empty. This counts it: for each official onset window, how many
release-side events lie on strictly earlier days of the same month.

    .venv/Scripts/python.exe analysis/t2_onset_supervision_budget.py
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
from tfb.data import RELEASE, load  # noqa: E402


def main() -> None:
    print(f"{'panel':<12}{'split':<12}{'onset w':>9}{'days':>7}{'events':>8}{'ev days':>9}"
          f"{'>=1 past':>10}{'>=2 past':>10}{'median':>8}")
    tot = np.zeros(4)
    for i, panel in enumerate(C.T2_PANELS):
        ctx = C.Ctx(panel, i)
        for split in C.SPLITS:
            z = load(panel, split)
            blank = official_blank(panel, split, z)
            ev = first_onsets(ctx.vf, z, blank=blank)
            Wi = pd.read_csv(RELEASE / "task2" / panel / split / "window_index.csv")
            d_of = {str(x): k for k, x in enumerate(z["dates"])}
            dd = np.array([d_of[str(x)] for x in Wi.date], int)
            on = (Wi.condition == "queue_onset").to_numpy()
            odays = np.unique(dd[on])
            n_past = np.array([int((ev["d"] < d).sum()) for d in dd[on]])
            print(f"{panel:<12}{split:<12}{int(on.sum()):>9}{len(odays):>7}{len(ev['d']):>8}"
                  f"{len(np.unique(ev['d'])):>9}{int((n_past >= 1).sum()):>10}"
                  f"{int((n_past >= 2).sum()):>10}{int(np.median(n_past)) if len(n_past) else 0:>8}")
            tot += [int(on.sum()), len(ev["d"]), int((n_past >= 1).sum()), int((n_past >= 2).sum())]
        ctx.z = ctx.q = ctx.el = None
    print(f"\n{int(tot[0])} official onset windows, {int(tot[1])} release-side events; "
          f"{int(tot[2])} windows ({tot[2] / max(tot[0], 1):.1%}) have at least one earlier event, "
          f"{int(tot[3])} ({tot[3] / max(tot[0], 1):.1%}) at least two")


if __name__ == "__main__":
    main()
