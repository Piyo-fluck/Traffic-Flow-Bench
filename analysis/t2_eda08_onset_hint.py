"""Task 2 EDA 08: check the organizers' onset hint (synthetic_release_v1.json, task2.condition_rationale):

    "Onset is solvable: an hour before a queue forms the road is already at 1.28x its queue threshold
     against 1.54x where none follows, and carries 6912 vph against 4620."

On train truth, for earliest-onset origins (queue first at T+30), look one hour before the onset (T-30),
and compare with "no-onset" origins at the same times of day (no queue in history or horizon):
  - road-level: min over links of v / v_cut, and the flow of that slowest link;
  - where: is the onset link the slowest / highest-flow link, or within k stations downstream / upstream?
Then score simple "where" rules at T+30 against the recurrence prior on official onset windows.

    py analysis/t2_eda08_onset_hint.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
from t2_eda02 import T2_PANELS  # noqa: E402
from t2_eda03 import F, H, earliest_onset, labels  # noqa: E402
from tfb.data import T, load  # noqa: E402
from tfb.t1eval import family_mean  # noqa: E402


def main() -> None:
    rows = {}
    for panel in T2_PANELS:
        z = load(panel, "train")
        q, el = labels(z)
        vf = z["free_speed"]
        ts, tq = z["true_speed"], z["true_flow"]
        vr = np.where(el, ts / (0.6 * vf), np.nan)       # speed relative to the queue threshold
        eo = earliest_onset(q)
        d_on, t_on = np.nonzero(eo)
        # no-onset origins: no queue anywhere in [T-60, T+30], same slots as the onset set
        quiet = []
        for t0 in np.unique(t_on):
            ok = ~q[:, t0 - H:t0 + F + 1].any((1, 2))
            quiet += [(d, t0) for d in np.flatnonzero(ok)]
        d_no, t_no = (np.array(x) for x in zip(*quiet)) if quiet else (np.array([], int),) * 2
        tb = lambda t: t - 6  # noqa: E731  one hour before onset at T+30 is T-30
        res = {}
        for name, d, t in (("onset", d_on, t_on), ("none", d_no, t_no)):
            v = vr[d, tb(t)]
            with np.errstate(all="ignore"):
                vmin = np.nanmin(v, 1)
                arg = np.nanargmin(np.where(np.isnan(v), np.inf, v), 1)
            res[name] = (np.nanmedian(vmin), np.nanmedian(tq[d, tb(t), arg]), np.nanmedian(np.nanmax(np.where(el[d, tb(t)], tq[d, tb(t)], np.nan), 1)))
        # where: onset links vs the slowest link one hour before (and at T-5)
        hits = {}
        for lag_name, tt in (("T-30", t_on - 6), ("T-5", t_on - 1)):
            v = np.where(np.isnan(vr[d_on, tt]), np.inf, vr[d_on, tt])
            arg = np.argmin(v, 1)
            onl = q[d_on, t_on + F]
            S = onl.shape[1]
            for k in (0, 2, 5):
                win = np.zeros_like(onl)
                for off in range(-k, k + 1):
                    j = np.clip(arg + off, 0, S - 1)
                    win[np.arange(len(arg)), j] = True
                hits[f"{lag_name} slowest+-{k}"] = float((win & onl).any(1).mean())
            # signed offset of nearest onset link from the slowest link (positive = downstream)
            off = [np.flatnonzero(onl[i])[np.argmin(np.abs(np.flatnonzero(onl[i]) - arg[i]))] - arg[i] for i in range(len(arg)) if onl[i].any()]
            hits[f"{lag_name} median offset"] = float(np.median(off)) if off else np.nan
        rows[panel] = (res, hits, len(d_on), len(d_no))
        print(f"{panel}: onset {len(d_on)} vs quiet {len(d_no)} | 1h before: min v/v_cut {res['onset'][0]:.2f} vs {res['none'][0]:.2f}; "
              f"flow at slowest {res['onset'][1]:.0f} vs {res['none'][1]:.0f}; max flow {res['onset'][2]:.0f} vs {res['none'][2]:.0f} | "
              + ", ".join(f"{k} {v:.2f}" for k, v in hits.items()), flush=True)


if __name__ == "__main__":
    main()
