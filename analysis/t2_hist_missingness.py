"""Task 2: how complete are the window histories? Train windows are built from train truth (only ineligible cells
missing); the scored windows come from the released, T1-masked data. The transformer's onset is at LightGBM's level on
days 1-14 but far below it on later days (note 46) -- is the history more masked there?

Per panel: the share of observed history cells (ok) in the last 3 slots and over the hour, for
    train      earliest-onset training windows (ctx.eo, truth + eligibility, as the models are trained)
    val / priv the blank-aware official-like onset windows, days 1-14 and days 15+

    .venv/Scripts/python.exe analysis/t2_hist_missingness.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "scripts"))
from t2_eda02 import T2_PANELS  # noqa: E402
from t2_first_onset_eval import first_onsets, official_blank  # noqa: E402
from t2_m1_models import Ctx, hist_from_truth  # noqa: E402
from tfb.data import load  # noqa: E402


def share(ok: np.ndarray) -> str:
    return f"{ok[:, -3:].mean():.3f}/{ok.mean():.3f}" if len(ok) else "  -  "


def main() -> None:
    print("share of observed history cells, last 3 slots / whole hour (n windows)")
    tot: dict = {}
    for i, panel in enumerate(T2_PANELS):
        ctx = Ctx(panel, i)
        d, t0 = ctx.eo
        _, _, ok_tr = hist_from_truth(ctx, d, t0)
        line = [f"train {share(ok_tr)} ({len(d)})"]
        tot.setdefault("train", []).append(ok_tr.mean())
        for split in ("validation", "private"):
            z = load(panel, split)
            on = first_onsets(ctx.vf, z, blank=official_blank(panel, split, z))
            for name, m in (("d1-14", on["d"] < 14), ("d15+", on["d"] >= 14)):
                line.append(f"{split[:4]} {name} {share(on['ok'][m])} ({int(m.sum())})")
                if m.any():
                    tot.setdefault(f"{split[:4]} {name}", []).append(on["ok"][m].mean())
            # the day-by-day share of observed cells in the whole scored month (all links, 04-14 h)
            el = z["elig"] & np.isfinite(z["obs_speed"]) & np.isfinite(z["obs_flow"])
            by_day = el[:, 48:168].mean((1, 2))
            line.append(f"{split[:4]} month d1-14 {by_day[:14].mean():.3f} d15+ {by_day[14:].mean():.3f}")
        print(f"{panel:<10} " + " | ".join(line), flush=True)
    print("\nmean over panels (whole hour): " + "  ".join(f"{k} {np.mean(v):.3f}" for k, v in tot.items()))


if __name__ == "__main__":
    main()
