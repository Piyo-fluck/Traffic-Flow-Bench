"""Cache the Task 1 base tables (weekday x tod x station mean over every train day) per panel.

Separate, torch-free step, so the CNN inference process never has to load the large train caches.
Writes work/cache/<panel>_t1tables.npz with tv, tq of shape (7, T, S).

    py scripts/t1_tables_cache.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tfb.data import CACHE, PANELS, load  # noqa: E402
from tfb.features import profile_table, truth_valid  # noqa: E402


def main() -> None:
    for panel in PANELS:
        z = load(panel, "train")
        alld = np.ones(len(z["dates"]), bool)
        valid = truth_valid(z)
        tv = profile_table(z["true_speed"], valid, z["weekday"], alld)
        tq = profile_table(z["true_flow"], valid, z["weekday"], alld)
        np.savez(CACHE / f"{panel}_t1tables.npz", tv=tv, tq=tq)
        print(f"{panel}: tables {tv.shape}", flush=True)
        del z


if __name__ == "__main__":
    main()
