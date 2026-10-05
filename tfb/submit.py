"""Write per-task submission files from dense-array predictions."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tfb.data import RELEASE

STATE_COLUMNS = ["panel", "timestamp", "station_id", "link_id", "mask_regime", "speed_kmh", "flow_vph"]


def state_frame(panel: str, z: dict, pred_speed: np.ndarray, pred_flow: np.ndarray) -> pd.DataFrame:
    """Task 1 rows for every target cell, keyed exactly like the published template."""
    d, t, s = np.nonzero(z["target"])
    # Plain Python string concatenation on object arrays; np.char on very long arrays crashed natively
    # on Windows (0xc000070a) for large panels.
    day_txt = np.array([f"{x}T" for x in z["dates"]], dtype=object)
    slot_txt = np.array([f"{k // 12:02d}:{k % 12 * 5:02d}:00Z" for k in range(288)], dtype=object)
    ts = day_txt[d] + slot_txt[t]
    v = np.clip(np.nan_to_num(pred_speed[d, t, s].astype(float), nan=0.0), 0, None)
    q = np.clip(np.nan_to_num(pred_flow[d, t, s].astype(float), nan=0.0), 0, None)
    return pd.DataFrame({"panel": panel, "timestamp": ts, "station_id": z["station_ids"][s],
                         "link_id": z["link_ids"][s], "mask_regime": z["regime"][d],
                         "speed_kmh": v, "flow_vph": q}, columns=STATE_COLUMNS)


def check_against_template(panel: str, split: str, frame: pd.DataFrame) -> None:
    """Same key set as task1/<panel>/<split>/sample_submission_state.csv, no duplicates."""
    tpl = pd.read_csv(RELEASE / "task1" / panel / split / "sample_submission_state.csv",
                      usecols=["timestamp", "station_id", "mask_regime"], dtype=str)
    key = ["timestamp", "station_id", "mask_regime"]
    a = pd.MultiIndex.from_frame(frame[key].astype(str))
    b = pd.MultiIndex.from_frame(tpl[key])
    if len(a) != len(b) or not a.is_unique or not a.sort_values().equals(b.sort_values()):
        raise ValueError(f"{panel}/{split}: state rows do not match the template "
                         f"({len(a):,} vs {len(b):,} rows)")


def append_csv(frame: pd.DataFrame, path: Path, first: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, mode="w" if first else "a", header=first, index=False)
