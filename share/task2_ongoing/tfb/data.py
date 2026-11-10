"""Dense (day, time-of-day, station) arrays for one panel and split, cached as .npz.

Stations are ordered along the corridor by the LWR topology's order_index, so
axis 2 runs upstream to downstream and neighbouring indices are neighbouring
detectors. Every array shares that axis order.

Keys (D days, T=288 slots, S stations):
  dates (D,) str, weekday (D,), regime (D,) str
  station_ids, link_ids (S,) str; milepost, lanes, free_speed (S,); has_sensor (S,)
  obs_speed, obs_flow, obs_occ (D,T,S) float32   masked layer as released (NaN = blank)
  pct (D,T,S) uint8, elig (D,T,S) bool           quality fields, identical in both layers
  target (D,T,S) bool                            Task 1 rows of the published template
  train only: true_speed, true_flow, true_occ    unmasked layer
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RELEASE = ROOT / "data" / "kaggle_public"
CACHE = ROOT / "work" / "cache"
T = 288
REGIMES = ("R1", "R2", "R3")
PANELS = ["D7_I10_E", "D7_I10_W", "D7_I210_E", "D7_I210_W", "D7_I405_N", "D7_I405_S",
          "D12_I5_N", "D12_I5_S", "D12_I405_N", "D12_I405_S"]
COLS = ["date", "timestamp", "station_id", "speed_kmh", "flow_vph", "occupancy",
        "pct_observed", "is_score_eligible"]


def _files(panel: str, split: str, layer: str) -> list[Path]:
    base = RELEASE / "corridors" / panel / split / layer
    return sorted(base.glob("**/*.parquet"), key=lambda p: p.name)


def _stations(panel: str, first_file: Path) -> pd.DataFrame:
    net = RELEASE / "corridors" / panel / "network"
    topo = pd.read_csv(net / "lwr_mainline_topology.csv", dtype={"link_id": str})
    links = pd.read_csv(net / "links.csv", dtype={"link_id": str})
    fd = pd.read_csv(net / "fd_parameters.csv", dtype={"link_id": str})
    st = (pd.read_parquet(first_file, columns=["station_id", "link_id", "milepost"])
          .astype({"station_id": str, "link_id": str})
          .drop_duplicates(["station_id", "link_id"]))
    st = st.merge(topo[["link_id", "order_index", "has_sensor"]], on="link_id", how="left")
    if st.order_index.isna().any():
        raise ValueError(f"{panel}: stations on links missing from the topology")
    st = st.sort_values(["order_index", "milepost"]).reset_index(drop=True)
    # Lane count exactly as the scorer resolves it: links.csv, overridden by
    # fd_parameters.csv (by link, then by station where that column exists).
    lanes = dict(zip(links.link_id, links.lanes))
    lanes.update(zip(fd.link_id, fd.lanes))
    st["lanes"] = st.link_id.map(lanes).astype(float)
    if "station_id" in fd.columns:
        st["lanes"] = st.station_id.map(dict(zip(fd.station_id.astype(str), fd.lanes))).fillna(st.lanes)
    st["lanes"] = st.lanes.where((st.lanes >= 1) & (st.lanes <= 16), 1.0)
    st["free_speed"] = st.link_id.map(dict(zip(links.link_id, links.free_speed_kmh))).astype(float)
    return st


def _slots(frame: pd.DataFrame, station_index: pd.Index,
           day: np.ndarray | int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Time slot and station index per row; rejects unknown stations and duplicate cells."""
    ts = frame["timestamp"].astype(str)
    t = ts.str.slice(11, 13).astype(int).to_numpy() * 12 + ts.str.slice(14, 16).astype(int).to_numpy() // 5
    s = station_index.get_indexer(frame["station_id"].astype(str))
    if (s < 0).any():
        raise ValueError("unknown station id")
    cell = (np.asarray(day, dtype=np.int64) * T + t) * len(station_index) + s
    if len(np.unique(cell)) != len(frame):
        raise ValueError("duplicate (day, timestamp, station) rows")
    return t, s


def _read_layer(files: list[Path], station_index: pd.Index, with_regime: bool) -> dict:
    D, S = len(files), len(station_index)
    out = {k: np.full((D, T, S), np.nan, np.float32) for k in ("speed", "flow", "occ")}
    out["pct"] = np.zeros((D, T, S), np.uint8)
    out["elig"] = np.zeros((D, T, S), bool)
    dates, regimes = [], []
    for d, f in enumerate(files):
        x = pd.read_parquet(f, columns=COLS + (["mask_regime"] if with_regime else []))
        if x["date"].nunique() != 1:
            raise ValueError(f"{f.name}: more than one date in a daily partition")
        t, s = _slots(x, station_index)
        out["speed"][d, t, s] = x.speed_kmh.to_numpy(np.float32)
        out["flow"][d, t, s] = x.flow_vph.to_numpy(np.float32)
        out["occ"][d, t, s] = x.occupancy.to_numpy(np.float32)
        out["pct"][d, t, s] = x.pct_observed.to_numpy().astype(np.uint8)
        out["elig"][d, t, s] = x.is_score_eligible.to_numpy().astype(bool)
        dates.append(str(x["date"].iloc[0]))
        regimes.append(str(x["mask_regime"].iloc[0]) if with_regime else "")
    out["dates"] = np.array(dates)
    out["regime"] = np.array(regimes)
    return out


def _targets(panel: str, split: str, dates: np.ndarray, regime: np.ndarray,
             station_index: pd.Index) -> np.ndarray:
    tpl = pd.read_csv(RELEASE / "task1" / panel / split / "sample_submission_state.csv",
                      usecols=["timestamp", "station_id", "mask_regime"], dtype=str)
    d = pd.Index(dates).get_indexer(tpl.timestamp.str.slice(0, 10))
    if (d < 0).any():
        raise ValueError("template row on a date with no partition")
    t, s = _slots(tpl, station_index, day=d)
    if (regime[d] != tpl.mask_regime.to_numpy()).any():
        raise ValueError("template regime disagrees with the partition's regime")
    target = np.zeros((len(dates), T, len(station_index)), bool)
    target[d, t, s] = True
    return target


def build(panel: str, split: str) -> dict:
    masked = _files(panel, split, "mainline_states_masked")
    st = _stations(panel, masked[0])
    idx = pd.Index(st.station_id)
    m = _read_layer(masked, idx, with_regime=True)
    arrays = {
        "dates": m["dates"], "regime": m["regime"],
        "weekday": pd.to_datetime(m["dates"]).weekday.to_numpy().astype(np.int8),
        "station_ids": st.station_id.to_numpy().astype(str), "link_ids": st.link_id.to_numpy().astype(str),
        "milepost": st.milepost.to_numpy(float), "lanes": st.lanes.to_numpy(float),
        "free_speed": st.free_speed.to_numpy(float), "has_sensor": st.has_sensor.to_numpy().astype(np.int8),
        "obs_speed": m["speed"], "obs_flow": m["flow"], "obs_occ": m["occ"],
        "pct": m["pct"], "elig": m["elig"],
        "target": _targets(panel, split, m["dates"], m["regime"], idx),
    }
    truth = _files(panel, split, "mainline_states")
    if truth:
        by_name = {p.name: p for p in truth}
        tr = _read_layer([by_name[p.name] for p in masked], idx, with_regime=False)
        if not (tr["dates"] == m["dates"]).all():
            raise ValueError("truth and masked layers are not aligned by date")
        arrays.update(true_speed=tr["speed"], true_flow=tr["flow"], true_occ=tr["occ"])
    # Days sorted by date, not by the regime directory they were published in.
    order = np.argsort(arrays["dates"])
    for k, v in arrays.items():
        if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == len(order) and k not in (
                "station_ids", "link_ids", "milepost", "lanes", "free_speed", "has_sensor"):
            arrays[k] = v[order]
    return arrays


def load(panel: str, split: str, refresh: bool = False) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{panel}_{split}.npz"
    if path.exists() and not refresh:
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    arrays = build(panel, split)
    np.savez(path, **arrays)
    return arrays
