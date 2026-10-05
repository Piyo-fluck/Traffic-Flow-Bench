"""Replace one task's values in an assembled upload with the values of per-task files (lightweight).

Avoids the full string merge of scripts/assemble_submission.py. That merge crashed natively under load on
Windows (0xc000070a). Only that task's rows of submission_key.csv are kept, streamed in chunks. The base
upload is read with numeric columns, and the new values are joined by submission_id.

    py scripts/swap_task_rows.py --base work/sub_F1/submission.csv --task queue \
        --files work/sub_t2m1_month/queue_validation.csv work/sub_t2m1_month/queue_private.csv --out work/sub_F2/submission.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tfb.data import RELEASE  # noqa: E402

KEYS = {"queue": ["window_id", "timestamp", "link_id"], "odme": ["panel", "departure_time", "path_id"],
        "state": ["panel", "timestamp", "station_id", "link_id", "mask_regime"]}
VALUES = {"queue": ["queue_pred"], "odme": ["path_flow"], "state": ["speed_kmh", "flow_vph"]}


def norm(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for c in cols:
        out[c] = pd.to_datetime(out[c], utc=True) if c == "timestamp" else out[c].astype(str)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--task", choices=list(KEYS), required=True)
    ap.add_argument("--files", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    keys, vals = KEYS[args.task], VALUES[args.task]
    parts = []
    for chunk in pd.read_csv(RELEASE / "submission_key.csv", usecols=["submission_id", "task"] + keys,
                             dtype=str, chunksize=500_000):
        parts.append(chunk[chunk.task == args.task])
    key = norm(pd.concat(parts, ignore_index=True), keys)
    new = norm(pd.concat([pd.read_csv(f) for f in args.files], ignore_index=True), keys)
    m = key.merge(new[keys + vals], on=keys, how="left", validate="one_to_one")
    if m[vals].isna().any().any():
        raise SystemExit(f"{int(m[vals].isna().any(axis=1).sum())} {args.task} rows have no new value")
    m["submission_id"] = m.submission_id.astype(np.int64)
    base = pd.read_csv(args.base, dtype={"task": "category"})
    idx = pd.Index(base.submission_id.to_numpy())
    pos = idx.get_indexer(m.submission_id.to_numpy())
    if (pos < 0).any():
        raise SystemExit("some key ids are missing from the base upload")
    if not (base.task.to_numpy()[pos] == args.task).all():
        raise SystemExit("task label mismatch between key and base")
    changed = 0
    for v in vals:
        newv = m[v].to_numpy()
        if args.task == "queue":
            newv = newv.astype(int)
            if not np.isin(newv, [0, 1]).all():
                raise SystemExit("queue_pred outside {0, 1}")
        changed += int((base[v].to_numpy()[pos] != newv).sum())
        col = base[v].to_numpy().copy()
        col[pos] = newv
        base[v] = col
    base["queue_pred"] = base.queue_pred.astype(int)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(args.out, index=False)
    print(f"replaced {len(m):,} {args.task} rows ({changed:,} values changed); wrote {args.out} "
          f"({args.out.stat().st_size / 2**20:.0f} MB); rows {len(base):,}, share queued {base.loc[base.task == 'queue', 'queue_pred'].mean():.4f}")


if __name__ == "__main__":
    main()
