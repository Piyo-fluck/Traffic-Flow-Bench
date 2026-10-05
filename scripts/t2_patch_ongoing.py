"""Take a finished upload's queue rows and change only the ONGOING ones.

Candidate Z's onset half must survive byte for byte, so the queue files are reconstructed from Z itself (through
submission_key.csv) and only the rows listed in the ongoing file are overwritten. Writes queue_<split>.csv for
scripts/swap_task_rows.py.

    py scripts/t2_patch_ongoing.py --base work/sub_Z_ttt3600_t2adapt_t4ref/submission.csv \
        --ongoing work/sub_t2occ/ongoing_validation.csv work/sub_t2occ/ongoing_private.csv --out-dir work/sub_t2AH
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tfb.data import RELEASE  # noqa: E402

KEYS = ["window_id", "timestamp", "link_id"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--ongoing", type=Path, nargs="+", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    for chunk in pd.read_csv(RELEASE / "submission_key.csv", usecols=["submission_id", "task"] + KEYS,
                             dtype=str, chunksize=500_000):
        parts.append(chunk[chunk.task == "queue"])
    key = pd.concat(parts, ignore_index=True).drop(columns="task")
    key["submission_id"] = key.submission_id.astype("int64")
    base = pd.read_csv(args.base, usecols=["submission_id", "task", "queue_pred"])
    base = base[base.task == "queue"][["submission_id", "queue_pred"]]
    q = key.merge(base, on="submission_id", how="left", validate="one_to_one")
    assert not q.queue_pred.isna().any(), "the base upload is missing queue rows"
    q["queue_pred"] = q.queue_pred.astype(int)
    new = pd.concat([pd.read_csv(f) for f in args.ongoing], ignore_index=True)
    for c in KEYS:
        new[c] = new[c].astype(str)
    q["timestamp"] = pd.to_datetime(q.timestamp, utc=True)
    new["timestamp"] = pd.to_datetime(new.timestamp, utc=True)
    m = q.merge(new.rename(columns={"queue_pred": "new_pred"}), on=KEYS, how="left", validate="one_to_one")
    hit = m.new_pred.notna()
    ch = int((m.loc[hit, "new_pred"].astype(int) != m.loc[hit, "queue_pred"]).sum())
    m.loc[hit, "queue_pred"] = m.loc[hit, "new_pred"].astype(int)
    print(f"queue rows {len(m):,}; ongoing rows matched {int(hit.sum()):,} of {len(new):,}; values changed {ch}")
    m["split"] = m.window_id.str.extract(r"_(validation|private)_", expand=False)
    assert m.split.notna().all(), "a window_id has no split"
    for s, g in m.groupby("split"):
        out = g[KEYS + ["queue_pred"]].copy()
        out["queue_pred"] = out.queue_pred.astype(int)
        out.to_csv(args.out_dir / f"queue_{s}.csv", index=False)
        print(f"wrote queue_{s}.csv: {len(out):,} rows, share 1 = {out.queue_pred.mean():.4f}")


if __name__ == "__main__":
    main()
