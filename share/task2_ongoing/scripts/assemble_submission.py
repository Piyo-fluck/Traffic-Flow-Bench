"""Assemble one Kaggle upload from per-task, per-split files and validate it.

The upload covers both scored months, so each task arrives as one file per
split. They are concatenated as raw text (timestamps must stay byte-for-byte),
merged onto submission_key.csv by the organizer's merge_submissions.py, checked
row-for-row against sample_submission.csv, and written with rounded values.

    py scripts/assemble_submission.py \
      --state work/sub/state_validation.csv work/sub/state_private.csv \
      --queue work/sub/queue_validation.csv work/sub/queue_private.csv \
      --odme  work/sub/t4_validation/baseline_submission.csv work/sub/t4_private/baseline_submission.csv \
      --output work/sub/submission.csv
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RELEASE = ROOT / "data" / "kaggle_public"
MERGE = ROOT / "trafficflowbench-public" / "src" / "merge_submissions.py"
VALUE_COLUMNS = ["speed_kmh", "flow_vph", "queue_pred", "path_flow"]


def concat_csv(paths: list[Path], out: Path) -> None:
    """Concatenate CSVs that share one header, without parsing them."""
    first_header = None
    with open(out, "wb") as w:
        for p in paths:
            with open(p, "rb") as r:
                header = r.readline()
                if first_header is None:
                    first_header = header
                    w.write(header)
                elif header != first_header:
                    raise ValueError(f"header of {p} differs from {paths[0]}")
                shutil.copyfileobj(r, w, 1 << 24)


def merge(state: Path, queue: Path, odme: Path, key: Path, out: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(MERGE), "--state", str(state), "--queue", str(queue),
         "--odme", str(odme), "--key", str(key), "--output", str(out)],
        capture_output=True, text=True,
    )
    print(result.stdout.replace("\r", "\n").strip().splitlines()[-1] if result.stdout else "")
    if result.returncode != 0:
        raise SystemExit(f"merge_submissions.py failed:\n{result.stderr}")
    if "WARNING" in result.stdout:
        raise SystemExit(f"merge left unmatched cells:\n{result.stdout}")


def validate(sub: pd.DataFrame, sample: Path) -> None:
    template = pd.read_csv(sample, usecols=["submission_id", "task"])
    problems = []
    if len(sub) != len(template):
        problems.append(f"row count {len(sub):,} != template {len(template):,}")
    else:
        if not np.array_equal(sub.submission_id.to_numpy(), template.submission_id.to_numpy()):
            problems.append("submission_id order differs from the template")
        if not np.array_equal(sub.task.to_numpy(), template.task.to_numpy()):
            problems.append("task column differs from the template")
    if sub.submission_id.duplicated().any():
        problems.append("duplicate submission_id")
    values = sub[VALUE_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        problems.append("non-finite or empty value present")
    state, queue, odme = (sub.task == t for t in ("state", "queue", "odme"))
    if not sub.loc[queue, "queue_pred"].isin([0, 1]).all():
        problems.append("queue_pred outside {0, 1}")
    if (sub.loc[odme, "path_flow"] < 0).any():
        problems.append("negative path_flow")
    if (sub.loc[state, ["speed_kmh", "flow_vph"]] < 0).any().any():
        problems.append("negative speed or flow")
    if problems:
        raise SystemExit("validation failed:\n  - " + "\n  - ".join(problems))

    print("rows by task:", sub.task.value_counts().to_dict())
    s = sub.loc[state]
    print(f"state: speed mean {s.speed_kmh.mean():.2f} km/h, flow mean {s.flow_vph.mean():.1f} vph, "
          f"rows with speed==0: {(s.speed_kmh == 0).sum():,}, flow==0: {(s.flow_vph == 0).sum():,}")
    print(f"queue: share predicted queued {sub.loc[queue, 'queue_pred'].mean():.4f}")
    print(f"odme: path_flow total {sub.loc[odme, 'path_flow'].sum():,.1f}, "
          f"zero paths {(sub.loc[odme, 'path_flow'] == 0).sum():,}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", type=Path, nargs="+", required=True)
    ap.add_argument("--queue", type=Path, nargs="+", required=True)
    ap.add_argument("--odme", type=Path, nargs="+", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--release-root", type=Path, default=RELEASE)
    args = ap.parse_args()

    work = args.output.parent / "_assemble"
    work.mkdir(parents=True, exist_ok=True)
    combined = {}
    for task, paths in (("state", args.state), ("queue", args.queue), ("odme", args.odme)):
        combined[task] = work / f"{task}_all.csv"
        concat_csv(paths, combined[task])
    raw = work / "merged_raw.csv"
    merge(combined["state"], combined["queue"], combined["odme"],
          args.release_root / "submission_key.csv", raw)

    sub = pd.read_csv(raw)
    validate(sub, args.release_root / "sample_submission.csv")
    # Full float precision triples the file for no score change.
    sub["speed_kmh"] = sub.speed_kmh.round(3)
    sub["flow_vph"] = sub.flow_vph.round(2)
    sub["path_flow"] = sub.path_flow.round(4)
    sub["queue_pred"] = sub.queue_pred.astype(int)
    sub.to_csv(args.output, index=False)
    print(f"wrote {args.output} ({args.output.stat().st_size / 2**20:.0f} MB)")


if __name__ == "__main__":
    main()
