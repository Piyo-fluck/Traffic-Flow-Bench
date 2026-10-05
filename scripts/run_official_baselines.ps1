# Rebuild the official-baseline upload from scratch (submitted 2026-09-14 as ref 56225746).
# Run from the project root:  powershell -File scripts\run_official_baselines.ps1
$ErrorActionPreference = 'Stop'
$env:PYTHONIOENCODING = 'utf-8'
$py  = '.\.venv\Scripts\python.exe'
$src = 'trafficflowbench-public\src'
$rel = 'data\kaggle_public'
$out = 'work\sub_baseline'
New-Item -ItemType Directory -Force $out | Out-Null

foreach ($s in 'validation', 'private') {
    & $py "$src\task1\build_task1_baseline_submission.py" --release-root $rel --split $s --output "$out\state_$s.csv"
    & $py "$src\task2\build_task2_persistence_submission.py" --release-root $rel --split $s --output "$out\queue_$s.csv"
    & $py "$src\task4\build_task4_odme_artifacts.py" --release-root $rel --split $s --output-root "$out\t4_$s"
}

& $py scripts\assemble_submission.py `
    --state "$out\state_validation.csv" "$out\state_private.csv" `
    --queue "$out\queue_validation.csv" "$out\queue_private.csv" `
    --odme  "$out\t4_validation\baseline_submission.csv" "$out\t4_private\baseline_submission.csv" `
    --output "$out\submission.csv"
