# Rebuild submission A (2026-09-14): Task 1 = M3 v1 CNN, Task 2 = persistence, Task 4 = baseline NNLS.
# Submission B is the same with Task 4 = public_reference.csv (see the last block).
# Run from the project root:  powershell -File scripts\run_submission_A.ps1
# Needs the official-baseline per-task files from scripts\run_official_baselines.ps1 (work\sub_baseline).
$ErrorActionPreference = 'Stop'
$env:PYTHONIOENCODING = 'utf-8'
Remove-Item Env:M3_V2, Env:M3_DNLOSS, Env:M3_WIDTH -ErrorAction SilentlyContinue
$py = '.\.venv\Scripts\python.exe'
$b  = 'work\sub_baseline'

# Task 1: pooled CNN over all 10 panels, every train day, 40 epochs, seed 0 (about 25 min alone on an RTX 4090).
& $py scripts\t1_m3_predict.py --epochs 40 --seed 0 --out-dir work\sub_m3v1

& $py scripts\assemble_submission.py `
    --state work\sub_m3v1\state_validation.csv work\sub_m3v1\state_private.csv `
    --queue "$b\queue_validation.csv" "$b\queue_private.csv" `
    --odme  "$b\t4_validation\baseline_submission.csv" "$b\t4_private\baseline_submission.csv" `
    --output work\sub_A_m3v1\submission.csv

# Submission B: Task 4 from the split's own released counts and prior.
& $py scripts\assemble_submission.py `
    --state work\sub_m3v1\state_validation.csv work\sub_m3v1\state_private.csv `
    --queue "$b\queue_validation.csv" "$b\queue_private.csv" `
    --odme  "$b\t4_validation\public_reference.csv" "$b\t4_private\public_reference.csv" `
    --output work\sub_B_m3v1_t4ref\submission.csv
