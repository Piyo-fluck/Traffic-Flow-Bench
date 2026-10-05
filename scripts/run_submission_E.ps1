# Rebuild submission E (2026-09-14):
#   T1 = mean of 3 CNNs (width 64 seeds 0/1 + width 128 seed 0; notes 03/07)
#   T2 = learned onset ranker + ongoing per-cell model (note 15; alpha 1.2, tau 0.65)
#   T4 = public_reference (NNLS on each split's own released counts; note 06)
# Run from the project root:  powershell -File scripts\run_submission_E.ps1
# Needs work\sub_baseline (scripts\run_official_baselines.ps1).
$ErrorActionPreference = 'Stop'
$env:PYTHONIOENCODING = 'utf-8'
Remove-Item Env:M3_V2, Env:M3_DNLOSS, Env:M3_WIDTH -ErrorAction SilentlyContinue
$py = '.\.venv\Scripts\python.exe'
$b  = 'work\sub_baseline'

# Task 1: three production CNNs (about 25 + 25 + 55 min on an RTX 4090), then their mean.
& $py scripts\t1_m3_predict.py --epochs 40 --seed 0 --out-dir work\sub_m3v1
& $py scripts\t1_m3_predict.py --epochs 40 --seed 1 --out-dir work\sub_m3v1_s1
$env:M3_WIDTH = '128'
& $py scripts\t1_m3_predict.py --epochs 40 --seed 0 --out-dir work\sub_m3w128
Remove-Item Env:M3_WIDTH
& $py scripts\t1_ensemble_predict.py --runs work\sub_m3v1 work\sub_m3v1_s1 work\sub_m3w128 --out-dir work\sub_m3ens3

# Task 2: learned models trained on every train day.
& $py scripts\t2_m1_predict.py --rounds-on 600 --rounds-og 3300 --alpha 1.2 --tau 0.65 --out-dir work\sub_t2m1

& $py scripts\assemble_submission.py `
    --state work\sub_m3ens3\state_validation.csv work\sub_m3ens3\state_private.csv `
    --queue work\sub_t2m1\queue_validation.csv work\sub_t2m1\queue_private.csv `
    --odme  "$b\t4_validation\public_reference.csv" "$b\t4_private\public_reference.csv" `
    --output work\sub_E_ens3_t2m1_t4ref\submission.csv
