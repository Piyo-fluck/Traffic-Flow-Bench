# Rebuild submissions F1 and F2 (prepared 2026-09-14 for the 2026-09-15 slots). Notes 17-18.
#   F1 = E with T1 month base: mean of 3 CNNs on base = train table + month deviation (kappa speed 2, flow 10)
#   F2 = F1 with T2 month-past priors: saved T2 models, P_q from the scored month's past days (kappa 4)
#   T4 = public_reference in both.
# Run from the project root:  powershell -File scripts\run_submission_F.ps1
# Needs: work\sub_m3v1, work\sub_m3v1_s1, work\sub_m3w128 (CNN checkpoints), work\sub_t2m1 (T2 models + queue
# files), work\sub_baseline (T4). Build them with scripts\run_submission_E.ps1.
# Windows: python.exe crashed natively (0xc000070a) intermittently under load, so every step retries.
$ErrorActionPreference = 'Continue'
$env:PYTHONIOENCODING = 'utf-8'
$env:OMP_NUM_THREADS = '2'; $env:ARROW_NUM_THREADS = '2'; $env:MKL_NUM_THREADS = '2'
$py = '.\.venv\Scripts\python.exe'
$b  = 'work\sub_baseline'

function Invoke-Retry([scriptblock]$cmd, [int]$tries = 3) {
    for ($i = 1; $i -le $tries; $i++) { & $cmd; if ($LASTEXITCODE -eq 0) { return }; Write-Host "retry $i failed ($LASTEXITCODE)" }
    throw "step failed after $tries tries"
}

# T1 with month base: tables (torch-free), then one inference process per panel, then CSVs.
Invoke-Retry { & $py scripts\t1_tables_cache.py }
foreach ($p in 'D7_I10_E','D7_I10_W','D7_I210_E','D7_I210_W','D7_I405_N','D7_I405_S','D12_I5_N','D12_I5_S','D12_I405_N','D12_I405_S') {
    Invoke-Retry { & $py scripts\t1_predict_ckpt.py --ckpt work\sub_m3v1\t1_m3_seed0.pt:64 work\sub_m3v1_s1\t1_m3_seed1.pt:64 `
        work\sub_m3w128\t1_m3_seed0.pt:128 --month-kappa 2 --month-kappa-q 10 --out-dir work\sub_m3ens3_month --panels $p --npz-only }
}
Invoke-Retry { & $py scripts\t1_npz_to_state.py --pred-dir work\sub_m3ens3_month }

# F1
Invoke-Retry { & $py scripts\assemble_submission.py `
    --state work\sub_m3ens3_month\state_validation.csv work\sub_m3ens3_month\state_private.csv `
    --queue work\sub_t2m1\queue_validation.csv work\sub_t2m1\queue_private.csv `
    --odme  "$b\t4_validation\public_reference.csv" "$b\t4_private\public_reference.csv" `
    --output work\sub_F1_ens3month_t2m1_t4ref\submission.csv }

# T2 with month-past priors (saved models), then F2 = F1 with the queue rows swapped.
Invoke-Retry { & $py scripts\t2_m1_predict.py --load-models work\sub_t2m1 --month-kappa 4 --alpha 1.2 --tau 0.65 --out-dir work\sub_t2m1_month }
Invoke-Retry { & $py scripts\swap_task_rows.py --base work\sub_F1_ens3month_t2m1_t4ref\submission.csv --task queue `
    --files work\sub_t2m1_month\queue_validation.csv work\sub_t2m1_month\queue_private.csv `
    --out work\sub_F2_ens3month_t2m1month_t4ref\submission.csv }
