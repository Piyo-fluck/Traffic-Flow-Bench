# Build candidate L = K with Task 1 replaced by the test-time-trained CNNs (note 32).
#   K = F1 + T2 (onset: expected-IoU ratio c 1.0 on H's probs; ongoing: I) (note 31)
#   T1 = the 3 production CNNs fine-tuned per scored month (scripts/t1_ttt_train.py), month base kappa 2 / 10
# Run from the project root:  powershell -File scripts\run_submission_L.ps1
# Windows: python.exe crashed natively (0xc000070a) intermittently under load, so every step retries,
# and inference runs one panel per process.
$ErrorActionPreference = 'Continue'
$env:PYTHONIOENCODING = 'utf-8'
$env:OMP_NUM_THREADS = '2'; $env:ARROW_NUM_THREADS = '2'; $env:MKL_NUM_THREADS = '2'
$py  = '.\.venv\Scripts\python.exe'
$out = 'work\sub_m3ttt'

function Invoke-Retry([scriptblock]$cmd, [int]$tries = 3) {
    for ($i = 1; $i -le $tries; $i++) { & $cmd; if ($LASTEXITCODE -eq 0) { return }; Write-Host "retry $i failed ($LASTEXITCODE)" }
    throw "step failed after $tries tries"
}

# 1. Test-time training, one scored month at a time.
foreach ($s in 'validation','private') {
    Invoke-Retry { & $py scripts\t1_ttt_train.py --split $s --out-dir $out }
}

# 2. Inference with each month's own fine-tuned checkpoints, one panel per process.
$panels = 'D7_I10_E','D7_I10_W','D7_I210_E','D7_I210_W','D7_I405_N','D7_I405_S','D12_I5_N','D12_I5_S','D12_I405_N','D12_I405_S'
foreach ($s in 'validation','private') {
    foreach ($p in $panels) {
        Invoke-Retry { & $py scripts\t1_predict_ckpt.py --ckpt "$out\ttt_${s}_0.pt:64" "$out\ttt_${s}_1.pt:64" "$out\ttt_${s}_2.pt:128" `
            --month-kappa 2 --month-kappa-q 10 --out-dir $out --panels $p --splits $s --npz-only }
    }
}

# 3. State CSVs (torch-free), then L = K with the state rows swapped.
Invoke-Retry { & $py scripts\t1_npz_to_state.py --pred-dir $out }
Invoke-Retry { & $py scripts\swap_task_rows.py --base work\sub_K_ens3month_t2K_t4ref\submission.csv --task state `
    --files "$out\state_validation.csv" "$out\state_private.csv" --out work\sub_L_ttt_t2K_t4ref\submission.csv }

# 4. Check: only speed/flow may differ from K; no NaN.
& $py -c @"
import pandas as pd
a = pd.read_csv(r'work\sub_K_ens3month_t2K_t4ref\submission.csv'); b = pd.read_csv(r'work\sub_L_ttt_t2K_t4ref\submission.csv')
print(a.shape, b.shape, list(a.columns) == list(b.columns), (a.iloc[:, 0].values == b.iloc[:, 0].values).all())
d = (a.drop(columns=a.columns[0]).fillna(-1) != b.drop(columns=b.columns[0]).fillna(-1))
print('L vs K changed per column:', d.sum().to_dict(), '| nan L', int(b.isna().sum().sum()))
"@
Write-Host 'candidate L done'
