# Rebuild submission C' (2026-09-14): T1 = M3 v1 CNN, T2 = rules v2, T4 = public_reference.
# Run from the project root:  powershell -File scripts\run_submission_C2.ps1
# Needs work\sub_m3v1 (scripts\run_submission_A.ps1) and work\sub_baseline (scripts\run_official_baselines.ps1).
$ErrorActionPreference = 'Stop'
$env:PYTHONIOENCODING = 'utf-8'
$py = '.\.venv\Scripts\python.exe'
$b  = 'work\sub_baseline'

# Task 2 rules (notes 09-11). Onset: top-3 links of P_on + 0.5*P_q at T+30, with weekday/weekend
# priors smoothed over +-1 slot. Ongoing: min v/vf over the last 3 history steps < 0.8.
& $py scripts\t2_predict_rules.py --k 3 --alpha 0.5 --smooth 1 --group wt2 --vcut 0.8 --last 3 --out-dir work\sub_t2rules_v2

& $py scripts\assemble_submission.py `
    --state work\sub_m3v1\state_validation.csv work\sub_m3v1\state_private.csv `
    --queue work\sub_t2rules_v2\queue_validation.csv work\sub_t2rules_v2\queue_private.csv `
    --odme  "$b\t4_validation\public_reference.csv" "$b\t4_private\public_reference.csv" `
    --output work\sub_C2_m3v1_t2rulesv2_t4ref\submission.csv
