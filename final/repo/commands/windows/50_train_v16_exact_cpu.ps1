$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root
$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) { $Python = "python" }
& $Python .\scripts\train_mdeberta_ru_prompt_injection_option_b.py `
  --device cpu `
  --student-model .\work\models\mdeberta-ru-prompt-injection-v13-critical-correction-ft `
  --prepared-dataset-dir .\datasets\prepared\training-dataset-v16-critical-recall-restoration-windowed `
  --output-dir .\work\models\mdeberta-ru-prompt-injection-v16-critical-recall-restoration-ft `
  --learning-rate 8e-7 `
  --epochs 1 `
  --distill-weight 0.0 `
  --skip-teacher `
  --last-n-layers 4 `
  --train-batch-size 32 `
  --eval-batch-size 128 `
  --gradient-accumulation-steps 1 `
  --checkpoint-steps 250 `
  --save-total-limit 8 `
  --optim adamw_torch_fused `
  --group-by-length `
  --torch-num-threads 6 `
  --preflight-steps 2
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
