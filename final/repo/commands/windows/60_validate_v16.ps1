param([ValidateSet("cpu", "cuda", "auto")][string]$Device = "cuda")
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root
$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) { $Python = "python" }
& $Python .\scripts\run_core_validation.py `
  --v13-model .\work\models\mdeberta-ru-prompt-injection-v13-critical-correction-ft `
  --v16-model .\work\models\mdeberta-ru-prompt-injection-v16-critical-recall-restoration-ft `
  --output-dir .\work\validation-v13-v16-core `
  --device $Device `
  --window-batch-size 64
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
