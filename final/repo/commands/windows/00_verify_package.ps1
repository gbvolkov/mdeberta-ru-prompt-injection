$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root
$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) { $Python = "python" }
& $Python .\scripts\verify_v16_package.py --root $Root
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
if (Test-Path .\manifests\SHA256SUMS.txt) {
  & $Python .\scripts\verify_sha256_manifest.py --root $Root
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
