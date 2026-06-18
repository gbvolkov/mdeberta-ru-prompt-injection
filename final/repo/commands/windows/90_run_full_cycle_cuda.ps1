$ErrorActionPreference = "Stop"
$Scripts = @(
  "10_train_v10_cuda.ps1",
  "20_train_v11_cuda.ps1",
  "30_build_v12_dataset.ps1",
  "31_train_v12_cuda.ps1",
  "40_train_v13_cuda.ps1",
  "51_train_v16_cuda.ps1",
  "60_validate_v16.ps1"
)
foreach ($Script in $Scripts) {
  Write-Host "[cycle] starting $Script"
  & (Join-Path $PSScriptRoot $Script)
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
  Write-Host "[cycle] completed $Script"
}
