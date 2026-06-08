param(
  [ValidateSet("auto", "cpu", "cuda")]
  [string]$Device = "cpu",

  [string]$OutputDir = ".\validation-comparison-v10-v14-core",

  [string]$Models = "v14",

  [switch]$CompareAll,

  [int]$WindowBatchSize = 64,

  [string]$Thresholds = "0.82,0.90,0.95,0.99,0.999,0.9995",

  [double]$PrimaryThreshold = 0.95,

  [int]$ProgressEveryDocs = 10,

  [int]$ProgressEveryWindows = 500,

  [switch]$Force
)

$ErrorActionPreference = "Stop"

if ($CompareAll) {
  $Models = "v10,v11,v12,v13,v14"
}

$compareArgs = @(
  "run", "python", "compare_v10_v13_validation_suite.py",
  "--suite", "core",
  "--models", $Models,
  "--output-dir", $OutputDir,
  "--thresholds", $Thresholds,
  "--primary-threshold", "$PrimaryThreshold",
  "--window-batch-size", "$WindowBatchSize",
  "--device", $Device,
  "--progress-every-docs", "$ProgressEveryDocs",
  "--progress-every-windows", "$ProgressEveryWindows"
)

if ($Force) {
  $compareArgs += "--force"
}

uv @compareArgs

$summaryJson = Join-Path $OutputDir "comparison-summary.json"
$gateJson = Join-Path $OutputDir "v14-gate-summary.json"
$gateMd = Join-Path $OutputDir "v14-gate-summary.md"

uv run python summarize_validation_comparison.py `
  --comparison-json $summaryJson `
  --model v14 `
  --output-json $gateJson `
  --output-md $gateMd
