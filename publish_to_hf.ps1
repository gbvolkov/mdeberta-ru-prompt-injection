<#
Prepare and upload the trained Russian prompt-injection detector to Hugging Face.

Examples:
  .\publish_to_hf.ps1 -RepoId "YOUR_HF_USERNAME/mdeberta-ru-prompt-injection" -SkipUpload
  .\publish_to_hf.ps1 -RepoId "YOUR_HF_USERNAME/mdeberta-ru-prompt-injection"

Before uploading:
  hf auth login
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$RepoId,

    [string]$SourceDir,

    [string]$StagingDir,

    [string]$ReadmePath,

    [switch]$SkipUpload,

    [switch]$IncludeTrainingScript,

    [switch]$IncludeTrainingArgs
)

$ErrorActionPreference = "Stop"

# The HF CLI is Python-based and may print Unicode status symbols. Force UTF-8
# so Windows consoles do not fail with charmap encoding errors.
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [System.Text.UTF8Encoding]::new()
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

if ([string]::IsNullOrWhiteSpace($SourceDir)) {
    $SourceDir = Join-Path $PSScriptRoot "mdeberta-ru-prompt-injection-35-65"
}
if ([string]::IsNullOrWhiteSpace($StagingDir)) {
    $StagingDir = Join-Path $PSScriptRoot "hf-upload"
}
if ([string]::IsNullOrWhiteSpace($ReadmePath)) {
    $ReadmePath = Join-Path $PSScriptRoot "README.md"
}

if ($RepoId -match "YOUR(_HF)?_USERNAME") {
    throw "Replace the placeholder namespace in -RepoId with your real Hugging Face username or organization. Example: volko/mdeberta-ru-prompt-injection"
}
if ($RepoId -notmatch "^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$") {
    throw "RepoId must look like 'namespace/model-name'. Got: $RepoId"
}
function Resolve-ExistingPath([string]$Path, [string]$Description) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "$Description does not exist: $Path"
    }
    return (Resolve-Path -LiteralPath $Path).Path
}

function Get-SafeStagingPath([string]$Path) {
    $projectRoot = (Resolve-Path -LiteralPath $PSScriptRoot).Path
    $fullPath = [System.IO.Path]::GetFullPath($Path)

    if ($fullPath -eq $projectRoot) {
        throw "StagingDir cannot be the project root: $fullPath"
    }
    if (-not $fullPath.StartsWith($projectRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "For safety, StagingDir must be inside the project root. Got: $fullPath"
    }
    return $fullPath
}

function Copy-IfExists([string]$FromDir, [string]$Pattern, [string]$ToDir) {
    $files = Get-ChildItem -LiteralPath $FromDir -File -Filter $Pattern -ErrorAction SilentlyContinue
    foreach ($file in $files) {
        Copy-Item -LiteralPath $file.FullName -Destination (Join-Path $ToDir $file.Name) -Force
    }
    return @($files).Count
}

function Require-StagedFile([string]$Pattern, [string]$Message) {
    $matches = @(Get-ChildItem -LiteralPath $script:StagePath -File -Filter $Pattern -ErrorAction SilentlyContinue)
    if ($matches.Count -eq 0) {
        throw $Message
    }
}
function Invoke-Hf {
    param(
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Arguments
    )

    & hf @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "hf $($Arguments -join ' ') failed with exit code $LASTEXITCODE"
    }
}

$SourcePath = Resolve-ExistingPath $SourceDir "SourceDir"
$CardPath = Resolve-ExistingPath $ReadmePath "ReadmePath"
$StagePath = Get-SafeStagingPath $StagingDir

Write-Host "Source model dir : $SourcePath"
Write-Host "Model card       : $CardPath"
Write-Host "Staging dir      : $StagePath"
Write-Host "Repo id          : $RepoId"

if (Test-Path -LiteralPath $StagePath) {
    Write-Host "Removing previous staging directory..."
    Remove-Item -LiteralPath $StagePath -Recurse -Force
}
New-Item -ItemType Directory -Path $StagePath | Out-Null

Copy-Item -LiteralPath $CardPath -Destination (Join-Path $StagePath "README.md") -Force

$patterns = @(
    "config.json",
    "model.safetensors",
    "model-*.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model-*.bin",
    "pytorch_model.bin.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "spm.model",
    "sentencepiece.bpe.model",
    "tokenizer.model",
    "vocab.*",
    "merges.txt",
    "threshold_recommendations.json",
    "student_eval_metrics.json",
    "teacher_baseline_metrics.json",
    "run_config.json",
    "inference_example.py"
)

foreach ($pattern in $patterns) {
    [void](Copy-IfExists $SourcePath $pattern $StagePath)
}

if ($IncludeTrainingArgs) {
    [void](Copy-IfExists $SourcePath "training_args.bin" $StagePath)
}

if ($IncludeTrainingScript) {
    $trainingScript = Join-Path $PSScriptRoot "train_mdeberta_ru_prompt_injection_option_b.py"
    if (Test-Path -LiteralPath $trainingScript) {
        Copy-Item -LiteralPath $trainingScript -Destination (Join-Path $StagePath "train_mdeberta_ru_prompt_injection_option_b.py") -Force
    }
}

$sampleScript = Join-Path $PSScriptRoot "sample.py"
if (Test-Path -LiteralPath $sampleScript) {
    Copy-Item -LiteralPath $sampleScript -Destination (Join-Path $StagePath "sample.py") -Force
}

Require-StagedFile "README.md" "README.md was not staged."
Require-StagedFile "config.json" "config.json was not found in source model directory."

$modelFiles = @(
    Get-ChildItem -LiteralPath $StagePath -File -Filter "model.safetensors" -ErrorAction SilentlyContinue
    Get-ChildItem -LiteralPath $StagePath -File -Filter "model-*.safetensors" -ErrorAction SilentlyContinue
    Get-ChildItem -LiteralPath $StagePath -File -Filter "pytorch_model.bin" -ErrorAction SilentlyContinue
    Get-ChildItem -LiteralPath $StagePath -File -Filter "pytorch_model-*.bin" -ErrorAction SilentlyContinue
)
if (@($modelFiles).Count -eq 0) {
    throw "No model weight file was staged. Expected model.safetensors or pytorch_model.bin."
}

$tokenizerFiles = @(
    Get-ChildItem -LiteralPath $StagePath -File -Filter "tokenizer.json" -ErrorAction SilentlyContinue
    Get-ChildItem -LiteralPath $StagePath -File -Filter "spm.model" -ErrorAction SilentlyContinue
    Get-ChildItem -LiteralPath $StagePath -File -Filter "vocab.*" -ErrorAction SilentlyContinue
)
if (@($tokenizerFiles).Count -eq 0) {
    throw "No tokenizer file was staged. Expected tokenizer.json, spm.model, or vocab.*."
}

Write-Host "`nStaged files:"
Get-ChildItem -LiteralPath $StagePath -File | Sort-Object Name | ForEach-Object {
    $mb = [Math]::Round($_.Length / 1MB, 2)
    Write-Host ("  {0} ({1} MB)" -f $_.Name, $mb)
}
$totalBytes = (Get-ChildItem -LiteralPath $StagePath -File | Measure-Object -Property Length -Sum).Sum
$totalGb = [Math]::Round($totalBytes / 1GB, 3)
Write-Host "Total staged size: $totalGb GB"

Write-Host "`nExcluded by construction: checkpoint-*/, stage-cache/, preflight-check*/, optimizer.pt, scheduler.pt, rng_state.pth."

$uploadCommand = "hf upload $RepoId `"$StagePath`" . --commit-message `"Upload mDeBERTa Russian prompt-injection detector`""
if ($SkipUpload) {
    Write-Host "`nSkipUpload set. Review staged files, then run:"
    Write-Host "  $uploadCommand"
    exit 0
}

if (-not (Get-Command hf -ErrorAction SilentlyContinue)) {
    throw "The 'hf' CLI was not found. Install/update huggingface_hub or run: uvx hf auth login"
}

Write-Host "`nChecking Hugging Face authentication..."
Invoke-Hf auth whoami

Write-Host "`nUploading to Hugging Face..."
Invoke-Hf upload $RepoId $StagePath "." --commit-message "Upload mDeBERTa Russian prompt-injection detector"

Write-Host "`nDone. Model repo: https://huggingface.co/$RepoId"

