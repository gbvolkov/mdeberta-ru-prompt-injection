<#
Prepare and upload the trained Russian prompt-injection detector to Hugging Face.

Examples:
  .\publish_to_hf.ps1 -RepoId "YOUR_HF_USERNAME/mdeberta-ru-prompt-injection" -SkipUpload
  .\publish_to_hf.ps1 -RepoId "YOUR_HF_USERNAME/mdeberta-ru-prompt-injection"
  powershell -NoProfile -ExecutionPolicy Bypass -File .\publish_to_hf.ps1 -RepoId "YOUR_HF_USERNAME/mdeberta-ru-prompt-injection" -SkipUpload

Defaults publish the v16 critical recall restoration fine-tune:
  .\mdeberta-ru-prompt-injection-v16-critical-recall-restoration-ft

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

    [string]$StressResultPath,

    [string]$DatasetValidatorReportPath,

    [string]$DatasetBuildReportPath,

    [string]$CriticalValidationSummaryPath,

    [string]$FalseNegativeReportPath,

    [string]$ValidationComparisonSummaryPath,

    [string]$ValidationGateSummaryPath,

    [string]$ValidationCorpusReportPath,

    [double]$RecommendedThreshold = 0.99,

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

$DefaultModelVersion = "v16-critical-recall-restoration-ft"
$DefaultValidationName = "v13-critical-ru-validation"
$ThresholdLabel = $RecommendedThreshold.ToString("0.######", [System.Globalization.CultureInfo]::InvariantCulture)

if ([string]::IsNullOrWhiteSpace($SourceDir)) {
    $SourceDir = Join-Path $PSScriptRoot "mdeberta-ru-prompt-injection-$DefaultModelVersion"
}
if ([string]::IsNullOrWhiteSpace($StagingDir)) {
    $StagingDir = Join-Path $PSScriptRoot "hf-upload-$DefaultModelVersion"
}
if ([string]::IsNullOrWhiteSpace($ReadmePath)) {
    $ReadmePath = Join-Path $PSScriptRoot "MODEL_CARD_V16.md"
}
if ([string]::IsNullOrWhiteSpace($StressResultPath)) {
    $preferredStressResult = Join-Path $PSScriptRoot "stress-$DefaultModelVersion-on-$DefaultValidationName-threshold-$ThresholdLabel.json"
    $legacyStressResult = Join-Path $PSScriptRoot "stress-$DefaultModelVersion-on-$DefaultValidationName-threshold-0.5.json"
    if (Test-Path -LiteralPath $preferredStressResult) {
        $StressResultPath = $preferredStressResult
    }
    elseif (Test-Path -LiteralPath $legacyStressResult) {
        # Keep compatibility with older locally named stress artifacts.
        $StressResultPath = $legacyStressResult
    }
}
if ([string]::IsNullOrWhiteSpace($DatasetValidatorReportPath)) {
    $DatasetValidatorReportPath = Join-Path $PSScriptRoot "training-dataset-v16-critical-recall-restoration-windowed-analysis.json"
}
if ([string]::IsNullOrWhiteSpace($DatasetBuildReportPath)) {
    $DatasetBuildReportPath = Join-Path $PSScriptRoot "training-dataset-v16-critical-recall-restoration-windowed-report.json"
}
if ([string]::IsNullOrWhiteSpace($CriticalValidationSummaryPath)) {
    $CriticalValidationSummaryPath = Join-Path $PSScriptRoot "validation-comparison-v13-v16-core\v16\v13_critical_ru-summary.json"
}
if ([string]::IsNullOrWhiteSpace($FalseNegativeReportPath)) {
    $FalseNegativeReportPath = ""
}
if ([string]::IsNullOrWhiteSpace($ValidationComparisonSummaryPath)) {
    $ValidationComparisonSummaryPath = Join-Path $PSScriptRoot "validation-comparison-v13-v16-core\comparison-summary.json"
}
if ([string]::IsNullOrWhiteSpace($ValidationGateSummaryPath)) {
    $ValidationGateSummaryPath = ""
}
if ([string]::IsNullOrWhiteSpace($ValidationCorpusReportPath)) {
    $ValidationCorpusReportPath = Join-Path $PSScriptRoot "validation-comparison-v13-v16-core\validation-suite-manifest.json"
}

if ($RepoId -match "YOUR(_HF)?_USERNAME") {
    throw "Replace the placeholder namespace in -RepoId with your real Hugging Face username or organization. Example: gbv/mdeberta-ru-prompt-injection"
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

function Copy-ProjectFileIfExists([string]$FileName, [string]$ToDir) {
    $filePath = Join-Path $PSScriptRoot $FileName
    if (Test-Path -LiteralPath $filePath) {
        Copy-Item -LiteralPath $filePath -Destination (Join-Path $ToDir $FileName) -Force
        return 1
    }
    return 0
}

function Copy-ArtifactIfExists([string]$Path, [string]$ToDir, [string]$StagedName) {
    if ([string]::IsNullOrWhiteSpace($Path)) {
        return 0
    }
    if (Test-Path -LiteralPath $Path) {
        $destinationName = if ([string]::IsNullOrWhiteSpace($StagedName)) {
            Split-Path -Leaf $Path
        }
        else {
            $StagedName
        }
        Copy-Item -LiteralPath $Path -Destination (Join-Path $ToDir $destinationName) -Force
        return 1
    }
    return 0
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

$stressStagedName = "stress-$DefaultModelVersion-on-$DefaultValidationName-threshold-$ThresholdLabel.json"
$stressMetrics = $null
if ((Copy-ArtifactIfExists $StressResultPath $StagePath $stressStagedName) -gt 0) {
    try {
        $stressMetrics = Get-Content -LiteralPath $StressResultPath -Raw -Encoding UTF8 | ConvertFrom-Json
        if ([Math]::Abs([double]$stressMetrics.threshold - $RecommendedThreshold) -gt 0.000001) {
            Write-Warning "StressResultPath threshold is $($stressMetrics.threshold), but RecommendedThreshold is $RecommendedThreshold."
        }
    }
    catch {
        Write-Warning "Could not parse stress result JSON at ${StressResultPath}: $($_.Exception.Message)"
    }
}

$criticalMetrics = $null
if ((Copy-ArtifactIfExists $CriticalValidationSummaryPath $StagePath "v16-v13-critical-ru-validation-summary.json") -gt 0) {
    try {
        $criticalSummary = Get-Content -LiteralPath $CriticalValidationSummaryPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $thresholdMetrics = $criticalSummary.threshold_metrics
        $metricProperty = $thresholdMetrics.PSObject.Properties[$ThresholdLabel]
        if ($null -ne $metricProperty) {
            $criticalMetrics = $metricProperty.Value
        }
        else {
            Write-Warning "Critical validation summary has no metrics for threshold $ThresholdLabel."
        }
    }
    catch {
        Write-Warning "Could not parse critical validation summary at ${CriticalValidationSummaryPath}: $($_.Exception.Message)"
    }
}

[void](Copy-ArtifactIfExists $DatasetValidatorReportPath $StagePath "")
[void](Copy-ArtifactIfExists $DatasetBuildReportPath $StagePath "")
[void](Copy-ArtifactIfExists $FalseNegativeReportPath $StagePath "")
[void](Copy-ArtifactIfExists $ValidationComparisonSummaryPath $StagePath "v16-validation-comparison-summary.json")
[void](Copy-ArtifactIfExists $ValidationGateSummaryPath $StagePath "v16-gate-summary.json")
[void](Copy-ArtifactIfExists $ValidationCorpusReportPath $StagePath "")

$thresholdPayload = [ordered]@{
    recommended_threshold = $RecommendedThreshold
    source = "v16-v13-critical-ru-validation-summary.json"
    note = "V16 diagnostic candidate threshold is 0.99. It improves V13 on the core diagnostic suite but still requires production calibration before deployment."
}
if ($null -ne $criticalMetrics) {
    $thresholdPayload["validation_at_recommended_threshold"] = [ordered]@{
        corpus = "v13_critical_ru"
        documents = $criticalMetrics.documents
        threshold = $criticalMetrics.threshold
        precision = $criticalMetrics.precision
        recall = $criticalMetrics.recall
        f1 = $criticalMetrics.f1
        true_positives = $criticalMetrics.tp
        false_positives = $criticalMetrics.fp
        false_negatives = $criticalMetrics.fn
        benign_false_positive_rate = $criticalMetrics.benign_fp_rate
    }
}
elseif ($null -ne $stressMetrics) {
    $thresholdPayload["validation_at_recommended_threshold"] = [ordered]@{
        rows = $stressMetrics.rows
        threshold = $stressMetrics.threshold
        accuracy = $stressMetrics.accuracy
        precision = $stressMetrics.precision
        recall = $stressMetrics.recall
        f1 = $stressMetrics.f1
        false_positives = $stressMetrics.false_positives
        false_negatives = $stressMetrics.false_negatives
        false_positive_rate = $stressMetrics.false_positive_rate
        false_negative_rate = $stressMetrics.false_negative_rate
    }
}
($thresholdPayload | ConvertTo-Json -Depth 6) | Set-Content -LiteralPath (Join-Path $StagePath "threshold_recommendations.json") -Encoding UTF8

$evaluationArtifacts = @()
foreach ($artifact in $evaluationArtifacts) {
    [void](Copy-ProjectFileIfExists $artifact $StagePath)
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

$commitMessage = "Upload mDeBERTa Russian prompt-injection detector $DefaultModelVersion"
$uploadCommand = "hf upload $RepoId `"$StagePath`" . --commit-message `"$commitMessage`""
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
Invoke-Hf upload $RepoId $StagePath "." --commit-message $commitMessage

Write-Host "`nDone. Model repo: https://huggingface.co/$RepoId"
