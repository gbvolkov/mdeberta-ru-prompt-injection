param(
  [string]$Corpus = "blind-acceptance-benign-documents",
  [string]$EvalSuiteDir = ".\arc\v18-pod-eval-20260615\v18-eval-suite",
  [string]$Stage1ModelId = "gbv/mdeberta-ru-prompt-injection",
  [string]$ReviewerModelId = ".\mdeberta-v16-positive-reviewer-oracle",
  [string]$OutputRoot = ".\policy-results-smoke",
  [int]$LimitDocuments = 20,
  [ValidateSet("auto", "cpu", "cuda")]
  [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $EvalSuiteDir)) {
  $fallback = ".\arc\v16-positive-reviewer-oracle-pod-20260616\eval-suite"
  if (Test-Path $fallback) {
    $EvalSuiteDir = $fallback
  } else {
    throw "Eval suite directory not found. Pass -EvalSuiteDir <path>."
  }
}

$inputJsonl = Join-Path $EvalSuiteDir "$Corpus.jsonl"
if (-not (Test-Path $inputJsonl)) {
  $available = Get-ChildItem -File $EvalSuiteDir -Filter "*.jsonl" | Select-Object -ExpandProperty BaseName
  throw "Corpus '$Corpus' not found at $inputJsonl. Available corpora: $($available -join ', ')"
}

if ($ReviewerModelId.StartsWith(".\") -or $ReviewerModelId.StartsWith("..\")) {
  if (-not (Test-Path $ReviewerModelId)) {
    throw "Reviewer model path '$ReviewerModelId' not found. Pass -ReviewerModelId <local path or HF model id>."
  }
}

$outDir = Join-Path $OutputRoot $Corpus
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$docOut = Join-Path $outDir "document-results.jsonl"
$winOut = Join-Path $outDir "window-results.jsonl"
$summaryOut = Join-Path $outDir "summary.json"

Write-Host "[smoke] corpus=$Corpus"
Write-Host "[smoke] input=$inputJsonl"
Write-Host "[smoke] output=$outDir"
Write-Host "[smoke] limit=$LimitDocuments device=$Device"

uv run python .\run_v16_family_gated_policy_eval.py `
  --stage1-model-id $Stage1ModelId `
  --reviewer-model-id $ReviewerModelId `
  --input-jsonl $inputJsonl `
  --v16-threshold-low 0.82 `
  --v16-threshold-protect 0.999 `
  --reviewer-threshold-block 0.30 `
  --reviewer-threshold-allow 0.10 `
  --reviewer-allowlist security_compliance_redaction_wrappers `
  --protected-family-policy-json .\policy\protected-family-policy.json `
  --slice-router-config-json .\policy\slice-router-config.json `
  --certified-slices-json .\policy\certified-slices-empty.json `
  --mode diagnostic `
  --output-jsonl $docOut `
  --window-output-jsonl $winOut `
  --summary-json $summaryOut `
  --device $Device `
  --limit-documents $LimitDocuments

$summary = Get-Content -Raw $summaryOut | ConvertFrom-Json
$docRows = Get-Content $docOut | Where-Object { $_.Trim() } | ForEach-Object { $_ | ConvertFrom-Json }
$winRows = Get-Content $winOut | Where-Object { $_.Trim() } | ForEach-Object { $_ | ConvertFrom-Json }

$failures = New-Object System.Collections.Generic.List[string]

if (($docRows | Where-Object { $null -ne $_.effective_action }).Count -ne 0) {
  $failures.Add("document_effective_action_not_null")
}
if (($winRows | Where-Object { $null -ne $_.effective_action }).Count -ne 0) {
  $failures.Add("window_effective_action_not_null")
}
if ([int]$summary.effective_attack_allow_policy_added -ne 0) {
  $failures.Add("effective_attack_allow_policy_added_nonzero")
}
if ([int]$summary.certified_auto_allow_count -ne 0) {
  $failures.Add("certified_auto_allow_count_nonzero")
}
if ([int]$summary.reviewer_effective_auto_allow_count -ne 0) {
  $failures.Add("reviewer_effective_auto_allow_count_nonzero")
}
if ([int]$summary.reviewer_scored_window_count -gt [int]$summary.reviewer_eligible_window_count) {
  $failures.Add("reviewer_scored_window_count_gt_eligible")
}

$requiredDocFields = @(
  "proposed_action",
  "effective_action",
  "mode",
  "policy_engine_version",
  "policy_config_hash",
  "router_version",
  "router_config_hash",
  "reviewer_model_id",
  "reviewer_calibration_id",
  "certified_slice_id",
  "policy_decision_trace",
  "document_router_categories",
  "document_router_semantic_families",
  "document_router_rule_ids",
  "document_has_security_compliance_redaction_wrapper",
  "document_has_direct_protected_attack_anchor",
  "document_primary_category_reason",
  "reviewer_max_score_on_eligible_windows"
)

$requiredWindowFields = @(
  "proposed_action",
  "effective_action",
  "mode",
  "policy_engine_version",
  "policy_config_hash",
  "router_version",
  "router_config_hash",
  "reviewer_model_id",
  "reviewer_calibration_id",
  "certified_slice_id",
  "policy_decision_trace",
  "reviewer_eligible_for_scoring",
  "reviewer_eligibility_reason"
)

if ($docRows.Count -gt 0) {
  $docFieldNames = $docRows[0].PSObject.Properties.Name
  foreach ($field in $requiredDocFields) {
    if ($field -notin $docFieldNames) {
      $failures.Add("missing_document_field:$field")
    }
  }
}

if ($winRows.Count -gt 0) {
  $winFieldNames = $winRows[0].PSObject.Properties.Name
  foreach ($field in $requiredWindowFields) {
    if ($field -notin $winFieldNames) {
      $failures.Add("missing_window_field:$field")
    }
  }
}

$check = [ordered]@{
  status = if ($failures.Count -eq 0) { "pass" } else { "fail" }
  failures = @($failures)
  corpus = $Corpus
  documents = $summary.documents
  windows = $summary.windows
  proposed_attack_allow_policy_added = $summary.proposed_attack_allow_policy_added
  effective_attack_allow_policy_added = $summary.effective_attack_allow_policy_added
  certified_auto_allow_count = $summary.certified_auto_allow_count
  reviewer_effective_auto_allow_count = $summary.reviewer_effective_auto_allow_count
  reviewer_potential_allow_count = $summary.reviewer_potential_allow_count
  reviewer_scored_window_count = $summary.reviewer_scored_window_count
  reviewer_eligible_window_count = $summary.reviewer_eligible_window_count
  output_dir = $outDir
}

$checkPath = Join-Path $outDir "smoke-check.json"
$check | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 $checkPath
$check | ConvertTo-Json -Depth 8

if ($failures.Count -ne 0) {
  throw "Smoke validation failed. See $checkPath"
}

Write-Host "[smoke] PASS. Check report: $checkPath"
