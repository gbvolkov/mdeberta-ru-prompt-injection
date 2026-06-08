# V14 Boundary Correction

V14 is a small corrective fine-tune from V13. It is built from V13 diagnostic misses and false positives, plus replay.

## Dataset

Prepared dataset:

```text
training-dataset-v14-boundary-correction-windowed
```

Current size:

```text
train:      61,794 rows
validation:  8,000 rows
total:      69,794 rows
```

Label mix:

```text
train benign:           28,147
train prompt injection: 33,647
```

Diagnostic errors used:

```text
critical RU false-negative docs: 72
benign production false-positive docs: 114
```

Important: this uses diagnostic/mining corpora. After V14 is trained, the same V10-V13 diagnostic comparison suite can show whether known failures moved, but it is not blind acceptance.

## Rebuild Dataset

```powershell
uv run python build_v14_boundary_correction_dataset.py `
  --tokenizer-id .\mdeberta-ru-prompt-injection-v13-critical-correction-ft `
  --base-dataset-dir .\training-dataset-v13-critical-russian-correction-windowed `
  --output-dir .\training-dataset-v14-boundary-correction-windowed `
  --validation-output-dir .\training-dataset-v14-boundary-correction-windowed-validation `
  --report-json .\training-dataset-v14-boundary-correction-windowed-report.json
```

## Train V14

```powershell
uv run python train_mdeberta_ru_prompt_injection_option_b.py `
  --device cuda `
  --bf16 `
  --student-model .\mdeberta-ru-prompt-injection-v13-critical-correction-ft `
  --prepared-dataset-dir .\training-dataset-v14-boundary-correction-windowed `
  --output-dir .\mdeberta-ru-prompt-injection-v14-boundary-correction-ft `
  --learning-rate 1e-6 `
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
  --tf32 `
  --torch-num-threads 6 `
  --preflight-steps 2
```

## Evaluate V14

Run the same core diagnostic suite:

```powershell
.\run_v14_core_validation.ps1 -Device cuda -WindowBatchSize 128
```

`compare_v10_v13_validation_suite.py` already includes the `v14` model key.

The wrapper writes:

```text
validation-comparison-v10-v14-core\comparison-summary.csv
validation-comparison-v10-v14-core\comparison-summary.json
validation-comparison-v10-v14-core\v14-gate-summary.md
validation-comparison-v10-v14-core\v14-gate-summary.json
```

To compare every historical model on the same suite:

```powershell
.\run_v14_core_validation.ps1 -Device cuda -WindowBatchSize 128 -CompareAll -OutputDir .\validation-comparison-v10-v14-core-all
```

Acceptance target for diagnostic selection:

```text
benign_prod_dev FPR       < 1%
v13_benign_windows FPR    < 1%
malicious_dev recall     >= 99%
v13_critical_ru recall   >= 99%
```
