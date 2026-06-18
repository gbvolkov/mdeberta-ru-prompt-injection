# V13 Critical Russian Correction

V13 is a targeted follow-up when V12 still needs stronger coverage for explicit Russian internal-prompt exfiltration.

## Build Dataset

```powershell
uv run python build_v13_critical_correction_dataset.py `
  --tokenizer-id .\mdeberta-ru-prompt-injection-v12-critical-correction-ft `
  --base-dataset-dir .\training-dataset-v12-russian-critical-correction-windowed `
  --carrier-jsonl .\false-positive-corpus-documents.jsonl `
  --locked-corpus-glob "v12-eval-suites\*locked*.jsonl" `
  --output-dir .\training-dataset-v13-critical-russian-correction-windowed `
  --validation-output-dir .\training-dataset-v13-critical-russian-correction-windowed-validation `
  --diagnostic-dev-jsonl .\v13-critical-ru-diagnostic-dev.jsonl `
  --report-json .\training-dataset-v13-critical-russian-correction-windowed-report.json `
  --allow-underfilled
```

## Leakage Check

```powershell
uv run python check_v12_leakage.py `
  --training-dataset-dir .\training-dataset-v13-critical-russian-correction-windowed `
  --locked-corpus-glob "v12-eval-suites\*locked*.jsonl" `
  --report-json .\v13-leakage-check-report.json
```

## Train

```powershell
uv run python train_mdeberta_ru_prompt_injection_option_b.py `
  --device cuda `
  --bf16 `
  --student-model .\mdeberta-ru-prompt-injection-v12-critical-correction-ft `
  --prepared-dataset-dir .\training-dataset-v13-critical-russian-correction-windowed `
  --output-dir .\mdeberta-ru-prompt-injection-v13-critical-russian-correction-ft `
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

## Diagnostic Evaluation

```powershell
uv run python run_blind_broad_eval.py `
  --model-id .\mdeberta-ru-prompt-injection-v13-critical-russian-correction-ft `
  --input-jsonl .\v13-critical-ru-diagnostic-dev.jsonl `
  --thresholds "0.82,0.90,0.95,0.99" `
  --primary-threshold 0.95 `
  --window-batch-size 128 `
  --output-jsonl .\v13-critical-ru-diagnostic-dev-results.jsonl `
  --summary-json .\v13-critical-ru-diagnostic-dev-summary.json `
  --device cuda
```
