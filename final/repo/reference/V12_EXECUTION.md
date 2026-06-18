# V12 Execution Recovery Notes

This file restores the operational sequence used for the V12 corrective run.

## Build Locked Evaluation Suites

Use explicit fresh/held-out JSONL inputs. Do not pass prior training or mining corpora as locked acceptance inputs.

```powershell
uv run python build_v12_frozen_eval_suites.py `
  --tokenizer-id microsoft/mdeberta-v3-base `
  --production-benign-jsonl .\fresh-production-domain-benign.jsonl `
  --external-benign-jsonl .\fresh-external-benign.jsonl `
  --carrier-jsonl .\fresh-production-domain-benign.jsonl `
  --output-dir .\v12-eval-suites `
  --allow-underfilled
```

## Build V12 Correction Dataset

```powershell
uv run python build_v12_correction_dataset.py `
  --tokenizer-id .\mdeberta-ru-prompt-injection-v11-fp-correction-ft `
  --v11-dataset-dir .\training-dataset-v11-fp-correction-windowed-200k `
  --carrier-jsonl .\false-positive-corpus-documents.jsonl `
  --locked-corpus-glob "v12-eval-suites\*locked*.jsonl" `
  --output-dir .\training-dataset-v12-russian-critical-correction-windowed `
  --validation-output-dir .\training-dataset-v12-russian-critical-correction-windowed-validation `
  --report-json .\training-dataset-v12-russian-critical-correction-windowed-report.json `
  --allow-underfilled
```

## Leakage Check

```powershell
uv run python check_v12_leakage.py `
  --training-dataset-dir .\training-dataset-v12-russian-critical-correction-windowed `
  --locked-corpus-glob "v12-eval-suites\*locked*.jsonl" `
  --report-json .\v12-leakage-check-report.json
```

## Train V12

```powershell
uv run python train_mdeberta_ru_prompt_injection_option_b.py `
  --device cuda `
  --bf16 `
  --student-model .\mdeberta-ru-prompt-injection-v11-fp-correction-ft `
  --prepared-dataset-dir .\training-dataset-v12-russian-critical-correction-windowed `
  --output-dir .\mdeberta-ru-prompt-injection-v12-critical-correction-ft `
  --learning-rate 2e-6 `
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

## Evaluate Checkpoints On Development Corpora

```powershell
uv run python evaluate_v12_checkpoints.py `
  --checkpoint-root .\mdeberta-ru-prompt-injection-v12-critical-correction-ft `
  --corpus-jsonl critical_ru_dev=v12-eval-suites\critical-ru-dev.jsonl `
  --corpus-jsonl production_dev=v12-eval-suites\production-domain-dev.jsonl `
  --corpus-jsonl external_dev=v12-eval-suites\external-blind-dev.jsonl `
  --output-dir .\v12-checkpoint-eval `
  --thresholds "0.82,0.90,0.95,0.99,0.999,0.9995" `
  --primary-threshold 0.95 `
  --window-batch-size 128 `
  --device cuda
```

Locked acceptance corpora are run only after checkpoint and threshold are fixed.
