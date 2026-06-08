# V10-V13 Validation Comparison

This is a diagnostic comparison suite for the four local model iterations:

- `v10`: `.\mdeberta-ru-prompt-injection-v10-benign-scratch`
- `v11`: `.\mdeberta-ru-prompt-injection-v11-fp-correction-ft`
- `v12`: `.\mdeberta-ru-prompt-injection-v12-critical-correction-ft`
- `v13`: `.\mdeberta-ru-prompt-injection-v13-critical-correction-ft`

It uses production-style windowing and max-window aggregation through `run_blind_broad_eval.py`.

## Quick Comparison

Runs only:

- V13 critical Russian attack validation
- V13 benign validation windows

```powershell
uv run python compare_v10_v13_validation_suite.py `
  --suite quick `
  --output-dir .\validation-comparison-v10-v13-quick `
  --thresholds "0.82,0.90,0.95,0.99" `
  --primary-threshold 0.95 `
  --window-batch-size 64 `
  --device cpu
```

Use `--device cuda` on a GPU machine.

## Core Comparison

Runs:

- V13 critical Russian attack validation
- V13 benign validation windows
- malicious document dev set
- benign production calibration dev set

```powershell
uv run python compare_v10_v13_validation_suite.py `
  --suite core `
  --output-dir .\validation-comparison-v10-v13-core `
  --thresholds "0.82,0.90,0.95,0.99,0.999,0.9995" `
  --primary-threshold 0.95 `
  --window-batch-size 64 `
  --device cpu
```

## Full Diagnostic Comparison

This is expensive on CPU. It adds:

- malicious document test set
- benign production locked-test style set
- full false-positive corpus

```powershell
uv run python compare_v10_v13_validation_suite.py `
  --suite full `
  --output-dir .\validation-comparison-v10-v13-full `
  --thresholds "0.82,0.90,0.95,0.99,0.999,0.9995" `
  --primary-threshold 0.95 `
  --window-batch-size 128 `
  --device cuda
```

## Smoke Test

Use this to verify the machinery without spending hours:

```powershell
uv run python compare_v10_v13_validation_suite.py `
  --suite quick `
  --output-dir .\validation-comparison-v10-v13-smoke `
  --thresholds "0.82,0.95" `
  --primary-threshold 0.95 `
  --window-batch-size 16 `
  --limit-documents 20 `
  --device cpu `
  --force
```

## Outputs

Each run writes:

- `validation-suite-manifest.json`
- per-model per-corpus document result JSONL files
- per-model per-corpus summary JSON files
- `comparison-summary.csv`
- `comparison-summary.json`

The current corpora are suitable for regression and diagnostic comparison. Do not call this a final blind acceptance suite if any included corpus has been used for training, mining, threshold selection, or previous inspection.
