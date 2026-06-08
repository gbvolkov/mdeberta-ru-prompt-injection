# V16 Critical Recall Restoration

V16 is a new corrective dataset for training from V13.

Do not train V16 from V15. V15 reduced benign false positives strongly, but it regressed critical Russian recall.

## Design

V16 changes the V15 pressure:

- parent remains `mdeberta-ru-prompt-injection-v13-critical-correction-ft`
- adds V15 regression-guard positives from critical Russian examples V15 missed
- keeps V13 critical false-negative anchors
- reduces semantic broad positives versus V15
- removes V15-style augmented benign hard-negative expansion by default
- does not include prepared prior training-dataset replay by default
- uses non-training-source benign context windows instead of V10/V11/V13 replay
- removes keyword-only benign hard negatives by default
- expands actual V13 benign false-positive document windows to roughly 1K rows
- uses `benign-prod-calibration-dev.jsonl` as the default embedded-attack carrier, not the older false-positive corpus carrier

Current generated shape:

```text
total rows:      43,675
train rows:      37,675
validation rows:  6,000

prompt_injection:     27,170
not_prompt_injection: 16,505
```

Main components:

| Component | Rows |
|---|---:|
| `critical_ru_exact_v13_miss` | 64 |
| `critical_ru_near_anchor_positive` | 12,998 |
| `critical_ru_semantic_hard_positive` | 3,500 |
| `critical_ru_embedded_positive` | 4,986 |
| `critical_ru_wrapper_positive` | 1,491 |
| `critical_ru_v15_regression_guard_positive` | 4,131 |
| `benign_actual_fp_hard_negative` | 976 |
| `benign_context_non_training_source` | 15,529 |

Prior prepared training replay is disabled:

```text
prior_training_replay.included: false
benign_replay_target:           0
attack_replay_target:           0
benign_keyword_target:          0
```

Diagnostic errors used:

```text
V13 critical false-negative docs:       64
V15 critical false-negative docs:      160
V13 benign-prod false-positive docs:   114
V13 benign-window false-positive docs:   3
```

Leakage marker audit:

```text
all_rows prior-training marker count:    0
train prior-training marker count:       0
validation prior-training marker count:  0
diagnostic-errors prior marker matches:  0
```

## Build Dataset

```powershell
uv run python build_v16_critical_recall_restoration_dataset.py `
  --tokenizer-id .\mdeberta-ru-prompt-injection-v13-critical-correction-ft `
  --output-dir .\training-dataset-v16-critical-recall-restoration-windowed `
  --validation-output-dir .\training-dataset-v16-critical-recall-restoration-windowed-validation `
  --report-json .\training-dataset-v16-critical-recall-restoration-windowed-report.json
```

## Analyze Dataset Before Training

Composition only:

```powershell
uv run python analyze_v15_dataset_before_training.py `
  --dataset-dir .\training-dataset-v16-critical-recall-restoration-windowed `
  --report-json .\training-dataset-v16-critical-recall-restoration-windowed-analysis.json
```

Score the validation split with V13 before training:

```powershell
uv run python analyze_v15_dataset_before_training.py `
  --dataset-dir .\training-dataset-v16-critical-recall-restoration-windowed `
  --model-id .\mdeberta-ru-prompt-injection-v13-critical-correction-ft `
  --tokenizer-id .\mdeberta-ru-prompt-injection-v13-critical-correction-ft `
  --score-split validation `
  --thresholds "0.82,0.95,0.99,0.999,0.9995" `
  --batch-size 64 `
  --device cpu `
  --scored-jsonl .\training-dataset-v16-v13-scored-validation.jsonl `
  --report-json .\training-dataset-v16-v13-scored-validation-analysis.json
```

Do not launch training until this pre-score report has been inspected.

## Train From V13

Start conservatively:

```powershell
uv run python train_mdeberta_ru_prompt_injection_option_b.py `
  --device cuda `
  --bf16 `
  --student-model .\mdeberta-ru-prompt-injection-v13-critical-correction-ft `
  --prepared-dataset-dir .\training-dataset-v16-critical-recall-restoration-windowed `
  --output-dir .\mdeberta-ru-prompt-injection-v16-critical-recall-restoration-ft `
  --learning-rate 8e-7 `
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

## Compare After Training

```powershell
uv run python compare_v10_v13_validation_suite.py `
  --suite core `
  --models v13,v16 `
  --output-dir .\validation-comparison-v13-v16-core `
  --thresholds "0.82,0.90,0.95,0.99,0.999,0.9995" `
  --primary-threshold 0.95 `
  --window-batch-size 64 `
  --device cpu `
  --progress-every-docs 25 `
  --progress-every-windows 1000
```

This comparison suite is diagnostic, not blind acceptance.
