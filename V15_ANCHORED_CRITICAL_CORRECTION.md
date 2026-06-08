# V15 Anchored Critical Correction

This dataset is for a new corrective run from V13, not from V14.

V14 is not continued because it only produced a weak high-threshold improvement and worsened low-threshold benign false positives.

## Build Dataset

```powershell
uv run python build_v15_anchored_critical_correction_dataset.py `
  --tokenizer-id .\mdeberta-ru-prompt-injection-v13-critical-correction-ft `
  --output-dir .\training-dataset-v15-anchored-critical-correction-windowed `
  --validation-output-dir .\training-dataset-v15-anchored-critical-correction-windowed-validation `
  --report-json .\training-dataset-v15-anchored-critical-correction-windowed-report.json
```

Expected shape:

```text
total rows:                      60K
anchored critical positives:     ~12K
semantic critical positives:      ~6K
embedded critical positives:     ~6.5K
wrapper critical positives:      ~2.5K
benign FP hard negatives:        ~4.4K
limited keyword benign:          ~1.5K
benign replay:                   ~19K
attack replay:                   ~8K-9K
```

The semantic critical component is separate from the anchored substitution
component. It uses intent/scenario frames such as forensic review, routing debug,
configuration migration, audit recovery, private-context reconstruction and tool
registry inspection, rather than only replacing verbs and targets in the same
sentence skeleton.

## Analyze Before Training

Composition-only analysis:

```powershell
uv run python analyze_v15_dataset_before_training.py `
  --dataset-dir .\training-dataset-v15-anchored-critical-correction-windowed `
  --report-json .\training-dataset-v15-anchored-critical-correction-windowed-analysis.json
```

Score the validation split with V13 before training:

```powershell
uv run python analyze_v15_dataset_before_training.py `
  --dataset-dir .\training-dataset-v15-anchored-critical-correction-windowed `
  --model-id .\mdeberta-ru-prompt-injection-v13-critical-correction-ft `
  --tokenizer-id .\mdeberta-ru-prompt-injection-v13-critical-correction-ft `
  --score-split validation `
  --thresholds "0.82,0.95,0.99,0.995,0.999" `
  --batch-size 64 `
  --device cpu `
  --scored-jsonl .\training-dataset-v15-v13-scored-validation.jsonl `
  --report-json .\training-dataset-v15-v13-scored-validation-analysis.json
```

Use CUDA if available:

```powershell
uv run python analyze_v15_dataset_before_training.py `
  --dataset-dir .\training-dataset-v15-anchored-critical-correction-windowed `
  --model-id .\mdeberta-ru-prompt-injection-v13-critical-correction-ft `
  --tokenizer-id .\mdeberta-ru-prompt-injection-v13-critical-correction-ft `
  --score-split validation `
  --thresholds "0.82,0.95,0.99,0.995,0.999" `
  --batch-size 128 `
  --device cuda `
  --scored-jsonl .\training-dataset-v15-v13-scored-validation.jsonl `
  --report-json .\training-dataset-v15-v13-scored-validation-analysis.json
```

The generated pre-training analysis summary is:

```text
V15_PRETRAIN_ANALYSIS.md
```

## Train From V13

```powershell
uv run python train_mdeberta_ru_prompt_injection_option_b.py `
  --device cuda `
  --bf16 `
  --student-model .\mdeberta-ru-prompt-injection-v13-critical-correction-ft `
  --prepared-dataset-dir .\training-dataset-v15-anchored-critical-correction-windowed `
  --output-dir .\mdeberta-ru-prompt-injection-v15-anchored-critical-correction-ft `
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
