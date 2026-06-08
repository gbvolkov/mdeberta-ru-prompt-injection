# V15 Pre-Training Analysis

V15 is built to train from `mdeberta-ru-prompt-injection-v13-critical-correction-ft`.

It should not use V14 as parent. V14 was weakly positive at threshold `0.99`, but not strong enough and worsened lower-threshold benign FPR.

## Dataset

```text
dataset:    training-dataset-v15-anchored-critical-correction-windowed
train:      53,000 rows
validation:  7,000 rows
total:      60,000 rows
```

Train labels:

```text
not_prompt_injection: 21,991
prompt_injection:    31,009
```

Validation labels:

```text
not_prompt_injection: 2,904
prompt_injection:    4,096
```

Components:

| Component | Rows |
|---|---:|
| `critical_ru_anchored_positive` | 11,998 |
| `critical_ru_semantic_paraphrase_positive` | 6,000 |
| `critical_ru_embedded_positive` | 6,500 |
| `critical_ru_wrapper_positive` | 2,463 |
| `critical_ru_exact_v13_miss` | 72 |
| `benign_actual_fp_hard_negative` | 4,431 |
| `benign_keyword_hard_negative` | 1,468 |
| `benign_replay` | 18,996 |
| `attack_replay` | 8,072 |

Diagnostic source rows used:

```text
critical V13 false-negative documents:      72
benign_prod_dev V13 false-positive docs:   114
v13_benign_windows false-positive docs:      5
```

Duplicate text rows:

```text
train:      0
validation: 0
```

## Difference From V14

V14 generated too many broad/regular phrases from a phrase grid. V15 now uses
three positive correction channels:

```text
actual V13 false negative -> near-preserved exact miss replay
actual V13 false negative -> anchored mutation around the failed intent
semantic intent frame -> different operational scenario and phrasing
```

The semantic channel is intentionally separate from `mutate_fragment()` and its
verb/target substitutions. It generates role/scenario forms such as routing
debug, forensic audit, tool-registry inspection, configuration migration, private
context reconstruction and hidden-policy recovery.

Broad keyword benign hard negatives were reduced:

```text
V14 broad keyword hard negatives: 11,466
V15 broad keyword hard negatives:  1,468
```

Generic attack replay was reduced:

```text
V14 attack replay: 20,000
V15 attack replay: 10,000
```

## V13 Pre-Score On V15 Validation

This checks whether V15 is merely relearning examples V13 already handles.

At threshold `0.99`:

```text
TP: 3,765
FN:   331
FP:   240
TN: 2,664
positive recall: 91.92%
benign FP rate:  8.26%
```

Component behavior at threshold `0.99`:

| Component | V13 TP/FP | V13 FN/TN | Interpretation |
|---|---:|---:|---|
| `critical_ru_exact_v13_miss` | 2 TP | 6 FN | Still hard; useful correction signal |
| `critical_ru_anchored_positive` | 1,286 TP | 115 FN | Mostly learned, but still has hard positives |
| `critical_ru_semantic_paraphrase_positive` | 612 TP | 88 FN | New semantic paraphrase channel; useful hard-positive signal |
| `critical_ru_embedded_positive` | 726 TP | 32 FN | Mostly learned, some useful hard signal |
| `critical_ru_wrapper_positive` | 274 TP | 13 FN | Mostly learned, some useful hard signal |
| `benign_actual_fp_hard_negative` | 235 FP | 282 TN | Very useful hard-negative signal |
| `benign_keyword_hard_negative` | 0 FP | 171 TN | Safe but not a strong correction signal |
| `benign_replay` | 5 FP | 2,211 TN | Mostly safe replay |
| `attack_replay` | 865 TP | 77 FN | Useful positive replay |

Score distributions:

```text
positive score p05: 0.5262
positive score median: 0.99997
benign score median: 0.0017
benign score p95: 0.99914
```

## Decision

This dataset is acceptable for a first V15 training run from V13.

It contains enough hard signal to avoid pure relearning:

```text
critical exact misses remain missed by V13
anchored positives include V13 FNs at target thresholds
semantic paraphrase positives add non-grid realizations of the same attack intent
actual benign false positives remain high-scored by V13
benign replay is mostly low-scored
```

The dataset is not perfect: many anchored positives are already detected by V13,
so the expected movement may still be moderate. If V15 under-moves the critical
RU recall again, the next change should not add more generic positives; it
should increase exact/near-exact anchored misses and higher-quality semantic
paraphrases while reducing already-easy positives.

## Training Command

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
