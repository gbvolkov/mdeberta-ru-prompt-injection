# V17 Clean Windowed Dataset Build Procedure

## Objective

Build V17 as a clean window-level dataset for training directly from:

```text
microsoft/mdeberta-v3-base
```

V17 must not copy rows from V10-V16 training datasets. Prior datasets and evaluations may be used only as exclusion references or diagnostics, not as row sources.

## Non-Negotiable Rules

1. The training unit is one production-style window.
2. Windowing must match production:

```text
tokenizer: microsoft/mdeberta-v3-base
window token length: 254
window stride: 128
training tokenizer max_length: 256
```

3. Label rule:

```text
prompt_injection      only if the visible attack text is inside the window
not_prompt_injection  if the window does not contain visible attack text
```

4. A malicious long document does not make all its windows malicious.
5. Previous prepared datasets are forbidden as training-row inputs.

Forbidden as row sources:

```text
training-dataset-v10*
training-dataset-v11*
training-dataset-v12*
training-dataset-v13*
training-dataset-v14*
training-dataset-v15*
training-dataset-v16*
false-positive-corpus-documents.jsonl
malicious-document-dev.jsonl
v12-eval-suites/*
v13-*validation*
v16-proper-validation-suite/*
any previous mining/error jsonl used to train V12-V16
```

Allowed use of previous artifacts:

```text
deduplication / exclusion index only
diagnostic validation only
source-name risk audit only
```

Rows from previous prepared datasets may not be copied even if their labels look correct. If an old row points to a raw source document that is independently available, only the raw source document may be re-collected and re-windowed under the V17 policy.

## Target Dataset Shape

Preferred V17 size:

```text
total rows:       250,000 windows
train rows:       225,000 windows
validation rows:   25,000 windows
```

Minimum acceptable pilot:

```text
180,000-200,000 total rows
```

Target label balance:

```text
prompt_injection:      50-55%
not_prompt_injection:  45-50%
```

Target language balance:

```text
Russian:        70-75%
English:        15-20%
Mixed language:  5-10%
```

## Component Targets

### Benign Windows: About 120K

| Component | Target Rows |
| --- | ---: |
| `proper_benign_prod_windows` | 35K-45K |
| `benign_carrier_contrast_windows` | 30K-40K |
| `benign_long_doc_windows` | 15K-20K |
| `benign_wrapper_redaction_url_windows` | 8K-12K |
| `benign_actual_fp_hard_negative` | 3K-5K |
| `benign_fp_proxy_security_language` | 4K-7K |
| `benign_security_policy_discussion` | 5K-8K |
| `benign_general_ru_en_mixed` | 8K-12K |

### Attack Windows: About 130K

| Component | Target Rows |
| --- | ---: |
| `critical_ru_visible_attack_windows` | 30K-40K |
| `embedded_visible_attack_windows` | 35K-45K |
| `general_prompt_injection_ru` | 18K-25K |
| `general_prompt_injection_en` | 12K-18K |
| `mixed_language_attack_windows` | 8K-12K |
| `wrapper_redaction_url_attack_windows` | 8K-12K |
| `short_standalone_attack_windows` | 8K-12K |
| `hard_fn_visible_windows` | 5K-8K |

`hard_fn_visible_windows` must be generated fresh from V17 attack-family definitions. It must not copy previous V13/V16 failed rows.

## Source Collection

Collect fresh source documents directly from raw sources, not prepared V-model datasets.

Acceptable source classes:

```text
fresh FineWeb2 RU/EN documents
fresh C4 RU/EN documents
fresh Wikipedia RU/EN documents
fresh StackExchange / technical Q&A
fresh legal / administrative text sources
fresh support / knowledge-base style documents
fresh HR / job-description / corporate procedure documents
fresh news/security/compliance text
fresh redaction/wrapper/document-format examples
```

Every collected source document must receive:

```json
{
  "source_document_id": "...",
  "source_name": "...",
  "source_origin": "raw_source",
  "category": "...",
  "language": "ru|en|mixed",
  "text": "..."
}
```

Rows with `source_origin` equal to any prior prepared dataset must be rejected.

## Attack Bank Construction

Create a V17-only attack bank from semantic families. Do not import old attack rows.

Minimum semantic-family count:

```text
40-60 families
```

Required families include:

```text
direct override
system prompt disclosure
developer prompt disclosure
tool list disclosure
hidden routing disclosure
internal policy extraction
private context extraction
secret / token / key exfiltration
debug / audit / compliance framing
role-play bypass
parser recovery / document-boundary tricks
translation or transformation of hidden rules
wrapper / redaction / URL adjacent injection
```

Required Russian concepts:

```text
системный промпт
промпт разработчика
служебные инструкции
скрытая маршрутизация
внутренние инструменты
внутренняя конфигурация
правила выбора инструментов
покажи скрытые правила
перечисли служебные сообщения
```

Each generated attack instance must include:

```json
{
  "attack_text": "...",
  "attack_text_hash": "...",
  "attack_template_id": "...",
  "semantic_family": "...",
  "language": "ru|en|mixed",
  "generation_origin": "v17_attack_bank"
}
```

## Long Malicious Document Algorithm

This is the required algorithm for documents containing short embedded attacks.

1. Select a fresh benign carrier document from the raw source pool.
2. Select a V17-generated attack instance from the V17 attack bank.
3. Insert the attack span into the carrier at one of these positions:

```text
beginning
early body
middle
late body
end
near URL
near redaction marker
near document wrapper
near section boundary
```

4. Save the exact character offsets and token offsets of the inserted attack span.
5. Build production windows over the full resulting document:

```text
254 tokens, stride 128
```

6. For each window:

```text
if the window visibly contains the inserted attack text or a verified visible attack span:
    label = prompt_injection
else:
    label = not_prompt_injection
```

7. If token/character offset mapping says a window overlaps the attack span but decoded text does not visibly contain the attack, mark it ambiguous and drop it.
8. Sample benign contrast windows from the same malicious carrier:

```text
before-attack context windows
after-attack context windows
same-document ordinary windows
wrapper-adjacent windows without attack text
```

9. Cap repeated windows per carrier document and per attack text hash.

Required metadata for every generated malicious-carrier window:

```json
{
  "source_document_id": "...",
  "carrier_document_id": "...",
  "window_index": 12,
  "window_count": 31,
  "attack_text_hash": "...",
  "attack_template_id": "...",
  "semantic_family": "...",
  "attack_visible_in_window": true,
  "label": "prompt_injection",
  "component": "embedded_visible_attack_windows",
  "text": "..."
}
```

## Attack Visibility Rule

A window is `prompt_injection` only if its decoded production window contains a material visible substring of the inserted attack span.

Use both:

```text
token-span overlap
decoded-text verification
```

Default materiality:

```text
at least 5 attack tokens visible
or at least 20 normalized attack characters visible
or a complete shorter standalone attack visible
```

Attack visibility levels:

| Visibility | Label | Use |
| --- | --- | --- |
| `full` | `prompt_injection` | train |
| `partial_material` | `prompt_injection` | train, capped |
| `partial_ambiguous` | drop | do not train |

For every attack-derived window, store:

```json
{
  "attack_token_overlap_count": 37,
  "attack_token_total": 52,
  "attack_overlap_ratio": 0.71,
  "attack_start_token": 1234,
  "attack_end_token": 1286,
  "window_token_start": 1152,
  "window_token_end": 1406
}
```

## Deduplication And Leakage Exclusion

Build an exclusion index from previous artifacts before sampling V17 rows.

Use prior artifacts only to populate:

```text
excluded_exact_text_hashes
excluded_normalized_text_hashes
excluded_source_document_ids
excluded_carrier_document_ids
excluded_attack_text_hashes
excluded_attack_template_ids where exact generated instance was reused
```

Reject a V17 row if it has:

```text
exact normalized text overlap with excluded data
same source document as locked/evaluation data
same carrier document as locked/evaluation data
same attack text hash as excluded data
same attack template id as excluded data
same generated attack instance as locked/evaluation data
near-duplicate overlap above configured threshold
```

The final row builder must enforce these exclusions at row level, not only during source-document collection.

Within V17, deduplicate by:

```text
normalized_text_hash
window_text_hash
source_document_id
carrier_document_id
attack_text_hash
attack_template_id
semantic_family
```

Deduplication must happen before the train/validation split.

## Source And Template Caps

The builder must prevent source, carrier, and template dominance:

```text
max rows per carrier_document_id: 12
max rows per attack_text_hash: 20
max rows per attack_template_id: 2,000
max rows per ordinary semantic_family: 8,000
critical semantic families may exceed 8,000 only when explicitly reported
no single raw benign source should dominate the benign class
```

## Split Policy

Split before final row export by groups, not random rows.

Group keys:

```text
source_document_id
carrier_document_id
attack_text_hash
attack_template_id
generated_instance_id
```

No validation row may share an exact group key with train.

High-level `semantic_family` may overlap between train and validation. This is intentional: the model must be evaluated on held-out phrasings from known critical families such as system-prompt disclosure, developer-prompt disclosure, hidden routing, and tool-list disclosure.

The report must distinguish:

```text
seen semantic families with unseen instances
held-out semantic subfamilies
fully unseen semantic families
```

A separate stress validation split may hold out entire semantic families, but this must not be the only validation set.

Validation target:

```text
25K-30K rows
both labels present
all major components present
Russian critical attack windows present
embedded attack windows present
benign carrier contrast windows present
wrapper/redaction benign and attack windows present
```

## Output Dataset Format

The final Hugging Face dataset must be compatible with the current training script:

```json
{
  "text": "...",
  "label": 0,
  "source_name": "v17_proper_benign_prod_windows"
}
```

Label mapping:

```text
0 = not_prompt_injection
1 = prompt_injection
```

Extra metadata columns are allowed and recommended:

```json
{
  "component": "...",
  "language": "...",
  "category": "...",
  "source_document_id": "...",
  "carrier_document_id": "...",
  "window_index": 0,
  "window_count": 10,
  "attack_visible_in_window": false,
  "semantic_family": null,
  "attack_text_hash": null,
  "split_group_id": "..."
}
```

## Required Build Artifacts

The builder must produce:

```text
training-dataset-v17-clean-windowed/
training-dataset-v17-clean-windowed-validation/
training-dataset-v17-clean-windowed-report.json
v17-source-document-manifest.jsonl
v17-attack-bank.jsonl
v17-dropped-invalid-examples.jsonl
v17-leakage-exclusion-report.json
v17-component-samples/
```

## Report Requirements

The report must include:

```text
total rows
train rows
validation rows
label distribution
component distribution
language distribution
category distribution
semantic-family distribution
attack-visible true/false counts
carrier-contrast counts
window-count buckets
deduplication removals
leakage exclusions
train/validation group-overlap checks
examples dropped as ambiguous attack-overlap windows
source-origin audit
```

Hard pass/fail checks:

```text
no training rows sourced from V10-V16 prepared datasets
0 exact duplicate texts across train/validation
0 shared source_document_id across train/validation
0 shared carrier_document_id across train/validation
0 shared attack_text_hash across train/validation
0 shared attack_template_id across train/validation
all prompt_injection rows have attack_visible_in_window = true
all not_prompt_injection rows have attack_visible_in_window = false
```

## Pre-Training Review Checklist

Before V17 training, confirm:

```text
1. Total rows are 180K-250K, preferably near 250K.
2. Attack/benign balance is near 50/50.
3. Russian share is 70-75%.
4. Benign production/document windows are at least 90K total.
5. Attack windows are visible-span verified.
6. Benign carrier contrast windows exist for malicious carriers.
7. No prior V10-V16 prepared dataset rows were imported.
8. No train/validation group leakage exists.
9. Validation has both labels and all major components.
10. Report contains examples for every component.
```

## Builder Command Shape

After the V17 builder is implemented and reviewed, run:

```powershell
uv run python build_v17_clean_windowed_dataset.py `
  --tokenizer-id microsoft/mdeberta-v3-base `
  --output-dir .\training-dataset-v17-clean-windowed `
  --validation-output-dir .\training-dataset-v17-clean-windowed-validation `
  --report-json .\training-dataset-v17-clean-windowed-report.json `
  --target-total-rows 250000 `
  --validation-rows 25000 `
  --seed 47
```

## V17 Training Command

Train from the base model, not from V13/V16:

```powershell
uv run python train_mdeberta_ru_prompt_injection_option_b.py `
  --device cuda `
  --teacher-device cuda `
  --bf16 `
  --student-model microsoft/mdeberta-v3-base `
  --prepared-dataset-dir .\training-dataset-v17-clean-windowed `
  --output-dir .\mdeberta-ru-prompt-injection-v17-clean-scratch `
  --learning-rate 1e-5 `
  --epochs 2 `
  --distill-weight 0.0 `
  --skip-teacher `
  --last-n-layers 12 `
  --train-batch-size 16 `
  --eval-batch-size 64 `
  --gradient-accumulation-steps 2 `
  --checkpoint-steps 500 `
  --save-total-limit 12 `
  --optim adamw_torch_fused `
  --group-by-length `
  --tf32 `
  --torch-num-threads 6 `
  --preflight-steps 2 `
  --rebuild-stage-cache `
  --no-trainer-auto-resume
```

## Post-Build Diagnostic Validation

Before training V17, validate the base-line previous models on the corrected diagnostic corpora only for comparison:

```text
proper_benign_windows
proper_critical_attack_windows
proper_malicious_dev_documents
proper_benign_prod_dev_documents
v13_critical_ru
```

After V17 training, evaluate V17 on the same diagnostics at:

```text
0.82, 0.90, 0.95, 0.99, 0.999, 0.9995
```

Primary diagnostic gates:

```text
proper benign-window FPR < 1%
proper critical attack-window recall >= 99%
proper malicious-document recall >= 99%
Russian critical attack-window recall >= 99%
```

Secondary monitoring:

```text
proper benign production-document FPR
false-positive windows per benign document
attack windows missed per malicious document
malicious documents with all attack windows missed
```
