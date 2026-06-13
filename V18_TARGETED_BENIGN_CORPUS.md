# V18 Targeted Benign Corpus

Use this process to enrich the underfilled benign categories separately from the broad FP corpus:

```text
hr_policies
corporate_procedures
safety_policies
support_documentation
```

Primary path: use `prepare_v18_direct_source_documents.py`. It fetches/crawls the configured source families directly and writes JSONL documents itself. Do not assume that `targeted_sources/*.jsonl` already exists.

Direct source families currently configured:

```text
GeoLibrary endpoints                         -> safety_policies, reports DNS/source failure if unavailable
Russian Legislative Corpus arXiv page        -> source discovery / metadata probe
Russian Legislative Corpus RusLawOD dataset  -> corporate_procedures, safety_policies
Professional standards registry              -> hr_policies, corporate_procedures
Yandex Cloud docs                            -> support_documentation, reports SSO/anti-bot redirect if blocked
Bitrix24 helpdesk                            -> support_documentation, corporate_procedures
Microsoft Learn ru-ru                        -> support_documentation, safety_policies
HF russian instruction sources               -> support_documentation, supplemental only
HH API vacancy probe                         -> hr_policies, reports 403 if blocked
```

Smoke-test every configured source:

```powershell
uv run python prepare_v18_direct_source_documents.py `
  --output-dir .\v18-direct-source-documents-smoke `
  --max-pages-per-source 25 `
  --max-depth 1 `
  --max-hf-scan-per-source 2000 `
  --hh-pages-per-query 1 `
  --max-documents-per-source 50 `
  --source-workers 4 `
  --no-progress `
  --allow-source-errors
```

Full targeted collection example:

```powershell
uv run python prepare_v18_direct_source_documents.py `
  --output-dir .\v18-direct-source-documents `
  --max-pages-per-source 5000 `
  --max-depth 3 `
  --max-hf-scan-per-source 300000 `
  --hh-pages-per-query 20 `
  --max-documents-per-source 50000 `
  --source-workers 6 `
  --no-progress `
  --allow-source-errors
```

The direct script writes:

```text
v18-direct-source-documents\v18-direct-source-documents.jsonl
v18-direct-source-documents\v18-direct-source-documents-report.json
v18-direct-source-documents\by_source\*.jsonl
v18-direct-source-documents\by_category\*.jsonl
```

Score the collected documents directly:

```powershell
uv run python run_false_positive_review.py `
  --model-id .\mdeberta-ru-prompt-injection-v16-critical-recall-restoration-ft `
  --input-jsonl .\v18-direct-source-documents\v18-direct-source-documents.jsonl `
  --threshold 0.82 `
  --window-batch-size 128 `
  --output-jsonl .\v18-fp-review-v16-direct-source-threshold-0.82.jsonl `
  --summary-json .\v18-fp-review-v16-direct-source-threshold-0.82-summary.json `
  --device cuda
```

Legacy path: `build_v18_targeted_benign_corpus.py` can still consume local exported files if you already have them, but it is not the primary process for fresh source preparation.

Local source spec format:

```text
path|category|source_name|origin|trust
```

`trust` is optional. Use `true` only when the source is already curated for that category. Without `trust`, rows must contain category keyword evidence.

Custom `--hf-source` inputs are restricted to `support_documentation`. If a Hugging Face source is intended for `hr_policies`, `corporate_procedures`, or `safety_policies`, first export/review/filter it to a local JSONL file and pass it through `--source-spec`.

Recommended source mapping:

```text
GeoLibrary / occupational safety exports                 -> safety_policies
Russian Legislative Corpus labor/safety/admin slices     -> corporate_procedures or safety_policies
Professional standards / labor function exports          -> hr_policies
Yandex Cloud / Bitrix24 / Microsoft Learn ru exports     -> support_documentation
```

Example:

```powershell
uv run python build_v18_targeted_benign_corpus.py `
  --output-jsonl .\v18-targeted-benign-corpus.jsonl `
  --report-json .\v18-targeted-benign-corpus-report.json `
  --tokenizer-id microsoft/mdeberta-v3-base `
  --source-spec ".\targeted_sources\geolibrary_ru.jsonl|safety_policies|geolibrary_ru|geolibrary_export|true" `
  --source-spec ".\targeted_sources\ru_legislative_safety.jsonl|safety_policies|ru_legislative_safety|russian_legislative_corpus|false" `
  --source-spec ".\targeted_sources\ru_legislative_admin.jsonl|corporate_procedures|ru_legislative_admin|russian_legislative_corpus|false" `
  --source-spec ".\targeted_sources\profstandards_ru.jsonl|hr_policies|profstandards_ru|professional_standards_export|true" `
  --source-spec ".\targeted_sources\support_docs_ru.jsonl|support_documentation|support_docs_ru|public_support_docs_export|true" `
  --include-hf-instruction-sources `
  --target-hr-policies 30000 `
  --target-corporate-procedures 50000 `
  --target-safety-policies 50000 `
  --target-support-documentation 50000 `
  --source-workers 4 `
  --checkpoint-dir .\v18-targeted-benign-corpus-checkpoints `
  --resume `
  --no-progress `
  --allow-source-errors `
  --allow-underfilled
```

Then score it separately:

```powershell
uv run python run_false_positive_review.py `
  --model-id .\mdeberta-ru-prompt-injection-v16-critical-recall-restoration-ft `
  --input-jsonl .\v18-targeted-benign-corpus.jsonl `
  --threshold 0.82 `
  --window-batch-size 128 `
  --output-jsonl .\v18-fp-review-v16-targeted-benign-threshold-0.82.jsonl `
  --summary-json .\v18-fp-review-v16-targeted-benign-threshold-0.82-summary.json `
  --device cuda
```

Pass both broad and targeted review files to external input preparation:

```powershell
--mined-review-jsonl .\v18-fp-review-v16-threshold-0.82.jsonl `
--mined-review-jsonl .\v18-fp-review-v16-targeted-benign-threshold-0.82.jsonl `
```
