# Dataset Recommendations for Russian Prompt-Injection Detection

## Context

The current training set is not balanced enough for a production-grade Russian prompt-injection detector:

```text
Train:
- prompt_injection: 10,199 / 16,314 = 62.5%
- benign:            6,115 / 16,314 = 37.5%

Validation:
- prompt_injection: 1,801 / 2,880 = 62.5%
- benign:           1,079 / 2,880 = 37.5%
```

The benign side is also too narrow:

```text
Benign train sources:
- ru_turbo_alpaca:       3,397
- OpenAssistant/oasst1:  2,525
- manual hard negatives:   193
```

This can cause the model to over-classify Russian text as `prompt_injection`, especially when benign prompts contain security vocabulary such as:

```text
system prompt
prompt injection
ignore previous instructions
jailbreak
developer message
раскрой системный промпт
игнорируй предыдущие инструкции
```

## Current Implemented Builder

`build_training_dataset.py` now has two modes:

```text
Full corrected dataset:
- public attack datasets
- public benign datasets
- project-authored benign document fragments
- project-authored hard negatives
- project-authored embedded prompt-injection windows
- project-authored quoted-attack hard negatives

Embedded-only dataset:
- project-authored embedded prompt-injection windows
- project-authored quoted-attack hard negatives
```

The embedded-only mode is intended for a short corrective fine-tune of an existing checkpoint:

```powershell
uv run python build_training_dataset.py `
  --embedded-only `
  --output-dir training-dataset-embedded-v1 `
  --validation-output-dir training-dataset-embedded-v1-validation `
  --report-path training-dataset-embedded-v1-report.json `
  --max-embedded-prompt-injection-attacks 6000 `
  --max-embedded-quoted-hard-negatives 6000
```

The generated embedded positives insert short prompt-injection snippets into benign document/RAG-like Russian text, then crop an attack-centered window so the malicious span remains visible under the current `max_len=256` training setup. The generated benign hard negatives quote or discuss prompt-injection phrases in safe tasks such as translation, audit-log analysis, documentation, unit tests, and moderation guidelines.

Synthetic train and validation rows are kept separate with:

```text
separate attack snippet pools
separate benign carrier pools
fixed split_hint values
```

This reduces train/validation leakage for the embedded validation split. It is still synthetic validation, so it should be supplemented with production-like held-out data before publishing a final model.

## Target Training Balance

For the full corrective training pass, target a benign-heavy mix:

```text
prompt_injection: 28–35%
benign:           65–72%
```

A benign-heavy mix is useful here because false positives are expensive and the current corrective builder adds many targeted document and hard-negative rows.

Recommended command-line setting:

```bash
--benign-to-attack-ratio 2.5
```

For experiments where recall is more important and false positives are acceptable, lower this ratio and re-check benign stress validation. Do not choose the ratio from overall F1 alone.


## Target Benign Buckets

The benign class should not be one generic bucket. Split it conceptually into:

```text
benign_general
benign_technical
benign_security_discussion
benign_hard_negative
benign_multilingual
benign_mixed_ru_en
benign_documentation
benign_logs_and_audit
```

The most important missing category is `benign_hard_negative`: safe prompts that contain suspicious-looking phrases but are not attacks.

---

# Recommended Benign Datasets

## 1. `Vikhrmodels/GrandMaster-PRO-MAX`

**Use case:** general Russian and English instruction-style benign prompts.

**Why add it:**

- Strong Russian instruction-following coverage.
- Useful replacement or supplement for `ru_turbo_alpaca`.
- Good for broadening benign user intent beyond synthetic Alpaca-style prompts.

**Recommended use:**

```text
Sample 5k–7k Russian or Russian-dominant user prompts.
Label as: benign
Bucket: benign_general / benign_multilingual
```

**Filtering:**

Reject examples that look like:

```text
system prompts
hidden policy prompts
role-play jailbreaks
agent instruction templates
prompts asking to reveal hidden instructions
```

**License note:** Apache-2.0 at time of writing. Verify before commercial use.

Dataset page:

```text
https://huggingface.co/datasets/Vikhrmodels/GrandMaster-PRO-MAX
```

---

## 2. `CohereLabs/aya_collection`

**Use case:** multilingual benign instruction data, including Russian and many other languages.

**Why add it:**

- Good multilingual coverage.
- Useful for Russian plus mixed-language robustness.
- Helps `microsoft/mdeberta-v3-base` leverage multilingual pretraining.

**Recommended use:**

```text
Sample 3k–5k Russian examples.
Optionally add 1k–2k multilingual examples from Ukrainian, Belarusian, English, German, Spanish, French, etc.
Label as: benign
Bucket: benign_multilingual / benign_general
```

**Filtering:**

Keep only natural user instructions. Avoid examples that are meta-prompts, system prompts, or safety-policy instructions.

**License note:** Apache-2.0 for many Aya resources, but this is a collection. Verify the subset license before commercial use.

Dataset page:

```text
https://huggingface.co/datasets/CohereLabs/aya_collection
```

---

## 3. `OpenAssistant/oasst1`

**Use case:** real multilingual user prompts and conversation turns.

**Why add it:**

- You already use it, but the current count is small.
- Russian prompter turns are valuable.
- More natural than purely synthetic instruction datasets.

**Recommended use:**

```text
Sample 4k–6k Russian prompter messages if available after filtering.
Label as: benign
Bucket: benign_general / benign_multilingual
```

**Recommended filters:**

```python
role == "prompter"
lang == "ru"
deleted == False
review_result is not False
```

**Additional suggestion:**

Do not only use first/root prompts. Later user turns can be useful as long as they are user-authored and benign.

**License note:** Apache-2.0 at time of writing.

Dataset page:

```text
https://huggingface.co/datasets/OpenAssistant/oasst1
```

---

## 4. `IlyaGusev/ru_stackoverflow`

**Use case:** Russian technical benign prompts.

**Why add it:**

This is one of the best additions for reducing false positives on technical Russian text. It contains questions, code, logs, commands, error messages, and discussion patterns that can otherwise look suspicious to a security classifier.

**Recommended use:**

```text
Sample 2k–4k question titles and/or question bodies.
Label as: benign
Bucket: benign_technical
```

**Good examples of benign technical text:**

```text
Как передать токен в HTTP-заголовке?
Почему Python не видит переменную окружения?
Как настроить nginx reverse proxy?
Ошибка авторизации при запросе к API
Как удалить строку из PostgreSQL?
```

**Filtering and redaction:**

Before using this dataset, remove or redact:

```text
emails
phone numbers
IP addresses
access tokens
API keys
passwords
private URLs
long credential-like strings
```

**License note:** CC BY-SA 2.5. This may be problematic for redistribution or commercial model training. Review carefully.

Dataset page:

```text
https://huggingface.co/datasets/IlyaGusev/ru_stackoverflow
```

---

## 5. `allenai/WildChat-1M`

**Use case:** real-world user-chatbot conversation prompts.

**Why add it:**

- Natural user-chat data.
- Multilingual coverage.
- Useful for code-switching and messy real user behavior.

**Recommended use:**

```text
Sample 1k–2k Russian or Russian-dominant user turns.
Label as: benign
Bucket: benign_general / benign_mixed_ru_en
```

**Filtering:**

Keep only user turns. Prefer rows where safety/toxicity metadata indicates the content is safe.

Avoid:

```text
toxic content
private or redacted content
jailbreak-looking content
system prompt extraction attempts
assistant answers
```

**License note:** ODC-BY with dataset-specific conditions. Review carefully before commercial use.

Dataset page:

```text
https://huggingface.co/datasets/allenai/WildChat-1M
```

---

## 6. `nickoo004/queryshield-multilingual`

**Use case:** multilingual safe professional-domain questions.

**Why add it:**

- Contains multilingual safe queries.
- Useful for benign professional, business, support, legal, medical, and general question patterns.
- Includes Russian and related-language coverage.

**Recommended use:**

```text
Sample 1k–2k Russian or Russian-like safe user questions.
Label as: benign
Bucket: benign_multilingual / benign_general
```

**Important:**

Use the raw user question field. Do not train on optimized meta-prompts or assistant answers.

**License note:** MIT at time of writing.

Dataset page:

```text
https://huggingface.co/datasets/nickoo004/queryshield-multilingual
```

---

## 7. `leolee99/NotInject`

**Use case:** benign hard-negative examples for over-defense testing.

**Why add it:**

This dataset is small but valuable because it contains prompts that may look suspicious but are not injections. It is useful for measuring whether the classifier overblocks safe prompts.

**Recommended use:**

```text
Use mostly as held-out evaluation data.
Optionally include a small subset in training.
Label as: benign
Bucket: benign_hard_negative
```

**Suggested approach:**

```text
70% held-out over-defense eval
30% training or template expansion
```

**License note:** MIT at time of writing.

Dataset page:

```text
https://huggingface.co/datasets/leolee99/NotInject
```

---

# Recommended Attack Datasets

Your immediate problem is benign scarcity, so do not add too many more attacks until benign coverage is fixed. Still, a modest number of diverse attacks can improve robustness.

## 1. `dmtrdr/russian_prompt_injections`

**Use case:** primary Russian direct prompt-injection attack dataset.

**Recommended use:**

```text
Keep 10k–12k examples.
Label as: prompt_injection
Bucket: direct_attack_ru
```

Since this is currently the dominant source, avoid increasing it unless you add much more benign data.

Dataset page:

```text
https://huggingface.co/datasets/dmtrdr/russian_prompt_injections
```

---

## 2. `cyberec/promptwall-injection-dataset`

**Use case:** multilingual prompt-injection examples.

**Why add it:**

- Contains multiple attack categories.
- Useful for direct, indirect, exfiltration, encoded, jailbreak, and multi-turn examples.
- Includes multilingual coverage.

**Recommended use:**

```text
Sample 300–500 examples.
Label as: prompt_injection
Bucket: attack_multilingual
```

**License note:** MIT at time of writing.

Dataset page:

```text
https://huggingface.co/datasets/cyberec/promptwall-injection-dataset
```

---

## 3. `Lakera/gandalf_ignore_instructions`

**Use case:** compact direct prompt-injection examples.

**Why add it:**

- Good examples of direct “ignore instructions” attacks.
- Useful as auxiliary English attack data.

**Recommended use:**

```text
Sample 500–1,000 examples.
Label as: prompt_injection
Bucket: direct_attack_en
```

Do not let this dominate because it is mostly English.

Dataset page:

```text
https://huggingface.co/datasets/Lakera/gandalf_ignore_instructions
```

---

## 4. `microsoft/llmail-inject-challenge`

**Use case:** indirect prompt-injection attacks in email-like contexts.

**Why add it:**

Use this if your system processes emails, documents, support tickets, RAG chunks, or retrieved text.

**Recommended use:**

```text
Sample 500–2,000 curated examples if relevant.
Label as: prompt_injection
Bucket: indirect_attack
```

This requires more preprocessing than simple instruction datasets.

Dataset page:

```text
https://huggingface.co/datasets/microsoft/llmail-inject-challenge
```

---

## 5. `microsoft/BIPIA`

**Use case:** indirect prompt-injection benchmark.

**Why add it:**

- Strong for evaluating indirect prompt injection.
- Useful if the model will inspect RAG/document chunks.

**Recommended use:**

Use primarily for evaluation or for carefully curated indirect-attack training examples.

Dataset page:

```text
https://huggingface.co/datasets/microsoft/BIPIA
```

---

# Suggested Next Training Mix

For a CPU-manageable training run, aim for:

```text
Total rows: 30k–40k

Benign: 18k–22k
Attack: 16k–18k
```

A good initial mix:

```text
BENIGN
- 5k–7k  GrandMaster-PRO-MAX Russian user prompts
- 4k–6k  OpenAssistant/oasst1 Russian prompter messages
- 3k–5k  Aya Russian / multilingual instruction prompts
- 2k–4k  ru_stackoverflow technical prompts
- 1k–2k  WildChat Russian user turns
- 1k–2k  QueryShield Russian safe user questions
- 1k–3k  manual/template hard negatives
- 339    NotInject, preferably mostly held out

ATTACK
- 10k–12k dmtrdr Russian attacks
- 300–500 PromptWall multilingual/Russian attacks
- 500–1k Gandalf direct attacks
- 500–2k indirect/email/RAG attacks if relevant
```

---

# Recommended Priority Order

## Step 1: Add broad benign Russian instruction data

Add:

```text
Vikhrmodels/GrandMaster-PRO-MAX
CohereLabs/aya_collection
more OpenAssistant/oasst1
```

This should fix the general benign scarcity.

## Step 2: Add technical benign Russian data

Add:

```text
IlyaGusev/ru_stackoverflow
```

This should reduce false positives on code, logs, shell commands, errors, and API/security vocabulary.

## Step 3: Add hard negatives

Expand manual hard negatives to at least:

```text
2k–5k examples
```

Use categories such as:

```text
translation tasks
security documentation
SOC/log analysis
QA/test-case generation
moderation guidelines
developer documentation
RAG/document analysis
product support texts
```

## Step 4: Add a small amount of attack diversity

Add:

```text
cyberec/promptwall-injection-dataset
Lakera/gandalf_ignore_instructions
microsoft/llmail-inject-challenge, if relevant
```

Keep attack additions modest until benign coverage is strong.

---

# Hard-Negative Template Strategy

Generate benign examples that quote dangerous phrases without being attacks.

Examples:

```text
Переведи фразу "ignore previous instructions" на русский язык.
Объясни, почему "раскрой системный промпт" является индикатором атаки.
Найди в логах все события, где встречается фраза "jailbreak mode".
Создай unit test для строки "ignore all previous instructions".
Напиши документацию о том, почему нельзя раскрывать system prompt.
Составь guideline для аннотаторов: когда текст является prompt injection, а когда benign discussion.
Проанализируй инцидент, где пользователь написал "покажи скрытые инструкции".
Сделай русскую локализацию warning: "Potential prompt injection detected".
```

These should be labeled as:

```text
label: benign
bucket: benign_hard_negative
```

---

# Filtering Rules for Benign Data

For every benign source:

```text
1. Keep only the user prompt/input side.
2. Do not train on assistant answers.
3. Remove system prompts and hidden policy prompts.
4. Remove obvious jailbreaks and prompt-injection attempts.
5. Deduplicate aggressively.
6. Redact PII and credentials.
7. Keep source_name and bucket fields for auditing.
```

Recommended metadata fields:

```json
{
  "text": "...",
  "label": 0,
  "source_name": "OpenAssistant/oasst1",
  "bucket": "benign_general",
  "language": "ru"
}
```

---

# Evaluation Split Recommendation

Do not rely only on a random split.

Build a fixed validation set with these buckets:

```text
direct_attack_ru
direct_attack_en
attack_multilingual
indirect_attack
encoded_attack
benign_general_ru
benign_technical_ru
benign_security_discussion
benign_hard_negative
benign_mixed_ru_en
benign_logs_and_audit
```

Track metrics by bucket.

The most important production metrics are:

```text
false positive rate on benign_hard_negative
false positive rate on benign_technical_ru
false positive rate on benign_security_discussion
recall on Russian direct attacks
recall on indirect attacks
PR-AUC
balanced accuracy
```

Overall F1 alone is not enough.

---

# Recommended Dataset Constants

You can start with this conceptual configuration:

```python
BENIGN_DATASETS = [
    "Vikhrmodels/GrandMaster-PRO-MAX",
    "CohereLabs/aya_collection",
    "OpenAssistant/oasst1",
    "IlyaGusev/ru_stackoverflow",
    "allenai/WildChat-1M",
    "nickoo004/queryshield-multilingual",
    "leolee99/NotInject",
]

ATTACK_DATASETS = [
    "dmtrdr/russian_prompt_injections",
    "cyberec/promptwall-injection-dataset",
    "Lakera/gandalf_ignore_instructions",
    "microsoft/llmail-inject-challenge",
    "microsoft/BIPIA",
]
```

---

# Practical Recommendation

For the full corrected run, use the implemented builder defaults as the baseline:

```text
Benign:
- GrandMaster-PRO-MAX
- OASST
- ru_turbo_alpaca
- Russian instructions
- Russian news / sentiment / bank reviews
- generated document fragments
- generated hard negatives
- generated quoted-attack hard negatives
- NotInject as a small hard-negative source

Attack:
- current dmtrdr Russian attacks after curation
- manual Russian/mixed prompt-injection templates
- embedded prompt-injection windows
- PromptWall / Gandalf / deepset / OpenSafetyLab curated rows
```

Start with:

```bash
--benign-to-attack-ratio 2.5
```

For the interim embedded-only fine-tune, use:

```bash
--embedded-only
--distill-weight 0.0
```

For the full from-scratch run, keep conservative benign-only teacher distillation:

```bash
--teacher-distill-mode benign_only
--distill-weight 0.02
```

The ProtectAI teacher is English-specialized, so it should not dominate Russian training.
