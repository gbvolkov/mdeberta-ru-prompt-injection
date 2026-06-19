# V16 Remaining Error Inspection

This is diagnostic/mining analysis over already inspected corpora, not blind acceptance evidence.

## critical_ru_fn_at_0.95

Rows: `27`

### by_tag

- `embedded_context`: 27
- `exact_old_v13_miss_unresolved`: 26
- `used_as_v16_training_diagnostic_seed`: 21
- `wrapper_or_marker_adjacent`: 9

### by_category

- `long_document_embedded_attack`: 6
- `job_descriptions`: 6
- `support_documentation`: 4
- `admin_instructions`: 4
- `technical_documentation`: 3
- `legal_templates`: 2
- `hr_policies`: 1
- `meeting_minutes`: 1

### by_source_name

- `v13_critical_ru_embedded_hard_positive_job_descriptions`: 5
- `v13_critical_ru_embedded_hard_positive_admin_instructions`: 4
- `v13_critical_ru_embedded_hard_positive_support_documentation`: 3
- `v10_attack_replay_v12_critical_ru_embedded_attack_legal_templates`: 2
- `v10_attack_replay_v12_critical_ru_embedded_attack_job_descriptions`: 2
- `v13_critical_ru_embedded_hard_positive_technical_documentation`: 2
- `v13_critical_ru_wrapper_hard_positive_legal_templates`: 2
- `v13_critical_ru_wrapper_hard_positive_technical_documentation`: 1
- `v10_attack_replay_v12_critical_ru_embedded_attack_knowledge_base`: 1
- `v13_critical_ru_wrapper_hard_positive_support_documentation`: 1
- `v10_attack_replay_v12_critical_ru_embedded_attack_corporate_procedures`: 1
- `v13_critical_ru_wrapper_hard_positive_job_descriptions`: 1
- `v13_critical_ru_embedded_hard_positive_hr_policies`: 1
- `v13_critical_ru_wrapper_hard_positive_meeting_minutes`: 1

### by_language

- `ru`: 21
- `en`: 6

### by_window_count_bucket

- `1`: 24
- `2-4`: 3

### by_score_band

- `score_lt_0.01`: 12
- `score_0.01_0.5`: 10
- `score_0.5_0.82`: 4
- `score_0.82_0.95`: 1

## benign_prod_dev_fp_at_0.95

Rows: `58`

### by_tag

- `ordinary_business_doc_or_unclear`: 34
- `system_policy_instruction_words`: 14
- `url_or_link_heavy`: 12
- `crime_security_investigation_language`: 8
- `secret_confidential_language`: 2

### by_category

- `job_descriptions`: 16
- `technical_documentation`: 11
- `admin_instructions`: 6
- `unsafe_non_injection`: 5
- `knowledge_base`: 4
- `meeting_minutes`: 4
- `safety_policies`: 4
- `corporate_procedures`: 3
- `support_documentation`: 3
- `hr_policies`: 1
- `legal_templates`: 1

### by_source_name

- `c4_ru`: 26
- `fineweb2_ru`: 19
- `fineweb_en`: 5
- `toxic_chat`: 4
- `stackexchange`: 2
- `wikipedia_ru`: 1
- `real_toxicity_prompts`: 1

### by_language

- `ru`: 46
- `en`: 12

### by_window_count_bucket

- `5-20`: 27
- `21-50`: 12
- `2-4`: 9
- `101+`: 5
- `51-100`: 4
- `1`: 1

### by_score_band

- `score_gte_0.99`: 35
- `score_0.95_0.99`: 23

## benign_prod_dev_fp_at_0.99

Rows: `35`

### by_tag

- `ordinary_business_doc_or_unclear`: 24
- `system_policy_instruction_words`: 8
- `url_or_link_heavy`: 5
- `crime_security_investigation_language`: 5
- `secret_confidential_language`: 2

### by_category

- `job_descriptions`: 8
- `technical_documentation`: 5
- `admin_instructions`: 4
- `safety_policies`: 4
- `meeting_minutes`: 3
- `support_documentation`: 3
- `unsafe_non_injection`: 3
- `knowledge_base`: 2
- `corporate_procedures`: 1
- `hr_policies`: 1
- `legal_templates`: 1

### by_source_name

- `c4_ru`: 15
- `fineweb2_ru`: 11
- `fineweb_en`: 4
- `stackexchange`: 2
- `toxic_chat`: 2
- `real_toxicity_prompts`: 1

### by_language

- `ru`: 26
- `en`: 9

### by_window_count_bucket

- `5-20`: 19
- `21-50`: 6
- `2-4`: 4
- `51-100`: 3
- `101+`: 2
- `1`: 1

### by_score_band

- `score_gte_0.99`: 35

