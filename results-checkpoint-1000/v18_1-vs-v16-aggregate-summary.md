# V18.1 Eval Aggregate Summary

Primary display threshold: `0.82`

| model | corpus | documents | fp | fn | precision | recall | f1 | benign_fp_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v18_1 | benign_stress_windows | 236 | 0 | 0 | 0 | 0 | 0 | 0 |
| v18_1 | blind_acceptance_benign_documents | 1500 | 0 | 0 | 0 | 0 | 0 | 0 |
| v18_1 | blind_acceptance_corporate_benign_windows | 400 | 40 | 0 | 0 | 0 | 0 | 0.1 |
| v18_1 | blind_acceptance_critical_ru_windows | 400 | 0 | 0 | 1 | 1 | 1 | 0 |
| v18_1 | blind_acceptance_malicious_documents | 800 | 0 | 242 | 1 | 0.6975 | 0.821797 | 0 |
| v18_1 | proper_benign_prod_dev_documents | 3000 | 0 | 0 | 0 | 0 | 0 | 0 |
| v18_1 | proper_benign_windows | 2266 | 0 | 0 | 0 | 0 | 0 | 0 |
| v18_1 | proper_critical_attack_windows | 3512 | 0 | 353 | 1 | 0.899487 | 0.947084 | 0 |
| v18_1 | proper_malicious_dev_documents | 1161 | 0 | 260 | 1 | 0.776055 | 0.873909 | 0 |
