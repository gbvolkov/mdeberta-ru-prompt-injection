# V16 Policy Diagnostic Summary

Status: `pass`

| Corpus | Docs | Windows | Proposed Policy-Added Attack Allow | Effective Policy-Added Attack Allow | Reviewer Scored Windows | Potential Allows |
|---|---:|---:|---:|---:|---:|---:|
| benign-stress-windows | 236 | 236 | 0 | 0 | 0 | 0 |
| blind-acceptance-benign-documents | 1500 | 3500 | 0 | 0 | 956 | 0 |
| blind-acceptance-corporate-benign-windows | 400 | 400 | 0 | 0 | 19 | 0 |
| blind-acceptance-critical-ru-windows | 400 | 400 | 0 | 0 | 0 | 0 |
| blind-acceptance-malicious-documents | 800 | 1891 | 0 | 0 | 219 | 0 |
| proper-benign-prod-dev-documents | 3000 | 54451 | 0 | 0 | 18 | 0 |
| proper-benign-windows | 2266 | 2772 | 0 | 0 | 0 | 0 |
| proper-critical-attack-windows | 3512 | 3966 | 0 | 0 | 15 | 0 |
| proper-malicious-dev-documents | 1161 | 16784 | 0 | 0 | 68 | 0 |

## Totals

```json
{
  "documents": 13275,
  "windows": 84400,
  "proposed_attack_allow_policy_added": 0,
  "effective_attack_allow_policy_added": 0,
  "proposed_attack_allow_baseline_v16_miss": 28,
  "effective_attack_allow_baseline_v16_miss": 0,
  "certified_auto_allow_count": 0,
  "reviewer_effective_auto_allow_count": 0,
  "reviewer_potential_allow_count": 0,
  "reviewer_scored_window_count": 1295,
  "reviewer_eligible_window_count": 1295,
  "direct_attack_anchor_block_count": 3843,
  "policy_security_mention_count": 9855
}
```
