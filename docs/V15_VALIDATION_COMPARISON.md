# V15 Validation Comparison

Compared models:

- `mdeberta-ru-prompt-injection-v12-critical-correction-ft`
- `mdeberta-ru-prompt-injection-v13-critical-correction-ft`
- `mdeberta-ru-prompt-injection-v15-anchored-critical-correction-ft`

Source:

```text
validation-comparison-v10-v13-core/comparison-summary.json
```

This is a diagnostic comparison over previously used or inspected corpora, not a blind acceptance report.

## Decision

V15 is not a release candidate.

V15 strongly improves benign false-positive behavior versus V13, but it reduces critical Russian recall. The regression is most visible at high thresholds, exactly where we wanted a better operating point.

No V12/V13/V15 threshold satisfies all required gates:

```text
benign_prod_dev FPR < 1%
v13_benign_windows FPR < 1%
malicious_dev recall >= 99%
v13_critical_ru recall >= 99%
```

## Main Comparison

| Threshold | Model | benign_prod_dev FPR | v13_benign_windows FPR | malicious_dev recall | v13_critical_ru recall |
|---:|---|---:|---:|---:|---:|
| 0.82 | V12 | 8.03% | 1.97% | 99.93% | 98.37% |
| 0.82 | V13 | 5.17% | 0.77% | 99.93% | 98.86% |
| 0.82 | V15 | 1.13% | 0.43% | 99.93% | 98.37% |
| 0.95 | V12 | 4.87% | 0.77% | 99.93% | 98.01% |
| 0.95 | V13 | 3.80% | 0.43% | 99.93% | 98.58% |
| 0.95 | V15 | 0.40% | 0.17% | 99.87% | 97.84% |
| 0.99 | V12 | 2.10% | 0.34% | 99.93% | 97.28% |
| 0.99 | V13 | 1.83% | 0.34% | 99.93% | 98.37% |
| 0.99 | V15 | 0.13% | 0.09% | 99.73% | 96.67% |
| 0.999 | V12 | 0.23% | 0.09% | 99.73% | 93.86% |
| 0.999 | V13 | 0.33% | 0.09% | 99.87% | 97.07% |
| 0.999 | V15 | 0.00% | 0.00% | 97.67% | 91.66% |
| 0.9995 | V12 | 0.10% | 0.00% | 99.47% | 90.52% |
| 0.9995 | V13 | 0.10% | 0.00% | 99.80% | 96.14% |
| 0.9995 | V15 | 0.00% | 0.00% | 95.60% | 87.47% |

## V15 Versus V13

| Threshold | Metric | V13 | V15 | Delta, pp |
|---:|---|---:|---:|---:|
| 0.82 | benign_prod_dev FPR | 5.17% | 1.13% | -4.03 |
| 0.82 | v13_benign_windows FPR | 0.77% | 0.43% | -0.34 |
| 0.82 | malicious_dev recall | 99.93% | 99.93% | 0.00 |
| 0.82 | critical_ru recall | 98.86% | 98.37% | -0.49 |
| 0.95 | benign_prod_dev FPR | 3.80% | 0.40% | -3.40 |
| 0.95 | v13_benign_windows FPR | 0.43% | 0.17% | -0.26 |
| 0.95 | malicious_dev recall | 99.93% | 99.87% | -0.07 |
| 0.95 | critical_ru recall | 98.58% | 97.84% | -0.73 |
| 0.99 | benign_prod_dev FPR | 1.83% | 0.13% | -1.70 |
| 0.99 | v13_benign_windows FPR | 0.34% | 0.09% | -0.26 |
| 0.99 | malicious_dev recall | 99.93% | 99.73% | -0.20 |
| 0.99 | critical_ru recall | 98.37% | 96.67% | -1.71 |
| 0.999 | benign_prod_dev FPR | 0.33% | 0.00% | -0.33 |
| 0.999 | v13_benign_windows FPR | 0.09% | 0.00% | -0.09 |
| 0.999 | malicious_dev recall | 99.87% | 97.67% | -2.20 |
| 0.999 | critical_ru recall | 97.07% | 91.66% | -5.41 |
| 0.9995 | benign_prod_dev FPR | 0.10% | 0.00% | -0.10 |
| 0.9995 | v13_benign_windows FPR | 0.00% | 0.00% | 0.00 |
| 0.9995 | malicious_dev recall | 99.80% | 95.60% | -4.20 |
| 0.9995 | critical_ru recall | 96.14% | 87.47% | -8.66 |

## Exact Counts At Key Thresholds

| Threshold | Model | benign_prod_dev FP | v13_benign_windows FP | malicious_dev TP/FN | critical_ru TP/FN |
|---:|---|---:|---:|---:|---:|
| 0.82 | V12 | 241/3000 | 23/1166 | 1499/1 | 2419/40 |
| 0.82 | V13 | 155/3000 | 9/1166 | 1499/1 | 2431/28 |
| 0.82 | V15 | 34/3000 | 5/1166 | 1499/1 | 2419/40 |
| 0.95 | V12 | 146/3000 | 9/1166 | 1499/1 | 2410/49 |
| 0.95 | V13 | 114/3000 | 5/1166 | 1499/1 | 2424/35 |
| 0.95 | V15 | 12/3000 | 2/1166 | 1498/2 | 2406/53 |
| 0.99 | V12 | 63/3000 | 4/1166 | 1499/1 | 2392/67 |
| 0.99 | V13 | 55/3000 | 4/1166 | 1499/1 | 2419/40 |
| 0.99 | V15 | 4/3000 | 1/1166 | 1496/4 | 2377/82 |

## Interpretation

V15 learned the benign correction too strongly:

- benign production FPR at `0.95`: V13 `3.80%` -> V15 `0.40%`
- benign production FPR at `0.99`: V13 `1.83%` -> V15 `0.13%`
- benign validation-window FPR is also lower at every threshold

But the same shift moved too many critical Russian attack examples below the decision boundary:

- critical Russian recall at `0.95`: V13 `98.58%` -> V15 `97.84%`
- critical Russian recall at `0.99`: V13 `98.37%` -> V15 `96.67%`
- critical Russian recall at `0.999`: V13 `97.07%` -> V15 `91.66%`

The result is a conservative model that suppresses false positives but loses the critical family we were trying to preserve.

## Recommendation

Do not use V15 as the next parent model.

Use V13 as the parent for the next correction. If V15 data is reused, reduce the benign hard-negative pressure and rebalance toward hard critical Russian positives that V13 misses at `0.95` to `0.999`, not broad/easy positives.
