# Benign Stress Validation Results

Dataset: `benign-stress-validation`

This is a benign-only stress set for overblocking checks. It targets Russian factual/document fragments such as historical address bullets, biographical notes, catalog snippets, local-history text, and cultural references.

Rows: 236

The current version contains:

- 186 `full_text` rows
- 50 `sentence` rows derived from full text with `split_text_by_sentences`

All rows are labeled `benign`, so `precision`, `recall`, and `f1` for the positive class are not meaningful here. The useful metric is false positives:

```text
false_positive_rate = false_positives / rows
```

## Threshold Comparison

| Threshold | Benign passed | False positives | False positive rate |
| --------: | ------------: | --------------: | ------------------: |
| 0.500000  |           123 |             113 |              47.88% |

## Threshold 0.5 By Bucket

| Bucket                              | Rows | False positives | False positive rate |
| ----------------------------------- | ---: | --------------: | ------------------: |
| `benign_historical_address_fragment` |   98 |              55 |              56.12% |
| `benign_biographical_note`          |   58 |              28 |              48.28% |
| `benign_document_bullet_fragment`   |   30 |              30 |             100.00% |
| `benign_catalog_fragment`           |   30 |               0 |               0.00% |
| `benign_local_history`              |    9 |               0 |               0.00% |
| `benign_literature_art_fragment`    |   11 |               0 |               0.00% |

## Interpretation

The model overblocks a specific benign pattern: short Russian document fragments with bullets, addresses, house numbers, names, dates, and incomplete historical notes. The sentence rows are less fragile than full bullet/address rows, but still show false positives: `manual_benign_stress_generated_sentence` has 12 false positives out of 50 rows at threshold `0.5`.

This is not a false-negative problem. These are false positives: benign texts classified as `prompt_injection`.

## Recommended Follow-Up

Add more benign training coverage for:

- historical address fragments
- bullet-list document snippets
- biographical notes with initials, dates, addresses, and house numbers
- local-history text with old street names and destroyed buildings
- incomplete factual snippets copied from generated summaries or OCR/document chunks

Keep this stress set, or a manually reviewed version of it, as a held-out regression validation set. Do not rely only on the original random validation split because it hides this overblocking pattern.
