#!/usr/bin/env python3
"""Prepare window-level statistics for the V16 FP/FN evaluation subset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_THRESHOLDS = (0.5, 0.82, 0.9, 0.95, 0.99, 0.999, 0.9995)
SCORE_BANDS = (
    ("<0.001", None, 0.001),
    ("0.001-0.01", 0.001, 0.01),
    ("0.01-0.10", 0.01, 0.10),
    ("0.10-0.50", 0.10, 0.50),
    ("0.50-0.82", 0.50, 0.82),
    ("0.82-0.90", 0.82, 0.90),
    ("0.90-0.95", 0.90, 0.95),
    ("0.95-0.99", 0.95, 0.99),
    ("0.99-0.999", 0.99, 0.999),
    ("0.999-0.9995", 0.999, 0.9995),
    (">=0.9995", 0.9995, None),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-document-results-jsonl",
        default="v16-blind-balanced-eval-results-20260617/document-results.jsonl",
    )
    parser.add_argument(
        "--error-document-results-jsonl",
        default="v16-error-window-results/document-results.jsonl",
    )
    parser.add_argument(
        "--error-window-results-jsonl",
        default="v16-error-window-results/window-results.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="v16-error-window-results/prepared-window-statistics",
    )
    parser.add_argument("--primary-threshold", type=float, default=0.82)
    parser.add_argument(
        "--thresholds",
        default=",".join(str(value) for value in DEFAULT_THRESHOLDS),
    )
    parser.add_argument("--progress-every", type=int, default=25000)
    parser.add_argument("--top-fp-windows", type=int, default=1000)
    return parser.parse_args()


def parse_thresholds(value: str) -> list[float]:
    thresholds = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    if not thresholds:
        raise ValueError("At least one threshold is required")
    return thresholds


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc


def safe_div(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def score_band(score: float) -> str:
    for label, lower, upper in SCORE_BANDS:
        if lower is None and score < upper:
            return label
        if upper is None and score >= lower:
            return label
        if lower is not None and upper is not None and lower <= score < upper:
            return label
    return "unknown"


def position_bucket(index: int, count: int) -> str:
    if count <= 1:
        return "only"
    if index == 0:
        return "first"
    if index == count - 1:
        return "last"
    ratio = index / (count - 1)
    if ratio < 0.25:
        return "early"
    if ratio < 0.75:
        return "middle"
    return "late"


def count_bucket(value: int) -> str:
    if value == 0:
        return "0"
    if value == 1:
        return "1"
    if value == 2:
        return "2"
    if value <= 5:
        return "3-5"
    if value <= 10:
        return "6-10"
    if value <= 25:
        return "11-25"
    if value <= 50:
        return "26-50"
    return "51+"


def share_bucket(value: float) -> str:
    if value <= 0.01:
        return "<=1%"
    if value <= 0.05:
        return "1-5%"
    if value <= 0.10:
        return "5-10%"
    if value <= 0.25:
        return "10-25%"
    if value <= 0.50:
        return "25-50%"
    if value < 1.0:
        return "50-<100%"
    return "100%"


def confusion_row(scope: str, threshold: float, counts: Counter[str]) -> dict[str, Any]:
    tp = counts["tp"]
    fp = counts["fp"]
    tn = counts["tn"]
    fn = counts["fn"]
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    fpr = safe_div(fp, fp + tn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return {
        "scope": scope,
        "threshold": threshold,
        "windows": tp + fp + tn + fn,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "benign_window_fpr": fpr,
        "f1": f1,
    }


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> int:
    materialized = list(rows)
    if fieldnames is None:
        fieldnames = []
        seen: set[str] = set()
        for row in materialized:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)
    return len(materialized)


def metric_rows(
    denominator: dict[Any, int],
    numerator: dict[Any, int],
    key_names: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows = []
    keys = sorted(set(denominator) | set(numerator), key=lambda item: str(item))
    for key in keys:
        values = key if isinstance(key, tuple) else (key,)
        row = {name: value for name, value in zip(key_names, values)}
        total = denominator.get(key, 0)
        positive = numerator.get(key, 0)
        row.update(
            {
                "benign_windows": total,
                "false_positive_windows": positive,
                "true_negative_windows": total - positive,
                "benign_window_fpr": safe_div(positive, total),
            }
        )
        rows.append(row)
    rows.sort(key=lambda row: (-row["false_positive_windows"],) + tuple(str(row[name]) for name in key_names))
    return rows


def main() -> None:
    args = parse_args()
    thresholds = parse_thresholds(args.thresholds)
    primary = float(args.primary_threshold)
    if not math.isclose(primary, 0.82, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            "This error-document subset was selected at threshold 0.82; "
            "use --primary-threshold 0.82 or rebuild the subset for another threshold."
        )
    if primary not in thresholds:
        thresholds.append(primary)
        thresholds.sort()

    full_document_path = Path(args.full_document_results_jsonl)
    error_document_path = Path(args.error_document_results_jsonl)
    error_window_path = Path(args.error_window_results_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    print("[window-stats] indexing full 300K document results", flush=True)
    full_documents = 0
    full_label_documents = Counter()
    full_label_windows = Counter()
    full_benign_by_category: dict[str, int] = defaultdict(int)
    full_benign_by_source: dict[str, int] = defaultdict(int)
    full_benign_by_language: dict[str, int] = defaultdict(int)
    full_benign_by_category_language: dict[tuple[str, str], int] = defaultdict(int)
    full_benign_by_window_count_bucket: dict[str, int] = defaultdict(int)
    full_benign_by_position: dict[str, int] = defaultdict(int)
    full_attack_multi_window_documents = 0
    full_attack_fn_by_threshold = Counter()
    expected_error_documents: dict[str, dict[str, Any]] = {}

    for row in read_jsonl(full_document_path):
        full_documents += 1
        label = row.get("document_label", "unknown")
        windows = int(row.get("window_count") or 0)
        document_id = row.get("document_id")
        score = float(row.get("document_max_prompt_injection_score") or 0.0)
        if not document_id:
            raise ValueError("Full document result missing document_id")
        full_label_documents[label] += 1
        full_label_windows[label] += windows
        if label == "prompt_injection" and windows != 1:
            full_attack_multi_window_documents += 1
        if label == "prompt_injection":
            for threshold in thresholds:
                if score < threshold:
                    full_attack_fn_by_threshold[threshold] += 1
        predicted_attack = score >= primary
        is_original_error = (
            label == "not_prompt_injection" and predicted_attack
        ) or (
            label == "prompt_injection" and not predicted_attack
        )
        if is_original_error:
            if document_id in expected_error_documents:
                raise ValueError(f"Duplicate full-corpus error document_id: {document_id}")
            expected_error_documents[document_id] = {
                "document_label": label,
                "document_max_prompt_injection_score": score,
                "window_count": windows,
                "best_window_index": row.get("best_window_index"),
                "best_window_text_hash": row.get("best_window_text_hash"),
            }
        if label == "not_prompt_injection":
            category = row.get("category", "unknown")
            source = row.get("source_name", "unknown")
            language = row.get("language", "unknown")
            bucket = row.get("window_count_bucket", "unknown")
            full_benign_by_category[category] += windows
            full_benign_by_source[source] += windows
            full_benign_by_language[language] += windows
            full_benign_by_category_language[(category, language)] += windows
            full_benign_by_window_count_bucket[bucket] += windows
            for index in range(windows):
                full_benign_by_position[position_bucket(index, windows)] += 1
        if args.progress_every and full_documents % args.progress_every == 0:
            print(
                f"[window-stats] full_docs={full_documents:,} full_windows={sum(full_label_windows.values()):,}",
                flush=True,
            )

    print("[window-stats] indexing 6,300 error-document results", flush=True)
    error_documents: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(read_jsonl(error_document_path), 1):
        document_id = row.get("document_id")
        if not document_id:
            raise ValueError("Error-document result missing document_id")
        if document_id in error_documents:
            raise ValueError(f"Duplicate error-document result: {document_id}")
        error_documents[document_id] = row
        if args.progress_every and index % args.progress_every == 0:
            print(f"[window-stats] error_docs={index:,}", flush=True)

    expected_ids = set(expected_error_documents)
    actual_ids = set(error_documents)
    missing_error_document_ids = sorted(expected_ids - actual_ids)
    extra_error_document_ids = sorted(actual_ids - expected_ids)
    rerun_score_mismatches = 0
    rerun_threshold_flips = 0
    rerun_window_count_mismatches = 0
    rerun_best_index_mismatches = 0
    rerun_best_hash_mismatches = 0
    rerun_label_mismatches = 0
    for document_id in sorted(expected_ids & actual_ids):
        expected = expected_error_documents[document_id]
        actual = error_documents[document_id]
        expected_score = float(expected["document_max_prompt_injection_score"])
        actual_score = float(actual.get("document_max_prompt_injection_score") or 0.0)
        if not math.isclose(expected_score, actual_score, rel_tol=0.0, abs_tol=1e-8):
            rerun_score_mismatches += 1
        if (expected_score >= primary) != (actual_score >= primary):
            rerun_threshold_flips += 1
        if expected["document_label"] != actual.get("document_label"):
            rerun_label_mismatches += 1
        if int(expected["window_count"] or 0) != int(actual.get("window_count") or 0):
            rerun_window_count_mismatches += 1
        if int(expected["best_window_index"] or 0) != int(actual.get("best_window_index") or 0):
            rerun_best_index_mismatches += 1
        if expected["best_window_text_hash"] != actual.get("best_window_text_hash"):
            rerun_best_hash_mismatches += 1

    threshold_counts = {threshold: Counter() for threshold in thresholds}
    score_band_counts: dict[tuple[str, str], int] = defaultdict(int)
    error_window_by_category: dict[str, int] = defaultdict(int)
    error_window_by_source: dict[str, int] = defaultdict(int)
    error_window_by_language: dict[str, int] = defaultdict(int)
    error_window_by_category_language: dict[tuple[str, str], int] = defaultdict(int)
    error_window_by_window_count_bucket: dict[str, int] = defaultdict(int)
    error_window_by_position: dict[str, int] = defaultdict(int)
    fp_window_by_category: dict[str, int] = defaultdict(int)
    fp_window_by_source: dict[str, int] = defaultdict(int)
    fp_window_by_language: dict[str, int] = defaultdict(int)
    fp_window_by_category_language: dict[tuple[str, str], int] = defaultdict(int)
    fp_window_by_window_count_bucket: dict[str, int] = defaultdict(int)
    fp_window_by_position: dict[str, int] = defaultdict(int)
    document_scores: dict[str, dict[int, float]] = defaultdict(dict)
    document_positive_indexes: dict[str, list[int]] = defaultdict(list)
    observed_window_counts = Counter()
    seen_window_keys: set[tuple[str, int]] = set()
    consistency_failures: list[str] = []
    fp_detail_path = output_dir / "false_positive_windows_threshold_0.82.csv"
    fn_detail_path = output_dir / "false_negative_windows_threshold_0.82.csv"
    combined_detail_path = output_dir / "all_fp_fn_windows_threshold_0.82.csv"
    detail_fields = [
        "document_id",
        "document_label",
        "category",
        "source_name",
        "language",
        "semantic_family",
        "window_index",
        "window_count",
        "window_position_bucket",
        "p_prompt_injection",
        "window_text_hash",
        "window_text",
    ]
    fp_detail_handle = fp_detail_path.open("w", encoding="utf-8-sig", newline="")
    fn_detail_handle = fn_detail_path.open("w", encoding="utf-8-sig", newline="")
    combined_detail_handle = combined_detail_path.open("w", encoding="utf-8-sig", newline="")
    fp_writer = csv.DictWriter(fp_detail_handle, fieldnames=detail_fields)
    fn_writer = csv.DictWriter(fn_detail_handle, fieldnames=detail_fields)
    combined_writer = csv.DictWriter(
        combined_detail_handle,
        fieldnames=["error_type", *detail_fields],
    )
    fp_writer.writeheader()
    fn_writer.writeheader()
    combined_writer.writeheader()
    top_fp_heap: list[tuple[float, int, dict[str, Any]]] = []
    serial = 0
    window_rows = 0
    fp_window_count = 0
    fn_window_count = 0

    print("[window-stats] scanning detailed window results", flush=True)
    try:
        for row in read_jsonl(error_window_path):
            window_rows += 1
            document_id = row.get("document_id")
            if document_id not in error_documents:
                consistency_failures.append(f"window_missing_error_document:{document_id}")
                continue
            meta = error_documents[document_id]
            label = row.get("document_label", "unknown")
            actual_attack = label == "prompt_injection"
            score = float(row.get("p_prompt_injection") or 0.0)
            window_index = int(row.get("window_index") or 0)
            window_count = int(row.get("window_count") or 0)
            window_key = (document_id, window_index)
            if window_key in seen_window_keys:
                consistency_failures.append(f"duplicate_window:{document_id}:{window_index}")
            seen_window_keys.add(window_key)
            observed_window_counts[document_id] += 1
            document_scores[document_id][window_index] = score
            category = row.get("category", meta.get("category", "unknown"))
            source = row.get("source_name", meta.get("source_name", "unknown"))
            language = row.get("language", meta.get("language", "unknown"))
            family = meta.get("semantic_family", "unknown")
            bucket = meta.get("window_count_bucket", "unknown")
            position = position_bucket(window_index, window_count)

            for threshold in thresholds:
                predicted_attack = score >= threshold
                if actual_attack and predicted_attack:
                    threshold_counts[threshold]["tp"] += 1
                elif actual_attack:
                    threshold_counts[threshold]["fn"] += 1
                elif predicted_attack:
                    threshold_counts[threshold]["fp"] += 1
                else:
                    threshold_counts[threshold]["tn"] += 1

            score_band_counts[(label, score_band(score))] += 1
            if not actual_attack:
                error_window_by_category[category] += 1
                error_window_by_source[source] += 1
                error_window_by_language[language] += 1
                error_window_by_category_language[(category, language)] += 1
                error_window_by_window_count_bucket[bucket] += 1
                error_window_by_position[position] += 1

            is_primary_positive = score >= primary
            detail = {
                "document_id": document_id,
                "document_label": label,
                "category": category,
                "source_name": source,
                "language": language,
                "semantic_family": family,
                "window_index": window_index,
                "window_count": window_count,
                "window_position_bucket": position,
                "p_prompt_injection": score,
                "window_text_hash": row.get("window_text_hash"),
                "window_text": row.get("window_text", ""),
            }
            if not actual_attack and is_primary_positive:
                fp_window_count += 1
                document_positive_indexes[document_id].append(window_index)
                fp_window_by_category[category] += 1
                fp_window_by_source[source] += 1
                fp_window_by_language[language] += 1
                fp_window_by_category_language[(category, language)] += 1
                fp_window_by_window_count_bucket[bucket] += 1
                fp_window_by_position[position] += 1
                fp_writer.writerow(detail)
                combined_writer.writerow({"error_type": "false_positive", **detail})
                serial += 1
                heap_item = (score, serial, detail)
                if len(top_fp_heap) < args.top_fp_windows:
                    heapq.heappush(top_fp_heap, heap_item)
                elif score > top_fp_heap[0][0]:
                    heapq.heapreplace(top_fp_heap, heap_item)
            elif actual_attack and not is_primary_positive:
                fn_window_count += 1
                fn_writer.writerow(detail)
                combined_writer.writerow({"error_type": "false_negative", **detail})

            if args.progress_every and window_rows % args.progress_every == 0:
                print(
                    f"[window-stats] windows={window_rows:,} fp_windows={fp_window_count:,} "
                    f"fn_windows={fn_window_count:,}",
                    flush=True,
                )
    finally:
        fp_detail_handle.close()
        fn_detail_handle.close()
        combined_detail_handle.close()

    print("[window-stats] validating document/window consistency", flush=True)
    max_score_mismatches = 0
    count_mismatches = 0
    best_index_mismatches = 0
    window_index_set_mismatches = 0
    document_rows = []
    positive_count_distribution = Counter()
    positive_share_distribution = Counter()
    for document_id, meta in error_documents.items():
        score_by_index = document_scores.get(document_id, {})
        scores = list(score_by_index.values())
        expected_count = int(meta.get("window_count") or 0)
        observed_count = observed_window_counts[document_id]
        if observed_count != expected_count:
            count_mismatches += 1
        if set(score_by_index) != set(range(expected_count)):
            window_index_set_mismatches += 1
        if not scores:
            consistency_failures.append(f"document_without_windows:{document_id}")
            continue
        max_score = max(scores)
        max_index = max(score_by_index, key=lambda index: score_by_index[index])
        reported_max = float(meta.get("document_max_prompt_injection_score") or 0.0)
        reported_best = int(meta.get("best_window_index") or 0)
        if not math.isclose(max_score, reported_max, rel_tol=0.0, abs_tol=1e-8):
            max_score_mismatches += 1
        if max_index != reported_best:
            best_index_mismatches += 1
        positives = document_positive_indexes.get(document_id, [])
        positive_count = len(positives)
        positive_share = safe_div(positive_count, observed_count)
        label = meta.get("document_label", "unknown")
        if label == "not_prompt_injection":
            positive_count_distribution[count_bucket(positive_count)] += 1
            positive_share_distribution[share_bucket(positive_share)] += 1
        document_rows.append(
            {
                "document_id": document_id,
                "document_label": label,
                "category": meta.get("category", "unknown"),
                "source_name": meta.get("source_name", "unknown"),
                "language": meta.get("language", "unknown"),
                "semantic_family": meta.get("semantic_family", "unknown"),
                "window_count": observed_count,
                "positive_windows_at_0.82": positive_count,
                "positive_window_share_at_0.82": positive_share,
                "min_window_score": min(scores),
                "mean_window_score": statistics.fmean(scores),
                "median_window_score": statistics.median(scores),
                "max_window_score": max_score,
                "first_positive_window_index": min(positives) if positives else None,
                "last_positive_window_index": max(positives) if positives else None,
                "reported_best_window_index": reported_best,
            }
        )

    if count_mismatches:
        consistency_failures.append(f"window_count_mismatches:{count_mismatches}")
    if max_score_mismatches:
        consistency_failures.append(f"max_score_mismatches:{max_score_mismatches}")
    if best_index_mismatches:
        consistency_failures.append(f"best_index_mismatches:{best_index_mismatches}")
    if window_index_set_mismatches:
        consistency_failures.append(f"window_index_set_mismatches:{window_index_set_mismatches}")
    if missing_error_document_ids:
        consistency_failures.append(f"missing_error_document_ids:{len(missing_error_document_ids)}")
    if extra_error_document_ids:
        consistency_failures.append(f"extra_error_document_ids:{len(extra_error_document_ids)}")
    if rerun_score_mismatches:
        consistency_failures.append(f"rerun_score_mismatches:{rerun_score_mismatches}")
    if rerun_threshold_flips:
        consistency_failures.append(f"rerun_threshold_flips:{rerun_threshold_flips}")
    if rerun_window_count_mismatches:
        consistency_failures.append(f"rerun_window_count_mismatches:{rerun_window_count_mismatches}")
    if rerun_best_index_mismatches:
        consistency_failures.append(f"rerun_best_index_mismatches:{rerun_best_index_mismatches}")
    if rerun_best_hash_mismatches:
        consistency_failures.append(f"rerun_best_hash_mismatches:{rerun_best_hash_mismatches}")
    if rerun_label_mismatches:
        consistency_failures.append(f"rerun_label_mismatches:{rerun_label_mismatches}")

    error_subset_threshold_rows = [
        confusion_row("error_document_subset", threshold, threshold_counts[threshold])
        for threshold in thresholds
    ]

    full_benign_windows = full_label_windows["not_prompt_injection"]
    full_attack_windows = full_label_windows["prompt_injection"]
    full_primary_counts = Counter(
        {
            "fp": fp_window_count,
            "tn": full_benign_windows - fp_window_count,
            "fn": fn_window_count,
            "tp": full_attack_windows - fn_window_count,
        }
    )
    full_primary_row = confusion_row("reconstructed_full_300k_corpus", primary, full_primary_counts)
    subset_reconciliation_passed = not any(
        (
            missing_error_document_ids,
            extra_error_document_ids,
            rerun_score_mismatches,
            rerun_threshold_flips,
            rerun_window_count_mismatches,
            rerun_best_index_mismatches,
            rerun_best_hash_mismatches,
            rerun_label_mismatches,
        )
    )
    reconstruction_exact = full_attack_multi_window_documents == 0 and subset_reconciliation_passed
    full_primary_row["reconstruction_exact"] = reconstruction_exact
    full_primary_row["reconstruction_note"] = (
        "Exact: every attack document has one production window; all benign documents outside the error subset have max score below 0.82."
        if reconstruction_exact
        else "Reconstruction is not exact; inspect window_consistency_report.json."
    )

    benign_threshold_rows = []
    full_threshold_rows = []
    for threshold in thresholds:
        if threshold < primary:
            continue
        counts = threshold_counts[threshold]
        fp = counts["fp"]
        benign_threshold_rows.append(
            {
                "scope": "reconstructed_full_300k_benign_windows",
                "threshold": threshold,
                "benign_windows": full_benign_windows,
                "false_positive_windows": fp,
                "true_negative_windows": full_benign_windows - fp,
                "benign_window_fpr": safe_div(fp, full_benign_windows),
                "reconstruction_exact": True,
                "note": "Exact for thresholds at or above 0.82 because all omitted benign documents have document max score below 0.82.",
            }
        )
        full_fn = full_attack_fn_by_threshold[threshold]
        full_counts = Counter(
            {
                "fp": fp,
                "tn": full_benign_windows - fp,
                "fn": full_fn,
                "tp": full_attack_windows - full_fn,
            }
        )
        full_row = confusion_row("reconstructed_full_300k_corpus", threshold, full_counts)
        full_row["reconstruction_exact"] = reconstruction_exact
        full_threshold_rows.append(full_row)

    attack_threshold_rows = []
    for threshold in thresholds:
        full_fn = full_attack_fn_by_threshold[threshold]
        attack_threshold_rows.append(
            {
                "scope": "full_300k_attack_windows",
                "threshold": threshold,
                "attack_windows": full_attack_windows,
                "true_positive_windows": full_attack_windows - full_fn,
                "false_negative_windows": full_fn,
                "attack_window_recall": safe_div(full_attack_windows - full_fn, full_attack_windows),
                "reconstruction_exact": full_attack_multi_window_documents == 0,
            }
        )

    category_rows = metric_rows(full_benign_by_category, fp_window_by_category, ("category",))
    source_rows = metric_rows(full_benign_by_source, fp_window_by_source, ("source_name",))
    language_rows = metric_rows(full_benign_by_language, fp_window_by_language, ("language",))
    category_language_rows = metric_rows(
        full_benign_by_category_language,
        fp_window_by_category_language,
        ("category", "language"),
    )
    window_bucket_rows = metric_rows(
        full_benign_by_window_count_bucket,
        fp_window_by_window_count_bucket,
        ("document_window_count_bucket",),
    )
    position_rows = metric_rows(full_benign_by_position, fp_window_by_position, ("window_position_bucket",))

    score_band_rows = []
    for (label, band), count in sorted(score_band_counts.items()):
        label_total = sum(value for (candidate_label, _), value in score_band_counts.items() if candidate_label == label)
        score_band_rows.append(
            {
                "document_label": label,
                "score_band": band,
                "windows": count,
                "share_within_label_in_error_subset": safe_div(count, label_total),
            }
        )

    positive_count_rows = [
        {"positive_window_count_bucket": key, "false_positive_documents": value}
        for key, value in positive_count_distribution.items()
    ]
    positive_count_order = {label: index for index, label in enumerate(("0", "1", "2", "3-5", "6-10", "11-25", "26-50", "51+"))}
    positive_count_rows.sort(key=lambda row: positive_count_order.get(row["positive_window_count_bucket"], 99))
    positive_share_rows = [
        {"positive_window_share_bucket": key, "false_positive_documents": value}
        for key, value in positive_share_distribution.items()
    ]
    positive_share_order = {label: index for index, label in enumerate(("<=1%", "1-5%", "5-10%", "10-25%", "25-50%", "50-<100%", "100%"))}
    positive_share_rows.sort(key=lambda row: positive_share_order.get(row["positive_window_share_bucket"], 99))

    top_fp_rows = [item[2] for item in sorted(top_fp_heap, key=lambda item: item[0], reverse=True)]
    document_rows.sort(key=lambda row: (row["document_label"], -row["max_window_score"], row["document_id"]))
    false_positive_document_rows = [
        row for row in document_rows if row["document_label"] == "not_prompt_injection"
    ]

    print("[window-stats] writing CSV and JSON artifacts", flush=True)
    write_csv(output_dir / "error_subset_window_threshold_metrics.csv", error_subset_threshold_rows)
    write_csv(output_dir / "full_corpus_window_metrics_threshold_0.82.csv", [full_primary_row])
    write_csv(output_dir / "full_corpus_window_metrics_thresholds_ge_0.82.csv", full_threshold_rows)
    write_csv(output_dir / "full_corpus_attack_window_recall_all_thresholds.csv", attack_threshold_rows)
    write_csv(output_dir / "full_corpus_benign_window_fpr_thresholds_ge_0.82.csv", benign_threshold_rows)
    write_csv(output_dir / "full_corpus_fp_windows_by_category_threshold_0.82.csv", category_rows)
    write_csv(output_dir / "full_corpus_fp_windows_by_source_threshold_0.82.csv", source_rows)
    write_csv(output_dir / "full_corpus_fp_windows_by_language_threshold_0.82.csv", language_rows)
    write_csv(output_dir / "full_corpus_fp_windows_by_category_language_threshold_0.82.csv", category_language_rows)
    write_csv(output_dir / "full_corpus_fp_windows_by_document_window_count_bucket_threshold_0.82.csv", window_bucket_rows)
    write_csv(output_dir / "full_corpus_fp_windows_by_position_threshold_0.82.csv", position_rows)
    write_csv(output_dir / "error_subset_window_score_bands.csv", score_band_rows)
    write_csv(output_dir / "error_document_window_statistics.csv", document_rows)
    write_csv(output_dir / "false_positive_document_window_statistics.csv", false_positive_document_rows)
    write_csv(output_dir / "false_positive_document_positive_window_count_distribution.csv", positive_count_rows)
    write_csv(output_dir / "false_positive_document_positive_window_share_distribution.csv", positive_share_rows)
    write_csv(output_dir / "top_1000_false_positive_windows_threshold_0.82.csv", top_fp_rows, detail_fields)

    consistency = {
        "status": "pass" if not consistency_failures else "fail",
        "failures": consistency_failures[:100],
        "full_documents": full_documents,
        "full_windows": sum(full_label_windows.values()),
        "error_documents": len(error_documents),
        "error_windows": window_rows,
        "unique_window_keys": len(seen_window_keys),
        "expected_error_windows": sum(int(row.get("window_count") or 0) for row in error_documents.values()),
        "window_count_mismatches": count_mismatches,
        "max_score_mismatches": max_score_mismatches,
        "best_window_index_mismatches": best_index_mismatches,
        "window_index_set_mismatches": window_index_set_mismatches,
        "full_attack_multi_window_documents": full_attack_multi_window_documents,
        "expected_original_error_documents": len(expected_error_documents),
        "rerun_error_documents": len(error_documents),
        "missing_error_document_ids": missing_error_document_ids[:100],
        "extra_error_document_ids": extra_error_document_ids[:100],
        "rerun_score_mismatches": rerun_score_mismatches,
        "rerun_threshold_flips": rerun_threshold_flips,
        "rerun_window_count_mismatches": rerun_window_count_mismatches,
        "rerun_best_window_index_mismatches": rerun_best_index_mismatches,
        "rerun_best_window_hash_mismatches": rerun_best_hash_mismatches,
        "rerun_label_mismatches": rerun_label_mismatches,
        "subset_reconciliation_passed": subset_reconciliation_passed,
        "reconstruction_exact": reconstruction_exact,
    }
    with (output_dir / "window_consistency_report.json").open("w", encoding="utf-8") as handle:
        json.dump(consistency, handle, ensure_ascii=False, indent=2)

    false_positive_documents = sum(
        1 for row in error_documents.values() if row.get("document_label") == "not_prompt_injection"
    )
    summary = {
        "status": "pass" if not consistency_failures else "fail",
        "model_id": "gbv/mdeberta-ru-prompt-injection",
        "primary_threshold": primary,
        "inputs": {
            "full_document_results_jsonl": str(full_document_path.resolve()),
            "error_document_results_jsonl": str(error_document_path.resolve()),
            "error_window_results_jsonl": str(error_window_path.resolve()),
        },
        "full_corpus": {
            "documents": full_documents,
            "windows": sum(full_label_windows.values()),
            "documents_by_label": dict(full_label_documents),
            "windows_by_label": dict(full_label_windows),
            "window_metrics_at_0.82": full_primary_row,
            "window_metrics_at_thresholds_ge_0.82": full_threshold_rows,
            "attack_window_recall_all_thresholds": attack_threshold_rows,
        },
        "error_subset": {
            "documents": len(error_documents),
            "windows": window_rows,
            "false_positive_windows_at_0.82": fp_window_count,
            "false_negative_windows_at_0.82": fn_window_count,
            "threshold_metrics": error_subset_threshold_rows,
        },
        "key_findings": {
            "full_benign_window_fpr_at_0.82": full_primary_row["benign_window_fpr"],
            "full_attack_window_recall_at_0.82": full_primary_row["recall"],
            "false_positive_windows_at_0.82": fp_window_count,
            "false_positive_documents_at_0.82": false_positive_documents,
            "average_fp_windows_per_fp_document": safe_div(fp_window_count, false_positive_documents),
            "share_of_error_subset_benign_windows_positive": safe_div(
                fp_window_count,
                threshold_counts[primary]["fp"] + threshold_counts[primary]["tn"],
            ),
        },
        "consistency": consistency,
        "elapsed_seconds": time.time() - started,
    }
    with (output_dir / "window-statistics-summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    top_categories = category_rows[:5]
    top_sources = source_rows[:8]
    markdown = [
        "# V16 Window-Level Error Statistics",
        "",
        f"- Model: `gbv/mdeberta-ru-prompt-injection`",
        f"- Primary threshold: `{primary}`",
        f"- Full corpus: {full_documents:,} documents / {sum(full_label_windows.values()):,} windows",
        f"- Error subset: {len(error_documents):,} documents / {window_rows:,} windows",
        "",
        "## Full-Corpus Window Metrics At 0.82",
        "",
        "These metrics are reconstructed exactly. Every attack document has one window, and every benign document omitted from the error subset has a maximum window score below 0.82.",
        "",
        "| TP windows | FP windows | TN windows | FN windows | precision | recall | benign window FPR | F1 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {full_primary_row['tp']:,} | {full_primary_row['fp']:,} | {full_primary_row['tn']:,} | "
            f"{full_primary_row['fn']:,} | {full_primary_row['precision']:.2%} | {full_primary_row['recall']:.4%} | "
            f"{full_primary_row['benign_window_fpr']:.4%} | {full_primary_row['f1']:.2%} |"
        ),
        "",
        "## Error-Subset Threshold Sweep",
        "",
        "| threshold | TP | FP | TN | FN | precision | recall | benign window FPR |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in error_subset_threshold_rows:
        markdown.append(
            f"| {row['threshold']} | {row['tp']:,} | {row['fp']:,} | {row['tn']:,} | {row['fn']:,} | "
            f"{row['precision']:.2%} | {row['recall']:.2%} | {row['benign_window_fpr']:.2%} |"
        )
    markdown.extend(
        [
            "",
            "## Exact Full-Corpus Threshold Sweep At Or Above 0.82",
            "",
            "| threshold | TP | FP | TN | FN | precision | recall | benign window FPR |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in full_threshold_rows:
        markdown.append(
            f"| {row['threshold']} | {row['tp']:,} | {row['fp']:,} | {row['tn']:,} | {row['fn']:,} | "
            f"{row['precision']:.2%} | {row['recall']:.4%} | {row['benign_window_fpr']:.4%} |"
        )
    markdown.extend(
        [
            "",
            "## Highest FP-Window Categories In Full Corpus",
            "",
            "| category | benign windows | FP windows | window FPR |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in top_categories:
        markdown.append(
            f"| {row['category']} | {row['benign_windows']:,} | {row['false_positive_windows']:,} | {row['benign_window_fpr']:.4%} |"
        )
    markdown.extend(
        [
            "",
            "## Highest FP-Window Sources In Full Corpus",
            "",
            "| source | benign windows | FP windows | window FPR |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in top_sources:
        markdown.append(
            f"| {row['source_name']} | {row['benign_windows']:,} | {row['false_positive_windows']:,} | {row['benign_window_fpr']:.4%} |"
        )
    markdown.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- The {fp_window_count:,} false-positive windows are contained in {false_positive_documents:,} false-positive documents.",
            f"- This is {safe_div(fp_window_count, full_benign_windows):.4%} of all {full_benign_windows:,} benign production windows.",
            f"- The average false-positive document contains {safe_div(fp_window_count, false_positive_documents):.2f} positive windows, but the distribution is reported separately because long documents dominate the selected subset.",
            f"- All {fn_window_count:,} false-negative windows are the single windows of the 18 false-negative attack documents.",
            "- Threshold metrics below 0.82 are complete only for the selected error-document subset. Full-corpus benign FPR is exact only at thresholds at or above 0.82 using these detailed window outputs.",
            "",
            "## Consistency",
            "",
            f"- Status: `{consistency['status']}`",
            f"- Window-count mismatches: {count_mismatches}",
            f"- Document-max score mismatches: {max_score_mismatches}",
            f"- Best-window index mismatches: {best_index_mismatches}",
        ]
    )
    with (output_dir / "window-statistics-summary.md").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(markdown) + "\n")

    digest_builder = hashlib.sha256()
    with error_window_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest_builder.update(block)
    digest = digest_builder.hexdigest().upper()
    manifest = {
        "window_results_sha256": digest,
        "output_files": sorted(path.name for path in output_dir.iterdir() if path.is_file()),
    }
    with (output_dir / "window-statistics-manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    print(
        json.dumps(
            {
                "status": summary["status"],
                "output_dir": str(output_dir.resolve()),
                "full_windows": sum(full_label_windows.values()),
                "error_windows": window_rows,
                "fp_windows_at_0.82": fp_window_count,
                "fn_windows_at_0.82": fn_window_count,
                "full_benign_window_fpr_at_0.82": full_primary_row["benign_window_fpr"],
                "full_attack_window_recall_at_0.82": full_primary_row["recall"],
                "elapsed_seconds": summary["elapsed_seconds"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    if consistency_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
