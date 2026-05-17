# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from sample import (
    classify,
    classification_counts,
    grouped_metrics,
    length_bin_for_text,
    length_bin_for_value,
    load_model,
    load_validation_rows,
    summarize_numeric,
    window_preview,
)


DEFAULT_MODEL_ID = "./mdeberta-ru-prompt-injection-v8-complete-ft"
DEFAULT_OUTPUT_DIR = "validation-regression-v8-complete-ft"
DEFAULT_THRESHOLD = 0.5
DEFAULT_BATCH_SIZE = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run sliding-window validation regression across one or more prepared validation datasets. "
            "Writes aggregate reports and false-positive/false-negative JSONL files."
        )
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--dataset-dir",
        action="append",
        default=[],
        help=(
            "Validation dataset directory. Repeat to test multiple datasets. "
            "If omitted, all non-temporary training-dataset*-validation directories are used."
        ),
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional per-dataset row limit for quick smoke checks.",
    )
    parser.add_argument(
        "--include-full-text",
        action="store_true",
        help="Include full text in FP/FN JSONL rows. By default only previews are written.",
    )
    parser.add_argument(
        "--include-window-details",
        action="store_true",
        help="Include per-window scores for each FP/FN row.",
    )
    parser.add_argument(
        "--max-error-rows",
        type=int,
        default=0,
        help="Maximum FP and FN rows to write per dataset. 0 means no limit.",
    )
    parser.add_argument("--min-accuracy", type=float, default=None)
    parser.add_argument("--min-precision", type=float, default=None)
    parser.add_argument("--min-recall", type=float, default=None)
    parser.add_argument("--min-f1", type=float, default=None)
    parser.add_argument(
        "--min-bucket-recall",
        action="append",
        default=[],
        metavar="BUCKET=VALUE",
        help=(
            "Require a minimum recall for a bucket, for example "
            "deep_embedded_indirect_attack_short=0.99. Repeat as needed."
        ),
    )
    return parser.parse_args()


def discover_validation_dirs(root: Path) -> list[Path]:
    paths = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        name = path.name
        if name.startswith("tmp-"):
            continue
        if name.startswith("training-dataset") and name.endswith("-validation"):
            paths.append(path)
    return sorted(paths, key=lambda item: natural_sort_key(item.name))


def natural_sort_key(value: str) -> list[int | str]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def safe_name(value: str) -> str:
    value = value.replace("\\", "/").rstrip("/")
    name = Path(value).name or value
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def parse_bucket_recall_gates(values: list[str]) -> dict[str, float]:
    gates: dict[str, float] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --min-bucket-recall value: {value!r}. Expected BUCKET=VALUE.")
        bucket, threshold = value.split("=", 1)
        bucket = bucket.strip()
        if not bucket:
            raise ValueError(f"Invalid --min-bucket-recall value: {value!r}. Bucket is empty.")
        gates[bucket] = float(threshold)
    return gates


def label_name(label: int) -> str:
    return "prompt_injection" if label == 1 else "benign"


def dataset_values(dataset: Any, column: str, default: str = "") -> list[str]:
    if column not in dataset.column_names:
        return [default] * len(dataset)
    return [str(value) for value in dataset[column]]


def compute_report(
    dataset_name: str,
    dataset_path: Path,
    dataset: Any,
    labels: list[int],
    pred_labels: list[int],
    texts: list[str],
    token_lengths: list[int],
    threshold: float,
) -> dict[str, Any]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        pred_labels,
        average="binary",
        zero_division=0,
    )
    report: dict[str, Any] = {
        "dataset": dataset_name,
        "dataset_path": str(dataset_path),
        "rows": len(labels),
        "threshold": threshold,
        "text_length_stats": summarize_numeric([len(text) for text in texts]),
        "text_token_length_stats": summarize_numeric(token_lengths),
        "accuracy": accuracy_score(labels, pred_labels),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        **classification_counts(labels, pred_labels),
    }
    report["by_length_bin"] = grouped_metrics(
        [length_bin_for_text(text) for text in texts],
        labels,
        pred_labels,
    )
    report["by_token_length_bin"] = grouped_metrics(
        [length_bin_for_value(length) for length in token_lengths],
        labels,
        pred_labels,
    )
    if "bucket" in dataset.column_names:
        report["by_bucket"] = grouped_metrics(dataset_values(dataset, "bucket"), labels, pred_labels)
    if "source_name" in dataset.column_names:
        report["by_source"] = grouped_metrics(dataset_values(dataset, "source_name"), labels, pred_labels)
    return report


def write_error_rows(
    path: Path,
    dataset_name: str,
    dataset: Any,
    texts: list[str],
    labels: list[int],
    pred_labels: list[int],
    token_lengths: list[int],
    predictions: list[dict[str, Any]],
    target_true: int,
    target_pred: int,
    include_full_text: bool,
    include_window_details: bool,
    max_rows: int,
) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for idx, (true_label, pred_label) in enumerate(zip(labels, pred_labels)):
            if true_label != target_true or pred_label != target_pred:
                continue
            row = make_error_row(
                dataset_name=dataset_name,
                dataset=dataset,
                index=idx,
                text=texts[idx],
                true_label=true_label,
                pred_label=pred_label,
                token_length=token_lengths[idx],
                prediction=predictions[idx],
                include_full_text=include_full_text,
                include_window_details=include_window_details,
            )
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
            if max_rows and count >= max_rows:
                break
    return count


def make_error_row(
    dataset_name: str,
    dataset: Any,
    index: int,
    text: str,
    true_label: int,
    pred_label: int,
    token_length: int,
    prediction: dict[str, Any],
    include_full_text: bool,
    include_window_details: bool,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "dataset": dataset_name,
        "index": index,
        "true_label": label_name(true_label),
        "predicted_label": label_name(pred_label),
        "p_prompt_injection": prediction["p_prompt_injection"],
        "p_benign": prediction["p_benign"],
        "threshold": prediction["threshold"],
        "text_length": len(text),
        "text_token_length": token_length,
        "text_preview": window_preview(text, limit=500),
    }
    for column in ["source_name", "bucket", "language", "text_unit", "parent_id", "split_hint"]:
        if column in dataset.column_names:
            row[column] = dataset[index][column]
    for key in ["window_count", "window_strategy", "best_window_index"]:
        if key in prediction:
            row[key] = prediction[key]
    if include_full_text:
        row["text"] = text
    if include_window_details and "window_scores" in prediction:
        row["window_scores"] = prediction["window_scores"]
    return row


def write_markdown_summary(path: Path, reports: list[dict[str, Any]], failures: list[str]) -> None:
    lines = [
        "# Validation Regression Summary",
        "",
        "| Dataset | Rows | Accuracy | Precision | Recall | F1 | FP | FN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for report in reports:
        lines.append(
            "| {dataset} | {rows} | {accuracy:.4f} | {precision:.4f} | {recall:.4f} | "
            "{f1:.4f} | {false_positives} | {false_negatives} |".format(**report)
        )
    if failures:
        lines.extend(["", "## Gate Failures", ""])
        lines.extend(f"- {failure}" for failure in failures)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_gates(
    report: dict[str, Any],
    min_accuracy: float | None,
    min_precision: float | None,
    min_recall: float | None,
    min_f1: float | None,
    min_bucket_recall: dict[str, float],
) -> list[str]:
    failures: list[str] = []
    scalar_gates = {
        "accuracy": min_accuracy,
        "precision": min_precision,
        "recall": min_recall,
        "f1": min_f1,
    }
    for metric, minimum in scalar_gates.items():
        if minimum is not None and float(report[metric]) < minimum:
            failures.append(
                f"{report['dataset']}: {metric}={float(report[metric]):.6f} below {minimum:.6f}"
            )
    bucket_metrics = report.get("by_bucket", {})
    for bucket, minimum in min_bucket_recall.items():
        if bucket not in bucket_metrics:
            failures.append(f"{report['dataset']}: bucket {bucket!r} is missing")
            continue
        recall = float(bucket_metrics[bucket]["recall"])
        if recall < minimum:
            failures.append(
                f"{report['dataset']}: bucket {bucket} recall={recall:.6f} below {minimum:.6f}"
            )
    return failures


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    root = Path.cwd()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "errors").mkdir(parents=True, exist_ok=True)

    dataset_paths = [Path(value) for value in args.dataset_dir]
    if not dataset_paths:
        dataset_paths = discover_validation_dirs(root)
    if not dataset_paths:
        raise SystemExit("No validation datasets found. Pass --dataset-dir at least once.")

    bucket_gates = parse_bucket_recall_gates(args.min_bucket_recall)
    tokenizer, model = load_model(args.model_id)

    reports: list[dict[str, Any]] = []
    failures: list[str] = []
    for dataset_path in dataset_paths:
        dataset_path = dataset_path.resolve()
        dataset_name = safe_name(str(dataset_path))
        print(f"Validating {dataset_name} ({dataset_path})...")
        dataset = load_validation_rows(str(dataset_path))
        if args.limit is not None:
            dataset = dataset.select(range(min(args.limit, len(dataset))))

        texts = [str(text) for text in dataset["text"]]
        labels = [int(label) for label in dataset["label"]]
        token_lengths = [len(tokenizer(text, add_special_tokens=False)["input_ids"]) for text in texts]
        predictions = classify(
            texts,
            tokenizer,
            model,
            args.threshold,
            batch_size=args.batch_size,
            show_progress=True,
            include_window_details=args.include_window_details,
            include_full_window_text=args.include_full_text,
        )
        pred_labels = [1 if row["label"] == "prompt_injection" else 0 for row in predictions]
        report = compute_report(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            dataset=dataset,
            labels=labels,
            pred_labels=pred_labels,
            texts=texts,
            token_lengths=token_lengths,
            threshold=args.threshold,
        )
        reports.append(report)

        fn_path = output_dir / "errors" / f"{dataset_name}.false_negatives.jsonl"
        fp_path = output_dir / "errors" / f"{dataset_name}.false_positives.jsonl"
        written_fn = write_error_rows(
            path=fn_path,
            dataset_name=dataset_name,
            dataset=dataset,
            texts=texts,
            labels=labels,
            pred_labels=pred_labels,
            token_lengths=token_lengths,
            predictions=predictions,
            target_true=1,
            target_pred=0,
            include_full_text=args.include_full_text,
            include_window_details=args.include_window_details,
            max_rows=args.max_error_rows,
        )
        written_fp = write_error_rows(
            path=fp_path,
            dataset_name=dataset_name,
            dataset=dataset,
            texts=texts,
            labels=labels,
            pred_labels=pred_labels,
            token_lengths=token_lengths,
            predictions=predictions,
            target_true=0,
            target_pred=1,
            include_full_text=args.include_full_text,
            include_window_details=args.include_window_details,
            max_rows=args.max_error_rows,
        )
        report["written_false_negative_rows"] = written_fn
        report["written_false_positive_rows"] = written_fp

        dataset_failures = evaluate_gates(
            report=report,
            min_accuracy=args.min_accuracy,
            min_precision=args.min_precision,
            min_recall=args.min_recall,
            min_f1=args.min_f1,
            min_bucket_recall=bucket_gates,
        )
        failures.extend(dataset_failures)
        print(
            "  accuracy={accuracy:.4f} precision={precision:.4f} recall={recall:.4f} "
            "f1={f1:.4f} fp={false_positives} fn={false_negatives}".format(**report)
        )

    summary = {
        "model_id": args.model_id,
        "threshold": args.threshold,
        "batch_size": args.batch_size,
        "limit": args.limit,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "datasets": reports,
        "gate_failures": failures,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown_summary(output_dir / "summary.md", reports, failures)

    print(f"Wrote summary: {summary_path}")
    print(f"Wrote markdown: {output_dir / 'summary.md'}")
    if failures:
        print("Gate failures:")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
