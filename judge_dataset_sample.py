# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_from_disk

from judge_label_candidates import (
    DEFAULT_MODEL,
    OPENAI_RESPONSES_URL,
    batched,
    candidate_id,
    compact_judgments_file,
    estimate_cost,
    judge_batch_until_complete,
    load_env_file,
    load_existing_judgment_ids,
    normalize_text,
)


LABEL_NAMES = {
    0: "benign",
    1: "prompt_injection",
}
LENGTH_BINS = [
    (0, 128, "0000-0128"),
    (129, 256, "0129-0256"),
    (257, 512, "0257-0512"),
    (513, 1024, "0513-1024"),
    (1025, 2048, "1025-2048"),
    (2049, None, "2049+"),
]


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def length_bin_for_text(text: str) -> str:
    length = len(text)
    for lower, upper, label in LENGTH_BINS:
        if length >= lower and (upper is None or length <= upper):
            return label
    return LENGTH_BINS[-1][2]


def load_dataset_split(path: Path, split_name: str) -> Dataset:
    dataset = load_from_disk(str(path))
    if isinstance(dataset, DatasetDict):
        if split_name not in dataset:
            raise ValueError(f"DatasetDict at {path} does not contain split {split_name!r}.")
        dataset = dataset[split_name]
    if not isinstance(dataset, Dataset):
        raise TypeError(f"Expected Dataset or DatasetDict at {path}, got {type(dataset).__name__}")
    missing = {"text", "label"}.difference(dataset.column_names)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    return dataset


def prepare_dataset_rows(dataset: Dataset) -> list[dict[str, Any]]:
    rows = []
    for idx, row in enumerate(dataset):
        text = normalize_text(row.get("text"))
        if not text:
            continue
        label = int(row["label"])
        source = normalize_text(row.get("source_name")) or "unknown"
        clean = {
            "candidate_id": candidate_id(source, text),
            "dataset_index": idx,
            "dataset_label": label,
            "dataset_label_name": LABEL_NAMES.get(label, str(label)),
            "source_name": source,
            "bucket": normalize_text(row.get("bucket")) or "unknown",
            "language": normalize_text(row.get("language")) or "unknown",
            "text_unit": normalize_text(row.get("text_unit")) or "full_text",
            "length_bin": length_bin_for_text(text),
            "text": text,
        }
        rows.append(clean)
    return rows


def select_sample(rows: list[dict[str, Any]], sample_fraction: float, max_rows: int, seed: int) -> list[dict[str, Any]]:
    if not 0 < sample_fraction <= 1:
        raise ValueError("--sample-fraction must be > 0 and <= 1.")
    target = max(1, round(len(rows) * sample_fraction))
    if max_rows > 0:
        target = min(target, max_rows)
    indices = list(range(len(rows)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    selected = [rows[idx] for idx in indices[:target]]
    selected.sort(key=lambda row: row["dataset_index"])
    return selected


def judgment_to_binary(label: str) -> int | None:
    if label == "prompt_injection":
        return 1
    if label in {"benign", "unsafe_non_injection"}:
        return 0
    return None


def enrich_judgment(row: dict[str, Any], judgment: dict[str, Any]) -> dict[str, Any]:
    judgment_label = normalize_text(judgment.get("label"))
    binary = judgment_to_binary(judgment_label)
    dataset_label = int(row["dataset_label"])
    confidence = float(judgment.get("confidence", 0.0) or 0.0)
    return {
        **row,
        "judgment_label": judgment_label,
        "judgment_binary_label": binary,
        "judgment_confidence": confidence,
        "agreement": binary == dataset_label if binary is not None else None,
        "evidence_span": normalize_text(judgment.get("evidence_span")),
        "rationale_short": normalize_text(judgment.get("rationale_short")),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL in {path} at line {line_number}") from exc
    return rows


def group_summary(rows: list[dict[str, Any]], key: str, confidence_threshold: float) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, "unknown"))].append(row)
    return {group: summarize_rows(group_rows, confidence_threshold) for group, group_rows in sorted(grouped.items())}


def summarize_rows(rows: list[dict[str, Any]], confidence_threshold: float) -> dict[str, Any]:
    judged = [
        row
        for row in rows
        if row.get("judgment_label") in {"prompt_injection", "unsafe_non_injection", "benign", "ambiguous"}
    ]
    binary = [row for row in judged if row.get("judgment_binary_label") is not None]
    high_conf_binary = [row for row in binary if float(row.get("judgment_confidence", 0.0)) >= confidence_threshold]
    disagreements = [row for row in binary if row.get("agreement") is False]
    high_conf_disagreements = [row for row in high_conf_binary if row.get("agreement") is False]
    return {
        "rows": len(rows),
        "judged_rows": len(judged),
        "dataset_labels": dict(Counter(row.get("dataset_label_name", "unknown") for row in rows)),
        "judgment_labels": dict(Counter(row.get("judgment_label", "missing") for row in judged)),
        "binary_judged_rows": len(binary),
        "ambiguous_rows": sum(1 for row in judged if row.get("judgment_label") == "ambiguous"),
        "agreements": sum(1 for row in binary if row.get("agreement") is True),
        "disagreements": len(disagreements),
        "disagreement_rate": len(disagreements) / len(binary) if binary else 0.0,
        "high_conf_binary_rows": len(high_conf_binary),
        "high_conf_disagreements": len(high_conf_disagreements),
        "high_conf_disagreement_rate": len(high_conf_disagreements) / len(high_conf_binary)
        if high_conf_binary
        else 0.0,
    }


def write_summary(path: Path, output_rows: list[dict[str, Any]], confidence_threshold: float) -> None:
    summary = {
        "confidence_threshold": confidence_threshold,
        "overall": summarize_rows(output_rows, confidence_threshold),
        "by_dataset_label": group_summary(output_rows, "dataset_label_name", confidence_threshold),
        "by_source": group_summary(output_rows, "source_name", confidence_threshold),
        "by_bucket": group_summary(output_rows, "bucket", confidence_threshold),
        "by_length_bin": group_summary(output_rows, "length_bin", confidence_threshold),
        "high_conf_disagreements": [
            {
                "candidate_id": row["candidate_id"],
                "dataset_index": row["dataset_index"],
                "dataset_label_name": row["dataset_label_name"],
                "judgment_label": row["judgment_label"],
                "judgment_confidence": row["judgment_confidence"],
                "source_name": row["source_name"],
                "bucket": row["bucket"],
                "length_bin": row["length_bin"],
                "evidence_span": row["evidence_span"],
                "rationale_short": row["rationale_short"],
                "text": row["text"],
            }
            for row in output_rows
            if row.get("agreement") is False
            and row.get("judgment_binary_label") is not None
            and float(row.get("judgment_confidence", 0.0)) >= confidence_threshold
        ],
    }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM-audit a random sample from a prepared binary prompt-injection dataset."
    )
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--sample-fraction", type=float, default=0.10)
    parser.add_argument("--max-rows", type=int, default=0, help="Optional cap after applying sample fraction. 0 means no cap.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--env-file", default=".env", help="Path to .env file containing OPENAI_API_KEY.")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--url", default=OPENAI_RESPONSES_URL)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--reasoning-effort", default="minimal")
    parser.add_argument("--max-output-tokens-per-row", type=int, default=450)
    parser.add_argument("--min-split-batch-size", type=int, default=1)
    parser.add_argument("--missing-retries", type=int, default=2)
    parser.add_argument("--confidence-threshold", type=float, default=0.80)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    configure_stdout()
    args = parse_args()
    load_env_file(Path(args.env_file))

    output_path = Path(args.output)
    summary_path = Path(args.summary_output) if args.summary_output else output_path.with_suffix(".summary.json")
    compact_judgments_file(output_path)

    dataset = load_dataset_split(Path(args.dataset_dir), args.split)
    rows = prepare_dataset_rows(dataset)
    selected = select_sample(rows, args.sample_fraction, args.max_rows, args.seed)
    existing_ids = load_existing_judgment_ids(output_path)
    pending = [row for row in selected if row["candidate_id"] not in existing_ids]

    print(f"dataset rows      : {len(rows):,}")
    print(f"split             : {args.split}")
    print(f"selected rows     : {len(selected):,}")
    print(f"existing judgments: {len(existing_ids):,}")
    print(f"pending rows      : {len(pending):,}")
    print(f"selected labels   : {dict(Counter(row['dataset_label_name'] for row in selected))}")
    print(f"selected bins     : {dict(Counter(row['length_bin'] for row in selected))}")
    print(f"estimated cost    : {estimate_cost(selected, args.model)}")

    if args.dry_run:
        return

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise EnvironmentError(f"Set {args.api_key_env} before running without --dry-run.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written_ids = set(existing_ids)
    rows_by_id = {row["candidate_id"]: row for row in selected}
    written = 0
    with output_path.open("a", encoding="utf-8") as out:
        for batch_number, batch in enumerate(batched(pending, args.batch_size), start=1):
            judgments = judge_batch_until_complete(
                api_key=api_key,
                url=args.url,
                model=args.model,
                batch=batch,
                timeout=args.timeout,
                reasoning_effort=args.reasoning_effort,
                max_output_tokens_per_row=args.max_output_tokens_per_row,
                min_split_batch_size=args.min_split_batch_size,
                missing_retries=args.missing_retries,
            )
            for judgment in judgments:
                row = rows_by_id.get(str(judgment.get("candidate_id", "")))
                if not row or row["candidate_id"] in written_ids:
                    continue
                out.write(json.dumps(enrich_judgment(row, judgment), ensure_ascii=False) + "\n")
                written_ids.add(row["candidate_id"])
                written += 1
            out.flush()
            print(f"batch {batch_number}: wrote {written:,} judgments")
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

    output_rows = [row for row in read_jsonl(output_path) if row.get("candidate_id") in rows_by_id]
    write_summary(summary_path, output_rows, args.confidence_threshold)
    print(f"summary written   : {summary_path.resolve()}")

    missing = {row["candidate_id"] for row in selected} - load_existing_judgment_ids(output_path)
    if missing:
        print(f"warning: {len(missing):,} selected rows did not receive judgments")


if __name__ == "__main__":
    main()
