# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_from_disk


DEFAULT_DATASET_SPLIT = (
    "mdeberta-ru-prompt-injection-option-b"
    "/stage-cache/dataset_split-aab4dbcd753070da"
)


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def percentile(values: list[int], pct: float) -> int:
    if not values:
        return 0
    values = sorted(values)
    idx = round((len(values) - 1) * pct)
    return values[idx]


def print_counter(title: str, counter: Counter[Any]) -> None:
    print(title)
    total = counter.total()
    for key, count in counter.most_common():
        share = 100 * count / total if total else 0
        print(f"  {key}: {count:,} ({share:.1f}%)")


def describe_lengths(rows: list[dict[str, Any]]) -> str:
    lengths = [len(row["text"]) for row in rows]
    return (
        f"avg={statistics.mean(lengths):.1f}, "
        f"p50={percentile(lengths, 0.50)}, "
        f"p90={percentile(lengths, 0.90)}, "
        f"min={min(lengths)}, max={max(lengths)}"
    )


def analyze_split(name: str, rows: list[dict[str, Any]]) -> None:
    print(f"\n== {name} ==")
    print(f"rows: {len(rows):,}")

    print_counter("labels:", Counter(row["label"] for row in rows))
    print_counter("sources:", Counter(row["source_name"] for row in rows))
    if "bucket" in rows[0]:
        print_counter("buckets:", Counter(row["bucket"] for row in rows))
    if "language" in rows[0]:
        print_counter("languages:", Counter(row["language"] for row in rows))

    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["label"]), row["source_name"])].append(row)

    print("label/source lengths:")
    for (label, source), group_rows in sorted(grouped.items()):
        print(f"  label={label} source={source}: n={len(group_rows):,}, {describe_lengths(group_rows)}")

    if "bucket" in rows[0]:
        print("label/bucket lengths:")
        bucket_grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            bucket_grouped[(int(row["label"]), row["bucket"])].append(row)
        for (label, bucket), group_rows in sorted(bucket_grouped.items()):
            print(f"  label={label} bucket={bucket}: n={len(group_rows):,}, {describe_lengths(group_rows)}")


def load_dataset_split(path: Path) -> Dataset | DatasetDict:
    dataset = load_from_disk(str(path))
    if not isinstance(dataset, Dataset | DatasetDict):
        raise TypeError(f"Expected Dataset or DatasetDict at {path}, got {type(dataset).__name__}")
    return dataset


def main() -> None:
    configure_stdout()

    parser = argparse.ArgumentParser(description="Inspect cached training/validation dataset distribution.")
    parser.add_argument(
        "dataset_split",
        nargs="?",
        default=DEFAULT_DATASET_SPLIT,
        help="Path to a cached dataset_split-* directory.",
    )
    args = parser.parse_args()

    path = Path(args.dataset_split)
    dataset = load_dataset_split(path)

    print(f"dataset: {path.resolve()}")
    if isinstance(dataset, DatasetDict):
        for split_name in dataset:
            analyze_split(split_name, list(dataset[split_name]))
    else:
        analyze_split("dataset", list(dataset))


if __name__ == "__main__":
    main()
