# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_from_disk


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


def length_bin_for_text(text: str) -> str:
    length = len(text)
    for lower, upper, label in LENGTH_BINS:
        if length >= lower and (upper is None or length <= upper):
            return label
    return LENGTH_BINS[-1][2]


def print_length_bin_stats(rows: list[dict[str, Any]]) -> None:
    print("length bins:")
    bin_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        bin_rows[length_bin_for_text(str(row["text"]))].append(row)

    for _, _, label in LENGTH_BINS:
        group_rows = bin_rows.get(label, [])
        if not group_rows:
            continue
        labels = Counter(int(row["label"]) for row in group_rows)
        benign = labels.get(0, 0)
        attacks = labels.get(1, 0)
        ratio = benign / attacks if attacks else None
        ratio_text = f"{ratio:.2f}" if ratio is not None else "n/a"
        print(
            f"  {label}: rows={len(group_rows):,}, "
            f"benign={benign:,}, attack={attacks:,}, "
            f"benign/attack={ratio_text}, {describe_lengths(group_rows)}"
        )

    print("label/length bins:")
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["label"]), length_bin_for_text(str(row["text"])))].append(row)
    for (label, length_bin), group_rows in sorted(grouped.items()):
        print(f"  label={label} length_bin={length_bin}: n={len(group_rows):,}, {describe_lengths(group_rows)}")


def analyze_split(name: str, rows: list[dict[str, Any]]) -> None:
    print(f"\n== {name} ==")
    print(f"rows: {len(rows):,}")
    if not rows:
        return

    print_counter("labels:", Counter(row["label"] for row in rows))
    print_counter("sources:", Counter(row["source_name"] for row in rows))
    if "bucket" in rows[0]:
        print_counter("buckets:", Counter(row["bucket"] for row in rows))
    if "language" in rows[0]:
        print_counter("languages:", Counter(row["language"] for row in rows))
    if "text_unit" in rows[0]:
        print_counter("text units:", Counter(row["text_unit"] for row in rows))
    print_length_bin_stats(rows)

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


def analyze_dataset_path(path: Path, display_name: str | None = None) -> None:
    dataset = load_dataset_split(path)

    name = display_name or str(path.resolve())
    print(f"\ndataset: {name}")
    print(f"path: {path.resolve()}")
    if isinstance(dataset, DatasetDict):
        for split_name in dataset:
            analyze_split(split_name, list(dataset[split_name]))
    else:
        analyze_split("dataset", list(dataset))


def main() -> None:
    configure_stdout()

    parser = argparse.ArgumentParser(description="Inspect training and validation dataset distribution.")
    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Path to the prepared training Dataset or DatasetDict directory.",
    )
    parser.add_argument(
        "--validation-dataset-dir",
        default=None,
        help="Optional path to a standalone validation Dataset or DatasetDict directory.",
    )
    args = parser.parse_args()

    analyze_dataset_path(Path(args.dataset_dir), "training")
    if args.validation_dataset_dir:
        analyze_dataset_path(Path(args.validation_dataset_dir), "validation")


if __name__ == "__main__":
    main()
