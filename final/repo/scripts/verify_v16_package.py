#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_from_disk


EXPECTED_DATASETS = {
    "training-dataset-v10-benign-coverage": {
        "train": (127408, {0: 99817, 1: 27591}),
        "validation": (22485, {0: 17616, 1: 4869}),
    },
    "training-dataset-v11-fp-correction-windowed-200k": {
        "train": (200000, {0: 130000, 1: 70000}),
        "validation": (25000, {0: 16250, 1: 8750}),
    },
    "training-dataset-v13-critical-russian-correction-windowed": {
        "train": (35382, {0: 10316, 1: 25066}),
        "validation": (4000, {0: 1166, 1: 2834}),
    },
    "training-dataset-v16-critical-recall-restoration-windowed": {
        "train": (37675, {0: 14239, 1: 23436}),
        "validation": (6000, {0: 2266, 1: 3734}),
    },
}

REQUIRED_SCRIPTS = {
    "build_training_dataset.py",
    "build_v9_coverage_dataset.py",
    "build_v10_benign_coverage_dataset.py",
    "build_false_positive_corpus.py",
    "run_false_positive_review.py",
    "build_v11_windowed_dataset.py",
    "build_v12_correction_dataset.py",
    "build_v13_critical_correction_dataset.py",
    "build_v16_critical_recall_restoration_dataset.py",
    "build_v15_anchored_critical_correction_dataset.py",
    "v12_pipeline_utils.py",
    "train_mdeberta_ru_prompt_injection_option_b.py",
}

REQUIRED_RAW = {
    "false-positive-corpus-documents.jsonl",
    "false-positive-review-v10-threshold-0.82.jsonl",
    "benign-prod-calibration-dev.jsonl",
    "benign-prod-locked-test.jsonl",
    "malicious-document-dev.jsonl",
    "malicious-document-test.jsonl",
    "short_exfiltration_phrase_variants_v3.json",
    "training-dataset-v4-label-judgments.jsonl",
    "training-dataset-v5-balanced-llm-audit.jsonl",
    "v13-critical-ru-validation-corpus.jsonl",
    "v13-critical-results-v13.jsonl",
    "v13-critical-results-v15.jsonl",
    "v13-benign-validation-corpus.jsonl",
    "v13-benign-window-results-v13.jsonl",
    "benign-prod-results-v13.jsonl",
}

MODEL_ARTIFACT_NAMES = {
    "model.safetensors",
    "pytorch_model.bin",
    "optimizer.pt",
    "scheduler.pt",
    "training_args.bin",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the V16 full-cycle package without training.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--allow-generated-work-models", action="store_true")
    return parser.parse_args()


def inspect_dataset(path: Path) -> dict[str, Any]:
    dataset = load_from_disk(str(path))
    report: dict[str, Any] = {}
    for split_name, split in dataset.items():
        report[split_name] = {
            "rows": len(split),
            "labels": dict(Counter(int(value) for value in split["label"])),
            "columns": split.column_names,
        }
    return report


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    failures: list[str] = []

    scripts_dir = root / "scripts"
    for name in sorted(REQUIRED_SCRIPTS):
        if not (scripts_dir / name).is_file():
            failures.append(f"missing_script:{name}")

    raw_dir = root / "datasets/raw"
    for name in sorted(REQUIRED_RAW):
        if not (raw_dir / name).is_file():
            failures.append(f"missing_raw_input:{name}")

    prepared_dir = root / "datasets/prepared"
    dataset_report: dict[str, Any] = {}
    for name, expected_splits in EXPECTED_DATASETS.items():
        path = prepared_dir / name
        if not path.is_dir():
            failures.append(f"missing_prepared_dataset:{name}")
            continue
        actual = inspect_dataset(path)
        dataset_report[name] = actual
        for split_name, (expected_rows, expected_labels) in expected_splits.items():
            if split_name not in actual:
                failures.append(f"missing_split:{name}:{split_name}")
                continue
            if actual[split_name]["rows"] != expected_rows:
                failures.append(
                    f"row_count:{name}:{split_name}:{actual[split_name]['rows']}!={expected_rows}"
                )
            if actual[split_name]["labels"] != expected_labels:
                failures.append(
                    f"label_count:{name}:{split_name}:{actual[split_name]['labels']}!={expected_labels}"
                )

    forbidden: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.name not in MODEL_ARTIFACT_NAMES:
            continue
        relative = path.relative_to(root)
        if args.allow_generated_work_models and relative.parts and relative.parts[0] == "work":
            continue
        forbidden.append(relative.as_posix())
    if forbidden:
        failures.extend(f"bundled_model_artifact:{path}" for path in forbidden)

    report = {
        "status": "pass" if not failures else "fail",
        "root": str(root),
        "base_model": "microsoft/mdeberta-v3-base",
        "bundled_model_artifacts": forbidden,
        "datasets": dataset_report,
        "failures": failures,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
