from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_from_disk

from v12_pipeline_utils import iter_jsonl, normalize_text, text_hash, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check exact leakage from V12/V13 training rows into locked corpora.")
    parser.add_argument("--training-dataset-dir", required=True)
    parser.add_argument("--locked-corpus-glob", action="append", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--max-examples", type=int, default=20)
    return parser.parse_args()


def long_signature(text: str, *, n: int = 200) -> str:
    norm = normalize_text(text).lower()
    if len(norm) <= n:
        return text_hash(norm)
    return text_hash(norm[:n] + " ... " + norm[-n:])


def collect_locked(patterns: list[str]) -> dict[str, Any]:
    index = {
        "text_hashes": {},
        "long_signatures": {},
        "document_ids": {},
        "attack_text_hashes": {},
        "paths": [],
        "rows": 0,
    }
    for pattern in patterns:
        for path_text in glob.glob(pattern):
            path = Path(path_text)
            if not path.exists() or path.is_dir():
                continue
            index["paths"].append(str(path))
            for row_idx, row in enumerate(iter_jsonl(path)):
                index["rows"] += 1
                ref = {"path": str(path), "row": row_idx, "document_id": row.get("document_id")}
                text = row.get("text") or row.get("document_text") or row.get("window_text") or ""
                if normalize_text(text):
                    index["text_hashes"].setdefault(text_hash(text), ref)
                    index["long_signatures"].setdefault(long_signature(text), ref)
                if row.get("document_id") is not None:
                    index["document_ids"].setdefault(str(row["document_id"]), ref)
                if row.get("attack_text_hash"):
                    index["attack_text_hashes"].setdefault(str(row["attack_text_hash"]), ref)
    return index


def collect_training(dataset_dir: str | Path) -> list[dict[str, Any]]:
    data = load_from_disk(str(dataset_dir))
    rows: list[dict[str, Any]] = []
    for split_name, split in data.items():
        for idx, row in enumerate(split):
            rows.append({"split": split_name, "row": idx, **row})
    return rows


def main() -> None:
    args = parse_args()
    locked = collect_locked(args.locked_corpus_glob)
    training_rows = collect_training(args.training_dataset_dir)

    hard_errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    counts = Counter()

    for row in training_rows:
        text = row.get("text") or ""
        row_ref = {"split": row.get("split"), "row": row.get("row"), "source_name": row.get("source_name")}
        if normalize_text(text):
            h = text_hash(text)
            if h in locked["text_hashes"]:
                hard_errors.append({"check": "exact_text_hash", "training": row_ref, "locked": locked["text_hashes"][h]})
                counts["exact_text_hash"] += 1
            sig = long_signature(text)
            if sig in locked["long_signatures"]:
                warnings.append({"check": "long_signature", "training": row_ref, "locked": locked["long_signatures"][sig]})
                counts["long_signature"] += 1
        doc_id = row.get("document_id") or row.get("carrier_document_id")
        if doc_id is not None and str(doc_id) in locked["document_ids"]:
            hard_errors.append({"check": "document_id", "training": row_ref, "locked": locked["document_ids"][str(doc_id)]})
            counts["document_id"] += 1
        attack_hash = row.get("attack_text_hash")
        if attack_hash and str(attack_hash) in locked["attack_text_hashes"]:
            hard_errors.append({"check": "attack_text_hash", "training": row_ref, "locked": locked["attack_text_hashes"][str(attack_hash)]})
            counts["attack_text_hash"] += 1

    report = {
        "locked_rows": locked["rows"],
        "locked_paths": sorted(locked["paths"]),
        "training_rows": len(training_rows),
        "hard_error_checks": len(hard_errors),
        "warning_checks": len(warnings),
        "counts": dict(counts),
        "hard_error_examples": hard_errors[: args.max_examples],
        "warning_examples": warnings[: args.max_examples],
        "status": "pass" if not hard_errors else "fail",
    }
    write_json(args.report_json, report)
    print(json.dumps({k: report[k] for k in ("locked_rows", "training_rows", "hard_error_checks", "warning_checks", "status")}, ensure_ascii=False, indent=2))
    if hard_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
