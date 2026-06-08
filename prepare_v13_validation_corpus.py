from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_from_disk

from v12_pipeline_utils import LABEL_ATTACK, LABEL_BENIGN, infer_language, label_name, text_hash, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the V13 validation split to JSONL for document/window-level scoring.")
    parser.add_argument("--dataset-dir", default="training-dataset-v13-critical-russian-correction-windowed")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output-jsonl", default="v13-validation-window-corpus.jsonl")
    parser.add_argument("--report-json", default="v13-validation-window-corpus-report.json")
    return parser.parse_args()


def row_label(value: Any) -> str:
    return LABEL_ATTACK if int(value) == 1 else LABEL_BENIGN


def main() -> None:
    args = parse_args()
    dataset = load_from_disk(args.dataset_dir)
    split = dataset[args.split] if hasattr(dataset, "keys") else dataset

    rows = []
    for idx, row in enumerate(split):
        text = row.get("text", "")
        label = row_label(row.get("label", 0))
        rows.append(
            {
                "document_id": f"v13_validation_{idx:06d}_{text_hash(text)}",
                "document_label": label,
                "label": label,
                "text": text,
                "source_name": row.get("source_name") or "v13_validation",
                "category": row.get("category") or row.get("bucket") or "unknown",
                "language": row.get("language") or infer_language(text),
                "bucket": row.get("bucket"),
                "semantic_family": row.get("semantic_family_id") or row.get("semantic_family") or "unknown",
                "attack_text_hash": row.get("attack_text_hash"),
                "attack_template_id": row.get("attack_template_id"),
                "source_doc_id": row.get("source_doc_id"),
                "carrier_document_id": row.get("carrier_document_id"),
                "window_index": row.get("window_index"),
                "text_hash": text_hash(text),
            }
        )

    report = {
        "dataset_dir": args.dataset_dir,
        "split": args.split,
        "output_jsonl": args.output_jsonl,
        "rows": len(rows),
        "labels": dict(Counter(row["document_label"] for row in rows)),
        "sources": dict(Counter(row["source_name"] for row in rows)),
        "categories": dict(Counter(row["category"] for row in rows)),
        "languages": dict(Counter(row["language"] for row in rows)),
        "semantic_families": dict(Counter(row["semantic_family"] for row in rows)),
        "note": "Each row is one V13 validation training window represented as a scoring document.",
    }
    write_jsonl(args.output_jsonl, rows)
    write_json(args.report_json, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
