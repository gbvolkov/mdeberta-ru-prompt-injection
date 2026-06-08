from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.ipc as ipc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export exact model input texts labelled as attack.")
    parser.add_argument("--dataset-dir", default="training-dataset-v16-critical-recall-restoration-windowed")
    parser.add_argument("--critical-error-jsonl", default="v16-error-inspection-critical-ru-fn-at-0.95.jsonl")
    parser.add_argument("--critical-corpus-jsonl", default="v13-critical-ru-validation-corpus.jsonl")
    parser.add_argument("--output-prefix", default="v16-exact-attack-inputs")
    return parser.parse_args()


def iter_jsonl(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(path: str | Path, payload: Any) -> None:
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_arrow_split(split_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arrow_path in sorted(split_dir.glob("data-*.arrow")):
        with arrow_path.open("rb") as f:
            table = ipc.open_stream(f).read_all()
        columns = table.column_names
        data = table.to_pylist()
        for row in data:
            rows.append({key: row.get(key) for key in columns})
    return rows


def export_dataset_attack_rows(dataset_dir: Path) -> dict[str, list[dict[str, Any]]]:
    exported: dict[str, list[dict[str, Any]]] = {}
    for split_name in ("train", "validation"):
        split_dir = dataset_dir / split_name
        if not split_dir.exists():
            continue
        rows = []
        for row in read_arrow_split(split_dir):
            if int(row.get("label") or 0) != 1:
                continue
            rows.append(
                {
                    "split": split_name,
                    "label": 1,
                    "document_id": row.get("document_id"),
                    "text": row.get("text"),
                    "source_name": row.get("source_name"),
                    "component": row.get("component"),
                    "category": row.get("category"),
                    "language": row.get("language"),
                    "text_hash": row.get("text_hash"),
                    "attack_text_hash": row.get("attack_text_hash"),
                    "semantic_family": row.get("semantic_family"),
                    "generation_type": row.get("generation_type"),
                    "anchor_document_id": row.get("anchor_document_id"),
                    "anchor_text_hash": row.get("anchor_text_hash"),
                    "source_document_id": row.get("source_document_id"),
                    "window_index": row.get("window_index"),
                    "window_count": row.get("window_count"),
                    "v13_score": row.get("v13_score"),
                    "score_band": row.get("score_band"),
                    "review_status": row.get("review_status"),
                }
            )
        exported[split_name] = rows
    return exported


def export_critical_error_inputs(error_jsonl: Path, corpus_jsonl: Path) -> list[dict[str, Any]]:
    error_by_id = {row["document_id"]: row for row in iter_jsonl(error_jsonl)}
    rows = []
    for row in iter_jsonl(corpus_jsonl):
        document_id = row.get("document_id")
        if document_id not in error_by_id:
            continue
        error = error_by_id[document_id]
        rows.append(
            {
                "split": "v13_critical_ru_validation_corpus",
                "label": row.get("label") or row.get("document_label"),
                "document_label": row.get("document_label"),
                "document_id": document_id,
                "text": row.get("text"),
                "source_name": row.get("source_name"),
                "category": row.get("category"),
                "language": row.get("language"),
                "bucket": row.get("bucket"),
                "text_hash": row.get("text_hash"),
                "attack_text_hash": row.get("attack_text_hash"),
                "attack_template_id": row.get("attack_template_id"),
                "semantic_family": row.get("semantic_family"),
                "source_doc_id": row.get("source_doc_id"),
                "carrier_document_id": row.get("carrier_document_id"),
                "window_index": row.get("window_index"),
                "v16_score": error.get("score"),
                "v16_tags": error.get("tags"),
            }
        )
    return sorted(rows, key=lambda row: float(row.get("v16_score") or 0.0))


def write_critical_md(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Exact Model Inputs: 27 V16 Critical-RU False Negatives",
        "",
        "These are the exact `text` values from `v13-critical-ru-validation-corpus.jsonl` that were passed to the model as attack-labelled validation documents.",
        "",
        "No text has been reconstructed or completed.",
        "",
    ]
    for index, row in enumerate(rows, 1):
        text = str(row.get("text") or "").replace("```", "` ` `")
        lines.extend(
            [
                f"## {index}. `{row.get('document_id')}`",
                "",
                f"- Label passed: `{row.get('label')}`",
                f"- V16 score: `{float(row.get('v16_score') or 0.0):.12f}`",
                f"- Category: `{row.get('category')}`",
                f"- Source: `{row.get('source_name')}`",
                f"- Language: `{row.get('language')}`",
                f"- Bucket: `{row.get('bucket')}`",
                f"- Attack template ID: `{row.get('attack_template_id') or ''}`",
                f"- Attack text hash: `{row.get('attack_text_hash') or ''}`",
                f"- Text hash: `{row.get('text_hash') or ''}`",
                "",
                "```text",
                text,
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "by_component": dict(Counter(str(row.get("component")) for row in rows).most_common()),
        "by_source_name": dict(Counter(str(row.get("source_name")) for row in rows).most_common()),
        "by_category": dict(Counter(str(row.get("category")) for row in rows).most_common()),
        "by_language": dict(Counter(str(row.get("language")) for row in rows).most_common()),
        "by_semantic_family": dict(Counter(str(row.get("semantic_family")) for row in rows).most_common()),
    }


def main() -> None:
    args = parse_args()
    prefix = Path(args.output_prefix)
    dataset_dir = Path(args.dataset_dir)

    attack_rows_by_split = export_dataset_attack_rows(dataset_dir)
    train_rows = attack_rows_by_split.get("train", [])
    validation_rows = attack_rows_by_split.get("validation", [])
    critical_rows = export_critical_error_inputs(Path(args.critical_error_jsonl), Path(args.critical_corpus_jsonl))

    train_path = prefix.with_name(prefix.name + "-train.jsonl")
    validation_path = prefix.with_name(prefix.name + "-validation.jsonl")
    critical_jsonl_path = prefix.with_name(prefix.name + "-27-critical-ru-fn.jsonl")
    critical_md_path = prefix.with_name(prefix.name + "-27-critical-ru-fn.md")
    report_path = prefix.with_name(prefix.name + "-report.json")

    write_jsonl(train_path, train_rows)
    write_jsonl(validation_path, validation_rows)
    write_jsonl(critical_jsonl_path, critical_rows)
    write_critical_md(critical_md_path, critical_rows)

    report = {
        "dataset_dir": str(dataset_dir),
        "files": {
            "train_attack_inputs": str(train_path),
            "validation_attack_inputs": str(validation_path),
            "critical_ru_fn_attack_inputs_jsonl": str(critical_jsonl_path),
            "critical_ru_fn_attack_inputs_md": str(critical_md_path),
        },
        "train": summarize(train_rows),
        "validation": summarize(validation_rows),
        "critical_ru_fn_27": {
            "rows": len(critical_rows),
            "source": str(args.critical_corpus_jsonl),
            "note": "Exact text values passed as attack-labelled validation documents.",
        },
    }
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
