from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.ipc as ipc

from build_strict_exact_window_dataset import visible_attack


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a proper V16 diagnostic validation suite from exact windows/documents.")
    parser.add_argument("--strict-dataset-dir", default="training-dataset-v16-strict-exact-windowed")
    parser.add_argument("--benign-prod-jsonl", default="benign-prod-calibration-dev.jsonl")
    parser.add_argument("--malicious-dev-jsonl", default="malicious-document-dev.jsonl")
    parser.add_argument("--output-dir", default="v16-proper-validation-suite")
    parser.add_argument("--overwrite", action="store_true")
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
            rows.extend(ipc.open_stream(f).read_all().to_pylist())
    return rows


def normalize_window_row(row: dict[str, Any], *, split: str, label_name: str) -> dict[str, Any]:
    text = str(row.get("text") or "")
    return {
        "document_id": row.get("document_id") or row.get("text_hash"),
        "document_label": label_name,
        "label": label_name,
        "text": text,
        "category": row.get("category") or "unknown",
        "source_name": row.get("source_name") or "strict_exact_window",
        "language": row.get("language") or "unknown",
        "split": split,
        "component": row.get("component"),
        "generation_type": row.get("generation_type"),
        "semantic_family": row.get("semantic_family"),
        "text_hash": row.get("text_hash"),
        "attack_text_hash": row.get("attack_text_hash"),
        "source_document_id": row.get("source_document_id"),
        "window_index": row.get("window_index"),
        "window_count": row.get("window_count"),
        "v13_score": row.get("v13_score"),
        "score_band": row.get("score_band"),
        "validation_note": "exact strict window row; no reassembly",
    }


def build_window_corpora(strict_dataset_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    validation_rows = read_arrow_split(strict_dataset_dir / "validation")
    attack_rows = []
    benign_rows = []
    for row in validation_rows:
        label = int(row.get("label") or 0)
        if label == 1:
            # Assert the strict invariant again while building validation.
            if not visible_attack(str(row.get("text") or "")):
                continue
            attack_rows.append(normalize_window_row(row, split="strict_validation", label_name="prompt_injection"))
        else:
            benign_rows.append(normalize_window_row(row, split="strict_validation", label_name="not_prompt_injection"))
    return attack_rows, benign_rows


def count_words(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def estimate_window_count_by_words(text: str, *, window_words: int = 190, stride_words: int = 95) -> int:
    words = count_words(text)
    if words <= 0:
        return 0
    if words <= window_words:
        return 1
    return 1 + max(0, (words - window_words + stride_words - 1) // stride_words)


def window_bucket(count: Any) -> str:
    try:
        value = int(count)
    except (TypeError, ValueError):
        return "unknown"
    if value <= 1:
        return "1"
    if value <= 4:
        return "2-4"
    if value <= 20:
        return "5-20"
    if value <= 50:
        return "21-50"
    if value <= 100:
        return "51-100"
    return "101+"


def normalize_benign_doc(row: dict[str, Any]) -> dict[str, Any]:
    text = str(row.get("text") or row.get("document_text") or "")
    out = dict(row)
    out["document_label"] = "not_prompt_injection"
    out["label"] = "not_prompt_injection"
    out["text"] = text
    out["validation_note"] = "existing benign production dev document; text unchanged"
    out["estimated_word_window_count"] = estimate_window_count_by_words(text)
    out["visible_attack_cue_in_full_text"] = visible_attack(text)
    return out


def normalize_malicious_doc(row: dict[str, Any]) -> dict[str, Any]:
    text = str(row.get("text") or row.get("document_text") or "")
    out = dict(row)
    out["document_label"] = "prompt_injection"
    out["label"] = "prompt_injection"
    out["text"] = text
    out["validation_note"] = "existing malicious dev document kept only if full text visibly contains attack intent"
    out["estimated_word_window_count"] = estimate_window_count_by_words(text)
    out["visible_attack_cue_in_full_text"] = visible_attack(text)
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "labels": dict(Counter(str(row.get("document_label") or row.get("label")) for row in rows)),
        "categories": dict(Counter(str(row.get("category")) for row in rows).most_common()),
        "sources": dict(Counter(str(row.get("source_name")) for row in rows).most_common(50)),
        "languages": dict(Counter(str(row.get("language")) for row in rows).most_common()),
        "components": dict(Counter(str(row.get("component")) for row in rows if row.get("component")).most_common()),
        "estimated_word_window_buckets": dict(
            Counter(window_bucket(row.get("estimated_word_window_count")) for row in rows if row.get("estimated_word_window_count") is not None).most_common()
        ),
        "production_window_count_buckets": dict(
            Counter(window_bucket(row.get("production_window_count")) for row in rows if row.get("production_window_count") is not None).most_common()
        ),
        "attack_window_index_buckets": dict(
            Counter(window_bucket((int(row.get("attack_window_index")) + 1) if row.get("attack_window_index") is not None else None) for row in rows if row.get("attack_window_index") is not None).most_common()
        ),
    }


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} already exists. Use --overwrite to replace it.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    prepare_output(output_dir, args.overwrite)

    attack_windows, benign_windows = build_window_corpora(Path(args.strict_dataset_dir))
    benign_docs = [normalize_benign_doc(row) for row in iter_jsonl(args.benign_prod_jsonl)]

    malicious_docs_all = [normalize_malicious_doc(row) for row in iter_jsonl(args.malicious_dev_jsonl)]
    malicious_docs = [row for row in malicious_docs_all if row["visible_attack_cue_in_full_text"]]
    dropped_malicious_docs = [row for row in malicious_docs_all if not row["visible_attack_cue_in_full_text"]]

    files = {
        "proper_critical_attack_windows": output_dir / "proper-critical-attack-windows.jsonl",
        "proper_benign_windows": output_dir / "proper-benign-windows.jsonl",
        "proper_malicious_dev_documents": output_dir / "proper-malicious-dev-documents.jsonl",
        "proper_benign_prod_dev_documents": output_dir / "proper-benign-prod-dev-documents.jsonl",
        "dropped_malicious_dev_documents": output_dir / "dropped-malicious-dev-documents-without-visible-attack.jsonl",
    }
    write_jsonl(files["proper_critical_attack_windows"], attack_windows)
    write_jsonl(files["proper_benign_windows"], benign_windows)
    write_jsonl(files["proper_malicious_dev_documents"], malicious_docs)
    write_jsonl(files["proper_benign_prod_dev_documents"], benign_docs)
    write_jsonl(files["dropped_malicious_dev_documents"], dropped_malicious_docs)

    manifest = {
        "suite": "v16_proper_diagnostic",
        "note": (
            "Diagnostic validation suite with exact strict attack windows and malicious documents filtered "
            "to require visible attack intent in the evaluated text. This is not blind acceptance data."
        ),
        "thresholds": "0.82,0.90,0.95,0.99,0.999,0.9995",
        "primary_threshold": 0.95,
        "corpora": {
            "proper_critical_attack_windows": {
                "path": str(files["proper_critical_attack_windows"]),
                "kind": "strict_attack_window_regression",
                "rows": len(attack_windows),
                "note": "Exact validation windows labelled attack and passing visible-attack filter.",
            },
            "proper_benign_windows": {
                "path": str(files["proper_benign_windows"]),
                "kind": "strict_benign_window_regression",
                "rows": len(benign_windows),
                "note": "Exact validation windows labelled benign.",
            },
            "proper_malicious_dev_documents": {
                "path": str(files["proper_malicious_dev_documents"]),
                "kind": "visible_attack_long_document_dev",
                "rows": len(malicious_docs),
                "dropped_without_visible_attack": len(dropped_malicious_docs),
                "note": "Existing malicious dev docs kept only if full text visibly contains attack intent.",
            },
            "proper_benign_prod_dev_documents": {
                "path": str(files["proper_benign_prod_dev_documents"]),
                "kind": "benign_long_document_dev",
                "rows": len(benign_docs),
                "note": "Existing benign production dev docs, unchanged, with estimated word-window counts.",
            },
        },
    }
    report = {
        "manifest": str(output_dir / "validation-suite-manifest.json"),
        "files": {key: str(path) for key, path in files.items()},
        "corpora": {
            "proper_critical_attack_windows": summarize(attack_windows),
            "proper_benign_windows": summarize(benign_windows),
            "proper_malicious_dev_documents": summarize(malicious_docs),
            "proper_benign_prod_dev_documents": summarize(benign_docs),
            "dropped_malicious_dev_documents": summarize(dropped_malicious_docs),
        },
        "checks": {
            "attack_windows_all_visible_attack": all(visible_attack(row["text"]) for row in attack_windows),
            "malicious_docs_all_visible_attack": all(row["visible_attack_cue_in_full_text"] for row in malicious_docs),
            "benign_windows_attack_labels": Counter(row["document_label"] for row in benign_windows),
            "attack_windows_attack_labels": Counter(row["document_label"] for row in attack_windows),
        },
    }
    write_json(output_dir / "validation-suite-manifest.json", manifest)
    write_json(output_dir / "validation-suite-report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
