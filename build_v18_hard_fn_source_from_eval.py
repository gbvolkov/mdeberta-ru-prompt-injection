# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ATTACK_LABELS = {"prompt_injection", "attack", "malicious", "1", "true"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a V18 hard-FN source JSONL from a fresh malicious attack bank "
            "scored by run_blind_broad_eval.py."
        )
    )
    parser.add_argument("--attack-bank-jsonl", required=True)
    parser.add_argument("--eval-document-jsonl", required=True)
    parser.add_argument("--eval-window-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--threshold", type=float, default=0.82)
    parser.add_argument("--id-column", default="attack_template_id")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--source-name", default="fresh_external_attack_bank_hard_fn")
    parser.add_argument("--source-origin", default="fresh_external_attack_bank_previous_model_misses")
    return parser.parse_args()


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\x00", " ").split())


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "reviewed", "trusted"}


def label_value(row: dict[str, Any]) -> str:
    return normalize_text(row.get("document_label") or row.get("label")).lower()


def text_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def short_hash(text: str) -> str:
    return text_hash(text)[:20]


def load_attack_bank(path: str | Path, id_column: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for idx, row in enumerate(iter_jsonl(path)):
        doc_id = str(row.get(id_column) or row.get("document_id") or row.get("id") or f"doc_{idx}")
        rows[doc_id] = row
    return rows


def missed_documents(path: str | Path, threshold: float) -> set[str]:
    missed: set[str] = set()
    for row in iter_jsonl(path):
        if label_value(row) not in ATTACK_LABELS:
            continue
        score = float(row.get("document_max_prompt_injection_score", 0.0) or 0.0)
        if score < threshold:
            missed.add(str(row.get("document_id") or ""))
    missed.discard("")
    return missed


def anchor_visible(anchor: str, window_text: str) -> bool:
    if not anchor:
        return False
    return anchor.casefold() in window_text.casefold()


def build_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    attack_bank = load_attack_bank(args.attack_bank_jsonl, args.id_column)
    missed = missed_documents(args.eval_document_jsonl, args.threshold)
    rows: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    accepted_by_family: Counter[str] = Counter()
    accepted_by_language: Counter[str] = Counter()
    seen_text_hashes: set[str] = set()

    for window in iter_jsonl(args.eval_window_jsonl):
        doc_id = str(window.get("document_id") or "")
        if doc_id not in missed:
            continue
        source = attack_bank.get(doc_id)
        if not source:
            rejected["missing_attack_bank_source"] += 1
            continue
        score = float(window.get("p_prompt_injection", 0.0) or 0.0)
        if score >= args.threshold:
            rejected["window_not_missed"] += 1
            continue
        window_text = normalize_text(window.get("window_text"))
        if not window_text:
            rejected["missing_window_text"] += 1
            continue
        attack_text = normalize_text(source.get("attack_text") or source.get("text") or window_text)
        anchor = normalize_text(source.get("attack_anchor_text") or source.get("anchor_text") or source.get("attack_span_text"))
        if not anchor:
            rejected["missing_attack_anchor_text"] += 1
            continue
        if not anchor_visible(anchor, window_text):
            rejected["anchor_not_visible_in_window"] += 1
            continue
        th = text_hash(window_text)
        if th in seen_text_hashes:
            rejected["duplicate_window_text"] += 1
            continue
        seen_text_hashes.add(th)

        family = str(source.get("semantic_family") or window.get("semantic_family") or "hard_false_negative_visible_attack")
        language = str(source.get("language") or window.get("language") or "unknown")
        out = {
            "window_text": window_text,
            "text": window_text,
            "label": "prompt_injection",
            "document_label": "prompt_injection",
            "attack_text": attack_text,
            "attack_anchor_text": anchor,
            "manual_reviewed_visible_attack": True,
            "attack_visible_in_window": True,
            "language": language,
            "semantic_family": family,
            "semantic_subfamily": str(source.get("semantic_subfamily") or f"{family}:{language}"),
            "source_pool": "external_mining_only",
            "source_name": str(source.get("source_name") or args.source_name),
            "source_origin": str(source.get("source_origin") or args.source_origin),
            "source_document_id": doc_id,
            "original_document_id": doc_id,
            "document_id": f"v18_hard_fn_visible:{doc_id}:{window.get('window_index', 0)}:{short_hash(window_text)}",
            "category": str(source.get("category") or window.get("category") or "hard_false_negative_attack"),
            "window_index": int(window.get("window_index", 0) or 0),
            "window_count": int(window.get("window_count", 1) or 1),
            "previous_model_score": score,
            "threshold": args.threshold,
            "text_hash": th,
            "normalized_text_hash": th,
            "dedupe_cluster_id": th[:24],
            "attack_template_id": doc_id,
            "manual_reviewed_attack": truthy(source.get("manual_reviewed_attack")),
            "trusted_attack": truthy(source.get("trusted_attack")),
        }
        rows.append(out)
        accepted_by_family[family] += 1
        accepted_by_language[language] += 1
        if args.max_rows is not None and len(rows) >= args.max_rows:
            break

    return rows, {
        "attack_bank_rows": len(attack_bank),
        "missed_documents": len(missed),
        "accepted_rows": len(rows),
        "accepted_by_family": dict(accepted_by_family.most_common()),
        "accepted_by_language": dict(accepted_by_language.most_common()),
        "rejected": dict(rejected),
    }


def main() -> None:
    args = parse_args()
    rows, report = build_rows(args)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    report["output_jsonl"] = str(output_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
