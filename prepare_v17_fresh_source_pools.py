# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from transformers import AutoTokenizer

from v12_pipeline_utils import (
    count_windows,
    extract_text,
    infer_language,
    iter_jsonl,
    normalize_text,
    text_hash,
    window_count_bucket,
    write_json,
    write_jsonl,
)


WEAK_PRODUCTION_CATEGORIES = (
    "job_descriptions",
    "hr_policies",
    "corporate_procedures",
)

FP_CANDIDATE_CATEGORIES = (
    "job_descriptions",
    "hr_policies",
    "corporate_procedures",
    "support_documentation",
    "admin_instructions",
    "legal_templates",
    "technical_documentation",
    "security_compliance_redaction_wrappers",
)

EXCLUSION_SET_KEYS = (
    "text_hashes",
    "source_document_ids",
    "carrier_document_ids",
    "attack_text_hashes",
    "attack_template_ids",
    "generated_instance_ids",
)

FORBIDDEN_INPUT_MARKERS = (
    "training-dataset-v",
    "false-positive-corpus",
    "false-positive-review",
    "malicious-document-dev",
    "malicious-document-test",
    "benign-prod-calibration",
    "benign-prod-locked",
    "v12-eval-suites",
    "v13-",
    "v14-",
    "v15-",
    "v16-",
    "validation-comparison-",
)


@dataclass(frozen=True)
class SourceSpec:
    name: str
    repo_id: str
    config: str | None
    split: str
    language: str
    streaming: bool = True


@dataclass(frozen=True)
class CategoryRule:
    strong: tuple[str, ...]
    medium: tuple[str, ...]


HF_SOURCES = (
    SourceSpec("fresh_fineweb2_ru", "HuggingFaceFW/fineweb-2", "rus_Cyrl", "train", "ru"),
    SourceSpec("fresh_c4_ru", "allenai/c4", "ru", "train", "ru"),
    SourceSpec("fresh_wikipedia_ru", "wikimedia/wikipedia", "20231101.ru", "train", "ru"),
    SourceSpec("fresh_stackexchange", "common-pile/stackexchange_filtered", None, "train", "mixed"),
    SourceSpec("fresh_fineweb_en", "HuggingFaceFW/fineweb", "sample-10BT", "train", "en"),
    SourceSpec("fresh_c4_en", "allenai/c4", "en", "train", "en"),
    SourceSpec("fresh_wikipedia_en", "wikimedia/wikipedia", "20231101.en", "train", "en"),
    SourceSpec("fresh_legal_case_summaries", "joelniklaus/legal_case_document_summarization", None, "train", "en"),
    SourceSpec("fresh_cnn_dailymail", "abisee/cnn_dailymail", "3.0.0", "train", "en"),
    SourceSpec("fresh_ag_news", "fancyzhx/ag_news", None, "train", "en"),
)


CATEGORY_RULES = {
    "job_descriptions": CategoryRule(
        strong=(
            "job description",
            "role description",
            "position description",
            "\u0434\u043e\u043b\u0436\u043d\u043e\u0441\u0442\u043d\u0430\u044f \u0438\u043d\u0441\u0442\u0440\u0443\u043a\u0446\u0438\u044f",
            "\u0434\u043e\u043b\u0436\u043d\u043e\u0441\u0442\u043d\u044b\u0435 \u043e\u0431\u044f\u0437\u0430\u043d\u043d\u043e\u0441\u0442\u0438",
            "\u0444\u0443\u043d\u043a\u0446\u0438\u043e\u043d\u0430\u043b\u044c\u043d\u044b\u0435 \u043e\u0431\u044f\u0437\u0430\u043d\u043d\u043e\u0441\u0442\u0438",
        ),
        medium=(
            "responsibilities",
            "required qualifications",
            "candidate requirements",
            "reports to",
            "employment type",
            "\u0442\u0440\u0435\u0431\u043e\u0432\u0430\u043d\u0438\u044f \u043a \u043a\u0430\u043d\u0434\u0438\u0434\u0430\u0442\u0443",
            "\u043a\u0432\u0430\u043b\u0438\u0444\u0438\u043a\u0430\u0446\u0438\u043e\u043d\u043d\u044b\u0435 \u0442\u0440\u0435\u0431\u043e\u0432\u0430\u043d\u0438\u044f",
            "\u043f\u043e\u0434\u0447\u0438\u043d\u044f\u0435\u0442\u0441\u044f",
            "\u043f\u0440\u0430\u0432\u0430 \u0438 \u043e\u0442\u0432\u0435\u0442\u0441\u0442\u0432\u0435\u043d\u043d\u043e\u0441\u0442\u044c",
        ),
    ),
    "hr_policies": CategoryRule(
        strong=(
            "hr policy",
            "employee handbook",
            "personnel policy",
            "\u043a\u0430\u0434\u0440\u043e\u0432\u0430\u044f \u043f\u043e\u043b\u0438\u0442\u0438\u043a\u0430",
            "\u043f\u0440\u0430\u0432\u0438\u043b\u0430 \u0432\u043d\u0443\u0442\u0440\u0435\u043d\u043d\u0435\u0433\u043e \u0442\u0440\u0443\u0434\u043e\u0432\u043e\u0433\u043e \u0440\u0430\u0441\u043f\u043e\u0440\u044f\u0434\u043a\u0430",
            "\u043e\u0446\u0435\u043d\u043a\u0430 \u043f\u0435\u0440\u0441\u043e\u043d\u0430\u043b\u0430",
        ),
        medium=(
            "vacation",
            "sick leave",
            "dismissal",
            "employee data",
            "hiring policy",
            "\u043e\u0442\u043f\u0443\u0441\u043a",
            "\u0431\u043e\u043b\u044c\u043d\u0438\u0447\u043d\u044b\u0439",
            "\u043f\u0440\u0438\u0435\u043c \u043d\u0430 \u0440\u0430\u0431\u043e\u0442\u0443",
            "\u0443\u0432\u043e\u043b\u044c\u043d\u0435\u043d\u0438\u0435",
            "\u043f\u0435\u0440\u0441\u043e\u043d\u0430\u043b\u044c\u043d\u044b\u0435 \u0434\u0430\u043d\u043d\u044b\u0435 \u0441\u043e\u0442\u0440\u0443\u0434\u043d\u0438\u043a\u0430",
        ),
    ),
    "corporate_procedures": CategoryRule(
        strong=(
            "standard operating procedure",
            "internal procedure",
            "approval procedure",
            "\u0440\u0435\u0433\u043b\u0430\u043c\u0435\u043d\u0442",
            "\u043f\u043e\u0440\u044f\u0434\u043e\u043a \u0441\u043e\u0433\u043b\u0430\u0441\u043e\u0432\u0430\u043d\u0438\u044f",
            "\u0432\u043d\u0443\u0442\u0440\u0435\u043d\u043d\u044f\u044f \u043f\u0440\u043e\u0446\u0435\u0434\u0443\u0440\u0430",
            "\u043f\u0440\u043e\u0446\u0435\u0434\u0443\u0440\u0430 \u0443\u0442\u0432\u0435\u0440\u0436\u0434\u0435\u043d\u0438\u044f",
        ),
        medium=(
            "responsible persons",
            "process steps",
            "execution control",
            "approval workflow",
            "\u043e\u0442\u0432\u0435\u0442\u0441\u0442\u0432\u0435\u043d\u043d\u044b\u0435 \u043b\u0438\u0446\u0430",
            "\u044d\u0442\u0430\u043f\u044b \u043f\u0440\u043e\u0446\u0435\u0441\u0441\u0430",
            "\u043a\u043e\u043d\u0442\u0440\u043e\u043b\u044c \u0438\u0441\u043f\u043e\u043b\u043d\u0435\u043d\u0438\u044f",
        ),
    ),
    "support_documentation": CategoryRule(
        strong=("support article", "knowledge base article", "troubleshooting", "\u0438\u043d\u0441\u0442\u0440\u0443\u043a\u0446\u0438\u044f \u043f\u043e\u0434\u0434\u0435\u0440\u0436\u043a\u0438"),
        medium=("faq", "customer support", "help desk", "issue resolution", "\u0447\u0430\u0441\u0442\u044b\u0435 \u0432\u043e\u043f\u0440\u043e\u0441\u044b", "\u0440\u0435\u0448\u0435\u043d\u0438\u0435 \u043f\u0440\u043e\u0431\u043b\u0435\u043c"),
    ),
    "admin_instructions": CategoryRule(
        strong=("administrative instruction", "office procedure", "\u0430\u0434\u043c\u0438\u043d\u0438\u0441\u0442\u0440\u0430\u0442\u0438\u0432\u043d\u0430\u044f \u0438\u043d\u0441\u0442\u0440\u0443\u043a\u0446\u0438\u044f"),
        medium=("submit request", "approval form", "records management", "\u0437\u0430\u044f\u0432\u043a\u0430", "\u0441\u043e\u0433\u043b\u0430\u0441\u043e\u0432\u0430\u043d\u0438\u0435", "\u0434\u0435\u043b\u043e\u043f\u0440\u043e\u0438\u0437\u0432\u043e\u0434\u0441\u0442\u0432\u043e"),
    ),
    "legal_templates": CategoryRule(
        strong=("contract template", "legal notice", "agreement template", "\u0448\u0430\u0431\u043b\u043e\u043d \u0434\u043e\u0433\u043e\u0432\u043e\u0440\u0430", "\u044e\u0440\u0438\u0434\u0438\u0447\u0435\u0441\u043a\u043e\u0435 \u0443\u0432\u0435\u0434\u043e\u043c\u043b\u0435\u043d\u0438\u0435"),
        medium=("party shall", "terms and conditions", "liability", "\u0441\u0442\u043e\u0440\u043e\u043d\u044b", "\u043e\u0442\u0432\u0435\u0442\u0441\u0442\u0432\u0435\u043d\u043d\u043e\u0441\u0442\u044c", "\u0443\u0441\u043b\u043e\u0432\u0438\u044f"),
    ),
    "technical_documentation": CategoryRule(
        strong=("api documentation", "technical documentation", "configuration guide", "\u0442\u0435\u0445\u043d\u0438\u0447\u0435\u0441\u043a\u0430\u044f \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u0430\u0446\u0438\u044f"),
        medium=("parameter", "endpoint", "configuration", "installation", "\u043f\u0430\u0440\u0430\u043c\u0435\u0442\u0440", "\u043a\u043e\u043d\u0444\u0438\u0433\u0443\u0440\u0430\u0446\u0438\u044f", "\u0443\u0441\u0442\u0430\u043d\u043e\u0432\u043a\u0430"),
    ),
    "security_compliance_redaction_wrappers": CategoryRule(
        strong=("security policy", "compliance report", "redacted", "<redacted", "\u043e\u0442\u0447\u0435\u0442 \u043e \u0431\u0435\u0437\u043e\u043f\u0430\u0441\u043d\u043e\u0441\u0442\u0438"),
        medium=("audit", "confidential", "masked", "incident", "undisclosed", "\u0430\u0443\u0434\u0438\u0442", "\u043a\u043e\u043d\u0444\u0438\u0434\u0435\u043d\u0446\u0438\u0430\u043b\u044c\u043d", "\u0438\u043d\u0446\u0438\u0434\u0435\u043d\u0442"),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare fresh V17 source pools without copying prior training rows.")
    parser.add_argument("--tokenizer-id", default="microsoft/mdeberta-v3-base")
    parser.add_argument("--output-dir", default="fresh_sources")
    parser.add_argument("--input-jsonl", action="append", default=[], help="Optional fresh raw JSONL inputs. May be repeated.")
    parser.add_argument("--target-per-production-category", type=int, default=1200)
    parser.add_argument("--fp-candidate-target", type=int, default=6000)
    parser.add_argument("--exclude-prepared-dataset-dir", action="append", default=[])
    parser.add_argument("--exclude-jsonl-glob", action="append", default=[])
    parser.add_argument("--max-scan-per-source", type=int, default=250_000)
    parser.add_argument("--min-document-chars", type=int, default=350)
    parser.add_argument("--max-document-tokens", type=int, default=8192)
    parser.add_argument("--hf-sources", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-source-errors", action="store_true")
    parser.add_argument("--allow-risky-inputs", action="store_true")
    parser.add_argument("--allow-underfilled", action="store_true")
    parser.add_argument("--seed", type=int, default=47)
    return parser.parse_args()


def empty_exclusion_index() -> dict[str, set[str]]:
    return {key: set() for key in EXCLUSION_SET_KEYS}


def extract_source_text(row: dict[str, Any]) -> str:
    text = extract_text(row)
    if text:
        return text
    parts: list[str] = []
    for key in ("article", "highlights", "abstract", "document", "case_text", "review"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    return "\n\n".join(parts)


def forbidden_path_reason(path: str | Path) -> str | None:
    value = str(path).replace("\\", "/").lower()
    for marker in FORBIDDEN_INPUT_MARKERS:
        if marker in value:
            return marker
    return None


def require_allowed_input(path: str | Path, *, allow_risky: bool) -> None:
    reason = forbidden_path_reason(path)
    if reason and not allow_risky:
        raise ValueError(f"Refusing input path {path!s}; it matches prior-artifact marker {reason!r}.")


def iter_loaded_dataset_rows(dataset_obj: Any) -> Iterator[dict[str, Any]]:
    if isinstance(dataset_obj, DatasetDict):
        for split_name in dataset_obj:
            for row in dataset_obj[split_name]:
                yield dict(row)
    elif isinstance(dataset_obj, Dataset):
        for row in dataset_obj:
            yield dict(row)
    else:
        for row in dataset_obj:
            if isinstance(row, dict):
                yield row


def add_exclusion_row(exclusion: dict[str, set[str]], row: dict[str, Any]) -> bool:
    added = False
    text = extract_source_text(row)
    if text:
        exclusion["text_hashes"].add(text_hash(text))
        added = True
    metadata_map = {
        "source_document_ids": ("source_document_id", "source_doc_id", "document_id"),
        "carrier_document_ids": ("carrier_document_id", "carrier_doc_id"),
        "attack_text_hashes": ("attack_text_hash",),
        "attack_template_ids": ("attack_template_id", "template_id"),
        "generated_instance_ids": ("generated_instance_id", "attack_instance_id"),
    }
    for target_key, row_keys in metadata_map.items():
        for row_key in row_keys:
            value = row.get(row_key)
            if value:
                exclusion[target_key].add(str(value))
                added = True
    return added


def build_exclusion_index(args: argparse.Namespace) -> tuple[dict[str, set[str]], dict[str, Any]]:
    exclusion = empty_exclusion_index()
    jsonl_reports = []
    for pattern in args.exclude_jsonl_glob:
        for path in sorted(Path(".").glob(pattern)):
            if not path.is_file():
                continue
            count = 0
            for row in iter_jsonl(path):
                if add_exclusion_row(exclusion, row):
                    count += 1
            jsonl_reports.append({"path": str(path), "exists": True, "rows_indexed": count})

    prepared_reports = []
    for value in args.exclude_prepared_dataset_dir:
        path = Path(value)
        if not path.exists():
            report = {"path": str(path), "exists": False, "rows_indexed": 0}
            prepared_reports.append(report)
            if not args.allow_source_errors:
                raise FileNotFoundError(path)
            continue
        dataset_obj = load_from_disk(str(path))
        count = 0
        for row in iter_loaded_dataset_rows(dataset_obj):
            if add_exclusion_row(exclusion, row):
                count += 1
        prepared_reports.append({"path": str(path), "exists": True, "rows_indexed": count})

    return exclusion, {
        "excluded_previous_artifacts": {key: len(values) for key, values in exclusion.items()},
        "jsonl_files": jsonl_reports,
        "prepared_dataset_dirs": prepared_reports,
        "note": "Prior artifacts are used only for exclusion. They are never used as fresh row sources.",
    }


def iter_hf_rows(spec: SourceSpec) -> Iterable[dict[str, Any]]:
    kwargs = {"split": spec.split, "streaming": spec.streaming}
    if spec.config:
        return load_dataset(spec.repo_id, spec.config, **kwargs)
    return load_dataset(spec.repo_id, **kwargs)


def normalized_match_text(text: str) -> str:
    return normalize_text(text).lower()


def count_hits(text: str, phrases: tuple[str, ...]) -> int:
    return sum(1 for phrase in phrases if phrase and phrase.lower() in text)


def category_scores(text: str) -> dict[str, dict[str, int | bool]]:
    normalized = normalized_match_text(text)
    scores: dict[str, dict[str, int | bool]] = {}
    for category, rule in CATEGORY_RULES.items():
        strong = count_hits(normalized, rule.strong)
        medium = count_hits(normalized, rule.medium)
        accepted = strong >= 1 or medium >= 2
        if strong or medium:
            scores[category] = {
                "strong": strong,
                "medium": medium,
                "score": strong * 3 + medium,
                "accepted": accepted,
            }
    return scores


def best_category(text: str) -> tuple[str | None, dict[str, int | bool] | None]:
    scores = category_scores(text)
    accepted = {category: data for category, data in scores.items() if data["accepted"]}
    if not accepted:
        return None, None
    priority = {category: idx for idx, category in enumerate(FP_CANDIDATE_CATEGORIES)}
    category = max(
        accepted,
        key=lambda item: (
            int(accepted[item]["score"]),
            int(accepted[item]["strong"]),
            -priority.get(item, 999),
        ),
    )
    return category, accepted[category]


def fp_category_targets(total: int) -> dict[str, int]:
    weights = {
        "job_descriptions": 1000,
        "hr_policies": 1000,
        "corporate_procedures": 1000,
        "support_documentation": 800,
        "admin_instructions": 600,
        "legal_templates": 600,
        "technical_documentation": 600,
        "security_compliance_redaction_wrappers": 400,
    }
    base = sum(weights.values())
    targets: dict[str, int] = {}
    assigned = 0
    for category, weight in weights.items():
        value = max(1, round(total * weight / base))
        targets[category] = value
        assigned += value
    delta = total - assigned
    ordered = sorted(weights, key=weights.get, reverse=True)
    step = 1 if delta > 0 else -1
    for idx in range(abs(delta)):
        category = ordered[idx % len(ordered)]
        targets[category] = max(1, targets[category] + step)
    return targets


def source_doc_id(source_name: str, index: int, text: str) -> str:
    return f"v17fresh_{source_name}_{index}_{text_hash(text)}"


def output_row(
    *,
    row: dict[str, Any],
    text: str,
    category: str,
    language: str,
    source_name: str,
    source_path: str,
    source_origin: str,
    index: int,
    tokenizer: Any,
) -> dict[str, Any]:
    document_id = str(row.get("document_id") or row.get("id") or source_doc_id(source_name, index, text))
    windows = count_windows(text, tokenizer)
    return {
        "document_id": document_id,
        "document_label": "not_prompt_injection",
        "source_name": source_name,
        "source_path": source_path,
        "source_origin": source_origin,
        "category": category,
        "language": row.get("language") or language or infer_language(text),
        "text": normalize_text(text),
        "text_hash": text_hash(text),
        "window_count": windows,
        "window_count_bucket": window_count_bucket(windows),
    }


def process_candidate(
    *,
    row: dict[str, Any],
    source_name: str,
    source_path: str,
    source_origin: str,
    index: int,
    language_hint: str,
    tokenizer: Any,
    args: argparse.Namespace,
    exclusion: dict[str, set[str]],
    seen_hashes: set[str],
    category_docs: dict[str, list[dict[str, Any]]],
    target_by_category: dict[str, int],
    stats: dict[str, Counter],
    row_sources: Counter,
) -> None:
    text = extract_source_text(row)
    if not text:
        return
    category, score = best_category(text)
    if not category:
        return
    stats[category]["raw_candidates_seen"] += 1
    stats[category]["accepted_by_category_score"] += 1
    if len(normalize_text(text)) < args.min_document_chars:
        stats[category]["rejected_min_chars"] += 1
        return
    stats[category]["accepted_after_min_chars"] += 1

    token_count = len(tokenizer.encode(text, add_special_tokens=False))
    if token_count <= 0 or token_count > args.max_document_tokens:
        stats[category]["rejected_token_limit"] += 1
        return
    stats[category]["accepted_after_token_limit"] += 1

    h = text_hash(text)
    document_id = str(row.get("document_id") or row.get("id") or source_doc_id(source_name, index, text))
    if h in exclusion["text_hashes"] or document_id in exclusion["source_document_ids"] or document_id in exclusion["carrier_document_ids"]:
        stats[category]["rejected_by_exclusion"] += 1
        return
    if h in seen_hashes:
        stats[category]["duplicates_removed"] += 1
        return
    seen_hashes.add(h)

    if len(category_docs[category]) >= target_by_category.get(category, 0):
        stats[category]["accepted_after_dedup_over_target"] += 1
        return

    out = output_row(
        row={**row, "document_id": document_id},
        text=text,
        category=category,
        language=language_hint,
        source_name=source_name,
        source_path=source_path,
        source_origin=source_origin,
        index=index,
        tokenizer=tokenizer,
    )
    category_docs[category].append(out)
    stats[category]["written_rows"] += 1
    row_sources[source_origin] += 1


def all_targets_met(category_docs: dict[str, list[dict[str, Any]]], target_by_category: dict[str, int]) -> bool:
    return all(len(category_docs[category]) >= target for category, target in target_by_category.items())


def iter_local_rows(paths: list[str], *, allow_risky: bool) -> Iterator[tuple[dict[str, Any], str, str, str, str]]:
    for path_value in paths:
        require_allowed_input(path_value, allow_risky=allow_risky)
        path = Path(path_value)
        if not path.exists():
            raise FileNotFoundError(path)
        for index, row in enumerate(iter_jsonl(path)):
            source_name = str(row.get("source_name") or row.get("source") or path.stem)
            yield row, source_name, str(path), "raw_jsonl", str(row.get("language") or "")


def collect_sources(
    args: argparse.Namespace,
    tokenizer: Any,
    exclusion: dict[str, set[str]],
    target_by_category: dict[str, int],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    stats: dict[str, Counter] = {category: Counter() for category in target_by_category}
    category_docs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_hashes = set(exclusion["text_hashes"])
    row_sources: Counter = Counter()
    source_errors = []
    source_scan_counts: Counter = Counter()

    for index, (row, source_name, source_path, source_origin, language_hint) in enumerate(iter_local_rows(args.input_jsonl, allow_risky=args.allow_risky_inputs)):
        process_candidate(
            row=row,
            source_name=source_name,
            source_path=source_path,
            source_origin=source_origin,
            index=index,
            language_hint=language_hint,
            tokenizer=tokenizer,
            args=args,
            exclusion=exclusion,
            seen_hashes=seen_hashes,
            category_docs=category_docs,
            target_by_category=target_by_category,
            stats=stats,
            row_sources=row_sources,
        )
        source_scan_counts[source_name] += 1

    if args.hf_sources and not all_targets_met(category_docs, target_by_category):
        for spec in HF_SOURCES:
            if all_targets_met(category_docs, target_by_category):
                break
            print(f"Collecting {spec.name}...")
            try:
                for index, row in enumerate(iter_hf_rows(spec)):
                    if index >= args.max_scan_per_source:
                        break
                    if index and index % 25_000 == 0:
                        counts = {category: len(category_docs[category]) for category in target_by_category}
                        print(f"  {spec.name}: scanned {index:,}; accepted {json.dumps(counts, ensure_ascii=False)}")
                    process_candidate(
                        row=dict(row),
                        source_name=spec.name,
                        source_path=f"hf://{spec.repo_id}/{spec.config or 'default'}/{spec.split}#{index}",
                        source_origin="hf_streaming_source",
                        index=index,
                        language_hint=spec.language,
                        tokenizer=tokenizer,
                        args=args,
                        exclusion=exclusion,
                        seen_hashes=seen_hashes,
                        category_docs=category_docs,
                        target_by_category=target_by_category,
                        stats=stats,
                        row_sources=row_sources,
                    )
                    source_scan_counts[spec.name] += 1
                    if all_targets_met(category_docs, target_by_category):
                        break
            except Exception as exc:
                message = f"{spec.name}: {type(exc).__name__}: {exc}"
                source_errors.append(message)
                if not args.allow_source_errors:
                    raise
                print(f"  skipped {message}")

    report = {
        "category_stats": {category: dict(stats[category]) for category in target_by_category},
        "row_sources": dict(row_sources),
        "source_scan_counts": dict(source_scan_counts),
        "source_errors": source_errors,
    }
    return category_docs, report


def build_fp_candidates(
    category_docs: dict[str, list[dict[str, Any]]],
    fp_targets: dict[str, int],
    total_target: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rnd = random.Random(seed + 100)
    rows: list[dict[str, Any]] = []
    report = {}
    used_hashes: set[str] = set()
    for category, target in fp_targets.items():
        candidates = list(category_docs.get(category, []))
        rnd.shuffle(candidates)
        selected = []
        for row in candidates:
            h = row["text_hash"]
            if h in used_hashes:
                continue
            selected.append(row)
            used_hashes.add(h)
            if len(selected) >= target:
                break
        rows.extend(selected)
        report[category] = {"target": target, "written_rows": len(selected)}
    top_up_rows = 0
    if len(rows) < total_target:
        overflow = []
        for category in fp_targets:
            for row in category_docs.get(category, []):
                if row["text_hash"] not in used_hashes:
                    overflow.append(row)
        rnd.shuffle(overflow)
        for row in overflow:
            if len(rows) >= total_target:
                break
            h = row["text_hash"]
            if h in used_hashes:
                continue
            rows.append(row)
            used_hashes.add(h)
            top_up_rows += 1
            category = str(row.get("category") or "unknown")
            if category not in report:
                report[category] = {"target": 0, "written_rows": 0}
            report[category]["written_rows"] += 1
    report["_top_up"] = {
        "target_total": total_target,
        "rows_before_top_up": len(rows) - top_up_rows,
        "top_up_rows": top_up_rows,
        "final_rows": len(rows),
        "note": "Top-up uses unused fresh category rows only; previous prepared datasets remain exclusion-only.",
    }
    return rows, report


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)
    exclusion, exclusion_report = build_exclusion_index(args)
    fp_targets = fp_category_targets(args.fp_candidate_target)
    target_by_category = dict(fp_targets)
    for category in WEAK_PRODUCTION_CATEGORIES:
        target_by_category[category] = max(target_by_category.get(category, 0), args.target_per_production_category)

    category_docs, source_report = collect_sources(args, tokenizer, exclusion, target_by_category)
    rnd = random.Random(args.seed)
    for rows in category_docs.values():
        rnd.shuffle(rows)

    production_outputs = {
        category: category_docs.get(category, [])[: args.target_per_production_category]
        for category in WEAK_PRODUCTION_CATEGORIES
    }
    fp_candidates, fp_report = build_fp_candidates(category_docs, fp_targets, args.fp_candidate_target, args.seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    for category, rows in production_outputs.items():
        write_jsonl(output_dir / f"{category}.jsonl", rows)
    write_jsonl(output_dir / "fresh_fp_candidate_corpus.jsonl", fp_candidates)

    report = {
        "outputs": {
            **{f"{category}.jsonl": len(rows) for category, rows in production_outputs.items()},
            "fresh_fp_candidate_corpus.jsonl": len(fp_candidates),
        },
        "targets": {
            "target_per_production_category": args.target_per_production_category,
            "fp_candidate_target": args.fp_candidate_target,
            "fp_category_targets": fp_targets,
        },
        "fresh_source_pool": source_report,
        "fp_candidate_report": fp_report,
        "exclusion_report": exclusion_report,
        "acceptance": {
            "production_category_written_rows": {
                category: {
                    "written_rows": len(rows),
                    "minimum": args.target_per_production_category,
                    "status": "pass" if len(rows) >= args.target_per_production_category else "underfilled",
                }
                for category, rows in production_outputs.items()
            },
            "fp_candidate_corpus": {
                "written_rows": len(fp_candidates),
                "minimum": args.fp_candidate_target,
                "status": "pass" if len(fp_candidates) >= args.fp_candidate_target else "underfilled",
            },
            "previous_prepared_dataset_as_row_source": 0,
        },
    }
    write_json(output_dir / "v17_fresh_source_pools_report.json", report)
    print(json.dumps(report["acceptance"], ensure_ascii=False, indent=2))

    failures = []
    for category, info in report["acceptance"]["production_category_written_rows"].items():
        if info["status"] != "pass":
            failures.append(f"{category} {info['written_rows']}/{info['minimum']}")
    if report["acceptance"]["fp_candidate_corpus"]["status"] != "pass":
        info = report["acceptance"]["fp_candidate_corpus"]
        failures.append(f"fresh_fp_candidate_corpus {info['written_rows']}/{info['minimum']}")
    missing_exclusions = [
        item["path"]
        for item in exclusion_report["prepared_dataset_dirs"]
        if not item["exists"] or int(item["rows_indexed"]) <= 0
    ]
    if missing_exclusions:
        failures.append(f"missing_or_empty_exclusion_dirs={missing_exclusions}")
    if failures and not args.allow_underfilled:
        raise ValueError(
            "V17 fresh source pools underfilled or exclusion preflight failed: "
            + "; ".join(failures)
            + f". Inspect {output_dir / 'v17_fresh_source_pools_report.json'}."
        )


if __name__ == "__main__":
    main()
