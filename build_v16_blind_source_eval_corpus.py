# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator


LABEL_BENIGN = "not_prompt_injection"
LABEL_ATTACK = "prompt_injection"

TEXT_FIELDS = (
    "attack_text",
    "text",
    "window_text",
    "document_text",
    "content",
    "body",
    "article",
    "prompt",
    "instruction",
    "question",
    "answer",
    "summary",
    "description",
)

DEFAULT_CATEGORY_WEIGHTS = {
    "job_descriptions": 45_000,
    "corporate_procedures": 45_000,
    "hr_policies": 30_000,
    "admin_instructions": 30_000,
    "legal_templates": 30_000,
    "support_documentation": 30_000,
    "technical_documentation": 30_000,
    "safety_policies": 30_000,
    "meeting_minutes": 15_000,
    "knowledge_base": 15_000,
}

GENERATED_METADATA_MARKERS = (
    "generated",
    "synthetic",
    "template",
    "attack_bank",
    "attackbank",
    "manual_attack",
    "manual_prompt",
    "v17_blind_acceptance",
    "blind_acceptance_benign_generated",
    "blind_acceptance_corporate_benign_generated",
    "corporate_benign_generated",
)

GENERATED_TEXT_MARKERS = (
    "benign corporate acceptance sample",
    "corporate acceptance sample",
    "Control line:",
    "Контрольная строка:",
    "======== BEGIN OF DOCUMENT ========",
    "======== END OF DOCUMENT ========",
)

HASH_FIELDS = (
    "text_hash",
    "normalized_text_hash",
    "attack_text_hash",
    "source_text_hash",
    "original_text_hash",
    "original_normalized_text_hash",
)

ID_FIELDS = (
    "document_id",
    "source_document_id",
    "original_document_id",
    "source_row_id",
    "id",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a non-generated blind source-document evaluation corpus, using previous "
            "V16-and-earlier prepared datasets only as exclusion indexes."
        )
    )
    parser.add_argument("--input-jsonl", action="append", default=None)
    parser.add_argument("--output-dir", default="v16-blind-source-eval-150k")
    parser.add_argument("--output-jsonl", default=None)
    parser.add_argument("--report-json", default=None)
    parser.add_argument("--report-md", default=None)
    parser.add_argument("--category-csv", default=None)
    parser.add_argument("--source-csv", default=None)
    parser.add_argument("--source-category-csv", default=None)
    parser.add_argument("--target-documents", type=int, default=150_000)
    parser.add_argument("--attack-jsonl", action="append", default=[])
    parser.add_argument("--target-attack-documents", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument("--max-source-share", type=float, default=0.25)
    parser.add_argument("--min-document-chars", type=int, default=80)
    parser.add_argument("--max-document-chars", type=int, default=100_000)
    parser.add_argument("--exclude-prepared-dataset-dir", action="append", default=[])
    parser.add_argument(
        "--auto-exclude-v16-and-earlier",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically index local training-dataset-v* directories with version <= 16.",
    )
    parser.add_argument("--exclude-jsonl", action="append", default=[])
    parser.add_argument("--progress-interval", type=int, default=25_000)
    parser.add_argument("--allow-underfilled", action="store_true")
    args = parser.parse_args()
    if not args.input_jsonl:
        args.input_jsonl = ["v18-fp-candidate-corpus.jsonl"]
    return args


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").replace(chr(160), " ")).strip()


def normalized_for_hash(text: str) -> str:
    return normalize_text(text).lower()


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def hash_variants(text: str) -> set[str]:
    full = stable_hash(normalized_for_hash(text))
    return {full, full[:16], full[:20], full[:24]}


def row_hash_values(row: dict[str, Any], text: str = "") -> set[str]:
    values = {str(row.get(key)).strip() for key in HASH_FIELDS if row.get(key)}
    if text:
        values.update(hash_variants(text))
    return {value for value in values if value and value.lower() != "none"}


def extract_text(row: dict[str, Any]) -> str:
    for field in TEXT_FIELDS:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def label_name(row: dict[str, Any]) -> str:
    value = str(row.get("document_label", row.get("label", LABEL_BENIGN))).strip().lower()
    if value in {"0", "benign", "not_prompt_injection", "non_injection", "safe", ""}:
        return LABEL_BENIGN
    if value in {"1", "attack", "malicious", "prompt_injection", "injection"}:
        return LABEL_ATTACK
    return value


def infer_language(text: str) -> str:
    cyr = len(re.findall(r"[А-Яа-яЁё]", text))
    lat = len(re.findall(r"[A-Za-z]", text))
    if cyr and lat and min(cyr, lat) / max(cyr, lat) >= 0.12:
        return "mixed"
    if cyr > lat:
        return "ru"
    if lat:
        return "en"
    return "unknown"


def canonical_language(value: Any, text: str) -> str:
    raw = str(value or "").strip().lower().replace("-", "_")
    if raw in {"ru", "rus", "russian", "ru_ru", "rus_cyrl"}:
        return "ru"
    if raw in {"en", "eng", "english", "en_us", "en_gb"}:
        return "en"
    if raw in {"mixed", "multi", "multilingual", "ru_en", "en_ru"}:
        return "mixed"
    return infer_language(text)


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL") from exc
            if isinstance(row, dict):
                yield row


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            f.write("\n")


def metadata_blob(row: dict[str, Any]) -> str:
    parts = []
    for key in (
        "source_name",
        "source_origin",
        "source_path",
        "source_dataset",
        "source_config",
        "component",
        "generation_type",
    ):
        value = row.get(key)
        if value:
            parts.append(str(value))
    return " ".join(parts).lower()


def generated_reason(row: dict[str, Any], text: str) -> str | None:
    blob = metadata_blob(row)
    for marker in GENERATED_METADATA_MARKERS:
        if marker in blob:
            return f"generated_metadata_marker:{marker}"
    for marker in GENERATED_TEXT_MARKERS:
        if marker in text:
            return f"generated_text_marker:{marker}"
    return None


def attack_source_is_generated(row: dict[str, Any]) -> bool:
    blob = metadata_blob(row)
    return "generated" in blob or "trusted_attack_bank" in blob or "attack_bank" in blob


def stable_candidate_key(row: dict[str, Any], text: str, input_path: str, row_number: int) -> str:
    existing = str(row.get("document_id") or row.get("source_document_id") or row.get("source_row_id") or "").strip()
    if existing:
        return existing
    return f"{Path(input_path).name}:{row_number}:{stable_hash(normalized_for_hash(text))[:20]}"


def deterministic_random_key(seed: int, key: str) -> str:
    return stable_hash(f"{seed}:{key}")


def iter_arrow_rows(dataset_dir: Path) -> Iterator[dict[str, Any]]:
    try:
        import pyarrow.ipc as ipc
    except Exception as exc:
        raise RuntimeError("pyarrow is required to index saved DatasetDict Arrow files") from exc

    for arrow_path in sorted(dataset_dir.glob("**/*.arrow")):
        with arrow_path.open("rb") as f:
            try:
                reader = ipc.open_stream(f)
                for batch in reader:
                    for row in batch.to_pylist():
                        if isinstance(row, dict):
                            yield row
                continue
            except Exception:
                f.seek(0)
            reader = ipc.open_file(f)
            for index in range(reader.num_record_batches):
                batch = reader.get_batch(index)
                for row in batch.to_pylist():
                    if isinstance(row, dict):
                        yield row


def auto_exclude_dirs() -> list[Path]:
    dirs: list[Path] = []
    for path in sorted(Path(".").glob("training-dataset-v*")):
        if not path.is_dir():
            continue
        match = re.match(r"training-dataset-v(\d+)", path.name)
        if match and int(match.group(1)) <= 16:
            dirs.append(path)
    return dirs


def build_exclusion_index(args: argparse.Namespace) -> tuple[dict[str, set[str]], dict[str, Any]]:
    exclusion = {"hashes": set(), "ids": set()}
    reports: list[dict[str, Any]] = []

    dataset_dirs = [Path(value) for value in args.exclude_prepared_dataset_dir]
    if args.auto_exclude_v16_and_earlier:
        seen = {str(path.resolve()).lower() for path in dataset_dirs if path.exists()}
        for path in auto_exclude_dirs():
            key = str(path.resolve()).lower()
            if key not in seen:
                dataset_dirs.append(path)
                seen.add(key)

    for path in dataset_dirs:
        started = time.time()
        rows = 0
        hashes_before = len(exclusion["hashes"])
        ids_before = len(exclusion["ids"])
        if not path.exists():
            reports.append({"path": str(path), "exists": False, "rows_indexed": 0})
            continue
        for row in iter_arrow_rows(path):
            rows += 1
            text = extract_text(row)
            for value in row_hash_values(row, text):
                exclusion["hashes"].add(value)
            for key in ID_FIELDS:
                value = row.get(key)
                if value:
                    exclusion["ids"].add(str(value))
        reports.append(
            {
                "path": str(path),
                "exists": True,
                "rows_indexed": rows,
                "hashes_added": len(exclusion["hashes"]) - hashes_before,
                "ids_added": len(exclusion["ids"]) - ids_before,
                "seconds": round(time.time() - started, 2),
            }
        )
        print(
            f"[exclusion] {path}: rows={rows:,} hashes={len(exclusion['hashes']):,} ids={len(exclusion['ids']):,}",
            flush=True,
        )

    jsonl_reports: list[dict[str, Any]] = []
    for value in args.exclude_jsonl:
        for path in sorted(Path(".").glob(value)):
            if not path.is_file():
                continue
            rows = 0
            for row in iter_jsonl(path):
                rows += 1
                text = extract_text(row)
                for hash_value in row_hash_values(row, text):
                    exclusion["hashes"].add(hash_value)
                for key in ID_FIELDS:
                    if row.get(key):
                        exclusion["ids"].add(str(row[key]))
            jsonl_reports.append({"path": str(path), "rows_indexed": rows})
            print(f"[exclusion] {path}: rows={rows:,}", flush=True)

    return exclusion, {
        "prepared_dataset_dirs": reports,
        "jsonl_files": jsonl_reports,
        "hashes": len(exclusion["hashes"]),
        "ids": len(exclusion["ids"]),
        "note": "Prior datasets are indexed only for exclusion and are never used as evaluation row sources.",
    }


def eligible_candidate(
    row: dict[str, Any],
    *,
    text: str,
    exclusion: dict[str, set[str]],
    min_chars: int,
    max_chars: int,
) -> tuple[bool, str]:
    if label_name(row) != LABEL_BENIGN:
        return False, "non_benign_label"
    if not text:
        return False, "missing_text"
    if len(text) < min_chars:
        return False, "too_short"
    if len(text) > max_chars:
        return False, "too_long"
    generated = generated_reason(row, text)
    if generated:
        return False, generated
    for value in row_hash_values(row, text):
        if value in exclusion["hashes"]:
            return False, "excluded_hash_overlap_v16_or_earlier"
    for key in ID_FIELDS:
        value = row.get(key)
        if value and str(value) in exclusion["ids"]:
            return False, f"excluded_{key}_overlap_v16_or_earlier"
    return True, "eligible"


def eligible_attack_candidate(
    row: dict[str, Any],
    *,
    text: str,
    exclusion: dict[str, set[str]],
    min_chars: int,
    max_chars: int,
) -> tuple[bool, str]:
    if label_name(row) != LABEL_ATTACK:
        return False, "non_attack_label"
    if not text:
        return False, "missing_text"
    if len(text) < min_chars:
        return False, "too_short"
    if len(text) > max_chars:
        return False, "too_long"
    anchor = normalize_text(str(row.get("attack_anchor_text") or ""))
    if not anchor:
        return False, "missing_attack_anchor_text"
    if anchor.lower() not in text.lower():
        return False, "attack_anchor_not_visible"
    for value in row_hash_values(row, text):
        if value in exclusion["hashes"]:
            return False, "excluded_hash_overlap_v16_or_earlier"
    for key in ID_FIELDS:
        value = row.get(key)
        if value and str(value) in exclusion["ids"]:
            return False, f"excluded_{key}_overlap_v16_or_earlier"
    return True, "eligible"


def candidate_from_row(row: dict[str, Any], input_path: str, row_number: int, seed: int) -> dict[str, Any]:
    text = normalize_text(extract_text(row))
    key = stable_candidate_key(row, text, input_path, row_number)
    category = str(row.get("category") or "unknown").strip() or "unknown"
    source_name = str(row.get("source_name") or row.get("source") or Path(input_path).stem).strip()
    language = canonical_language(row.get("language"), text)
    return {
        "key": key,
        "random_key": deterministic_random_key(seed, key),
        "category": category,
        "source_name": source_name,
        "language": language,
        "text_hash": next(iter(hash_variants(text))),
        "document_id": str(row.get("document_id") or key),
    }


def scan_candidates(args: argparse.Namespace, exclusion: dict[str, set[str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    rejected = Counter()
    read_rows = Counter()
    eligible_counts = {
        "category": Counter(),
        "source_name": Counter(),
        "language": Counter(),
        "source_category": Counter(),
    }
    started = time.time()

    for input_path in args.input_jsonl:
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(path)
        for row_number, row in enumerate(iter_jsonl(path), 1):
            if row_number % args.progress_interval == 0:
                elapsed = max(0.1, time.time() - started)
                print(
                    f"[scan] rows={sum(read_rows.values()):,} eligible={len(candidates):,} "
                    f"rate={sum(read_rows.values()) / elapsed:,.1f}/s",
                    flush=True,
                )
            read_rows[str(path)] += 1
            text = normalize_text(extract_text(row))
            ok, reason = eligible_candidate(
                row,
                text=text,
                exclusion=exclusion,
                min_chars=args.min_document_chars,
                max_chars=args.max_document_chars,
            )
            if not ok:
                rejected[reason] += 1
                continue
            candidate = candidate_from_row(row, str(path), row_number, args.seed)
            candidates.append(candidate)
            eligible_counts["category"][candidate["category"]] += 1
            eligible_counts["source_name"][candidate["source_name"]] += 1
            eligible_counts["language"][candidate["language"]] += 1
            eligible_counts["source_category"][(candidate["source_name"], candidate["category"])] += 1

    return candidates, {
        "read_rows": dict(read_rows),
        "eligible_rows": len(candidates),
        "rejected": dict(rejected.most_common()),
        "eligible_by_category": dict(eligible_counts["category"].most_common()),
        "eligible_by_source_name": dict(eligible_counts["source_name"].most_common()),
        "eligible_by_language": dict(eligible_counts["language"].most_common()),
        "seconds": round(time.time() - started, 2),
    }


def attack_candidate_from_row(row: dict[str, Any], input_path: str, row_number: int, seed: int) -> dict[str, Any]:
    text = normalize_text(extract_text(row))
    key = str(row.get("attack_template_id") or row.get("document_id") or row.get("id") or "").strip()
    if not key:
        key = f"{Path(input_path).name}:attack:{row_number}:{stable_hash(normalized_for_hash(text))[:20]}"
    source_name = str(row.get("source_name") or row.get("source") or Path(input_path).stem).strip()
    language = canonical_language(row.get("language"), text)
    family = str(row.get("semantic_family") or row.get("attack_family") or "unknown").strip() or "unknown"
    return {
        "key": key,
        "random_key": deterministic_random_key(seed, key),
        "semantic_family": family,
        "source_name": source_name,
        "language": language,
        "text_hash": next(iter(hash_variants(text))),
        "document_id": str(row.get("document_id") or key),
    }


def scan_attack_candidates(args: argparse.Namespace, exclusion: dict[str, set[str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    rejected = Counter()
    read_rows = Counter()
    eligible_counts = {
        "semantic_family": Counter(),
        "source_name": Counter(),
        "language": Counter(),
    }
    started = time.time()

    for input_path in args.attack_jsonl:
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(path)
        for row_number, row in enumerate(iter_jsonl(path), 1):
            if row_number % args.progress_interval == 0:
                elapsed = max(0.1, time.time() - started)
                print(
                    f"[scan-attack] rows={sum(read_rows.values()):,} eligible={len(candidates):,} "
                    f"rate={sum(read_rows.values()) / elapsed:,.1f}/s",
                    flush=True,
                )
            read_rows[str(path)] += 1
            text = normalize_text(extract_text(row))
            ok, reason = eligible_attack_candidate(
                row,
                text=text,
                exclusion=exclusion,
                min_chars=args.min_document_chars,
                max_chars=args.max_document_chars,
            )
            if not ok:
                rejected[reason] += 1
                continue
            candidate = attack_candidate_from_row(row, str(path), row_number, args.seed)
            candidates.append(candidate)
            eligible_counts["semantic_family"][candidate["semantic_family"]] += 1
            eligible_counts["source_name"][candidate["source_name"]] += 1
            eligible_counts["language"][candidate["language"]] += 1

    return candidates, {
        "read_rows": dict(read_rows),
        "eligible_rows": len(candidates),
        "rejected": dict(rejected.most_common()),
        "eligible_by_semantic_family": dict(eligible_counts["semantic_family"].most_common()),
        "eligible_by_source_name": dict(eligible_counts["source_name"].most_common()),
        "eligible_by_language": dict(eligible_counts["language"].most_common()),
        "seconds": round(time.time() - started, 2),
    }


def scaled_category_targets(target: int, availability: Counter[str]) -> dict[str, int]:
    weight_total = sum(DEFAULT_CATEGORY_WEIGHTS.values())
    desired = {
        category: int(round(target * weight / weight_total))
        for category, weight in DEFAULT_CATEGORY_WEIGHTS.items()
    }
    drift = target - sum(desired.values())
    if drift:
        first = max(DEFAULT_CATEGORY_WEIGHTS, key=DEFAULT_CATEGORY_WEIGHTS.get)
        desired[first] += drift

    targets = {category: min(desired.get(category, 0), availability.get(category, 0)) for category in desired}
    for category, count in availability.items():
        targets.setdefault(category, 0)

    while sum(targets.values()) < target:
        remaining = target - sum(targets.values())
        expandable = [
            category
            for category, count in availability.items()
            if targets.get(category, 0) < count
        ]
        if not expandable:
            break
        total_capacity = sum(availability[category] - targets.get(category, 0) for category in expandable)
        changed = 0
        for category in sorted(expandable, key=lambda item: availability[item] - targets.get(item, 0), reverse=True):
            capacity = availability[category] - targets.get(category, 0)
            share = max(1, int(round(remaining * capacity / max(1, total_capacity))))
            add = min(capacity, share, remaining - changed)
            if add <= 0:
                continue
            targets[category] = targets.get(category, 0) + add
            changed += add
            if changed >= remaining:
                break
        if changed <= 0:
            break
    return {category: count for category, count in targets.items() if count > 0}


def select_candidates(args: argparse.Namespace, candidates: list[dict[str, Any]]) -> tuple[set[str], dict[str, Any]]:
    availability = Counter(row["category"] for row in candidates)
    category_targets = scaled_category_targets(args.target_documents, availability)
    source_cap = max(1, int(round(args.target_documents * args.max_source_share)))

    selected: list[dict[str, Any]] = []
    selected_keys: set[str] = set()
    category_counts = Counter()
    source_counts = Counter()
    language_counts = Counter()
    source_category_counts = Counter()
    skipped = Counter()

    for row in sorted(candidates, key=lambda item: item["random_key"]):
        if len(selected) >= args.target_documents:
            break
        if row["key"] in selected_keys:
            skipped["duplicate_selection_key"] += 1
            continue
        category = row["category"]
        source_name = row["source_name"]
        if category_counts[category] >= category_targets.get(category, 0):
            skipped["category_target_full"] += 1
            continue
        if source_counts[source_name] >= source_cap:
            skipped["source_cap_full"] += 1
            continue
        selected.append(row)
        selected_keys.add(row["key"])
        category_counts[category] += 1
        source_counts[source_name] += 1
        language_counts[row["language"]] += 1
        source_category_counts[(source_name, category)] += 1

    if len(selected) < args.target_documents:
        for row in sorted(candidates, key=lambda item: item["random_key"]):
            if len(selected) >= args.target_documents:
                break
            if row["key"] in selected_keys:
                continue
            source_name = row["source_name"]
            if source_counts[source_name] >= source_cap:
                continue
            selected.append(row)
            selected_keys.add(row["key"])
            category_counts[row["category"]] += 1
            source_counts[source_name] += 1
            language_counts[row["language"]] += 1
            source_category_counts[(source_name, row["category"])] += 1

    report = {
        "target_documents": args.target_documents,
        "selected_documents": len(selected),
        "source_cap": source_cap,
        "max_source_share": args.max_source_share,
        "category_targets": category_targets,
        "selected_by_category": dict(category_counts.most_common()),
        "selected_by_source_name": dict(source_counts.most_common()),
        "selected_by_language": dict(language_counts.most_common()),
        "selected_by_source_category": {
            f"{source}||{category}": count
            for (source, category), count in source_category_counts.most_common()
        },
        "skipped_after_eligible": dict(skipped.most_common()),
    }
    return selected_keys, report


def select_attack_candidates(args: argparse.Namespace, candidates: list[dict[str, Any]]) -> tuple[set[str], dict[str, Any]]:
    target = args.target_attack_documents
    if target <= 0:
        return set(), {
            "target_attack_documents": 0,
            "selected_attack_documents": 0,
            "selected_by_semantic_family": {},
            "selected_by_source_name": {},
            "selected_by_language": {},
        }
    family_availability = Counter(row["semantic_family"] for row in candidates)
    families = sorted(family_availability)
    if not families:
        return set(), {
            "target_attack_documents": target,
            "selected_attack_documents": 0,
            "selected_by_semantic_family": {},
            "selected_by_source_name": {},
            "selected_by_language": {},
        }
    per_family = {family: target // len(families) for family in families}
    for family in families[: target % len(families)]:
        per_family[family] += 1
    for family in families:
        per_family[family] = min(per_family[family], family_availability[family])

    selected: list[dict[str, Any]] = []
    selected_keys: set[str] = set()
    family_counts = Counter()
    source_counts = Counter()
    language_counts = Counter()
    skipped = Counter()

    for row in sorted(candidates, key=lambda item: item["random_key"]):
        if len(selected) >= target:
            break
        if row["key"] in selected_keys:
            skipped["duplicate_selection_key"] += 1
            continue
        family = row["semantic_family"]
        if family_counts[family] >= per_family.get(family, 0):
            skipped["family_target_full"] += 1
            continue
        selected.append(row)
        selected_keys.add(row["key"])
        family_counts[family] += 1
        source_counts[row["source_name"]] += 1
        language_counts[row["language"]] += 1

    if len(selected) < target:
        for row in sorted(candidates, key=lambda item: item["random_key"]):
            if len(selected) >= target:
                break
            if row["key"] in selected_keys:
                continue
            selected.append(row)
            selected_keys.add(row["key"])
            family_counts[row["semantic_family"]] += 1
            source_counts[row["source_name"]] += 1
            language_counts[row["language"]] += 1

    return selected_keys, {
        "target_attack_documents": target,
        "selected_attack_documents": len(selected),
        "family_targets": per_family,
        "selected_by_semantic_family": dict(family_counts.most_common()),
        "selected_by_source_name": dict(source_counts.most_common()),
        "selected_by_language": dict(language_counts.most_common()),
        "skipped_after_eligible": dict(skipped.most_common()),
    }


def output_row(row: dict[str, Any], input_path: str, row_number: int, seed: int) -> dict[str, Any]:
    text = normalize_text(extract_text(row))
    key = stable_candidate_key(row, text, input_path, row_number)
    source_name = str(row.get("source_name") or row.get("source") or Path(input_path).stem).strip()
    source_dataset = str(row.get("source_dataset") or row.get("source_origin") or row.get("source_path") or "")
    category = str(row.get("category") or "unknown").strip() or "unknown"
    language = canonical_language(row.get("language"), text)
    return {
        "document_id": str(row.get("document_id") or key),
        "document_label": LABEL_BENIGN,
        "label": LABEL_BENIGN,
        "category": category,
        "language": language,
        "source_name": source_name,
        "source_dataset": source_dataset,
        "source_config": str(row.get("source_config") or ""),
        "source_split": str(row.get("source_split") or ""),
        "source_pool": "blind_external_source_only",
        "source_origin": str(row.get("source_origin") or source_dataset or source_name),
        "source_row_id": str(row.get("source_row_id") or row.get("id") or ""),
        "text_hash": stable_hash(normalized_for_hash(text))[:20],
        "normalized_text_hash": stable_hash(normalized_for_hash(text))[:24],
        "text_length": len(text),
        "text_token_length": int(row.get("text_token_length") or 0),
        "production_window_count": int(row.get("production_window_count") or row.get("window_count") or 0),
        "length_bucket": str(row.get("length_bucket") or ""),
        "keyword_hits": row.get("keyword_hits") if isinstance(row.get("keyword_hits"), list) else [],
        "selection_key": key,
        "selection_random_key": deterministic_random_key(seed, key),
        "generated_document": False,
        "excluded_v16_or_earlier_overlap": False,
        "text": text,
    }


def attack_output_row(row: dict[str, Any], input_path: str, row_number: int, seed: int) -> dict[str, Any]:
    text = normalize_text(extract_text(row))
    key = str(row.get("attack_template_id") or row.get("document_id") or row.get("id") or "").strip()
    if not key:
        key = f"{Path(input_path).name}:attack:{row_number}:{stable_hash(normalized_for_hash(text))[:20]}"
    source_name = str(row.get("source_name") or row.get("source") or Path(input_path).stem).strip()
    language = canonical_language(row.get("language"), text)
    family = str(row.get("semantic_family") or row.get("attack_family") or "unknown").strip() or "unknown"
    return {
        "document_id": str(row.get("document_id") or key),
        "document_label": LABEL_ATTACK,
        "label": LABEL_ATTACK,
        "category": str(row.get("category") or "prompt_injection_attack"),
        "language": language,
        "source_name": source_name,
        "source_dataset": str(row.get("source_dataset") or row.get("source_origin") or "trusted_attack_bank"),
        "source_config": str(row.get("source_config") or ""),
        "source_split": str(row.get("source_split") or ""),
        "source_pool": "blind_trusted_attack_bank",
        "source_origin": str(row.get("source_origin") or "trusted_controlled_template_generated_attack_bank"),
        "source_row_id": str(row.get("source_row_id") or row.get("id") or ""),
        "semantic_family": family,
        "semantic_subfamily": str(row.get("semantic_subfamily") or ""),
        "attack_anchor_text": normalize_text(str(row.get("attack_anchor_text") or "")),
        "attack_template_id": str(row.get("attack_template_id") or key),
        "trusted_attack": bool(row.get("trusted_attack", True)),
        "manual_reviewed_attack": bool(row.get("manual_reviewed_attack", False)),
        "text_hash": stable_hash(normalized_for_hash(text))[:20],
        "normalized_text_hash": stable_hash(normalized_for_hash(text))[:24],
        "attack_text_hash": str(row.get("attack_text_hash") or stable_hash(normalized_for_hash(text))[:20]),
        "text_length": len(text),
        "text_token_length": int(row.get("text_token_length") or 0),
        "production_window_count": int(row.get("production_window_count") or row.get("window_count") or 1),
        "length_bucket": str(row.get("length_bucket") or ""),
        "keyword_hits": row.get("keyword_hits") if isinstance(row.get("keyword_hits"), list) else [],
        "selection_key": key,
        "selection_random_key": deterministic_random_key(seed, key),
        "generated_document": attack_source_is_generated(row),
        "controlled_template_attack": attack_source_is_generated(row),
        "excluded_v16_or_earlier_overlap": False,
        "text": text,
    }


def write_selected_rows(args: argparse.Namespace, selected_keys: set[str], output_jsonl: Path) -> dict[str, Any]:
    written = 0
    remaining_keys = set(selected_keys)
    counts = {
        "category": Counter(),
        "source_name": Counter(),
        "language": Counter(),
        "source_category": Counter(),
    }
    started = time.time()
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as out:
        for input_path in args.input_jsonl:
            path = Path(input_path)
            for row_number, row in enumerate(iter_jsonl(path), 1):
                text = normalize_text(extract_text(row))
                key = stable_candidate_key(row, text, str(path), row_number)
                if key not in remaining_keys:
                    continue
                out_row = output_row(row, str(path), row_number, args.seed)
                out.write(json.dumps(out_row, ensure_ascii=False, sort_keys=True))
                out.write("\n")
                remaining_keys.remove(key)
                written += 1
                counts["category"][out_row["category"]] += 1
                counts["source_name"][out_row["source_name"]] += 1
                counts["language"][out_row["language"]] += 1
                counts["source_category"][(out_row["source_name"], out_row["category"])] += 1
                if written % args.progress_interval == 0:
                    elapsed = max(0.1, time.time() - started)
                    print(f"[write] rows={written:,} rate={written / elapsed:,.1f}/s", flush=True)

    return {
        "written_rows": written,
        "unwritten_selected_keys": len(remaining_keys),
        "by_category": dict(counts["category"].most_common()),
        "by_source_name": dict(counts["source_name"].most_common()),
        "by_language": dict(counts["language"].most_common()),
        "by_source_category": {
            f"{source}||{category}": count
            for (source, category), count in counts["source_category"].most_common()
        },
        "seconds": round(time.time() - started, 2),
    }


def write_selected_attack_rows(args: argparse.Namespace, selected_keys: set[str], output_jsonl: Path) -> dict[str, Any]:
    written = 0
    remaining_keys = set(selected_keys)
    counts = {
        "semantic_family": Counter(),
        "source_name": Counter(),
        "language": Counter(),
    }
    started = time.time()
    with output_jsonl.open("a", encoding="utf-8") as out:
        for input_path in args.attack_jsonl:
            path = Path(input_path)
            for row_number, row in enumerate(iter_jsonl(path), 1):
                text = normalize_text(extract_text(row))
                key = str(row.get("attack_template_id") or row.get("document_id") or row.get("id") or "").strip()
                if not key:
                    key = f"{Path(input_path).name}:attack:{row_number}:{stable_hash(normalized_for_hash(text))[:20]}"
                if key not in remaining_keys:
                    continue
                out_row = attack_output_row(row, str(path), row_number, args.seed)
                out.write(json.dumps(out_row, ensure_ascii=False, sort_keys=True))
                out.write("\n")
                remaining_keys.remove(key)
                written += 1
                counts["semantic_family"][out_row["semantic_family"]] += 1
                counts["source_name"][out_row["source_name"]] += 1
                counts["language"][out_row["language"]] += 1
                if written % args.progress_interval == 0:
                    elapsed = max(0.1, time.time() - started)
                    print(f"[write-attack] rows={written:,} rate={written / elapsed:,.1f}/s", flush=True)

    return {
        "written_rows": written,
        "unwritten_selected_keys": len(remaining_keys),
        "by_semantic_family": dict(counts["semantic_family"].most_common()),
        "by_source_name": dict(counts["source_name"].most_common()),
        "by_language": dict(counts["language"].most_common()),
        "seconds": round(time.time() - started, 2),
    }


def write_count_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_distribution_csvs(report: dict[str, Any], category_csv: Path, source_csv: Path, source_category_csv: Path) -> None:
    selected = report["write_report"]
    total = max(1, selected["written_rows"])
    category_rows = [
        {"category": key, "documents": value, "share": value / total}
        for key, value in selected["by_category"].items()
    ]
    source_rows = [
        {"source_name": key, "documents": value, "share": value / total}
        for key, value in selected["by_source_name"].items()
    ]
    source_category_rows = []
    for key, value in selected["by_source_category"].items():
        source, category = key.split("||", 1)
        source_category_rows.append(
            {"source_name": source, "category": category, "documents": value, "share": value / total}
        )
    write_count_csv(category_csv, category_rows, ["category", "documents", "share"])
    write_count_csv(source_csv, source_rows, ["source_name", "documents", "share"])
    write_count_csv(source_category_csv, source_category_rows, ["source_name", "category", "documents", "share"])


def report_markdown(report: dict[str, Any]) -> str:
    write_report = report["write_report"]
    lines = [
        "# V16 Blind Source Evaluation Corpus",
        "",
        "This corpus is built from non-generated external source documents. V16-and-earlier prepared datasets are used only as exclusion indexes.",
        "",
        "## Summary",
        "",
        f"- Target documents: {report['selection_report']['target_documents']:,}",
        f"- Written benign documents: {write_report['written_rows']:,}",
        f"- Written attack documents: {report.get('attack_write_report', {}).get('written_rows', 0):,}",
        f"- Exclusion hashes indexed: {report['exclusion_report']['hashes']:,}",
        f"- Exclusion ids indexed: {report['exclusion_report']['ids']:,}",
        f"- Max source share: {report['selection_report']['max_source_share']:.2%}",
        "",
        "## Categories",
        "",
    ]
    for category, count in write_report["by_category"].items():
        lines.append(f"- {category}: {count:,}")
    lines.extend(["", "## Sources", ""])
    for source_name, count in write_report["by_source_name"].items():
        lines.append(f"- {source_name}: {count:,}")
    lines.extend(["", "## Languages", ""])
    for language, count in write_report["by_language"].items():
        lines.append(f"- {language}: {count:,}")
    attack_write = report.get("attack_write_report") or {}
    if attack_write.get("written_rows"):
        lines.extend(["", "## Attack Semantic Families", ""])
        for family, count in attack_write.get("by_semantic_family", {}).items():
            lines.append(f"- {family}: {count:,}")
        lines.extend(["", "## Attack Provenance", ""])
        lines.append("Attack rows are selected from the trusted controlled template-generated attack bank.")
    if report["failures"]:
        lines.extend(["", "## Failures", ""])
        for failure in report["failures"]:
            lines.append(f"- {failure}")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_jsonl = Path(args.output_jsonl) if args.output_jsonl else output_dir / "v16-blind-source-eval-150k.jsonl"
    report_json = Path(args.report_json) if args.report_json else output_dir / "v16-blind-source-eval-150k-report.json"
    report_md = Path(args.report_md) if args.report_md else output_dir / "v16-blind-source-eval-150k-report.md"
    category_csv = Path(args.category_csv) if args.category_csv else output_dir / "category-distribution.csv"
    source_csv = Path(args.source_csv) if args.source_csv else output_dir / "source-distribution.csv"
    source_category_csv = Path(args.source_category_csv) if args.source_category_csv else output_dir / "source-category-distribution.csv"

    print("[start] building exclusion index", flush=True)
    exclusion, exclusion_report = build_exclusion_index(args)
    print("[start] scanning candidates", flush=True)
    candidates, scan_report = scan_candidates(args, exclusion)
    print(f"[select] eligible={len(candidates):,} target={args.target_documents:,}", flush=True)
    selected_keys, selection_report = select_candidates(args, candidates)
    if len(selected_keys) < args.target_documents and not args.allow_underfilled:
        raise ValueError(f"Only selected {len(selected_keys):,}/{args.target_documents:,} documents.")
    print("[start] writing selected corpus", flush=True)
    write_report = write_selected_rows(args, selected_keys, output_jsonl)

    attack_scan_report: dict[str, Any] = {}
    attack_selection_report: dict[str, Any] = {}
    attack_write_report: dict[str, Any] = {
        "written_rows": 0,
        "unwritten_selected_keys": 0,
        "by_semantic_family": {},
        "by_source_name": {},
        "by_language": {},
    }
    if args.target_attack_documents > 0:
        print("[start] scanning attack candidates", flush=True)
        attack_candidates, attack_scan_report = scan_attack_candidates(args, exclusion)
        print(f"[select-attack] eligible={len(attack_candidates):,} target={args.target_attack_documents:,}", flush=True)
        attack_selected_keys, attack_selection_report = select_attack_candidates(args, attack_candidates)
        if len(attack_selected_keys) < args.target_attack_documents and not args.allow_underfilled:
            raise ValueError(f"Only selected {len(attack_selected_keys):,}/{args.target_attack_documents:,} attack documents.")
        print("[start] appending selected attack corpus", flush=True)
        attack_write_report = write_selected_attack_rows(args, attack_selected_keys, output_jsonl)

    failures = []
    if write_report["written_rows"] != len(selected_keys):
        failures.append("written_row_count_mismatch")
    if write_report["written_rows"] < args.target_documents:
        failures.append("target_underfilled")
    if any("generated" in key.lower() for key in write_report["by_source_name"]):
        failures.append("generated_source_name_present")
    if args.target_attack_documents > 0 and attack_write_report["written_rows"] < args.target_attack_documents:
        failures.append("attack_target_underfilled")
    if args.target_attack_documents > 0 and attack_write_report["unwritten_selected_keys"]:
        failures.append("attack_written_row_count_mismatch")

    report = {
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "output_jsonl": str(output_jsonl),
        "report_json": str(report_json),
        "input_jsonl": args.input_jsonl,
        "non_generated_required": True,
        "attack_rows_note": (
            "Benign side is non-generated. Attack side, when requested, is selected from the trusted "
            "controlled template-generated attack bank because no local natural 150K attack document pool exists."
        ),
        "previous_v16_and_earlier_used_only_as_exclusion": True,
        "exclusion_report": exclusion_report,
        "scan_report": scan_report,
        "selection_report": selection_report,
        "write_report": write_report,
        "attack_scan_report": attack_scan_report,
        "attack_selection_report": attack_selection_report,
        "attack_write_report": attack_write_report,
    }
    write_json(report_json, report)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text(report_markdown(report), encoding="utf-8")
    write_distribution_csvs(report, category_csv, source_csv, source_category_csv)
    print(json.dumps({"status": report["status"], "failures": failures, "written_rows": write_report["written_rows"], "output_jsonl": str(output_jsonl)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
