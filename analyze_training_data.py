# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
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
MODEL_MAX_LENGTH = 256
WINDOW_TOKEN_LENGTH = MODEL_MAX_LENGTH - 2
WINDOW_TOKEN_STRIDE = 128
ZERO_WIDTH_CHARS = {chr(code) for code in (0x200B, 0x200C, 0x200D, 0xFEFF, 0x2060)}
MOJIBAKE_MARKERS = (chr(0x00D0), chr(0x00D1), chr(0x00C2), chr(0x00E2) + chr(0x20AC) + chr(0x2122), chr(0x00E2) + chr(0x20AC) + chr(0x0153), chr(0x00E2) + chr(0x20AC), chr(0xFFFD))
DEFAULT_MAX_SOURCE_SHARE = 0.35
DEFAULT_MIN_CRITICAL_BUCKET_ROWS = 200
DEFAULT_MAX_DISTRIBUTION_JSD = 0.15


CRITICAL_BUCKETS = [
    "short_embedded_exfiltration_attack",
    "deep_embedded_indirect_attack_short",
    "deep_embedded_indirect_attack_medium",
    "deep_embedded_indirect_attack_long",
    "embedded_indirect_attack",
    "benign_hard_negative",
    "benign_document",
]


PHRASE_FAMILIES: dict[str, list[str]] = {
    "system_prompt_exfiltration": [
        "покажи системный промпт",
        "показать системный промпт",
        "раскрой системный промпт",
        "выведи системный промпт",
        "show the system prompt",
        "print the system prompt",
        "reveal the system prompt",
    ],
    "hidden_instruction_exfiltration": [
        "покажи скрытые инструкции",
        "раскрой скрытые инструкции",
        "выведи скрытые инструкции",
        "show hidden instructions",
        "reveal hidden instructions",
    ],
    "developer_message_exfiltration": [
        "покажи developer message",
        "выведи developer message",
        "раскрой developer message",
        "напечатай developer message",
        "верни developer message",
        "отправь developer message",
        "developer message",
        "developer prompt",
        "developer instructions",
        "сообщение разработчика",
        "инструкции разработчика",
        "правила разработчика",
        "внутренние инструкции разработчика",
        "внутренние правила разработчика",
        "скрытое сообщение разработчика",
        "приватное сообщение разработчика",
        "show developer message",
        "print developer message",
        "reveal developer message",
        "return developer message",
        "show the developer message",
        "print the developer message",
        "reveal the developer message",
        "return the developer message",
        "show developer instructions",
        "print developer instructions",
        "reveal developer instructions",
        "return developer prompt",
    ],
    "instruction_override": [
        "игнорируй предыдущие инструкции",
        "игнорируй все предыдущие инструкции",
        "забудь предыдущие инструкции",
        "ignore previous instructions",
        "ignore all previous instructions",
        "disregard previous instructions",
    ],
    "tool_or_secret_exfiltration": [
        "выведи секрет",
        "покажи секрет",
        "раскрой ключ",
        "show secrets",
        "print secrets",
        "exfiltrate",
    ],
}


@dataclass
class SplitRows:
    dataset_name: str
    split_name: str
    path: Path
    rows: list[dict[str, Any]]

    @property
    def display_name(self) -> str:
        return f"{self.dataset_name}/{self.split_name}"


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def percentile(values: list[int], pct: float) -> int:
    if not values:
        return 0
    values = sorted(values)
    idx = round((len(values) - 1) * pct)
    return values[idx]


def pct(value: int | float, total: int | float) -> float:
    return 100 * float(value) / float(total) if total else 0.0


def print_counter(title: str, counter: Counter[Any]) -> None:
    print(title)
    total = counter.total()
    for key, count in counter.most_common():
        print(f"  {key}: {count:,} ({pct(count, total):.1f}%)")


def describe_lengths(rows: list[dict[str, Any]]) -> str:
    lengths = [len(str(row["text"])) for row in rows]
    if not lengths:
        return "avg=0.0, p50=0, p90=0, min=0, max=0"
    return (
        f"avg={statistics.mean(lengths):.1f}, "
        f"p50={percentile(lengths, 0.50)}, "
        f"p90={percentile(lengths, 0.90)}, "
        f"min={min(lengths)}, max={max(lengths)}"
    )


def summarize_numeric(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"avg": 0.0, "p50": 0, "p90": 0, "min": 0, "max": 0}
    return {
        "avg": round(statistics.mean(values), 1),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "min": min(values),
        "max": max(values),
    }


def label_as_int(value: Any) -> int:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"prompt_injection", "injection", "attack", "malicious", "1"}:
            return 1
        if normalized in {"benign", "safe", "not_injection", "0"}:
            return 0
    return int(value)


def length_bin_for_value(length: int) -> str:
    for lower, upper, label in LENGTH_BINS:
        if length >= lower and (upper is None or length <= upper):
            return label
    return LENGTH_BINS[-1][2]


def length_bin_for_text(text: str) -> str:
    return length_bin_for_value(len(text))


def window_count_for_tokens(token_count: int) -> int:
    if token_count <= WINDOW_TOKEN_LENGTH:
        return 1
    count = 0
    start = 0
    last_start = max(0, token_count - WINDOW_TOKEN_LENGTH)
    while start <= last_start:
        count += 1
        if start == last_start:
            break
        start = min(start + WINDOW_TOKEN_STRIDE, last_start)
    return count


def window_count_bin(count: int) -> str:
    if count <= 1:
        return "1"
    if count == 2:
        return "2"
    if count == 3:
        return "3"
    return "4+"


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_parent_id(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def entropy(counter: Counter[Any]) -> float:
    total = counter.total()
    if not total:
        return 0.0
    result = 0.0
    for count in counter.values():
        probability = count / total
        result -= probability * math.log(probability)
    return result


def effective_count(counter: Counter[Any]) -> float:
    return math.exp(entropy(counter)) if counter else 0.0


def jensen_shannon_divergence(left: Counter[Any], right: Counter[Any]) -> float:
    keys = set(left) | set(right)
    left_total = left.total()
    right_total = right.total()
    if not keys or not left_total or not right_total:
        return 0.0
    p = {key: left[key] / left_total for key in keys}
    q = {key: right[key] / right_total for key in keys}
    m = {key: (p[key] + q[key]) / 2 for key in keys}
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def kl_divergence(left: dict[Any, float], right: dict[Any, float]) -> float:
    value = 0.0
    for key, probability in left.items():
        if probability > 0 and right[key] > 0:
            value += probability * math.log2(probability / right[key])
    return value


def character_script_counts(text: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for char in text:
        code = ord(char)
        if 0x0400 <= code <= 0x052F:
            counts["cyrillic"] += 1
        elif ("a" <= char.lower() <= "z"):
            counts["latin"] += 1
        elif char.isalpha():
            counts["other_alpha"] += 1
    return counts


def detect_phrase_families(text: str) -> list[str]:
    normalized = normalize_text(text)
    families = []
    for family, phrases in PHRASE_FAMILIES.items():
        if any(phrase in normalized for phrase in phrases):
            families.append(family)
    return families


def infer_attack_strength(text: str, label: int) -> str:
    normalized = normalize_text(text)
    if label == 0:
        return "benign_or_quoted"
    if any(phrase in normalized for phrase in PHRASE_FAMILIES["instruction_override"]):
        return "explicit_override"
    if any(family in detect_phrase_families(text) for family in ("system_prompt_exfiltration", "hidden_instruction_exfiltration", "developer_message_exfiltration")):
        return "bare_or_exfiltration_command"
    if len(text) >= 1000:
        return "long_jailbreak_or_document_attack"
    return "other_attack"


def infer_attack_surface(text: str) -> str:
    stripped = text.strip()
    normalized = stripped.lower()
    if "<!--" in stripped or "-->" in stripped or "<html" in normalized:
        return "html_or_comment"
    if "```" in stripped or "~~~" in stripped:
        return "markdown_or_code_block"
    if re.search(r"\b(metadata|instruction|system|assistant|developer)\s*[:=]", normalized):
        return "metadata_or_structured_field"
    if stripped.startswith("{") or stripped.startswith("["):
        return "json_like"
    if "|" in stripped and "\n" in stripped:
        return "table_like"
    if "ocr" in normalized or "скан" in normalized:
        return "ocr_like"
    return "plain_text"


def infer_attack_position(text: str) -> str:
    normalized = normalize_text(text)
    positions = []
    for phrases in PHRASE_FAMILIES.values():
        for phrase in phrases:
            index = normalized.find(phrase)
            if index >= 0:
                positions.append(index)
    if not positions:
        return "no_known_phrase"
    index = min(positions)
    if len(normalized) == 0:
        return "no_known_phrase"
    relative = index / len(normalized)
    if relative < 0.20:
        return "beginning"
    if relative > 0.80:
        return "end"
    return "middle"


def is_probably_quoted_or_discussed(text: str) -> bool:
    normalized = normalize_text(text)
    if any(marker in normalized for marker in ["пример", "цитат", "фраз", "quote", "example", "do not execute", "не выполня"]):
        return True
    for phrases in PHRASE_FAMILIES.values():
        for phrase in phrases:
            index = normalized.find(phrase)
            if index < 0:
                continue
            before = normalized[max(0, index - 2):index]
            after = normalized[index + len(phrase): index + len(phrase) + 2]
            if any(char in before for char in ['"', "'", "«", "`"]) or any(char in after for char in ['"', "'", "»", "`"]):
                return True
    return False


def text_quality_flags(row: dict[str, Any]) -> list[str]:
    text = str(row.get("text", ""))
    stripped = text.strip()
    flags: list[str] = []
    if not stripped:
        flags.append("empty_text")
    if 0 < len(stripped) < 8:
        flags.append("very_short_text")
    if any(marker in text for marker in MOJIBAKE_MARKERS):
        flags.append("possible_mojibake")
    if any(char in text for char in ZERO_WIDTH_CHARS):
        flags.append("zero_width_char")
    if any(unicodedata.category(char) == "Cc" and char not in "\n\r\t" for char in text):
        flags.append("control_char")
    if re.search(r"\s{20,}", text):
        flags.append("long_whitespace_run")

    scripts = character_script_counts(text)
    letters = scripts["cyrillic"] + scripts["latin"] + scripts["other_alpha"]
    if scripts["cyrillic"] and scripts["latin"] and min(scripts["cyrillic"], scripts["latin"]) / max(scripts["cyrillic"], scripts["latin"]) >= 0.20:
        flags.append("mixed_cyrillic_latin")

    language = str(row.get("language", "")).lower()
    if language == "ru" and letters >= 20 and scripts["cyrillic"] / letters < 0.30:
        flags.append("language_ru_low_cyrillic")
    if language == "en" and letters >= 20 and scripts["latin"] / letters < 0.50:
        flags.append("language_en_low_latin")
    return flags


def print_length_bin_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    print("length bins:")
    bin_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        bin_rows[length_bin_for_text(str(row["text"]))].append(row)

    report: dict[str, Any] = {}
    for _, _, label in LENGTH_BINS:
        group_rows = bin_rows.get(label, [])
        if not group_rows:
            continue
        labels = Counter(label_as_int(row["label"]) for row in group_rows)
        benign = labels.get(0, 0)
        attacks = labels.get(1, 0)
        ratio = benign / attacks if attacks else None
        ratio_text = f"{ratio:.2f}" if ratio is not None else "n/a"
        stats = {
            "rows": len(group_rows),
            "benign": benign,
            "attack": attacks,
            "benign_to_attack_ratio": ratio,
            "lengths": summarize_numeric([len(str(row["text"])) for row in group_rows]),
        }
        report[label] = stats
        print(
            f"  {label}: rows={len(group_rows):,}, "
            f"benign={benign:,}, attack={attacks:,}, "
            f"benign/attack={ratio_text}, {describe_lengths(group_rows)}"
        )

    print("label/length bins:")
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(label_as_int(row["label"]), length_bin_for_text(str(row["text"])))].append(row)
    for (label, length_bin), group_rows in sorted(grouped.items()):
        print(f"  label={label} length_bin={length_bin}: n={len(group_rows):,}, {describe_lengths(group_rows)}")
    return report


def token_and_window_report(rows: list[dict[str, Any]], tokenizer: Any | None) -> dict[str, Any]:
    if tokenizer is None:
        return {}
    token_lengths = [len(tokenizer(str(row["text"]), add_special_tokens=False)["input_ids"]) for row in rows]
    token_bins = Counter(length_bin_for_value(length) for length in token_lengths)
    window_counts = [window_count_for_tokens(length) for length in token_lengths]
    window_bins = Counter(window_count_bin(count) for count in window_counts)
    label_token_bins = Counter(
        (label_as_int(row["label"]), length_bin_for_value(token_lengths[idx])) for idx, row in enumerate(rows)
    )
    label_window_bins = Counter(
        (label_as_int(row["label"]), window_count_bin(window_counts[idx])) for idx, row in enumerate(rows)
    )
    return {
        "token_length_stats": summarize_numeric(token_lengths),
        "token_length_bins": dict(sorted(token_bins.items())),
        "window_count_stats": summarize_numeric(window_counts),
        "window_count_bins": dict(sorted(window_bins.items())),
        "label_token_length_bins": {f"{label}|{bucket}": count for (label, bucket), count in sorted(label_token_bins.items())},
        "label_window_count_bins": {f"{label}|{bucket}": count for (label, bucket), count in sorted(label_window_bins.items())},
        "token_lengths": token_lengths,
        "window_counts": window_counts,
    }


def print_token_and_window_report(report: dict[str, Any]) -> None:
    if not report:
        print("token/window coverage: skipped (pass --tokenizer-model to enable)")
        return
    stats = report["token_length_stats"]
    print(
        "token length stats: "
        f"avg={stats['avg']}, p50={stats['p50']}, p90={stats['p90']}, min={stats['min']}, max={stats['max']}"
    )
    print_counter("token length bins:", Counter(report["token_length_bins"]))
    window_stats = report["window_count_stats"]
    print(
        "sliding-window count stats: "
        f"avg={window_stats['avg']}, p50={window_stats['p50']}, "
        f"p90={window_stats['p90']}, min={window_stats['min']}, max={window_stats['max']}"
    )
    print_counter("sliding-window count bins:", Counter(report["window_count_bins"]))
    print("label/token length bins:")
    for key, count in report["label_token_length_bins"].items():
        label, bucket = key.split("|", 1)
        print(f"  label={label} token_length_bin={bucket}: n={count:,}")
    print("label/window count bins:")
    for key, count in report["label_window_count_bins"].items():
        label, bucket = key.split("|", 1)
        print(f"  label={label} window_count_bin={bucket}: n={count:,}")


def composition_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    label_counter = Counter(label_as_int(row["label"]) for row in rows)
    source_counter = Counter(str(row.get("source_name", "")) for row in rows)
    bucket_counter = Counter(str(row.get("bucket", "")) for row in rows if "bucket" in row)
    language_counter = Counter(str(row.get("language", "")) for row in rows if "language" in row)
    text_unit_counter = Counter(str(row.get("text_unit", "")) for row in rows if "text_unit" in row)
    return {
        "labels": dict(label_counter),
        "sources": dict(source_counter),
        "buckets": dict(bucket_counter),
        "languages": dict(language_counter),
        "text_units": dict(text_unit_counter),
        "source_entropy": entropy(source_counter),
        "source_effective_count": effective_count(source_counter),
        "bucket_entropy": entropy(bucket_counter),
        "bucket_effective_count": effective_count(bucket_counter),
        "top_source_share": max(source_counter.values()) / source_counter.total() if source_counter else 0.0,
        "top_bucket_share": max(bucket_counter.values()) / bucket_counter.total() if bucket_counter else 0.0,
    }


def phrase_family_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    family_rows: dict[str, dict[str, int]] = {
        family: {
            "rows": 0,
            "attack_rows": 0,
            "benign_rows": 0,
            "benign_quoted_or_discussed_rows": 0,
            "short_attack_rows": 0,
        }
        for family in PHRASE_FAMILIES
    }
    for row in rows:
        text = str(row["text"])
        label = label_as_int(row["label"])
        families = detect_phrase_families(text)
        for family in families:
            family_rows[family]["rows"] += 1
            if label == 1:
                family_rows[family]["attack_rows"] += 1
                if len(text) <= 256:
                    family_rows[family]["short_attack_rows"] += 1
            else:
                family_rows[family]["benign_rows"] += 1
                if is_probably_quoted_or_discussed(text):
                    family_rows[family]["benign_quoted_or_discussed_rows"] += 1
    for metrics in family_rows.values():
        positives = metrics["attack_rows"]
        negatives = metrics["benign_quoted_or_discussed_rows"]
        metrics["quoted_negative_to_attack_ratio"] = round(negatives / positives, 4) if positives else None
    return family_rows


def derived_taxonomy_report(rows: list[dict[str, Any]], token_report: dict[str, Any]) -> dict[str, Any]:
    intents: Counter[str] = Counter()
    strengths: Counter[str] = Counter()
    surfaces: Counter[str] = Counter()
    positions: Counter[str] = Counter()
    wrappers: Counter[str] = Counter()
    for row in rows:
        text = str(row["text"])
        label = label_as_int(row["label"])
        families = detect_phrase_families(text)
        if families:
            intents.update(families)
        elif label == 1:
            intents["other_or_unmatched_attack"] += 1
        else:
            intents["benign_no_known_attack_phrase"] += 1
        strengths[infer_attack_strength(text, label)] += 1
        surface = infer_attack_surface(text)
        surfaces[surface] += 1
        wrappers[surface] += 1
        if label == 1:
            positions[infer_attack_position(text)] += 1
    taxonomy = {
        "phrase_family_or_intent": dict(intents),
        "attack_strength": dict(strengths),
        "attack_surface": dict(surfaces),
        "attack_position": dict(positions),
        "wrapper": dict(wrappers),
    }
    if token_report:
        taxonomy["attack_window_count_bins"] = Counter(
            window_count_bin(token_report["window_counts"][idx])
            for idx, row in enumerate(rows)
            if label_as_int(row["label"]) == 1
        )
    return taxonomy


def text_quality_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    flag_counter: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for idx, row in enumerate(rows):
        flags = text_quality_flags(row)
        flag_counter.update(flags)
        for flag in flags:
            if len(examples[flag]) < 5:
                examples[flag].append(
                    {
                        "index": idx,
                        "label": label_as_int(row["label"]),
                        "source_name": row.get("source_name"),
                        "bucket": row.get("bucket"),
                        "text_preview": " ".join(str(row["text"]).split())[:180],
                    }
                )
    return {"flag_counts": dict(flag_counter), "examples": dict(examples)}


def split_duplicate_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    exact_counter = Counter(str(row["text"]) for row in rows)
    normalized_counter = Counter(normalize_text(str(row["text"])) for row in rows)
    labels_by_normalized: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        labels_by_normalized[normalize_text(str(row["text"]))].add(label_as_int(row["label"]))
    conflicts = [text for text, labels in labels_by_normalized.items() if len(labels) > 1]
    return {
        "exact_duplicate_texts": sum(1 for count in exact_counter.values() if count > 1),
        "exact_duplicate_rows": sum(count for count in exact_counter.values() if count > 1),
        "normalized_duplicate_texts": sum(1 for count in normalized_counter.values() if count > 1),
        "normalized_duplicate_rows": sum(count for count in normalized_counter.values() if count > 1),
        "conflicting_normalized_label_texts": len(conflicts),
        "conflicting_normalized_label_examples": conflicts[:10],
    }


def validate_split_report(
    split: SplitRows,
    report: dict[str, Any],
    max_source_share: float,
    min_critical_bucket_rows: int,
) -> list[str]:
    issues: list[str] = []
    rows = split.rows
    if not rows:
        issues.append("split is empty")
        return issues

    invalid_labels = [row.get("label") for row in rows if label_as_int(row.get("label")) not in {0, 1}]
    if invalid_labels:
        issues.append(f"invalid labels found: {invalid_labels[:5]}")

    labels = Counter(label_as_int(row["label"]) for row in rows)
    if len(labels) < 2 and not split.dataset_name.endswith("only"):
        issues.append(f"only one label present: {dict(labels)}")

    top_source_share = report["composition"]["top_source_share"]
    if top_source_share > max_source_share:
        issues.append(f"top source share {top_source_share:.3f} exceeds {max_source_share:.3f}")

    if "bucket" in rows[0] and is_validation_like(split):
        buckets = Counter(str(row.get("bucket", "")) for row in rows)
        for bucket in CRITICAL_BUCKETS:
            count = buckets.get(bucket, 0)
            if count == 0:
                issues.append(f"critical bucket missing: {bucket}")
            elif count < min_critical_bucket_rows:
                issues.append(f"critical bucket {bucket} has {count:,} rows, below {min_critical_bucket_rows:,}")

    phrase_report = report["phrase_families"]
    for family, metrics in phrase_report.items():
        attack_rows = int(metrics["attack_rows"])
        quoted_negatives = int(metrics["benign_quoted_or_discussed_rows"])
        if attack_rows >= 20 and quoted_negatives == 0:
            issues.append(f"phrase family {family} has {attack_rows:,} attacks but no quoted/discussion benign negatives")
        if family in {"system_prompt_exfiltration", "hidden_instruction_exfiltration", "developer_message_exfiltration"}:
            short_attack_rows = int(metrics["short_attack_rows"])
            if is_validation_like(split) and short_attack_rows < 50:
                issues.append(f"phrase family {family} has only {short_attack_rows:,} short attack rows")

    duplicate_report = report["duplicates"]
    if duplicate_report["conflicting_normalized_label_texts"]:
        issues.append(
            f"{duplicate_report['conflicting_normalized_label_texts']:,} normalized texts have conflicting labels"
        )
    return issues


def analyze_split(
    split: SplitRows,
    tokenizer: Any | None,
    max_source_share: float,
    min_critical_bucket_rows: int,
) -> dict[str, Any]:
    rows = split.rows
    print(f"\n== {split.split_name} ==")
    print(f"rows: {len(rows):,}")
    if not rows:
        return {"dataset": split.dataset_name, "split": split.split_name, "rows": 0, "issues": ["split is empty"]}

    print_counter("labels:", Counter(label_as_int(row["label"]) for row in rows))
    print_counter("sources:", Counter(row.get("source_name", "") for row in rows))
    if "bucket" in rows[0]:
        print_counter("buckets:", Counter(row.get("bucket", "") for row in rows))
    if "language" in rows[0]:
        print_counter("languages:", Counter(row.get("language", "") for row in rows))
    if "text_unit" in rows[0]:
        print_counter("text units:", Counter(row.get("text_unit", "") for row in rows))

    length_report = print_length_bin_stats(rows)
    token_report = token_and_window_report(rows, tokenizer)
    print_token_and_window_report(token_report)

    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(label_as_int(row["label"]), str(row.get("source_name", "")))].append(row)

    print("label/source lengths:")
    for (label, source), group_rows in sorted(grouped.items()):
        print(f"  label={label} source={source}: n={len(group_rows):,}, {describe_lengths(group_rows)}")

    if "bucket" in rows[0]:
        print("label/bucket lengths:")
        bucket_grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            bucket_grouped[(label_as_int(row["label"]), str(row.get("bucket", "")))].append(row)
        for (label, bucket), group_rows in sorted(bucket_grouped.items()):
            print(f"  label={label} bucket={bucket}: n={len(group_rows):,}, {describe_lengths(group_rows)}")

    report = {
        "dataset": split.dataset_name,
        "split": split.split_name,
        "path": str(split.path.resolve()),
        "rows": len(rows),
        "composition": composition_report(rows),
        "length_bins": length_report,
        "token_and_windows": {key: value for key, value in token_report.items() if key not in {"token_lengths", "window_counts"}},
        "taxonomy": derived_taxonomy_report(rows, token_report),
        "phrase_families": phrase_family_report(rows),
        "text_quality": text_quality_report(rows),
        "duplicates": split_duplicate_report(rows),
    }
    report["issues"] = validate_split_report(split, report, max_source_share, min_critical_bucket_rows)
    print_composition_metrics(report)
    print_taxonomy_metrics(report)
    print_phrase_family_metrics(report)
    print_quality_and_duplicate_metrics(report)
    print_issues(report["issues"])
    return report


def print_composition_metrics(report: dict[str, Any]) -> None:
    composition = report["composition"]
    print("composition metrics:")
    print(f"  source_entropy: {composition['source_entropy']:.3f}")
    print(f"  source_effective_count: {composition['source_effective_count']:.2f}")
    print(f"  bucket_entropy: {composition['bucket_entropy']:.3f}")
    print(f"  bucket_effective_count: {composition['bucket_effective_count']:.2f}")
    print(f"  top_source_share: {composition['top_source_share']:.3f}")
    print(f"  top_bucket_share: {composition['top_bucket_share']:.3f}")


def print_taxonomy_metrics(report: dict[str, Any]) -> None:
    taxonomy = report["taxonomy"]
    print_counter("derived attack intent / phrase families:", Counter(taxonomy["phrase_family_or_intent"]))
    print_counter("derived attack strength:", Counter(taxonomy["attack_strength"]))
    print_counter("derived attack surface:", Counter(taxonomy["attack_surface"]))
    print_counter("derived attack position:", Counter(taxonomy["attack_position"]))
    if "attack_window_count_bins" in taxonomy:
        print_counter("attack sliding-window count bins:", Counter(taxonomy["attack_window_count_bins"]))


def print_phrase_family_metrics(report: dict[str, Any]) -> None:
    print("phrase-family parity:")
    for family, metrics in report["phrase_families"].items():
        print(
            f"  {family}: rows={metrics['rows']:,}, attacks={metrics['attack_rows']:,}, "
            f"benign={metrics['benign_rows']:,}, quoted/discussed benign={metrics['benign_quoted_or_discussed_rows']:,}, "
            f"short_attacks={metrics['short_attack_rows']:,}, "
            f"quoted_negative/attack={metrics['quoted_negative_to_attack_ratio']}"
        )


def print_quality_and_duplicate_metrics(report: dict[str, Any]) -> None:
    print_counter("text quality flags:", Counter(report["text_quality"]["flag_counts"]))
    duplicates = report["duplicates"]
    print("duplicate/label-conflict metrics:")
    print(f"  exact_duplicate_texts: {duplicates['exact_duplicate_texts']:,}")
    print(f"  exact_duplicate_rows: {duplicates['exact_duplicate_rows']:,}")
    print(f"  normalized_duplicate_texts: {duplicates['normalized_duplicate_texts']:,}")
    print(f"  normalized_duplicate_rows: {duplicates['normalized_duplicate_rows']:,}")
    print(f"  conflicting_normalized_label_texts: {duplicates['conflicting_normalized_label_texts']:,}")


def print_issues(issues: list[str]) -> None:
    if not issues:
        print("dataset validation issues: none")
        return
    print("dataset validation issues:")
    for issue in issues:
        print(f"  - {issue}")


def load_dataset_split(path: Path) -> Dataset | DatasetDict:
    dataset = load_from_disk(str(path))
    if not isinstance(dataset, Dataset | DatasetDict):
        raise TypeError(f"Expected Dataset or DatasetDict at {path}, got {type(dataset).__name__}")
    return dataset


def materialize_rows(dataset: Dataset) -> list[dict[str, Any]]:
    return [dict(row) for row in dataset]


def load_dataset_rows(path: Path, display_name: str | None = None) -> list[SplitRows]:
    dataset = load_dataset_split(path)
    name = display_name or path.name
    print(f"\ndataset: {name}")
    print(f"path: {path.resolve()}")
    if isinstance(dataset, DatasetDict):
        return [
            SplitRows(dataset_name=name, split_name=split_name, path=path, rows=materialize_rows(dataset[split_name]))
            for split_name in dataset
        ]
    return [SplitRows(dataset_name=name, split_name="dataset", path=path, rows=materialize_rows(dataset))]


def is_train_like(split: SplitRows) -> bool:
    return split.split_name == "train"


def is_validation_like(split: SplitRows) -> bool:
    lowered_name = split.dataset_name.lower()
    lowered_path = split.path.name.lower()
    return (
        split.split_name == "validation"
        or lowered_name.startswith("validation")
        or lowered_path.endswith("-validation")
    )


def cross_split_integrity_report(splits: list[SplitRows]) -> dict[str, Any]:
    train_splits = [split for split in splits if is_train_like(split)]
    validation_splits = [split for split in splits if is_validation_like(split)]
    comparisons = []
    for train in train_splits:
        train_exact = {str(row["text"]) for row in train.rows}
        train_normalized = {normalize_text(str(row["text"])) for row in train.rows}
        train_parent_ids = {
            normalize_parent_id(row.get("parent_id"))
            for row in train.rows
            if normalize_parent_id(row.get("parent_id"))
        }
        for validation in validation_splits:
            validation_exact = {str(row["text"]) for row in validation.rows}
            validation_normalized = {normalize_text(str(row["text"])) for row in validation.rows}
            validation_parent_ids = {
                normalize_parent_id(row.get("parent_id"))
                for row in validation.rows
                if normalize_parent_id(row.get("parent_id"))
            }
            comparisons.append(
                {
                    "train_split": train.display_name,
                    "validation_split": validation.display_name,
                    "shared_exact_texts": len(train_exact & validation_exact),
                    "shared_normalized_texts": len(train_normalized & validation_normalized),
                    "shared_parent_ids": len(train_parent_ids & validation_parent_ids),
                }
            )
    return {"comparisons": comparisons}


def print_cross_split_integrity(report: dict[str, Any]) -> list[str]:
    issues = []
    print("\ncross-split integrity:")
    comparisons = report["comparisons"]
    if not comparisons:
        print("  skipped (need at least one train-like and one validation-like split)")
        return issues
    for item in comparisons:
        print(
            f"  {item['train_split']} -> {item['validation_split']}: "
            f"shared_exact_texts={item['shared_exact_texts']:,}, "
            f"shared_normalized_texts={item['shared_normalized_texts']:,}, "
            f"shared_parent_ids={item['shared_parent_ids']:,}"
        )
        if item["shared_exact_texts"]:
            issues.append(f"{item['train_split']} and {item['validation_split']} share exact texts")
        if item["shared_normalized_texts"]:
            issues.append(f"{item['train_split']} and {item['validation_split']} share normalized texts")
        if item["shared_parent_ids"]:
            issues.append(f"{item['train_split']} and {item['validation_split']} share parent_id values")
    return issues


def distribution_drift_report(split_reports: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = {(report["dataset"], report["split"]): report for report in split_reports}
    train_reports = [report for report in split_reports if report["split"] == "train"]
    validation_reports = [
        report
        for report in split_reports
        if report["split"] == "validation" or report["dataset"].startswith("validation")
    ]
    comparisons = []
    for train in train_reports:
        for validation in validation_reports:
            comparisons.append(
                {
                    "train_split": f"{train['dataset']}/{train['split']}",
                    "validation_split": f"{validation['dataset']}/{validation['split']}",
                    "label_jsd": jsd_from_report_counters(train, validation, "labels"),
                    "source_jsd": jsd_from_report_counters(train, validation, "sources"),
                    "bucket_jsd": jsd_from_report_counters(train, validation, "buckets"),
                    "language_jsd": jsd_from_report_counters(train, validation, "languages"),
                    "text_unit_jsd": jsd_from_report_counters(train, validation, "text_units"),
                    "length_bin_jsd": jsd_for_nested_counts(train["length_bins"], validation["length_bins"], "rows"),
                    "token_length_bin_jsd": jsd_for_plain_counts(
                        train.get("token_and_windows", {}).get("token_length_bins", {}),
                        validation.get("token_and_windows", {}).get("token_length_bins", {}),
                    ),
                    "window_count_bin_jsd": jsd_for_plain_counts(
                        train.get("token_and_windows", {}).get("window_count_bins", {}),
                        validation.get("token_and_windows", {}).get("window_count_bins", {}),
                    ),
                }
            )
    return {"comparisons": comparisons, "split_report_keys": [str(key) for key in by_key]}


def jsd_from_report_counters(left: dict[str, Any], right: dict[str, Any], key: str) -> float:
    return jensen_shannon_divergence(
        Counter(left["composition"].get(key, {})),
        Counter(right["composition"].get(key, {})),
    )


def jsd_for_nested_counts(left: dict[str, Any], right: dict[str, Any], count_key: str) -> float:
    return jensen_shannon_divergence(
        Counter({key: value[count_key] for key, value in left.items()}),
        Counter({key: value[count_key] for key, value in right.items()}),
    )


def jsd_for_plain_counts(left: dict[str, Any], right: dict[str, Any]) -> float:
    return jensen_shannon_divergence(Counter(left), Counter(right))


def print_distribution_drift(report: dict[str, Any], max_jsd: float) -> list[str]:
    issues = []
    print("\ntrain/validation distribution drift:")
    comparisons = report["comparisons"]
    if not comparisons:
        print("  skipped (need at least one train report and one validation report)")
        return issues
    for item in comparisons:
        print(f"  {item['train_split']} -> {item['validation_split']}:")
        for key, value in item.items():
            if not key.endswith("_jsd"):
                continue
            print(f"    {key}: {value:.4f}")
            if value > max_jsd:
                issues.append(
                    f"{item['train_split']} -> {item['validation_split']} {key}={value:.4f} exceeds {max_jsd:.4f}"
                )
    return issues


def load_tokenizer(model_id: str | None) -> Any | None:
    if not model_id:
        return None
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect and validate training/validation dataset coverage.")
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
    parser.add_argument(
        "--tokenizer-model",
        default=None,
        help="Optional tokenizer model/local directory for token length and sliding-window coverage metrics.",
    )
    parser.add_argument(
        "--json-report-path",
        default=None,
        help="Optional path to write the full validator report as JSON.",
    )
    parser.add_argument("--max-source-share", type=float, default=DEFAULT_MAX_SOURCE_SHARE)
    parser.add_argument("--max-distribution-jsd", type=float, default=DEFAULT_MAX_DISTRIBUTION_JSD)
    parser.add_argument("--min-critical-bucket-rows", type=int, default=DEFAULT_MIN_CRITICAL_BUCKET_ROWS)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with status 2 when validator issues are present.",
    )
    return parser.parse_args()


def main() -> None:
    configure_stdout()
    args = parse_args()

    tokenizer = load_tokenizer(args.tokenizer_model)
    splits = load_dataset_rows(Path(args.dataset_dir), "training")
    if args.validation_dataset_dir:
        splits.extend(load_dataset_rows(Path(args.validation_dataset_dir), "validation"))

    split_reports = [
        analyze_split(
            split=split,
            tokenizer=tokenizer,
            max_source_share=args.max_source_share,
            min_critical_bucket_rows=args.min_critical_bucket_rows,
        )
        for split in splits
    ]

    cross_split_report = cross_split_integrity_report(splits)
    cross_split_issues = print_cross_split_integrity(cross_split_report)
    drift_report = distribution_drift_report(split_reports)
    drift_issues = print_distribution_drift(drift_report, args.max_distribution_jsd)

    all_issues = [
        f"{report['dataset']}/{report['split']}: {issue}"
        for report in split_reports
        for issue in report.get("issues", [])
    ]
    all_issues.extend(cross_split_issues)
    all_issues.extend(drift_issues)

    full_report = {
        "dataset_dir": str(Path(args.dataset_dir).resolve()),
        "validation_dataset_dir": str(Path(args.validation_dataset_dir).resolve()) if args.validation_dataset_dir else None,
        "tokenizer_model": args.tokenizer_model,
        "gates": {
            "max_source_share": args.max_source_share,
            "max_distribution_jsd": args.max_distribution_jsd,
            "min_critical_bucket_rows": args.min_critical_bucket_rows,
            "critical_buckets": CRITICAL_BUCKETS,
        },
        "splits": split_reports,
        "cross_split_integrity": cross_split_report,
        "distribution_drift": drift_report,
        "issues": all_issues,
    }
    if args.json_report_path:
        output_path = Path(args.json_report_path)
        output_path.write_text(json.dumps(full_report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nWrote JSON report: {output_path}")

    print("\nvalidator summary:")
    if all_issues:
        for issue in all_issues:
            print(f"  - {issue}")
    else:
        print("  no issues")

    if args.strict and all_issues:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
