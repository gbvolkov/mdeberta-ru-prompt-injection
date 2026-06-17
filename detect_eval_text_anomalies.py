from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TEXT_FIELD_CANDIDATES = (
    "text",
    "window_text",
    "best_window_text",
    "document_text",
)
SCORE_FIELD_CANDIDATES = (
    "v16_rating",
    "stage1_score",
    "stage1_max_score",
    "p_prompt_injection",
    "document_max_prompt_injection_score",
    "score",
)
GROUP_FIELDS = (
    "corpus",
    "_corpus",
    "category",
    "language",
    "source_name",
    "document_label",
)
MOJIBAKE_MARKERS = ("Ð", "Ñ", "�", "Â", "Ã")
TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё<>@:/._-]+|\d+")


DEFAULT_MARKERS = {
    "model_facing_boilerplate": "no model-facing instruction is present",
    "revision_note": "Revision note",
    "hidden_prompt_negation": "do not request hidden prompt disclosure",
    "ordinary_admin_vocab_boilerplate": "ordinary administrative vocabulary",
    "begin_document": "BEGIN OF DOCUMENT",
    "end_document": "END OF DOCUMENT",
    "example_invalid": "example.invalid",
    "redacted_placeholder": "<redacted-",
    "template_approval_boilerplate": "Any template changes require normal manager approval",
    "control_log_boilerplate": "control log records updates",
    "control_marker_ru": "Контрольная строка",
    "corporate_acceptance_sample": "corporate acceptance sample",
    "system_prompt_en": "system prompt",
    "hidden_prompt_en": "hidden prompt",
    "model_instruction_en": "model-facing instruction",
    "instruction_en": "instruction",
    "secret_en": "secret",
    "policy_en": "policy",
    "ignore_previous_en": "ignore previous",
    "reveal_en": "reveal",
    "системный_промпт": "системный промпт",
    "скрытый_промпт": "скрытый промпт",
    "инструкция": "инструкция",
    "секрет": "секрет",
    "политика": "политика",
    "игнорируй": "игнорируй",
    "раскрой": "раскрой",
}

DEFAULT_REGEX_MARKERS = {
    "direct_ignore_previous_en": r"\b(ignore|disregard)\s+(all\s+)?(previous|prior)\s+(instructions?|directives?)\b",
    "direct_reveal_prompt_en": r"\b(reveal|show|print|dump|display)\b.{0,80}\b(system|developer|hidden)\s+prompt\b",
    "direct_bypass_policy_en": r"\b(bypass|override|disable)\b.{0,60}\b(policy|safety|rules?|guardrails?)\b",
    "direct_dan_en": r"\b(DAN|do anything now|developer mode|jailbreak|stay in character)\b",
    "direct_ignore_previous_ru": r"\b(игнорируй|проигнорируй|забудь)\b.{0,60}\b(предыдущие|прошлые|ранние)?\s*(инструкции|указания|правила)\b",
    "direct_reveal_prompt_ru": r"\b(раскрой|покажи|выведи|напечатай)\b.{0,80}\b(системный|служебный|скрытый|developer)\s+промпт\b",
    "direct_bypass_policy_ru": r"\b(обойди|отключи|переопредели)\b.{0,80}\b(ограничения|политику|правила|защиту)\b",
}

TRIGGER_REGEXES = {
    "system_prompt_en": r"\b(system|developer|hidden)\s+prompt\b",
    "instruction_en": r"\b(instruction|instructions|directive|directives)\b",
    "secret_en": r"\b(secret|secrets|token|tokens|credential|credentials|api\s*key|private\s+context)\b",
    "policy_en": r"\b(policy|policies|safety|access|routing|tool|tools)\b",
    "prompt_ru": r"\b(системн|служебн|скрыт)\w*\s+промпт\w*\b|\bпромпт\w*\b",
    "instruction_ru": r"\bинструкц\w*\b|\bуказан\w*\b|\bправил\w*\b",
    "secret_ru": r"\bсекрет\w*\b|\bтокен\w*\b|\bключ\w*\b|\bучётн\w*\b|\bучетн\w*\b|\bприватн\w*\s+контекст\w*\b",
    "policy_ru": r"\bполитик\w*\b|\bдоступ\w*\b|\bмаршрутизац\w*\b|\bинструмент\w*\b",
}

BENIGN_NEGATION_REGEXES = {
    "no_model_facing_instruction": r"\bno\s+model-facing\s+instruction\s+is\s+present\b",
    "not_hidden_prompt_disclosure": r"\bdo\s+not\s+request\s+hidden\s+prompt\s+disclosure\b",
    "ordinary_admin_vocab": r"\bordinary\s+administrative\s+vocabulary\b",
    "ru_no_model_commands": r"\bбез\s+команд\s+для\s+модели\b",
    "ru_no_hidden_context_disclosure": r"\bне\s+требу\w*\s+раскрыт\w*\s+скрыт\w*\s+контекст\w*\b",
}

METADATA_SCAFFOLD_REGEXES = {
    "revision_note": r"\bRevision\s+note\s+\d+\b",
    "document_number": r"\bDocument\s+\d+\b",
    "begin_end_document": r"\b(BEGIN|END)\s+OF\s+DOCUMENT\b",
    "example_invalid": r"\bexample\.invalid\b",
    "redacted_placeholder": r"<redacted-[^>]+>",
    "control_id": r"\b[A-Z]{2,5}-\d{3,5}-\d{1,4}\b",
    "evidence_item": r"\bevidence\s+item\b|\breview\s+cadence\b|\binternal\s+reference\s+page\b",
}

NUMERIC_CODE_REGEXES = {
    "hex_hash": r"\b[a-fA-F0-9]{16,}\b",
    "float_sequence": r"\b\d+\.\d+(?:\s*,\s*\d+\.\d+){2,}\b",
    "json_like": r"[{}\[\]]\s*['\"]?\w+['\"]?\s*:",
    "route_like": r"\b(GET|POST|PUT|PATCH|DELETE)\s+/[A-Za-z0-9_./{}-]+",
    "sql_like": r"\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION)\b.{0,80}\b(FROM|WHERE|TABLE|VALUES)\b",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect statistical text anomalies in eval/result rows. "
            "Supports CSV and JSONL inputs, including window-results.jsonl and FP CSV exports."
        )
    )
    parser.add_argument("--input", action="append", required=True, help="Input CSV/JSONL file or directory. Repeatable.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--include-name", help="Only load files whose name contains this string, useful for window-results.jsonl.")
    parser.add_argument("--threshold", type=float, default=0.82)
    parser.add_argument("--label-field", default="document_label")
    parser.add_argument("--benign-label", default="not_prompt_injection")
    parser.add_argument("--text-field", default="auto")
    parser.add_argument("--score-field", default="auto")
    parser.add_argument(
        "--selected-mode",
        choices=("benign_fp", "score_ge_threshold", "all"),
        default="benign_fp",
        help=(
            "Rows selected for anomaly analysis. benign_fp selects benign rows with score >= threshold. "
            "Use score_ge_threshold for FP-only CSVs without labels."
        ),
    )
    parser.add_argument("--marker-json", help="Optional JSON object/list of marker names and literal marker strings.")
    parser.add_argument("--ngram-min", type=int, default=4)
    parser.add_argument("--ngram-max", type=int, default=8)
    parser.add_argument("--min-ngram-count", type=int, default=20)
    parser.add_argument("--top-n", type=int, default=200)
    parser.add_argument("--progress-every", type=int, default=50000)
    return parser.parse_args()


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            row.setdefault("_input_file", str(path))
            row.setdefault("_source_line", line_no)
            row.setdefault("corpus", path.parent.name)
            rows.append(row)
    return rows


def read_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for line_no, row in enumerate(csv.DictReader(f), 2):
            parsed = dict(row)
            parsed.setdefault("_input_file", str(path))
            parsed.setdefault("_source_line", line_no)
            parsed.setdefault("corpus", path.parent.name)
            rows.append(parsed)
    return rows


def input_files(paths: list[str], include_name: str | None) -> list[Path]:
    files: list[Path] = []
    for value in paths:
        path = Path(value)
        if path.is_dir():
            files.extend(sorted(path.rglob("*.jsonl")))
            files.extend(sorted(path.rglob("*.csv")))
        else:
            files.append(path)
    unique: dict[str, Path] = {}
    for path in files:
        if include_name and include_name not in path.name:
            continue
        unique[str(path.resolve())] = path
    return list(unique.values())


def load_rows(paths: list[str], progress_every: int, include_name: str | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in input_files(paths, include_name):
        if not path.exists():
            raise FileNotFoundError(path)
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            loaded = read_jsonl(path)
        elif suffix == ".csv":
            loaded = read_csv(path)
        else:
            continue
        rows.extend(loaded)
        if progress_every and len(rows) % progress_every < len(loaded):
            print(f"[anomaly] loaded rows={len(rows):,}")
    return rows


def infer_field(rows: list[dict[str, Any]], candidates: tuple[str, ...], explicit: str) -> str:
    if explicit != "auto":
        return explicit
    counts = Counter()
    for row in rows[:5000]:
        for field in candidates:
            if row.get(field) not in (None, ""):
                counts[field] += 1
    if not counts:
        raise ValueError(f"Could not infer field from candidates: {candidates}")
    return counts.most_common(1)[0][0]


def as_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def is_valid_float(value: Any) -> bool:
    if value is None or value == "":
        return True
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def compile_regex(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE | re.DOTALL)


def regex_count(text: str, patterns: dict[str, str]) -> tuple[int, list[str]]:
    count = 0
    hits: list[str] = []
    for name, pattern in patterns.items():
        matches = compile_regex(pattern).findall(text)
        if matches:
            hits.append(name)
            count += len(matches)
    return count, hits


def marker_present(text: str, marker: dict[str, Any]) -> bool:
    pattern = str(marker.get("pattern", ""))
    if not pattern:
        return False
    if marker.get("type") == "regex":
        return bool(compile_regex(pattern).search(text))
    return pattern.casefold() in text.casefold()


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"https?://\S+", "<url>", text)
    text = re.sub(r"\b[\w.+-]+@[\w.-]+\.\w+\b", "<email>", text)
    text = re.sub(r"\d+", "<num>", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def text_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def script_stats(text: str) -> dict[str, float]:
    letters = [ch for ch in text if ch.isalpha()]
    total = len(letters)
    if total == 0:
        return {"cyrillic_ratio": 0.0, "latin_ratio": 0.0, "letter_count": 0.0}
    cyr = sum(1 for ch in letters if "А" <= ch <= "я" or ch in "Ёё")
    lat = sum(1 for ch in letters if "A" <= ch <= "Z" or "a" <= ch <= "z")
    return {
        "cyrillic_ratio": cyr / total,
        "latin_ratio": lat / total,
        "letter_count": float(total),
    }


def language_mismatch(row: dict[str, Any], stats: dict[str, float]) -> bool:
    language = str(row.get("language", "")).lower()
    if language == "ru" and stats["latin_ratio"] > 0.55 and stats["cyrillic_ratio"] < 0.30:
        return True
    if language == "en" and stats["cyrillic_ratio"] > 0.40:
        return True
    if language == "mixed" and (stats["latin_ratio"] < 0.05 or stats["cyrillic_ratio"] < 0.05):
        return True
    return False


def quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p25": 0.0, "p50": 0.0, "p75": 0.0, "iqr": 0.0}
    sorted_values = sorted(values)
    p25 = sorted_values[int((len(sorted_values) - 1) * 0.25)]
    p50 = statistics.median(sorted_values)
    p75 = sorted_values[int((len(sorted_values) - 1) * 0.75)]
    return {"p25": p25, "p50": p50, "p75": p75, "iqr": p75 - p25}


def normalize_marker_specs(data: dict[str, Any]) -> dict[str, dict[str, str]]:
    markers: dict[str, dict[str, str]] = {}
    for name, value in data.items():
        if isinstance(value, dict):
            marker_type = str(value.get("type", "literal"))
            pattern = str(value.get("pattern") or value.get("text") or "")
            if marker_type not in {"literal", "regex"}:
                raise ValueError(f"Unsupported marker type for {name}: {marker_type}")
            markers[str(name)] = {"type": marker_type, "pattern": pattern}
        else:
            markers[str(name)] = {"type": "literal", "pattern": str(value)}
    return markers


def load_markers(path: str | None) -> dict[str, dict[str, str]]:
    data: dict[str, Any] = dict(DEFAULT_MARKERS)
    for name, pattern in DEFAULT_REGEX_MARKERS.items():
        data[name] = {"type": "regex", "pattern": pattern}
    if not path:
        return normalize_marker_specs(data)
    loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(loaded, dict):
        data.update(loaded)
        return normalize_marker_specs(data)
    if isinstance(loaded, list):
        for item in loaded:
            if isinstance(item, dict):
                marker_name = str(item.get("name") or item.get("marker"))
                data[marker_name] = {"type": str(item.get("type", "literal")), "pattern": str(item.get("text") or item.get("pattern"))}
            else:
                data[str(item)] = str(item)
        return normalize_marker_specs(data)
    raise ValueError("--marker-json must contain a JSON object or list")


def selected_row(row: dict[str, Any], score: float, args: argparse.Namespace) -> bool:
    if args.selected_mode == "all":
        return True
    if args.selected_mode == "score_ge_threshold":
        return score >= args.threshold
    return row.get(args.label_field) == args.benign_label and score >= args.threshold


def baseline_row(row: dict[str, Any], score: float, args: argparse.Namespace) -> bool:
    return row.get(args.label_field) == args.benign_label and score < args.threshold


def row_id(row: dict[str, Any]) -> str:
    for key in ("window_text_hash", "best_window_text_hash", "document_id"):
        if row.get(key):
            return str(row[key])
    return f"{row.get('_input_file', '')}:{row.get('_source_line', '')}"


def group_key(row: dict[str, Any], fields: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(str(row.get(field, "unknown") or "unknown") for field in fields)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def marker_stats(
    selected: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
    text_field: str,
    markers: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, marker in markers.items():
        selected_has = sum(1 for row in selected if marker_present(str(row.get(text_field, "")), marker))
        baseline_has = sum(1 for row in baseline if marker_present(str(row.get(text_field, "")), marker))
        selected_no = len(selected) - selected_has
        baseline_no = len(baseline) - baseline_has
        if selected_has == 0 and baseline_has == 0:
            odds_ratio = None
        elif baseline:
            odds_ratio = ((selected_has + 0.5) / (selected_no + 0.5)) / ((baseline_has + 0.5) / (baseline_no + 0.5))
        else:
            odds_ratio = None
        rows.append(
            {
                "marker": name,
                "type": marker["type"],
                "pattern": marker["pattern"],
                "selected_has": selected_has,
                "selected_total": len(selected),
                "selected_share": selected_has / len(selected) if selected else 0.0,
                "baseline_has": baseline_has,
                "baseline_total": len(baseline),
                "baseline_share": baseline_has / len(baseline) if baseline else 0.0,
                "odds_ratio_selected_vs_baseline": odds_ratio,
            }
        )
    rows.sort(
        key=lambda row: (
            row["selected_has"] > 0 or row["baseline_has"] > 0,
            row["odds_ratio_selected_vs_baseline"] if row["odds_ratio_selected_vs_baseline"] is not None else -1,
            row["selected_has"],
        ),
        reverse=True,
    )
    return rows


def ngram_stats(
    rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    text_field: str,
    min_n: int,
    max_n: int,
    min_count: int,
    top_n: int,
) -> list[dict[str, Any]]:
    occurrence_counts: Counter[str] = Counter()
    row_counts: Counter[str] = Counter()
    for row in rows:
        tokens = tokenize(str(row.get(text_field, "")))
        row_phrases: set[str] = set()
        for n in range(min_n, max_n + 1):
            if len(tokens) < n:
                continue
            for idx in range(0, len(tokens) - n + 1):
                phrase = " ".join(tokens[idx : idx + n])
                occurrence_counts[phrase] += 1
                row_phrases.add(phrase)
        row_counts.update(row_phrases)

    baseline_row_counts: Counter[str] = Counter()
    for row in baseline_rows:
        tokens = tokenize(str(row.get(text_field, "")))
        row_phrases = set()
        for n in range(min_n, max_n + 1):
            if len(tokens) < n:
                continue
            for idx in range(0, len(tokens) - n + 1):
                row_phrases.add(" ".join(tokens[idx : idx + n]))
        baseline_row_counts.update(row_phrases)

    stats = [
        {
            "phrase": phrase,
            "count": row_count,
            "selected_share": row_count / len(rows) if rows else 0.0,
            "baseline_count": baseline_row_counts[phrase],
            "baseline_share": baseline_row_counts[phrase] / len(baseline_rows) if baseline_rows else 0.0,
            "odds_ratio_selected_vs_baseline": (
                None
                if row_count == 0 and baseline_row_counts[phrase] == 0
                else ((row_count + 0.5) / (len(rows) - row_count + 0.5))
                / ((baseline_row_counts[phrase] + 0.5) / (len(baseline_rows) - baseline_row_counts[phrase] + 0.5))
                if baseline_rows
                else None
            ),
            "occurrence_count": occurrence_counts[phrase],
            "occurrences_per_selected_row": occurrence_counts[phrase] / len(rows) if rows else 0.0,
        }
        for phrase, row_count in row_counts.most_common(top_n)
        if row_count >= min_count
    ]
    stats.sort(
        key=lambda row: (
            row["odds_ratio_selected_vs_baseline"] if row["odds_ratio_selected_vs_baseline"] is not None else -1,
            row["count"],
        ),
        reverse=True,
    )
    return stats


def group_stats(rows: list[dict[str, Any]], score_field: str, group_fields: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[group_key(row, group_fields)].append(row)
    out: list[dict[str, Any]] = []
    for key, group_rows in groups.items():
        scores = [as_float(row.get(score_field)) for row in group_rows]
        row = {group_fields[idx]: key[idx] for idx in range(len(group_fields))}
        row.update(
            {
                "rows": len(group_rows),
                "share": len(group_rows) / len(rows) if rows else 0.0,
                "mean_score": sum(scores) / len(scores) if scores else 0.0,
                "max_score": max(scores) if scores else 0.0,
            }
        )
        out.append(row)
    out.sort(key=lambda row: row["rows"], reverse=True)
    return out


def trigger_features(text: str, word_count: int) -> dict[str, Any]:
    trigger_count, trigger_hits = regex_count(text, TRIGGER_REGEXES)
    return {
        "trigger_term_count": trigger_count,
        "trigger_density": trigger_count / max(1, word_count),
        "trigger_hits": ";".join(trigger_hits),
    }


def numeric_code_features(text: str, word_count: int) -> dict[str, Any]:
    hit_count, hits = regex_count(text, NUMERIC_CODE_REGEXES)
    digit_count = sum(1 for ch in text if ch.isdigit())
    code_symbol_count = sum(1 for ch in text if ch in "{}[]();=<>|`")
    return {
        "numeric_code_artifact_score": int(hit_count > 0 or digit_count / max(1, len(text)) > 0.20 or code_symbol_count / max(1, len(text)) > 0.12),
        "numeric_code_hits": ";".join(hits),
        "digit_share": digit_count / max(1, len(text)),
        "code_symbol_density": code_symbol_count / max(1, len(text)),
        "url_count": len(re.findall(r"https?://\S+", text, flags=re.IGNORECASE)),
    }


def anomaly_bucket(scores: dict[str, int]) -> str:
    if scores["label_contamination_score"]:
        return "label_contamination_candidate"
    if scores["short_high_score_score"]:
        return "short_structured_high_score"
    if scores["numeric_code_artifact_score"]:
        return "numeric_code_artifact"
    if scores["template_repetition_score"] >= 2 and scores["benign_negation_marker_score"]:
        return "template_boilerplate_artifact"
    if scores["benign_negation_marker_score"]:
        return "prompt_security_negation_artifact"
    if scores["mixed_language_score"] and scores["trigger_density_score"]:
        return "mixed_language_template"
    if scores["template_repetition_score"] >= 2:
        return "template_boilerplate_artifact"
    return "ordinary_false_positive"


def trigger_density_by_category(flagged_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in flagged_rows:
        groups[str(row.get("category") or "unknown")].append(row)
    rows: list[dict[str, Any]] = []
    for category, group_rows in groups.items():
        densities = sorted(float(row["trigger_density"]) for row in group_rows)
        rows.append(
            {
                "category": category,
                "rows": len(group_rows),
                "mean_trigger_density": sum(densities) / len(densities),
                "p50_trigger_density": statistics.median(densities),
                "p90_trigger_density": densities[int((len(densities) - 1) * 0.90)],
                "max_trigger_density": max(densities),
                "high_trigger_density_rows": sum(1 for value in densities if value >= 0.05),
            }
        )
    rows.sort(key=lambda row: (row["mean_trigger_density"], row["rows"]), reverse=True)
    return rows


def template_clusters(selected: list[dict[str, Any]], flagged_rows: list[dict[str, Any]], text_field: str, top_n: int) -> list[dict[str, Any]]:
    by_row_id = {row["row_id"]: row for row in flagged_rows}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        normalized = normalize_text(str(row.get(text_field, "")))
        cluster_id = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]
        flagged = by_row_id.get(row_id(row))
        if flagged:
            groups[cluster_id].append(flagged)

    rows: list[dict[str, Any]] = []
    for cluster_id, group_rows in groups.items():
        if len(group_rows) < 2:
            continue
        scores = [float(row["score"]) for row in group_rows]
        rows.append(
            {
                "template_cluster_id": cluster_id,
                "rows": len(group_rows),
                "mean_score": sum(scores) / len(scores),
                "max_score": max(scores),
                "categories": ";".join(sorted({str(row["category"]) for row in group_rows})),
                "languages": ";".join(sorted({str(row["language"]) for row in group_rows})),
                "buckets": ";".join(sorted({str(row["anomaly_bucket"]) for row in group_rows})),
                "example_text": group_rows[0]["text"],
            }
        )
    rows.sort(key=lambda row: (row["rows"], row["max_score"]), reverse=True)
    return rows[:top_n]


def write_markdown_summary(path: Path, summary: dict[str, Any]) -> None:
    top_markers = "\n".join(
        f"| {row['marker']} | {row['selected_has']} | {row['selected_share']:.2%} | {row['baseline_has']} | {row['baseline_share']:.2%} | {row['odds_ratio_selected_vs_baseline'] or ''} |"
        for row in summary["top_markers"][:15]
    )
    top_ngrams = "\n".join(
        f"| {row['phrase']} | {row['count']} | {row['selected_share']:.2%} | {row.get('baseline_count', '')} | {row.get('baseline_share', 0.0):.2%} | {row.get('odds_ratio_selected_vs_baseline') or ''} |"
        for row in summary["top_ngrams"][:15]
    )
    bucket_rows = summary.get("bucket_counts", [])
    buckets = "\n".join(f"| {row['anomaly_bucket']} | {row['rows']} | {row['share']:.2%} |" for row in bucket_rows)
    text = f"""# FP Text Anomaly Summary

Selected rows: `{summary['selected_rows']}`
Baseline rows: `{summary['baseline_rows']}`
Threshold: `{summary['threshold']}`
Score field: `{summary['score_field']}`
Text field: `{summary['text_field']}`

## Bucket Counts

| bucket | rows | share |
| --- | ---: | ---: |
{buckets}

## Top Markers

| marker | selected | selected share | baseline | baseline share | odds ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
{top_markers}

## Top N-Grams

| phrase | selected rows | selected share | baseline rows | baseline share | odds ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
{top_ngrams}
"""
    path.write_text(text, encoding="utf-8")


def build_row_flags(
    selected: list[dict[str, Any]],
    text_field: str,
    score_field: str,
    markers: dict[str, dict[str, str]],
    length_bounds: dict[str, float],
) -> list[dict[str, Any]]:
    exact_counts = Counter(str(row.get(text_field, "")) for row in selected)
    normalized_counts = Counter(normalize_text(str(row.get(text_field, ""))) for row in selected)
    prefix_counts = Counter(str(row.get(text_field, ""))[:80] for row in selected)
    suffix_counts = Counter(str(row.get(text_field, ""))[-100:] for row in selected)

    flagged: list[dict[str, Any]] = []
    for row in selected:
        text = str(row.get(text_field, ""))
        stats = script_stats(text)
        marker_hits = [name for name, marker in markers.items() if marker_present(text, marker)]
        mojibake_count = sum(text.count(marker) for marker in MOJIBAKE_MARKERS)
        normalized_duplicate_count = normalized_counts[normalize_text(text)]
        exact_duplicate_count = exact_counts[text]
        prefix80_count = prefix_counts[text[:80]]
        suffix100_count = suffix_counts[text[-100:]]
        char_len = len(text)
        word_count = len(tokenize(text))
        entropy = text_entropy(text)
        trigger = trigger_features(text, word_count)
        benign_negation_count, benign_negation_hits = regex_count(text, BENIGN_NEGATION_REGEXES)
        label_contamination_count, label_contamination_hits = regex_count(text, DEFAULT_REGEX_MARKERS)
        metadata_scaffold_count, metadata_scaffold_hits = regex_count(text, METADATA_SCAFFOLD_REGEXES)
        numeric_code = numeric_code_features(text, word_count)
        mixed_language_score = int(stats["latin_ratio"] >= 0.10 and stats["cyrillic_ratio"] >= 0.10)
        template_repetition_score = int(normalized_duplicate_count > 3) + int(prefix80_count > 10) + int(suffix100_count > 10)
        trigger_density_score = int(trigger["trigger_density"] >= 0.05)
        short_high_score_score = int(word_count < 20 and as_float(row.get(score_field)) > 0.95 and trigger["trigger_term_count"] == 0)
        component_scores = {
            "template_repetition_score": template_repetition_score,
            "trigger_density_score": trigger_density_score,
            "benign_negation_marker_score": int(benign_negation_count > 0),
            "mixed_language_score": mixed_language_score,
            "metadata_scaffold_score": min(3, metadata_scaffold_count),
            "short_high_score_score": short_high_score_score,
            "numeric_code_artifact_score": numeric_code["numeric_code_artifact_score"],
            "label_contamination_score": int(label_contamination_count > 0),
        }
        reasons: list[str] = []
        if marker_hits:
            reasons.append("marker_hit")
        if component_scores["label_contamination_score"]:
            reasons.append("label_contamination_candidate")
        if component_scores["trigger_density_score"]:
            reasons.append("high_trigger_density")
        if component_scores["benign_negation_marker_score"]:
            reasons.append("benign_negation_marker")
        if component_scores["metadata_scaffold_score"]:
            reasons.append("metadata_scaffold")
        if component_scores["numeric_code_artifact_score"]:
            reasons.append("numeric_code_artifact")
        if normalized_duplicate_count > 3:
            reasons.append("normalized_duplicate_cluster")
        if exact_duplicate_count > 1:
            reasons.append("exact_duplicate")
        if prefix80_count > 10:
            reasons.append("repeated_prefix")
        if suffix100_count > 10:
            reasons.append("repeated_suffix")
        if mojibake_count:
            reasons.append("mojibake")
        if char_len < length_bounds["low"]:
            reasons.append("short_length_outlier")
        if char_len > length_bounds["high"]:
            reasons.append("long_length_outlier")
        if language_mismatch(row, stats):
            reasons.append("language_script_mismatch")
        if as_float(row.get(score_field)) >= 0.999:
            reasons.append("ultra_high_score")

        anomaly_score = (
            3 * component_scores["template_repetition_score"]
            + 2 * component_scores["trigger_density_score"]
            + 2 * component_scores["benign_negation_marker_score"]
            + 2 * component_scores["mixed_language_score"]
            + 2 * component_scores["metadata_scaffold_score"]
            + 2 * component_scores["short_high_score_score"]
            + 2 * component_scores["numeric_code_artifact_score"]
            + 5 * component_scores["label_contamination_score"]
        )
        anomaly_score += min(4, len(marker_hits))
        anomaly_score += 3 if mojibake_count else 0
        anomaly_score += 2 if language_mismatch(row, stats) else 0
        anomaly_score += 1 if "ultra_high_score" in reasons else 0

        flagged.append(
            {
                "anomaly_score": anomaly_score,
                "anomaly_bucket": anomaly_bucket(component_scores),
                "anomaly_reasons": ";".join(reasons),
                "marker_hits": ";".join(marker_hits),
                "label_contamination_hits": ";".join(label_contamination_hits),
                "benign_negation_hits": ";".join(benign_negation_hits),
                "metadata_scaffold_hits": ";".join(metadata_scaffold_hits),
                "numeric_code_hits": numeric_code["numeric_code_hits"],
                "score": as_float(row.get(score_field)),
                "row_id": row_id(row),
                "corpus": row.get("corpus") or row.get("_corpus") or "",
                "category": row.get("category", ""),
                "language": row.get("language", ""),
                "source_name": row.get("source_name", ""),
                "document_id": row.get("document_id", ""),
                "window_index": row.get("window_index", ""),
                "window_count": row.get("window_count", ""),
                "window_text_hash": row.get("window_text_hash") or row.get("best_window_text_hash") or "",
                "char_len": char_len,
                "word_count": word_count,
                "trigger_term_count": trigger["trigger_term_count"],
                "trigger_density": trigger["trigger_density"],
                "trigger_hits": trigger["trigger_hits"],
                **component_scores,
                "digit_share": numeric_code["digit_share"],
                "code_symbol_density": numeric_code["code_symbol_density"],
                "url_count": numeric_code["url_count"],
                "entropy": entropy,
                "cyrillic_ratio": stats["cyrillic_ratio"],
                "latin_ratio": stats["latin_ratio"],
                "mojibake_count": mojibake_count,
                "exact_duplicate_count": exact_duplicate_count,
                "normalized_duplicate_count": normalized_duplicate_count,
                "prefix80_count": prefix80_count,
                "suffix100_count": suffix100_count,
                "text": text,
            }
        )
    flagged.sort(key=lambda row: (row["anomaly_score"], row["score"]), reverse=True)
    return flagged


def main() -> None:
    configure_stdout()
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[anomaly] loading inputs")
    rows = load_rows(args.input, args.progress_every, args.include_name)
    if not rows:
        raise ValueError("No rows loaded")

    text_field = infer_field(rows, TEXT_FIELD_CANDIDATES, args.text_field)
    score_field = infer_field(rows, SCORE_FIELD_CANDIDATES, args.score_field)
    markers = load_markers(args.marker_json)
    invalid_score_count = sum(1 for row in rows if not is_valid_float(row.get(score_field)))

    print(f"[anomaly] rows={len(rows):,} text_field={text_field} score_field={score_field}")
    if invalid_score_count:
        print(f"[anomaly] warning invalid numeric scores={invalid_score_count:,}; invalid values are treated as 0.0")
    selected = [row for row in rows if selected_row(row, as_float(row.get(score_field)), args)]
    baseline = [row for row in rows if baseline_row(row, as_float(row.get(score_field)), args)]
    print(f"[anomaly] selected={len(selected):,} baseline={len(baseline):,}")
    if not selected:
        raise ValueError("No rows selected for anomaly analysis")

    lengths = [len(str(row.get(text_field, ""))) for row in selected]
    length_q = quantiles([float(value) for value in lengths])
    length_bounds = {
        "low": max(0.0, length_q["p25"] - 1.5 * length_q["iqr"]),
        "high": length_q["p75"] + 1.5 * length_q["iqr"],
    }

    print("[anomaly] computing marker statistics")
    marker_rows = marker_stats(selected, baseline, text_field, markers)
    write_csv(
        out_dir / "marker_stats.csv",
        marker_rows,
        [
            "marker",
            "type",
            "pattern",
            "selected_has",
            "selected_total",
            "selected_share",
            "baseline_has",
            "baseline_total",
            "baseline_share",
            "odds_ratio_selected_vs_baseline",
        ],
    )

    print("[anomaly] computing repeated n-grams")
    phrase_rows = ngram_stats(selected, baseline, text_field, args.ngram_min, args.ngram_max, args.min_ngram_count, args.top_n)
    write_csv(
        out_dir / "ngram_stats.csv",
        phrase_rows,
        [
            "phrase",
            "count",
            "selected_share",
            "baseline_count",
            "baseline_share",
            "odds_ratio_selected_vs_baseline",
            "occurrence_count",
            "occurrences_per_selected_row",
        ],
    )

    print("[anomaly] computing row-level anomaly flags")
    flagged_rows = build_row_flags(selected, text_field, score_field, markers, length_bounds)
    row_flag_fields = [
        "anomaly_score",
        "anomaly_bucket",
        "anomaly_reasons",
        "marker_hits",
        "label_contamination_hits",
        "benign_negation_hits",
        "metadata_scaffold_hits",
        "numeric_code_hits",
        "score",
        "row_id",
        "corpus",
        "category",
        "language",
        "source_name",
        "document_id",
        "window_index",
        "window_count",
        "window_text_hash",
        "char_len",
        "word_count",
        "trigger_term_count",
        "trigger_density",
        "trigger_hits",
        "template_repetition_score",
        "trigger_density_score",
        "benign_negation_marker_score",
        "mixed_language_score",
        "metadata_scaffold_score",
        "short_high_score_score",
        "numeric_code_artifact_score",
        "label_contamination_score",
        "digit_share",
        "code_symbol_density",
        "url_count",
        "entropy",
        "cyrillic_ratio",
        "latin_ratio",
        "mojibake_count",
        "exact_duplicate_count",
        "normalized_duplicate_count",
        "prefix80_count",
        "suffix100_count",
        "text",
    ]
    write_csv(
        out_dir / "row_anomaly_flags.csv",
        flagged_rows,
        row_flag_fields,
    )
    write_csv(out_dir / "fp_rows_with_anomaly_labels.csv", flagged_rows, row_flag_fields)
    write_csv(
        out_dir / "label_contamination_candidates.csv",
        [row for row in flagged_rows if int(row["label_contamination_score"]) > 0],
        row_flag_fields,
    )
    write_csv(
        out_dir / "short_high_score_candidates.csv",
        [row for row in flagged_rows if int(row["short_high_score_score"]) > 0],
        row_flag_fields,
    )
    write_csv(
        out_dir / "trigger_density_by_category.csv",
        trigger_density_by_category(flagged_rows),
        ["category", "rows", "mean_trigger_density", "p50_trigger_density", "p90_trigger_density", "max_trigger_density", "high_trigger_density_rows"],
    )
    write_csv(
        out_dir / "template_clusters.csv",
        template_clusters(selected, flagged_rows, text_field, args.top_n),
        ["template_cluster_id", "rows", "mean_score", "max_score", "categories", "languages", "buckets", "example_text"],
    )

    print("[anomaly] computing group summaries")
    group_specs = {
        "by_corpus.csv": ("corpus",),
        "by_category.csv": ("category",),
        "by_language.csv": ("language",),
        "by_source_name.csv": ("source_name",),
        "by_corpus_category.csv": ("corpus", "category"),
        "by_category_language.csv": ("category", "language"),
        "by_corpus_language.csv": ("corpus", "language"),
    }
    group_outputs: dict[str, list[dict[str, Any]]] = {}
    for filename, fields in group_specs.items():
        stat_rows = group_stats(selected, score_field, fields)
        group_outputs[filename] = stat_rows
        write_csv(out_dir / filename, stat_rows, list(fields) + ["rows", "share", "mean_score", "max_score"])

    score_values = [as_float(row.get(score_field)) for row in selected]
    anomaly_scores = [row["anomaly_score"] for row in flagged_rows]
    bucket_counter = Counter(str(row["anomaly_bucket"]) for row in flagged_rows)
    bucket_counts = [
        {"anomaly_bucket": bucket, "rows": count, "share": count / len(flagged_rows) if flagged_rows else 0.0}
        for bucket, count in bucket_counter.most_common()
    ]
    summary = {
        "input_rows": len(rows),
        "selected_rows": len(selected),
        "baseline_rows": len(baseline),
        "selected_mode": args.selected_mode,
        "threshold": args.threshold,
        "text_field": text_field,
        "score_field": score_field,
        "invalid_score_count": invalid_score_count,
        "score_summary": {
            "min": min(score_values),
            "p50": statistics.median(score_values),
            "mean": sum(score_values) / len(score_values),
            "max": max(score_values),
        },
        "length_summary": {
            **length_q,
            "low_outlier_cutoff": length_bounds["low"],
            "high_outlier_cutoff": length_bounds["high"],
            "min": min(lengths),
            "max": max(lengths),
        },
        "anomaly_score_summary": {
            "min": min(anomaly_scores),
            "p50": statistics.median(anomaly_scores),
            "mean": sum(anomaly_scores) / len(anomaly_scores),
            "max": max(anomaly_scores),
            "zero_score_rows": sum(1 for value in anomaly_scores if value == 0),
            "score_ge_5_rows": sum(1 for value in anomaly_scores if value >= 5),
            "score_ge_10_rows": sum(1 for value in anomaly_scores if value >= 10),
        },
        "top_markers": marker_rows[:25],
        "top_ngrams": phrase_rows[:25],
        "bucket_counts": bucket_counts,
        "top_groups": {filename: rows[:20] for filename, rows in group_outputs.items()},
        "outputs": {
            "row_anomaly_flags_csv": str(out_dir / "row_anomaly_flags.csv"),
            "fp_rows_with_anomaly_labels_csv": str(out_dir / "fp_rows_with_anomaly_labels.csv"),
            "trigger_density_by_category_csv": str(out_dir / "trigger_density_by_category.csv"),
            "label_contamination_candidates_csv": str(out_dir / "label_contamination_candidates.csv"),
            "short_high_score_candidates_csv": str(out_dir / "short_high_score_candidates.csv"),
            "template_clusters_csv": str(out_dir / "template_clusters.csv"),
            "marker_stats_csv": str(out_dir / "marker_stats.csv"),
            "ngram_stats_csv": str(out_dir / "ngram_stats.csv"),
            "summary_md": str(out_dir / "fp_text_anomaly_summary.md"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown_summary(out_dir / "fp_text_anomaly_summary.md", summary)

    print(
        json.dumps(
            {
                "status": "pass",
                "output_dir": str(out_dir),
                "input_rows": len(rows),
                "selected_rows": len(selected),
                "baseline_rows": len(baseline),
                "row_flags": str(out_dir / "row_anomaly_flags.csv"),
                "summary": str(out_dir / "summary.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
