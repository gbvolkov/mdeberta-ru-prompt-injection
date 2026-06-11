from __future__ import annotations

import argparse
import json
import re
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


MOJIBAKE_MARKERS = (
    chr(0x00D0),
    chr(0x00D1),
    chr(0x00C2),
    chr(0x00E2),
    chr(0xFFFD),
    chr(0x00C3) + chr(0x0090),
    chr(0x00C3) + chr(0x0091),
    chr(0x00C3) + chr(0x0082),
    chr(0x00C3) + chr(0x00A2) + chr(0x00E2) + chr(0x0082) + chr(0x00AC),
)
WRAPPER_MARKERS = (
    "END OF DOCUMENT",
    "BEGIN OF DOCUMENT",
    "<redacted",
    "[redacted",
    "[guarded sentence removed]",
    "<masked",
    "internal parser note",
    "<!--",
    "-->",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract and summarize remaining V16 diagnostic errors.")
    parser.add_argument("--tokenizer-id", default="mdeberta-ru-prompt-injection-v16-critical-recall-restoration-ft")
    parser.add_argument("--critical-corpus-jsonl", default="v13-critical-ru-validation-corpus.jsonl")
    parser.add_argument(
        "--critical-results-jsonl",
        default="validation-comparison-v13-v16-core/v16/v13_critical_ru-documents.jsonl",
    )
    parser.add_argument("--benign-corpus-jsonl", default="benign-prod-calibration-dev.jsonl")
    parser.add_argument(
        "--benign-results-jsonl",
        default="validation-comparison-v13-v16-core/v16/benign_prod_dev-documents.jsonl",
    )
    parser.add_argument(
        "--v13-critical-fn-jsonl",
        default="v13-critical-ru-validation-false-negatives-threshold-0.95.jsonl",
    )
    parser.add_argument("--v16-diagnostic-training-jsonl", default="v16-diagnostic-errors-used-for-training.jsonl")
    parser.add_argument("--critical-threshold", type=float, default=0.95)
    parser.add_argument("--benign-thresholds", default="0.95,0.99")
    parser.add_argument("--output-prefix", default="v16-error-inspection")
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


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def load_by_document_id(path: str | Path) -> dict[str, dict[str, Any]]:
    return {str(row["document_id"]): row for row in iter_jsonl(path)}


def load_id_set(path: str | Path) -> set[str]:
    p = Path(path)
    if not p.exists():
        return set()
    return {str(row.get("document_id")) for row in iter_jsonl(p) if row.get("document_id")}


def load_hash_set(path: str | Path) -> set[str]:
    p = Path(path)
    if not p.exists():
        return set()
    values = set()
    for row in iter_jsonl(p):
        for key in ("attack_text_hash", "source_text_hash", "best_window_text_hash", "text_hash"):
            if row.get(key):
                values.add(str(row[key]))
    return values


def contains_any(text: str, markers: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(marker.lower() in lowered for marker in markers)


def looks_mojibake(text: str) -> bool:
    marker_hits = sum(text.count(marker) for marker in MOJIBAKE_MARKERS)
    if marker_hits >= 2:
        return True
    prefixes = {chr(0x00D0), chr(0x00D1)}
    latin1_cyrillic = {chr(code) for code in range(0x00D0, 0x0100)}
    for idx, char in enumerate(text[:-1]):
        if char in prefixes and (text[idx + 1].isalnum() or text[idx + 1] in latin1_cyrillic):
            return True
    return False

def classify_critical(row: dict[str, Any], source: dict[str, Any], old_v13_ids: set[str], training_ids: set[str]) -> list[str]:
    source_name = str(row.get("source_name", ""))
    category = str(row.get("category", ""))
    text = str(source.get("text", ""))
    tags: list[str] = []
    if row["document_id"] in old_v13_ids:
        tags.append("exact_old_v13_miss_unresolved")
    if row["document_id"] in training_ids:
        tags.append("used_as_v16_training_diagnostic_seed")
    if "standalone" in source_name:
        tags.append("standalone")
    if "embedded" in source_name or (category and category != "critical_ru_override_hidden_prompt_exfiltration"):
        tags.append("embedded_context")
    if "wrapper" in source_name or contains_any(text, WRAPPER_MARKERS):
        tags.append("wrapper_or_marker_adjacent")
    if looks_mojibake(text):
        tags.append("mojibake_or_malformed_encoding")
    token_count = int(source.get("text_token_length") or 0)
    if token_count and token_count < 32:
        tags.append("very_short")
    if len(text.strip()) < 120:
        tags.append("very_short")
    if not tags:
        tags.append("uncategorized")
    return tags


def classify_benign_fp(row: dict[str, Any], source: dict[str, Any], best_window_text: str) -> list[str]:
    text = f"{source.get('text', '')}\n{best_window_text}"
    lowered = text.lower()
    tags: list[str] = []
    if contains_any(text, WRAPPER_MARKERS):
        tags.append("wrapper_or_redaction_marker")
    if re.search(r"http[s]{0,1}://|www\.", text, flags=re.I):
        tags.append("url_or_link_heavy")
    if any(word in lowered for word in ("secret", "confidential", "undisclosed", "classified", "hidden", "internal")):
        tags.append("secret_confidential_language")
    if any(word in lowered for word in ("system", "policy", "instructions", "procedure", "prompt", "rules")):
        tags.append("system_policy_instruction_words")
    if any(word in lowered for word in ("crime", "police", "investigation", "weapon", "security", "attack", "threat", "fraud")):
        tags.append("crime_security_investigation_language")
    if looks_mojibake(text):
        tags.append("mojibake_or_malformed_encoding")
    if not tags:
        tags.append("ordinary_business_doc_or_unclear")
    return tags


def approximate_windows(text: str, window_count: int, *, max_words: int = 190, stride_words: int = 95) -> list[dict[str, Any]]:
    words = text.split()
    if not words:
        return []
    if window_count <= 1 or len(words) <= max_words:
        return [{"window_index": 0, "token_length": len(words), "text": text}]
    windows: list[dict[str, Any]] = []
    start = 0
    index = 0
    while start < len(words):
        chunk = words[start : start + max_words]
        windows.append({"window_index": index, "token_length": len(chunk), "text": " ".join(chunk)})
        if index + 1 >= window_count or start + max_words >= len(words):
            break
        start += stride_words
        index += 1
    while len(windows) < window_count:
        # Keep indices aligned even when whitespace words undercount subword-token windows.
        windows.append(windows[-1] | {"window_index": len(windows)})
    return windows


def best_window(source: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    text = str(source.get("text", ""))
    windows = approximate_windows(text, int(result.get("window_count") or source.get("production_window_count") or 1))
    if not windows:
        return {"window_index": None, "window_text": "", "window_text_hash": None, "token_length": 0}
    idx = max(0, min(len(windows) - 1, int(result.get("best_window_index") or 0)))
    window = windows[idx]
    return {
        "window_index": idx,
        "window_text": window["text"],
        "window_text_hash": text_hash(window["text"]),
        "token_length": window["token_length"],
        "token_start": None,
        "token_end": None,
        "note": "Approximate whitespace-window excerpt; exact error membership and scores come from saved document results.",
    }


def compact_row(result: dict[str, Any], source: dict[str, Any], window: dict[str, Any], tags: list[str]) -> dict[str, Any]:
    text = str(source.get("text", ""))
    return {
        "document_id": result.get("document_id"),
        "document_label": result.get("document_label"),
        "score": result.get("document_max_prompt_injection_score"),
        "category": result.get("category"),
        "source_name": result.get("source_name"),
        "language": result.get("language"),
        "semantic_family": result.get("semantic_family"),
        "attack_template_id": source.get("attack_template_id"),
        "attack_text_hash": result.get("attack_text_hash") or source.get("attack_text_hash"),
        "window_count": result.get("window_count"),
        "window_count_bucket": result.get("window_count_bucket"),
        "best_window_index": result.get("best_window_index"),
        "best_window_text_hash": result.get("best_window_text_hash"),
        "reconstructed_best_window_text_hash": window.get("window_text_hash"),
        "best_window_token_length": window.get("token_length"),
        "tags": tags,
        "text_length": len(text),
        "looks_mojibake": looks_mojibake(text),
        "best_window_text": window.get("window_text", ""),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tag_counter: Counter[str] = Counter()
    for row in rows:
        tag_counter.update(row.get("tags", []))
    by_score_band = Counter()
    for row in rows:
        score = float(row.get("score") or 0.0)
        if score < 0.01:
            by_score_band["score_lt_0.01"] += 1
        elif score < 0.5:
            by_score_band["score_0.01_0.5"] += 1
        elif score < 0.82:
            by_score_band["score_0.5_0.82"] += 1
        elif score < 0.95:
            by_score_band["score_0.82_0.95"] += 1
        elif score < 0.99:
            by_score_band["score_0.95_0.99"] += 1
        else:
            by_score_band["score_gte_0.99"] += 1
    return {
        "rows": len(rows),
        "by_category": dict(Counter(str(row.get("category")) for row in rows).most_common()),
        "by_source_name": dict(Counter(str(row.get("source_name")) for row in rows).most_common()),
        "by_language": dict(Counter(str(row.get("language")) for row in rows).most_common()),
        "by_window_count_bucket": dict(Counter(str(row.get("window_count_bucket")) for row in rows).most_common()),
        "by_tag": dict(tag_counter.most_common()),
        "by_score_band": dict(by_score_band.most_common()),
        "top_scores": [
            {
                "document_id": row["document_id"],
                "score": row["score"],
                "category": row["category"],
                "source_name": row["source_name"],
                "tags": row["tags"],
            }
            for row in sorted(rows, key=lambda r: float(r.get("score") or 0.0), reverse=True)[:10]
        ],
        "bottom_scores": [
            {
                "document_id": row["document_id"],
                "score": row["score"],
                "category": row["category"],
                "source_name": row["source_name"],
                "tags": row["tags"],
            }
            for row in sorted(rows, key=lambda r: float(r.get("score") or 0.0))[:10]
        ],
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# V16 Remaining Error Inspection",
        "",
        "This is diagnostic/mining analysis over already inspected corpora, not blind acceptance evidence.",
        "",
    ]
    for name, section in report["sets"].items():
        lines.extend([f"## {name}", "", f"Rows: `{section['rows']}`", ""])
        for key in ("by_tag", "by_category", "by_source_name", "by_language", "by_window_count_bucket", "by_score_band"):
            lines.append(f"### {key}")
            lines.append("")
            for k, v in section.get(key, {}).items():
                lines.append(f"- `{k}`: {v}")
            lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    critical_sources = load_by_document_id(args.critical_corpus_jsonl)
    critical_results = list(iter_jsonl(args.critical_results_jsonl))
    benign_sources = load_by_document_id(args.benign_corpus_jsonl)
    benign_results = list(iter_jsonl(args.benign_results_jsonl))
    old_v13_ids = load_id_set(args.v13_critical_fn_jsonl)
    training_ids = load_id_set(args.v16_diagnostic_training_jsonl)

    critical_rows: list[dict[str, Any]] = []
    for result in critical_results:
        if str(result.get("document_label")) != "prompt_injection":
            continue
        if float(result.get("document_max_prompt_injection_score") or 0.0) >= args.critical_threshold:
            continue
        source = critical_sources.get(str(result["document_id"]), {})
        window = best_window(source, result)
        tags = classify_critical(result, source, old_v13_ids, training_ids)
        critical_rows.append(compact_row(result, source, window, tags))

    benign_thresholds = [float(part.strip()) for part in args.benign_thresholds.split(",") if part.strip()]
    benign_sets: dict[str, list[dict[str, Any]]] = {}
    for threshold in benign_thresholds:
        rows: list[dict[str, Any]] = []
        for result in benign_results:
            if str(result.get("document_label")) != "not_prompt_injection":
                continue
            if float(result.get("document_max_prompt_injection_score") or 0.0) < threshold:
                continue
            source = benign_sources.get(str(result["document_id"]), {})
            window = best_window(source, result)
            tags = classify_benign_fp(result, source, str(window.get("window_text", "")))
            rows.append(compact_row(result, source, window, tags))
        benign_sets[f"benign_prod_dev_fp_at_{threshold:g}"] = rows

    prefix = Path(args.output_prefix)
    critical_path = prefix.with_name(prefix.name + f"-critical-ru-fn-at-{args.critical_threshold:g}.jsonl")
    write_jsonl(critical_path, critical_rows)

    set_summaries = {
        f"critical_ru_fn_at_{args.critical_threshold:g}": summarize(critical_rows),
    }
    output_files = {"critical_ru_false_negatives": str(critical_path)}

    for set_name, rows in benign_sets.items():
        threshold_label = set_name.rsplit("_", 1)[-1]
        path = prefix.with_name(prefix.name + f"-{set_name.replace('_', '-')}.jsonl")
        write_jsonl(path, rows)
        output_files[set_name] = str(path)
        set_summaries[set_name] = summarize(rows)

    report = {
        "inputs": {
            "critical_corpus_jsonl": args.critical_corpus_jsonl,
            "critical_results_jsonl": args.critical_results_jsonl,
            "benign_corpus_jsonl": args.benign_corpus_jsonl,
            "benign_results_jsonl": args.benign_results_jsonl,
            "v13_critical_fn_jsonl": args.v13_critical_fn_jsonl,
            "v16_diagnostic_training_jsonl": args.v16_diagnostic_training_jsonl,
        },
        "thresholds": {
            "critical_fn_threshold": args.critical_threshold,
            "benign_fp_thresholds": benign_thresholds,
        },
        "output_files": output_files,
        "sets": set_summaries,
    }
    report_json = prefix.with_name(prefix.name + "-report.json")
    report_md = prefix.with_name(prefix.name + "-report.md")
    write_json(report_json, report)
    write_markdown(report, report_md)
    print(json.dumps({"report_json": str(report_json), "report_md": str(report_md), "output_files": output_files}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
