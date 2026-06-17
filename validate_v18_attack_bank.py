# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


MODEL_CONTROL_RE = re.compile(
    r"("
    r"system\s+prompt|developer\s+prompt|hidden\s+instruction|internal\s+tool|hidden\s+routing|"
    r"ignore\s+(all\s+){0,1}(previous|prior)\s+instruction[s]{0,1}|jailbreak|prompt\s+injection|"
    r"reveal\s+(the\s+){0,1}(system|developer)|show\s+(the\s+){0,1}(system|developer)|"
    r"print\s+(the\s+){0,1}(system|developer|hidden)|display\s+(the\s+){0,1}(system|developer|hidden)|"
    r"override\s+(the\s+){0,1}(system|instruction)|"
    r"system\s+rules|active\s+instruction|instruction\s+hierarchy|mandatory\s+directive|"
    r"highest\s+priority\s+rule|maximum\s+priority\s+rule|follow\s+only\s+this\s+instruction|"
    r"do\s+not\s+follow.*rules|put\s+this\s+above\s+the\s+system\s+rules|"
    r"replace\s+the\s+previous\s+context|disable\s+earlier\s+response\s+constraints|"
    r"supersede\s+the\s+system\s+instructions|private\s+instruction\s+stack|hidden\s+response\s+rules|"
    r"output\s+.*instructions|output\s+.*инструкц|"
    r"developer\s+rules|developer\s+message|system\s+rules|"
    r"developer\s+instructions|system\s+message|"
    r"private\s+moderation\s+policy|confidential\s+safety\s+rules|internal\s+safety\s+rules|"
    r"hidden\s+tool\s+names|hidden\s+environment\s+commands|private\s+functions|"
    r"confidential\s+instructions|invisible\s+response\s+constraints|response\s+constraints|"
    r"ignore\s+.*инструкц|"
    r"routing\s+rules|internal\s+routing\s+rules|tool\s+selection\s+rules|"
    r"internal\s+policy\s+configuration|"
    r"internal\s+policy|main\s+response\s+policy|directive\s+as\s+mandatory|"
    r"classification\s+rules|system\s+tools|"
    r"system\s+instructions|system\s+command|routing\s+conditions|routing\s+tools|concealed\s+instructions|"
    r"routing\s+policy|directive\s+mandatory|"
    r"supersede\s+.*инструкц|disable\s+.*огранич|"
    r"only\s+this\s+instruction|request\s+routing\s+path|"
    r"систем\w*\s+промпт|"
    r"промпт\s+разработчик\w*|"
    r"скрыт\w*\s+инструкц|"
    r"скрыт\w*\s+маршрут|"
    r"внутрен\w*\s+инструмент|"
    r"внутрен\w*\s+политик|"
    r"служебн\w*\s+правил|"
    r"правил\w*\s+безопасност|"
    r"правил\w*\s+классификац|"
    r"приватн\w*\s+политик|"
    r"скрыт\w*\s+policy\s+rules|"
    r"следуй\s+только\s+этой\s+инструкц|"
    r"отмени\s+.*(правил|инструкц)|"
    r"не\s+выполняй\s+.*(огранич|правил)|"
    r"не\s+учитывай\s+.*правил|"
    r"замени\s+системн\w*\s+инструкц|"
    r"прими\s+.*системн\w*\s+команд|"
    r"главн\w*\s+правил\w*\s+ответ|"
    r"обязательн\w*\s+директив|"
    r"высш\w*\s+приоритет|"
    r"рабоч\w*\s+инструкц|"
    r"скрыт\w*\s+указан|"
    r"скрыт\w*\s+команд|"
    r"максимальн\w*\s+приоритет|"
    r"нов\w*\s+иерархи\w*\s+инструкц|"
    r"переопредел\w*\s+предыдущ\w*\s+контекст|"
    r"правил\w*\s+разработчик|"
    r"сообщени\w*\s+разработчик|"
    r"исходн\w*\s+системн\w*\s+правил|"
    r"приватн\w*\s+функц|"
    r"скрыт\w*\s+tool\s+names|"
    r"считай\s+.*директив\w*\s+обязательн|"
    r"скрыт\w*\s+огранич|"
    r"выше\s+системн\w*\s+правил|"
    r"отключи\s+прежн\w*\s+правил|"
    r"системн\w*\s+сообщени|"
    r"developer\s+instructions|"
    r"роутинг|маршрутизац|"
    r"невидим\w*\s+правил|"
    r"правил\w*\s+ответ|"
    r"системн\w*\s+правил|"
    r"прежн\w*\s+правил|"
    r"переопред\w*|"
    r"служебн\w*\s+политик\w*\s+ответ|"
    r"системн\w*\s+инструмент|"
    r"hidden\s+classification\s+rules|"
    r"игнорируй\s+.*prior\s+instructions|"
    r"rules\s+above|"
    r"маршрут\s+обработки\s+запрос|"
    r"считай\s+.*directive\s+mandatory|"
    r"игнор\w*\s+.*инструкц|"
    r"раскр\w*\s+.*(промпт|инструкц|инструмент|маршрут|политик|конфигурац)|"
    r"покаж\w*\s+.*(промпт|инструкц|инструмент|маршрут|политик|конфигурац)|"
    r"вывед\w*\s+.*(промпт|инструкц|инструмент|маршрут|политик|конфигурац)"
    r")",
    re.IGNORECASE,
)

FILLER_ANCHOR_RE = re.compile(
    r"("
    r"while\s+summari[sz]ing|inside\s+a\s+support\s+ticket|before\s+answering|"
    r"compliance\s+audit|use\s+compact\s+list|application\s+context|"
    r"support\s+ticket\s+response|surrounding\s+document|"
    r"аудит|контекст|документ|заявка поддержки|юридический шаблон|кадровая процедура"
    r")",
    re.IGNORECASE,
)

FORBIDDEN_SOURCE_MARKERS = (
    "training-dataset-v10",
    "training-dataset-v11",
    "training-dataset-v12",
    "training-dataset-v13",
    "training-dataset-v14",
    "training-dataset-v15",
    "training-dataset-v16",
    "training-dataset-v17",
    "mdeberta-ru-prompt-injection-v10",
    "mdeberta-ru-prompt-injection-v11",
    "mdeberta-ru-prompt-injection-v12",
    "mdeberta-ru-prompt-injection-v13",
    "mdeberta-ru-prompt-injection-v14",
    "mdeberta-ru-prompt-injection-v15",
    "mdeberta-ru-prompt-injection-v16",
    "mdeberta-ru-prompt-injection-v17",
    "v12-eval-suites",
    "locked",
    "acceptance",
    "validation-comparison",
    "v17-full-validation",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a V18 trusted external attack bank JSONL.")
    parser.add_argument("--attack-bank-jsonl", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--min-rows", type=int, default=200_000)
    parser.add_argument("--min-unique-texts", type=int, default=100_000)
    parser.add_argument("--min-anchor-share", type=float, default=0.90)
    parser.add_argument("--min-trusted-or-reviewed-share", type=float, default=1.0)
    parser.add_argument("--ru-min", type=float, default=0.60)
    parser.add_argument("--ru-max", type=float, default=0.75)
    parser.add_argument("--en-min", type=float, default=0.15)
    parser.add_argument("--en-max", type=float, default=0.25)
    parser.add_argument("--mixed-min", type=float, default=0.05)
    parser.add_argument("--mixed-max", type=float, default=0.15)
    parser.add_argument("--max-family-share", type=float, default=0.35)
    parser.add_argument("--max-source-name-share", type=float, default=0.35)
    parser.add_argument("--progress-every-rows", type=int, default=50_000)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if isinstance(value, dict):
                yield value


def progress_iter(iterable: Iterable[dict[str, Any]], *, args: argparse.Namespace, label: str) -> Iterable[dict[str, Any]]:
    if args.no_progress:
        yield from iterable
        return
    every = int(args.progress_every_rows or 0)
    started = time.time()
    count = 0
    for row in iterable:
        count += 1
        if every and (count == 1 or count % every == 0):
            elapsed = max(0.001, time.time() - started)
            print(f"[progress] {label}: {count:,} rows elapsed={elapsed/60:.1f}m rate={count/elapsed:.1f}/s", flush=True)
        yield row
    if count and every:
        elapsed = max(0.001, time.time() - started)
        print(f"[progress] {label}: {count:,} rows elapsed={elapsed/60:.1f}m done", flush=True)


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def text_hash(value: str) -> str:
    return hashlib.sha256(normalize_text(value).encode("utf-8")).hexdigest()


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def canonical_language(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"ru", "rus", "russian", "ru_ru", "rus_cyrl"}:
        return "ru"
    if raw in {"en", "eng", "english", "en_us", "en_gb"}:
        return "en"
    if raw in {"mixed", "multi", "multilingual", "ru_en", "en_ru"}:
        return "mixed"
    return raw or "unknown"


def good_anchor(text: str, anchor: str) -> bool:
    text_norm = normalize_text(text).lower()
    anchor_norm = normalize_text(anchor).lower()
    if not anchor_norm:
        return False
    if anchor_norm not in text_norm:
        return False
    if FILLER_ANCHOR_RE.search(anchor_norm):
        return False
    return bool(MODEL_CONTROL_RE.search(anchor_norm))


def source_markers(row: dict[str, Any]) -> list[str]:
    fields = [
        row.get("source_name"),
        row.get("source_origin"),
        row.get("source_family"),
        row.get("generation_method"),
        row.get("audit_file_path"),
        row.get("document_id"),
        row.get("source_document_id"),
    ]
    joined = " ".join(str(value or "").lower() for value in fields)
    return [marker for marker in FORBIDDEN_SOURCE_MARKERS if marker in joined]


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    path = Path(args.attack_bank_jsonl)
    if not path.exists():
        raise FileNotFoundError(path)

    rows = 0
    unique_texts: set[str] = set()
    unique_templates: set[str] = set()
    languages = Counter()
    families = Counter()
    source_names = Counter()
    rejected = Counter()
    samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    trusted_or_reviewed = 0
    good_anchor_rows = 0

    for row in progress_iter(iter_jsonl(path), args=args, label=f"Validate attack bank {path.name}"):
        rows += 1
        attack_text = normalize_text(row.get("attack_text") or row.get("text") or row.get("window_text"))
        anchor = normalize_text(row.get("attack_anchor_text") or row.get("anchor_text") or row.get("attack_span_text"))
        if not attack_text:
            rejected["missing_attack_text"] += 1
            continue
        unique_texts.add(text_hash(attack_text))
        template_id = normalize_text(row.get("attack_template_id") or row.get("template_id"))
        if template_id:
            unique_templates.add(template_id)
        language = canonical_language(row.get("language"))
        family = normalize_text(row.get("semantic_family") or row.get("attack_family") or "unknown")
        source_name = normalize_text(row.get("source_name") or "unknown")
        languages[language] += 1
        families[family] += 1
        source_names[source_name] += 1
        if truthy(row.get("trusted_attack")) or truthy(row.get("manual_reviewed_attack")) or truthy(row.get("reviewed_attack")):
            trusted_or_reviewed += 1
        else:
            rejected["missing_trusted_or_reviewed_signal"] += 1
            if len(samples["missing_trusted_or_reviewed_signal"]) < 10:
                samples["missing_trusted_or_reviewed_signal"].append({"template_id": template_id, "text_excerpt": attack_text[:300]})
        if good_anchor(attack_text, anchor):
            good_anchor_rows += 1
        else:
            rejected["bad_or_missing_anchor"] += 1
            if len(samples["bad_or_missing_anchor"]) < 10:
                samples["bad_or_missing_anchor"].append({"template_id": template_id, "anchor": anchor, "text_excerpt": attack_text[:300]})
        markers = source_markers(row)
        if markers:
            rejected["forbidden_source_marker"] += 1
            if len(samples["forbidden_source_marker"]) < 10:
                samples["forbidden_source_marker"].append({"template_id": template_id, "markers": markers, "source_name": source_name})

    anchor_share = good_anchor_rows / max(1, rows)
    trusted_share = trusted_or_reviewed / max(1, rows)
    language_shares = {key: value / max(1, rows) for key, value in languages.items()}
    top_family_share = families.most_common(1)[0][1] / max(1, rows) if families else 0.0
    top_source_name_share = source_names.most_common(1)[0][1] / max(1, rows) if source_names else 0.0

    failures: list[str] = []
    if rows < args.min_rows:
        failures.append("rows_below_min")
    if len(unique_texts) < args.min_unique_texts:
        failures.append("unique_attack_texts_below_min")
    if anchor_share < args.min_anchor_share:
        failures.append("good_anchor_share_below_min")
    if trusted_share < args.min_trusted_or_reviewed_share:
        failures.append("trusted_or_reviewed_share_below_min")
    if not (args.ru_min <= language_shares.get("ru", 0.0) <= args.ru_max):
        failures.append("ru_language_share_out_of_range")
    if not (args.en_min <= language_shares.get("en", 0.0) <= args.en_max):
        failures.append("en_language_share_out_of_range")
    if not (args.mixed_min <= language_shares.get("mixed", 0.0) <= args.mixed_max):
        failures.append("mixed_language_share_out_of_range")
    if top_family_share > args.max_family_share:
        failures.append("semantic_family_dominance")
    if top_source_name_share > args.max_source_name_share:
        failures.append("source_name_dominance")
    if rejected["forbidden_source_marker"]:
        failures.append("forbidden_source_marker_present")
    if rejected["bad_or_missing_anchor"]:
        failures.append("bad_or_missing_anchor_present")
    if rejected["missing_trusted_or_reviewed_signal"]:
        failures.append("missing_trusted_or_reviewed_signal_present")

    report = {
        "attack_bank_jsonl": str(path),
        "rows": rows,
        "unique_attack_text_hashes": len(unique_texts),
        "unique_attack_template_ids": len(unique_templates),
        "good_anchor_rows": good_anchor_rows,
        "good_anchor_share": anchor_share,
        "trusted_or_reviewed_rows": trusted_or_reviewed,
        "trusted_or_reviewed_share": trusted_share,
        "languages": dict(languages.most_common()),
        "language_shares": language_shares,
        "semantic_families": dict(families.most_common()),
        "top_source_names": dict(source_names.most_common(20)),
        "top_family_share": top_family_share,
        "top_source_name_share": top_source_name_share,
        "rejected": dict(rejected),
        "samples": dict(samples),
        "failures": failures,
        "status": "pass" if not failures else "fail",
    }
    write_json(Path(args.report_json), report)
    print(json.dumps({"report_json": args.report_json, "status": report["status"], "failures": failures}, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
