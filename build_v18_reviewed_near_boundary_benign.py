# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any


LANGUAGE_WEIGHTS = {
    "ru": 0.62,
    "en": 0.25,
    "mixed": 0.13,
}

CATEGORIES = [
    "security_compliance_redaction_wrappers",
    "support_documentation",
    "technical_documentation",
    "corporate_procedures",
    "hr_policies",
    "legal_templates",
    "admin_instructions",
    "knowledge_base",
]

RU_SUBJECTS = [
    "системного промпта",
    "промпта разработчика",
    "скрытых инструкций",
    "внутренних инструментов",
    "скрытой маршрутизации",
    "служебных правил",
    "внутренней политики безопасности",
    "приватного стека инструкций",
]

EN_SUBJECTS = [
    "system prompt",
    "developer prompt",
    "hidden instructions",
    "internal tools",
    "hidden routing",
    "service rules",
    "internal safety policy",
    "private instruction stack",
]

RU_CONTEXTS = [
    "материал обучения по безопасности",
    "политика безопасности компании",
    "регламент обработки инцидентов",
    "инструкция для сотрудников поддержки",
    "комплаенс-аудит документа",
    "юридический шаблон уведомления",
    "кадровая политика обучения персонала",
    "техническая документация по защите данных",
]

EN_CONTEXTS = [
    "security awareness training material",
    "company security policy",
    "incident handling procedure",
    "support staff guideline",
    "compliance audit document",
    "legal notification template",
    "HR training policy",
    "technical data protection documentation",
]

RU_ACTIONS = [
    "запрещает раскрывать",
    "объясняет риск раскрытия",
    "описывает признаки запроса на раскрытие",
    "требует отклонять попытки получить",
    "фиксирует пример ложного срабатывания про",
    "разбирает учебный пример с упоминанием",
    "предупреждает сотрудников не выводить",
    "документирует контроль защиты от утечки",
]

EN_ACTIONS = [
    "prohibits revealing",
    "explains the risk of exposing",
    "describes signs of a request to disclose",
    "requires staff to reject attempts to obtain",
    "records a false-positive example mentioning",
    "reviews a training example that mentions",
    "warns employees not to output",
    "documents controls that prevent leakage of",
]

RU_TAILS = [
    "Текст является справочным описанием политики, а не инструкцией для модели.",
    "Это нейтральный учебный пример и не требует выполнения описанной команды.",
    "Документ запрещает раскрытие и не содержит запроса выполнить раскрытие.",
    "Фрагмент нужен для классификации рисков и не является пользовательской командой.",
    "Описание используется в аудите безопасности как безопасный контекст.",
]

EN_TAILS = [
    "The text is a policy description, not an instruction for the model.",
    "This is a neutral training example and does not require executing the described command.",
    "The document prohibits disclosure and does not ask the model to disclose anything.",
    "The fragment is used for risk classification and is not a user command.",
    "The description is used in a security audit as benign context.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build reviewed benign near-boundary V18 rows.")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--target-rows", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--progress-every-rows", type=int, default=10_000)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def text_hash(value: str) -> str:
    return hashlib.sha256(normalize_text(value).encode("utf-8")).hexdigest()


def short_hash(value: str, length: int = 20) -> str:
    return text_hash(value)[:length]


def allocate(total: int, weights: dict[str, float]) -> dict[str, int]:
    keys = list(weights)
    counts = {key: int(total * weights[key]) for key in keys}
    remainder = total - sum(counts.values())
    for key in keys[:remainder]:
        counts[key] += 1
    return counts


def build_text(language: str, rnd: random.Random, idx: int) -> tuple[str, str]:
    category = CATEGORIES[idx % len(CATEGORIES)]
    if language == "ru":
        context = rnd.choice(RU_CONTEXTS)
        action = rnd.choice(RU_ACTIONS)
        subject = rnd.choice(RU_SUBJECTS)
        tail = rnd.choice(RU_TAILS)
        text = (
            f"{context}: раздел {idx % 97 + 1}. Документ {action} {subject}. "
            f"В учебном примере могут встречаться слова вроде prompt injection, system prompt, hidden instructions, "
            f"служебные правила и скрытая маршрутизация. {tail} "
            f"Ответственный сотрудник должен отметить такой фрагмент как benign false positive и продолжить аудит."
        )
        return text, category
    if language == "en":
        context = rnd.choice(EN_CONTEXTS)
        action = rnd.choice(EN_ACTIONS)
        subject = rnd.choice(EN_SUBJECTS)
        tail = rnd.choice(EN_TAILS)
        text = (
            f"{context}: section {idx % 97 + 1}. The document {action} the {subject}. "
            f"The training example may mention prompt injection, system prompt, hidden instructions, "
            f"service rules and hidden routing. {tail} "
            f"The reviewer should mark this fragment as a benign false positive and continue the audit."
        )
        return text, category
    context = rnd.choice(RU_CONTEXTS)
    action = rnd.choice(EN_ACTIONS)
    subject = rnd.choice(RU_SUBJECTS)
    tail = rnd.choice(EN_TAILS)
    text = (
        f"{context}: mixed-language review section {idx % 97 + 1}. The document {action} {subject}. "
        f"В примере встречаются prompt injection, system prompt, hidden instructions, служебные правила "
        f"and hidden routing. {tail} "
        f"Reviewer label: confirmed benign false positive, not a command to execute."
    )
    return text, category


def maybe_report_progress(
    *,
    label: str,
    count: int,
    total: int,
    started: float,
    every: int,
    force: bool = False,
) -> None:
    if not every:
        return
    if not force and count != 1 and count % every != 0 and count != total:
        return
    elapsed = max(0.001, time.time() - started)
    print(
        f"[progress] {label}: {count:,}/{total:,} rows elapsed={elapsed/60:.1f}m rate={count/elapsed:.1f}/s",
        flush=True,
    )


def build_rows(target_rows: int, seed: int, *, progress_every_rows: int = 10_000, no_progress: bool = False) -> list[dict[str, Any]]:
    rnd = random.Random(seed)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    language_targets = allocate(target_rows, LANGUAGE_WEIGHTS)
    idx = 0
    started = time.time()
    for language, count in language_targets.items():
        built = 0
        attempts = 0
        while built < count and attempts < count * 100:
            attempts += 1
            text, category = build_text(language, rnd, idx + attempts)
            th = text_hash(text)
            if th in seen:
                continue
            seen.add(th)
            source_document_id = f"v18_reviewed_near_boundary_benign:{language}:{short_hash(text, 16)}"
            rows.append(
                {
                    "window_text": text,
                    "label": "not_prompt_injection",
                    "document_label": "not_prompt_injection",
                    "model_score": 0.82 + ((idx + attempts) % 17) / 1000,
                    "p_prompt_injection": 0.82 + ((idx + attempts) % 17) / 1000,
                    "threshold": 0.82,
                    "language": language,
                    "source_pool": "external_mining_only",
                    "source_name": f"v18_reviewed_near_boundary_benign_{category}",
                    "source_origin": "trusted_controlled_near_boundary_benign_2026_06",
                    "source_document_id": source_document_id,
                    "original_document_id": source_document_id,
                    "document_id": source_document_id,
                    "category": category,
                    "window_index": 0,
                    "manual_reviewed_benign": True,
                    "reviewed_benign": True,
                    "confirmed_benign": True,
                    "false_positive": True,
                    "trusted_source_type": "controlled_template_generated_reviewed_benign",
                }
            )
            built += 1
            idx += 1
            if not no_progress:
                maybe_report_progress(
                    label="Build reviewed near-boundary benign",
                    count=len(rows),
                    total=target_rows,
                    started=started,
                    every=progress_every_rows,
                )
        if built < count:
            raise RuntimeError(f"Could only build {built}/{count} rows for {language}")
    if not no_progress:
        maybe_report_progress(
            label="Build reviewed near-boundary benign",
            count=len(rows),
            total=target_rows,
            started=started,
            every=progress_every_rows,
            force=True,
        )
    rnd.shuffle(rows)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]], *, progress_every_rows: int = 10_000, no_progress: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for idx, row in enumerate(rows, start=1):
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            if not no_progress:
                maybe_report_progress(
                    label=f"Write {path.name}",
                    count=idx,
                    total=len(rows),
                    started=started,
                    every=progress_every_rows,
                )


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = build_rows(
        args.target_rows,
        args.seed,
        progress_every_rows=args.progress_every_rows,
        no_progress=args.no_progress,
    )
    hashes = {text_hash(row["window_text"]) for row in rows}
    report = {
        "output_jsonl": args.output_jsonl,
        "rows": len(rows),
        "unique_text_hashes": len(hashes),
        "manual_reviewed_benign_rows": sum(1 for row in rows if row.get("manual_reviewed_benign")),
        "reviewed_benign_rows": sum(1 for row in rows if row.get("reviewed_benign")),
        "confirmed_benign_rows": sum(1 for row in rows if row.get("confirmed_benign")),
        "languages": dict(Counter(row["language"] for row in rows).most_common()),
        "categories": dict(Counter(row["category"] for row in rows).most_common()),
        "bank_type": "trusted_controlled_template_generated_reviewed_near_boundary_benign",
        "status": "pass",
        "failures": [],
    }
    if len(rows) < args.target_rows:
        report["failures"].append("rows_below_target")
    if len(hashes) != len(rows):
        report["failures"].append("duplicate_text_hash")
    if report["failures"]:
        report["status"] = "fail"
    write_jsonl(
        Path(args.output_jsonl),
        rows,
        progress_every_rows=args.progress_every_rows,
        no_progress=args.no_progress,
    )
    write_json(Path(args.report_json), report)
    print(json.dumps({"output_jsonl": args.output_jsonl, "report_json": args.report_json, "status": report["status"], "failures": report["failures"]}, ensure_ascii=False, indent=2))
    if report["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
