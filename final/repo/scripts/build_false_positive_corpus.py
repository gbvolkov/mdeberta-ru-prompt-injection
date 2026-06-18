# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer


DEFAULT_TOKENIZER_ID = "./mdeberta-ru-prompt-injection-v10-benign-scratch"
MODEL_MAX_LENGTH = 256
WINDOW_TOKEN_LENGTH = MODEL_MAX_LENGTH - 2
WINDOW_TOKEN_STRIDE = 128

DEFAULT_CATEGORY_WEIGHTS = {
    "job_descriptions": 0.15,
    "corporate_procedures": 0.15,
    "hr_policies": 0.10,
    "admin_instructions": 0.10,
    "legal_templates": 0.10,
    "support_documentation": 0.10,
    "technical_documentation": 0.10,
    "safety_policies": 0.10,
    "meeting_minutes": 0.05,
    "knowledge_base": 0.05,
}

DEFAULT_LENGTH_BUCKET_WEIGHTS = {
    "short": 0.25,  # one production window
    "medium": 0.30,  # 2-4 production windows
    "long": 0.35,  # 5-20 production windows
    "very_long": 0.10,  # 21+ production windows
}


@dataclass(frozen=True)
class SourceSpec:
    name: str
    repo_id: str
    config: str | None
    split: str
    language: str
    target_share: float
    streaming: bool = True


@dataclass
class SourceResult:
    source_name: str
    unsafe: bool
    rows: list[dict[str, Any]]
    scanned: int
    accepted: int
    complete: bool
    from_checkpoint: bool
    error: str | None = None


REAL_BENIGN_SOURCES = (
    SourceSpec("fineweb2_ru", "HuggingFaceFW/fineweb-2", "rus_Cyrl", "train", "ru", 0.22),
    SourceSpec("c4_ru", "allenai/c4", "ru", "train", "ru", 0.18),
    SourceSpec("wikipedia_ru", "wikimedia/wikipedia", "20231101.ru", "train", "ru", 0.16),
    SourceSpec("fineweb_en", "HuggingFaceFW/fineweb", "sample-10BT", "train", "en", 0.12),
    SourceSpec("c4_en", "allenai/c4", "en", "train", "en", 0.10),
    SourceSpec("wikipedia_en", "wikimedia/wikipedia", "20231101.en", "train", "en", 0.08),
    SourceSpec("stackexchange", "common-pile/stackexchange_filtered", None, "train", "mixed", 0.08),
    SourceSpec("legal_case_summaries", "joelniklaus/legal_case_document_summarization", None, "train", "en", 0.06),
)

OPTIONAL_UNSAFE_SOURCES = (
    SourceSpec("russian_toxic", "Mnwa/russian-toxic", None, "train", "ru", 0.35),
    SourceSpec("ru_paradetox", "s-nlp/ru_paradetox_toxicity", None, "train", "ru", 0.15),
    SourceSpec("real_toxicity_prompts", "ToxicityPrompts/RealToxicityPrompts", None, "train", "en", 0.20),
    SourceSpec("toxic_chat", "lmsys/toxic-chat", "toxicchat0124", "train", "en", 0.15),
    SourceSpec("beavertails", "PKU-Alignment/BeaverTails", None, "30k_train", "en", 0.15),
)

TEXT_COLUMNS = (
    "text",
    "content",
    "body",
    "document",
    "article",
    "question",
    "answer",
    "prompt",
    "instruction",
    "input",
    "title",
    "summary",
    "judgement",
    "judgment",
    "case_text",
    "chosen",
    "rejected",
)

RU_KEYWORDS = {
    "job_descriptions": (
        "должностная инструкция",
        "должностной инструкции",
        "должностных инструкций",
        "должностные обязанности",
        "обязанности работника",
        "обязанности сотрудника",
        "функциональные обязанности",
        "общие положения",
        "права и ответственность",
        "права работника",
        "ответственность работника",
        "квалификационные требования",
        "требования к квалификации",
        "требования к образованию",
        "требования к опыту",
        "порядок назначения",
        "назначается на должность",
        "освобождается от должности",
        "подчиняется",
        "секретарь",
        "специалист обязан",
        "работник обязан",
        "сотрудник обязан",
        "должен знать",
        "имеет право",
    ),
    "corporate_procedures": (
        "регламент",
        "положение о",
        "корпоративная процедура",
        "порядок согласования",
        "порядок обработки",
        "внутренний порядок",
        "процедура рассмотрения",
        "служебный порядок",
        "контроль исполнения",
    ),
    "hr_policies": (
        "кадровая политика",
        "правила внутреннего трудового распорядка",
        "персональные данные работников",
        "оформление отпуска",
        "прием на работу",
        "увольнение работника",
        "оценка персонала",
        "адаптация сотрудников",
    ),
    "admin_instructions": (
        "административная инструкция",
        "служебная записка",
        "приказ",
        "распоряжение",
        "порядок оформления",
        "делопроизводство",
        "входящая корреспонденция",
        "исходящая корреспонденция",
    ),
    "legal_templates": (
        "договор",
        "шаблон договора",
        "образец договора",
        "предмет договора",
        "стороны договора",
        "права и обязанности сторон",
        "ответственность сторон",
        "срок действия договора",
        "реквизиты сторон",
    ),
    "support_documentation": (
        "служба поддержки",
        "обращение в поддержку",
        "решение проблемы",
        "инструкция пользователя",
        "часто задаваемые вопросы",
        "не удается войти",
        "восстановление доступа",
        "создать заявку",
    ),
    "technical_documentation": (
        "техническое задание",
        "руководство пользователя",
        "техническая документация",
        "инструкция по настройке",
        "конфигурация",
        "администратор системы",
        "параметры подключения",
        "api",
    ),
    "safety_policies": (
        "охрана труда",
        "техника безопасности",
        "инструкция по охране труда",
        "пожарная безопасность",
        "безопасные условия труда",
        "производственная безопасность",
        "аварийная ситуация",
        "средства индивидуальной защиты",
    ),
    "meeting_minutes": (
        "протокол совещания",
        "повестка дня",
        "присутствовали",
        "слушали",
        "решили",
        "постановили",
        "итоги совещания",
        "срок исполнения",
    ),
    "knowledge_base": (
        "база знаний",
        "статья базы знаний",
        "справочная статья",
        "как настроить",
        "пошаговая инструкция",
        "типовая проблема",
        "рекомендации пользователю",
    ),
}

EN_KEYWORDS = {
    "job_descriptions": (
        "job description",
        "position description",
        "role description",
        "job responsibilities",
        "key responsibilities",
        "duties and responsibilities",
        "essential duties",
        "employee duties",
        "qualification requirements",
        "required qualifications",
        "experience requirements",
        "reports to",
        "position summary",
        "job summary",
        "responsibilities include",
        "the employee is responsible for",
    ),
    "corporate_procedures": (
        "standard operating procedure",
        "corporate procedure",
        "internal procedure",
        "approval process",
        "compliance procedure",
        "policy and procedure",
    ),
    "hr_policies": (
        "hr policy",
        "employee handbook",
        "leave policy",
        "hiring process",
        "performance review",
        "employee data",
        "termination procedure",
    ),
    "admin_instructions": (
        "administrative instruction",
        "office procedure",
        "memo",
        "records management",
        "document control",
        "filing procedure",
    ),
    "legal_templates": (
        "legal template",
        "contract template",
        "agreement template",
        "terms and conditions",
        "parties agree",
        "liability of the parties",
    ),
    "support_documentation": (
        "support article",
        "troubleshooting",
        "knowledge base article",
        "contact support",
        "reset your password",
        "support ticket",
    ),
    "technical_documentation": (
        "technical documentation",
        "user manual",
        "administrator guide",
        "configuration",
        "api documentation",
        "installation guide",
        "system requirements",
    ),
    "safety_policies": (
        "safety policy",
        "occupational safety",
        "workplace safety",
        "fire safety",
        "emergency procedure",
        "personal protective equipment",
    ),
    "meeting_minutes": (
        "meeting minutes",
        "agenda item",
        "attendees",
        "minutes of meeting",
        "action items",
        "resolved that",
    ),
    "knowledge_base": (
        "knowledge base",
        "faq",
        "how to configure",
        "step-by-step guide",
        "known issue",
        "recommended solution",
    ),
}

PROMPT_INJECTION_PATTERNS = (
    r"\b(ignore|disregard|forget|bypass|override|discard)\b.{0,80}\b(previous|prior|above|system|developer|instruction|rule|policy)\b",
    r"\b(reveal|print|show|display|dump|return|exfiltrate|leak)\b.{0,100}\b(system prompt|developer message|hidden instruction|internal policy|secret|api key|token|password)\b",
    r"\b(act as|you are now|switch to)\b.{0,60}\b(dan|jailbreak|uncensored|developer mode|root mode)\b",
    r"игнорируй.{0,80}(предыдущ|прошл|системн|инструкц|правил|политик)",
    r"забудь.{0,80}(инструкц|правил|контекст|политик)",
    r"(покажи|раскрой|выведи|напечатай|верни|слей).{0,100}(системн|скрыт|секрет|developer|разработчик|токен|парол|ключ)",
    r"(обойди|отмени|переопредели|нарушь).{0,80}(политик|огранич|правил|инструкц|безопас)",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a document-level false-positive review corpus from real benign datasets. "
            "Documents are kept as full text; later run_false_positive_review.py performs production windowing."
        )
    )
    parser.add_argument("--output-jsonl", default="false-positive-corpus-documents.jsonl")
    parser.add_argument("--report-json", default="false-positive-corpus-report.json")
    parser.add_argument("--target-documents", type=int, default=20_000)
    parser.add_argument("--unsafe-documents", type=int, default=0)
    parser.add_argument("--tokenizer-id", default=DEFAULT_TOKENIZER_ID)
    parser.add_argument("--min-document-chars", type=int, default=80)
    parser.add_argument(
        "--max-document-chars",
        type=int,
        default=80_000,
        help="Skip larger documents instead of prefix-truncating them. Use 0 for no limit.",
    )
    parser.add_argument("--candidate-oversample-factor", type=float, default=2.0)
    parser.add_argument("--max-scan-per-source", type=int, default=350_000)
    parser.add_argument("--source-pool", default="external_mining_only", choices=("external_mining_only", "train", "internal_validation"))
    parser.add_argument("--source-workers", type=int, default=1, help="Number of source datasets to collect in parallel.")
    parser.add_argument("--checkpoint-dir", default=None, help="Directory for per-source checkpoint JSONL/state files.")
    parser.add_argument("--resume", action="store_true", help="Reuse per-source checkpoint files and continue partial sources.")
    parser.add_argument("--no-checkpoint", action="store_true", help="Disable per-source checkpoint writes.")
    parser.add_argument("--checkpoint-scan-interval", type=int, default=5000)
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    parser.add_argument("--allow-source-errors", action="store_true")
    parser.add_argument("--allow-underfilled", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    category_targets = scaled_targets(DEFAULT_CATEGORY_WEIGHTS, args.target_documents)
    length_targets = nested_length_targets(category_targets)
    unsafe_targets = {"unsafe_non_injection": args.unsafe_documents} if args.unsafe_documents else {}
    if args.dry_run:
        print(
            json.dumps(
                {
                    "category_targets": category_targets,
                    "length_bucket_weights": DEFAULT_LENGTH_BUCKET_WEIGHTS,
                    "unsafe_targets": unsafe_targets,
                    "sources": [spec.__dict__ for spec in REAL_BENIGN_SOURCES],
                    "optional_unsafe_sources": [spec.__dict__ for spec in OPTIONAL_UNSAFE_SOURCES],
                    "checkpoint_dir": str(checkpoint_dir_for_args(args)) if not args.no_checkpoint else "",
                    "source_workers": args.source_workers,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    candidates: dict[str, dict[str, list[dict[str, Any]]]] = {
        category: {bucket: [] for bucket in DEFAULT_LENGTH_BUCKET_WEIGHTS}
        for category in category_targets
    }
    unsafe_candidates: list[dict[str, Any]] = []
    errors: list[str] = []
    source_reports: list[dict[str, Any]] = []

    jobs: list[tuple[SourceSpec, int, bool]] = [
        (spec, source_candidate_target(spec, args.target_documents), False)
        for spec in REAL_BENIGN_SOURCES
    ]
    if args.unsafe_documents:
        jobs.extend(
            (
                spec,
                max(100, int(math.ceil(args.unsafe_documents * spec.target_share * args.candidate_oversample_factor))),
                True,
            )
            for spec in OPTIONAL_UNSAFE_SOURCES
        )

    for result in collect_source_jobs(args, jobs):
        source_reports.append(source_result_report(result))
        if result.error:
            errors.append(result.error)
            continue
        if result.unsafe:
            unsafe_candidates.extend(result.rows)
            continue
        for row in result.rows:
            category = row["category"]
            length_bucket = row["length_bucket"]
            if category in candidates and length_bucket in candidates[category]:
                candidates[category][length_bucket].append(row)

    selected = select_documents(candidates, category_targets, length_targets, args.allow_underfilled)
    selected_hashes = {row["text_hash"] for row in selected}
    selected.extend(select_unsafe_documents(unsafe_candidates, args.unsafe_documents, args.allow_underfilled, selected_hashes))
    selected = sorted(selected, key=lambda row: row["document_id"])

    write_jsonl(Path(args.output_jsonl), selected)
    report = make_report(args, selected, candidates, unsafe_candidates, errors, category_targets, length_targets, source_reports)
    Path(args.report_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print_summary(report, args.output_jsonl, args.report_json)


def scaled_targets(weights: dict[str, float], total: int) -> dict[str, int]:
    exact = {key: total * value for key, value in weights.items()}
    targets = {key: int(value) for key, value in exact.items()}
    remainder = total - sum(targets.values())
    for key, _ in sorted(exact.items(), key=lambda item: item[1] - int(item[1]), reverse=True)[:remainder]:
        targets[key] += 1
    return targets


def nested_length_targets(category_targets: dict[str, int]) -> dict[str, dict[str, int]]:
    return {
        category: scaled_targets(DEFAULT_LENGTH_BUCKET_WEIGHTS, target)
        for category, target in category_targets.items()
    }


def source_candidate_target(spec: SourceSpec, total_documents: int) -> int:
    return max(500, int(math.ceil(total_documents * spec.target_share * 2.2)))


def collect_source_jobs(args: argparse.Namespace, jobs: list[tuple[SourceSpec, int, bool]]) -> Iterable[SourceResult]:
    workers = max(1, int(args.source_workers or 1))
    if workers == 1:
        for spec, target_candidates, unsafe in jobs:
            try:
                yield collect_source_job(args, spec, target_candidates, unsafe)
            except Exception as exc:
                if not args.allow_source_errors:
                    raise
                yield SourceResult(
                    source_name=spec.name,
                    unsafe=unsafe,
                    rows=[],
                    scanned=0,
                    accepted=0,
                    complete=False,
                    from_checkpoint=False,
                    error=source_error_message(spec, exc),
                )
        return

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_job = {
            executor.submit(collect_source_job, args, spec, target_candidates, unsafe): (spec, unsafe)
            for spec, target_candidates, unsafe in jobs
        }
        for future in as_completed(future_to_job):
            spec, unsafe = future_to_job[future]
            try:
                yield future.result()
            except Exception as exc:
                if not args.allow_source_errors:
                    raise
                yield SourceResult(
                    source_name=spec.name,
                    unsafe=unsafe,
                    rows=[],
                    scanned=0,
                    accepted=0,
                    complete=False,
                    from_checkpoint=False,
                    error=source_error_message(spec, exc),
                )


def collect_source_job(args: argparse.Namespace, spec: SourceSpec, target_candidates: int, unsafe: bool) -> SourceResult:
    row_path, state_path = checkpoint_paths(args, spec, unsafe)
    existing_rows: list[dict[str, Any]] = []
    state: dict[str, Any] = {}
    compatible_checkpoint = False
    if args.resume and row_path and state_path and row_path.exists() and state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        compatible_checkpoint = checkpoint_state_compatible(state, args, spec, unsafe)
        if compatible_checkpoint:
            existing_rows = read_jsonl(row_path)
            if state.get("complete") and checkpoint_satisfies_request(state, args, target_candidates, len(existing_rows)):
                return SourceResult(
                    source_name=spec.name,
                    unsafe=unsafe,
                    rows=existing_rows[:target_candidates],
                    scanned=int(state.get("scanned", 0) or 0),
                    accepted=len(existing_rows),
                    complete=True,
                    from_checkpoint=True,
                )

    if row_path and not compatible_checkpoint:
        row_path.parent.mkdir(parents=True, exist_ok=True)
        row_path.write_text("", encoding="utf-8")
        existing_rows = []

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)
    rows = list(existing_rows)
    local_seen_hashes = {row["text_hash"] for row in rows if row.get("text_hash")}
    resume_scanned = int(state.get("scanned", 0) or 0) if compatible_checkpoint else 0
    scanned = resume_scanned
    stop_reason = "unknown"
    raw = open_dataset(spec)
    mode = "a" if compatible_checkpoint else "a"
    handle = row_path.open(mode, encoding="utf-8") if row_path else None
    last_state_scan = resume_scanned
    write_checkpoint_state(state_path, args, spec, unsafe, scanned, len(rows), target_candidates, False, "started")
    try:
        progress = tqdm(raw, desc=f"Collecting {spec.name}", unit="row", total=args.max_scan_per_source, disable=args.no_progress)
        for source_index, source_row in enumerate(progress, start=1):
            if source_index <= resume_scanned:
                continue
            scanned = source_index
            if scanned > args.max_scan_per_source:
                stop_reason = "max_scan"
                break
            row_out = build_candidate_row(
                spec=spec,
                args=args,
                tokenizer=tokenizer,
                source_row=source_row,
                unsafe=unsafe,
                local_seen_hashes=local_seen_hashes,
            )
            if row_out:
                rows.append(row_out)
                local_seen_hashes.add(row_out["text_hash"])
                if handle:
                    handle.write(json.dumps(row_out, ensure_ascii=False) + "\n")
                    handle.flush()
                progress.set_postfix(accepted=len(rows))
                if len(rows) >= target_candidates:
                    stop_reason = "target"
                    break
            if scanned - last_state_scan >= max(1, args.checkpoint_scan_interval):
                write_checkpoint_state(state_path, args, spec, unsafe, scanned, len(rows), target_candidates, False, "scan")
                last_state_scan = scanned
        else:
            stop_reason = "source_exhausted"
    finally:
        if handle:
            handle.close()

    complete = stop_reason in {"target", "max_scan", "source_exhausted"}
    write_checkpoint_state(state_path, args, spec, unsafe, scanned, len(rows), target_candidates, complete, stop_reason)
    return SourceResult(
        source_name=spec.name,
        unsafe=unsafe,
        rows=rows[:target_candidates],
        scanned=scanned,
        accepted=len(rows),
        complete=complete,
        from_checkpoint=bool(existing_rows),
    )


def build_candidate_row(
    *,
    spec: SourceSpec,
    args: argparse.Namespace,
    tokenizer: AutoTokenizer,
    source_row: Any,
    unsafe: bool,
    local_seen_hashes: set[str],
) -> dict[str, Any] | None:
    text = extract_text(source_row)
    prepared = prepare_document_text(text, args)
    if not prepared or looks_like_prompt_injection(prepared):
        return None
    text_hash = stable_hash(prepared)
    if text_hash in local_seen_hashes:
        return None
    token_count = token_length(prepared, tokenizer)
    window_count = production_window_count(token_count)
    if unsafe:
        category = "unsafe_non_injection"
        keyword_hits: list[str] = []
    else:
        category, keyword_hits = classify_category(prepared, spec.language)
        if not category:
            return None
    return make_document_row(
        text=prepared,
        source=spec,
        category=category,
        language=spec.language if spec.language != "mixed" else infer_language(prepared),
        token_count=token_count,
        window_count=window_count,
        length_bucket=length_bucket_for_window_count(window_count),
        keyword_hits=keyword_hits,
        source_row=source_row,
        text_hash=text_hash,
        source_pool=args.source_pool,
    )


def checkpoint_dir_for_args(args: argparse.Namespace) -> Path:
    if args.checkpoint_dir:
        return Path(args.checkpoint_dir)
    output = Path(args.output_jsonl)
    return output.with_suffix("").parent / f"{output.with_suffix('').name}-checkpoints"


def checkpoint_paths(args: argparse.Namespace, spec: SourceSpec, unsafe: bool) -> tuple[Path | None, Path | None]:
    if args.no_checkpoint:
        return None, None
    directory = checkpoint_dir_for_args(args)
    name = f"{'unsafe' if unsafe else 'source'}-{safe_filename(spec.name)}"
    return directory / f"{name}.jsonl", directory / f"{name}.state.json"


def safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return cleaned or "source"


def checkpoint_state_compatible(state: dict[str, Any], args: argparse.Namespace, spec: SourceSpec, unsafe: bool) -> bool:
    return (
        state.get("source_name") == spec.name
        and state.get("repo_id") == spec.repo_id
        and state.get("config") == (spec.config or "")
        and state.get("split") == spec.split
        and bool(state.get("unsafe")) == unsafe
        and state.get("tokenizer_id") == args.tokenizer_id
        and int(state.get("min_document_chars", -1)) == args.min_document_chars
        and int(state.get("max_document_chars", -1)) == args.max_document_chars
        and state.get("source_pool") == args.source_pool
    )


def checkpoint_satisfies_request(state: dict[str, Any], args: argparse.Namespace, target_candidates: int, row_count: int) -> bool:
    stop_reason = str(state.get("stop_reason") or "")
    previous_max_scan = int(state.get("max_scan_per_source", 0) or 0)
    if row_count >= target_candidates:
        return True
    if stop_reason in {"max_scan", "source_exhausted"} and previous_max_scan >= args.max_scan_per_source:
        return True
    return False


def write_checkpoint_state(
    path: Path | None,
    args: argparse.Namespace,
    spec: SourceSpec,
    unsafe: bool,
    scanned: int,
    accepted: int,
    target_candidates: int,
    complete: bool,
    stop_reason: str,
) -> None:
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "source_name": spec.name,
        "repo_id": spec.repo_id,
        "config": spec.config or "",
        "split": spec.split,
        "unsafe": unsafe,
        "tokenizer_id": args.tokenizer_id,
        "min_document_chars": args.min_document_chars,
        "max_document_chars": args.max_document_chars,
        "source_pool": args.source_pool,
        "max_scan_per_source": args.max_scan_per_source,
        "target_candidates": target_candidates,
        "scanned": scanned,
        "accepted": accepted,
        "complete": complete,
        "stop_reason": stop_reason,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def source_result_report(result: SourceResult) -> dict[str, Any]:
    return {
        "source_name": result.source_name,
        "unsafe": result.unsafe,
        "rows": len(result.rows),
        "scanned": result.scanned,
        "accepted": result.accepted,
        "complete": result.complete,
        "from_checkpoint": result.from_checkpoint,
        "error": result.error,
    }


def source_error_message(spec: SourceSpec, exc: Exception) -> str:
    return f"{spec.name}: {type(exc).__name__}: {exc}"


def collect_from_source(
    *,
    spec: SourceSpec,
    args: argparse.Namespace,
    tokenizer: AutoTokenizer,
    candidates: dict[str, dict[str, list[dict[str, Any]]]],
    seen_hashes: set[str],
    target_candidates: int,
) -> None:
    raw = open_dataset(spec)
    accepted = 0
    scanned = 0
    progress = tqdm(raw, desc=f"Collecting {spec.name}", unit="row", total=args.max_scan_per_source, disable=args.no_progress)
    for row in progress:
        scanned += 1
        text = extract_text(row)
        prepared = prepare_document_text(text, args)
        if not prepared or looks_like_prompt_injection(prepared):
            if scanned >= args.max_scan_per_source:
                break
            continue
        text_hash = stable_hash(prepared)
        if text_hash in seen_hashes:
            if scanned >= args.max_scan_per_source:
                break
            continue
        category, keyword_hits = classify_category(prepared, spec.language)
        if not category:
            if scanned >= args.max_scan_per_source:
                break
            continue
        token_count = token_length(prepared, tokenizer)
        window_count = production_window_count(token_count)
        length_bucket = length_bucket_for_window_count(window_count)
        row_out = make_document_row(
            text=prepared,
            source=spec,
            category=category,
            language=spec.language if spec.language != "mixed" else infer_language(prepared),
            token_count=token_count,
            window_count=window_count,
            length_bucket=length_bucket,
            keyword_hits=keyword_hits,
            source_row=row,
            text_hash=text_hash,
            source_pool=args.source_pool,
        )
        candidates[category][length_bucket].append(row_out)
        seen_hashes.add(text_hash)
        accepted += 1
        progress.set_postfix(accepted=accepted)
        if accepted >= target_candidates:
            break
        if scanned >= args.max_scan_per_source:
            break


def collect_unsafe_from_source(
    *,
    spec: SourceSpec,
    args: argparse.Namespace,
    tokenizer: AutoTokenizer,
    unsafe_candidates: list[dict[str, Any]],
    seen_hashes: set[str],
    target_candidates: int,
) -> None:
    raw = open_dataset(spec)
    accepted = 0
    scanned = 0
    progress = tqdm(raw, desc=f"Collecting {spec.name}", unit="row", total=args.max_scan_per_source, disable=args.no_progress)
    for row in progress:
        scanned += 1
        text = extract_text(row)
        prepared = prepare_document_text(text, args)
        if not prepared or looks_like_prompt_injection(prepared):
            if scanned >= args.max_scan_per_source:
                break
            continue
        text_hash = stable_hash(prepared)
        if text_hash in seen_hashes:
            if scanned >= args.max_scan_per_source:
                break
            continue
        token_count = token_length(prepared, tokenizer)
        window_count = production_window_count(token_count)
        unsafe_candidates.append(
            make_document_row(
                text=prepared,
                source=spec,
                category="unsafe_non_injection",
                language=spec.language if spec.language != "mixed" else infer_language(prepared),
                token_count=token_count,
                window_count=window_count,
                length_bucket=length_bucket_for_window_count(window_count),
                keyword_hits=[],
                source_row=row,
                text_hash=text_hash,
                source_pool=args.source_pool,
            )
        )
        seen_hashes.add(text_hash)
        accepted += 1
        progress.set_postfix(accepted=accepted)
        if accepted >= target_candidates:
            break
        if scanned >= args.max_scan_per_source:
            break


def open_dataset(spec: SourceSpec) -> Iterable[dict[str, Any]]:
    kwargs: dict[str, Any] = {"split": spec.split, "streaming": spec.streaming}
    if spec.config:
        return load_dataset(spec.repo_id, spec.config, **kwargs)
    return load_dataset(spec.repo_id, **kwargs)


def extract_text(row: Any) -> str:
    if isinstance(row, str):
        return normalize_text(row)
    if isinstance(row, list):
        return normalize_text(" ".join(extract_text(item) for item in row))
    if not isinstance(row, dict):
        return ""

    for key in TEXT_COLUMNS:
        if key in row:
            text = normalize_text(row[key])
            if text:
                return text

    string_values = []
    for value in row.values():
        if isinstance(value, str):
            string_values.append(value)
        elif isinstance(value, dict):
            text = extract_text(value)
            if text:
                string_values.append(text)
    return normalize_text(" ".join(string_values))


def prepare_document_text(text: str, args: argparse.Namespace) -> str:
    text = normalize_text(text)
    if len(text) < args.min_document_chars:
        return ""
    if args.max_document_chars and len(text) > args.max_document_chars:
        return ""
    if looks_like_encoding_corrupted(text):
        return ""
    return text


def classify_category(text: str, language: str) -> tuple[str | None, list[str]]:
    keyword_table = combined_keywords(language)
    scores: list[tuple[int, str, list[str]]] = []
    for category, keywords in keyword_table.items():
        hits = [keyword for keyword in keywords if keyword_matches(text, keyword)]
        if hits:
            specificity = sum(len(keyword) for keyword in hits)
            scores.append((len(hits), specificity, category, hits[:8]))
    if not scores:
        return None, []
    scores.sort(reverse=True)
    _, _, category, hits = scores[0]
    return category, hits


def keyword_matches(text: str, keyword: str) -> bool:
    keyword = normalize_text(keyword).lower()
    if not keyword:
        return False
    text = text.lower()
    escaped = re.escape(keyword)
    escaped = re.sub(r"\\\s+", r"\\s+", escaped)
    pattern = rf"(^|[^\w-]){escaped}($|[^\w-])"
    return re.search(pattern, text, flags=re.IGNORECASE | re.UNICODE) is not None


def looks_like_encoding_corrupted(text: str) -> bool:
    artifacts = sum(text.count(marker) for marker in (chr(0x00E2), chr(0xFFFD), chr(0x00D0), chr(0x00D1)))
    if artifacts >= 4:
        return True
    quote_markers = (chr(0x20AC), chr(0x2122), chr(0x201C), chr(0x201D), chr(0x2013))
    return any(chr(0x00E2) + marker in text for marker in quote_markers)

def combined_keywords(language: str) -> dict[str, tuple[str, ...]]:
    if language == "ru":
        return RU_KEYWORDS
    if language == "en":
        return EN_KEYWORDS
    merged = {}
    for category in DEFAULT_CATEGORY_WEIGHTS:
        merged[category] = (*RU_KEYWORDS.get(category, ()), *EN_KEYWORDS.get(category, ()))
    return merged


def looks_like_prompt_injection(text: str) -> bool:
    lower = text.lower()
    return any(re.search(pattern, lower, flags=re.IGNORECASE) for pattern in PROMPT_INJECTION_PATTERNS)


def make_document_row(
    *,
    text: str,
    source: SourceSpec,
    category: str,
    language: str,
    token_count: int,
    window_count: int,
    length_bucket: str,
    keyword_hits: list[str],
    source_row: Any,
    text_hash: str,
    source_pool: str,
) -> dict[str, Any]:
    source_row_id = extract_source_row_id(source_row)
    document_id = f"{category}:{source.name}:{source_row_id or text_hash}"
    return {
        "document_id": document_id,
        "document_label": "not_prompt_injection",
        "category": category,
        "source_name": source.name,
        "source_dataset": source.repo_id,
        "source_config": source.config or "",
        "source_split": source.split,
        "source_pool": source_pool,
        "source_pool_assignment": source_pool,
        "source_row_id": source_row_id,
        "language": language,
        "text_length": len(text),
        "text_token_length": token_count,
        "production_window_count": window_count,
        "length_bucket": length_bucket,
        "keyword_hits": keyword_hits,
        "text_hash": text_hash,
        "text": text,
    }


def extract_source_row_id(row: Any) -> str:
    if not isinstance(row, dict):
        return ""
    for key in ("id", "doc_id", "document_id", "url", "uri", "source", "path", "sha"):
        value = normalize_text(row.get(key))
        if value:
            return stable_hash(value)
    return ""


def select_documents(
    candidates: dict[str, dict[str, list[dict[str, Any]]]],
    category_targets: dict[str, int],
    length_targets: dict[str, dict[str, int]],
    allow_underfilled: bool,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_hashes: set[str] = set()
    for category, target in category_targets.items():
        category_selected: list[dict[str, Any]] = []
        used_hashes: set[str] = set()
        for length_bucket, quota in length_targets[category].items():
            bucket_rows = candidates[category][length_bucket]
            chosen: list[dict[str, Any]] = []
            for row in bucket_rows:
                row_hash = row["text_hash"]
                if row_hash in selected_hashes or row_hash in used_hashes:
                    continue
                chosen.append(row)
                used_hashes.add(row_hash)
                if len(chosen) >= quota:
                    break
            category_selected.extend(chosen)
        if len(category_selected) < target:
            for bucket_rows in candidates[category].values():
                for row in bucket_rows:
                    row_hash = row["text_hash"]
                    if row_hash in used_hashes or row_hash in selected_hashes:
                        continue
                    category_selected.append(row)
                    used_hashes.add(row_hash)
                    if len(category_selected) >= target:
                        break
                if len(category_selected) >= target:
                    break
        chosen_final = category_selected[:target]
        selected.extend(chosen_final)
        selected_hashes.update(row["text_hash"] for row in chosen_final)

    total_target = sum(category_targets.values())
    if len(selected) < total_target:
        for category in category_targets:
            for bucket_rows in candidates[category].values():
                for row in bucket_rows:
                    row_hash = row["text_hash"]
                    if row_hash in selected_hashes:
                        continue
                    selected.append(row)
                    selected_hashes.add(row_hash)
                    if len(selected) >= total_target:
                        break
                if len(selected) >= total_target:
                    break
            if len(selected) >= total_target:
                break
    if len(selected) < total_target and not allow_underfilled:
        raise ValueError(f"Corpus only has {len(selected):,}/{total_target:,} documents after category redistribution.")
    return selected[:total_target]


def select_unsafe_documents(
    unsafe_candidates: list[dict[str, Any]],
    target: int,
    allow_underfilled: bool,
    exclude_hashes: set[str] | None = None,
) -> list[dict[str, Any]]:
    if target <= 0:
        return []
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in unsafe_candidates:
        by_source[row["source_name"]].append(row)
    selected: list[dict[str, Any]] = []
    used_hashes: set[str] = set(exclude_hashes or set())
    per_source = math.ceil(target / max(1, len(by_source)))
    for source_name in sorted(by_source):
        for row in by_source[source_name][:per_source]:
            if row["text_hash"] in used_hashes:
                continue
            selected.append(row)
            used_hashes.add(row["text_hash"])
            if len(selected) >= target:
                break
        if len(selected) >= target:
            break
    if len(selected) < target:
        for row in unsafe_candidates:
            if row["text_hash"] in used_hashes:
                continue
            selected.append(row)
            used_hashes.add(row["text_hash"])
            if len(selected) >= target:
                break
    if len(selected) < target and not allow_underfilled:
        raise ValueError(f"unsafe_non_injection only has {len(selected):,}/{target:,} documents.")
    return selected


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_report(
    args: argparse.Namespace,
    selected: list[dict[str, Any]],
    candidates: dict[str, dict[str, list[dict[str, Any]]]],
    unsafe_candidates: list[dict[str, Any]],
    errors: list[str],
    category_targets: dict[str, int],
    length_targets: dict[str, dict[str, int]],
    source_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "config": vars(args),
        "checkpoint_dir": "" if args.no_checkpoint else str(checkpoint_dir_for_args(args)),
        "targets": {
            "categories": category_targets,
            "length_buckets": length_targets,
            "unsafe_documents": args.unsafe_documents,
        },
        "selected": summarize_rows(selected),
        "candidate_counts": {
            category: {bucket: len(rows) for bucket, rows in buckets.items()}
            for category, buckets in candidates.items()
        },
        "unsafe_candidate_count": len(unsafe_candidates),
        "source_reports": source_reports,
        "errors": errors,
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "documents": len(rows),
        "categories": count_dict(Counter(row["category"] for row in rows)),
        "sources": count_dict(Counter(row["source_name"] for row in rows)),
        "languages": count_dict(Counter(row["language"] for row in rows)),
        "length_buckets": count_dict(Counter(row["length_bucket"] for row in rows)),
        "window_count": {
            "total": sum(int(row["production_window_count"]) for row in rows),
            "avg": round(sum(int(row["production_window_count"]) for row in rows) / len(rows), 2) if rows else 0,
            "max": max((int(row["production_window_count"]) for row in rows), default=0),
        },
    }


def count_dict(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in counter.most_common()}


def print_summary(report: dict[str, Any], output_jsonl: str, report_json: str) -> None:
    print(
        json.dumps(
            {
                "output_jsonl": output_jsonl,
                "report_json": report_json,
                **report["selected"],
                "errors": report["errors"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def handle_source_error(spec: SourceSpec, exc: Exception, args: argparse.Namespace, errors: list[str]) -> None:
    message = f"{spec.name}: {type(exc).__name__}: {exc}"
    if args.allow_source_errors:
        errors.append(message)
        print(f"Skipped {message}")
        return
    raise exc


def token_length(text: str, tokenizer: AutoTokenizer) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def production_window_count(token_count: int) -> int:
    if token_count <= WINDOW_TOKEN_LENGTH:
        return 1
    return 1 + math.ceil((token_count - WINDOW_TOKEN_LENGTH) / WINDOW_TOKEN_STRIDE)


def length_bucket_for_window_count(window_count: int) -> str:
    if window_count <= 1:
        return "short"
    if window_count <= 4:
        return "medium"
    if window_count <= 20:
        return "long"
    return "very_long"


def infer_language(text: str) -> str:
    has_ru = bool(re.search(r"[А-Яа-яЁё]", text))
    has_en = bool(re.search(r"[A-Za-z]", text))
    if has_ru and has_en:
        return "mixed"
    if has_ru:
        return "ru"
    if has_en:
        return "en"
    return "unknown"


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return normalize_text(" ".join(normalize_text(v) for v in value.values()))
    if isinstance(value, list):
        return normalize_text(" ".join(normalize_text(item) for item in value))
    text = str(value).replace("\x00", " ")
    return re.sub(r"\s+", " ", text).strip()


def stable_hash(value: str) -> str:
    return hashlib.sha1(normalize_text(value).lower().encode("utf-8")).hexdigest()[:20]


if __name__ == "__main__":
    main()
