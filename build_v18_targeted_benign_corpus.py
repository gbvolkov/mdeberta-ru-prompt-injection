# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer


MODEL_MAX_LENGTH = 256
WINDOW_TOKEN_LENGTH = MODEL_MAX_LENGTH - 2
WINDOW_TOKEN_STRIDE = 128

TARGET_CATEGORIES = (
    "hr_policies",
    "corporate_procedures",
    "safety_policies",
    "support_documentation",
)

DEFAULT_TARGETS = {
    "hr_policies": 30_000,
    "corporate_procedures": 50_000,
    "safety_policies": 50_000,
    "support_documentation": 50_000,
}

DEFAULT_LENGTH_BUCKET_WEIGHTS = {
    "short": 0.20,
    "medium": 0.30,
    "long": 0.35,
    "very_long": 0.15,
}

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
    "output",
    "title",
    "summary",
    "description",
    "markdown",
    "html",
)

PROMPT_INJECTION_RE = re.compile(
    "|".join(
        [
            r"\b(ignore|disregard|forget|bypass|override|discard)\b.{0,100}\b(previous|prior|above|system|developer|instruction|rule|policy)\b",
            r"\b(reveal|print|show|display|dump|return|exfiltrate|leak)\b.{0,120}\b(system prompt|developer message|hidden instruction|internal policy|secret|api key|token|password)\b",
            r"\b(act as|you are now|switch to)\b.{0,80}\b(dan|jailbreak|uncensored|developer mode|root mode)\b",
            r"игнорируй.{0,100}(предыдущ|прошл|системн|инструкц|правил|политик)",
            r"забудь.{0,100}(инструкц|правил|контекст|политик)",
            r"(покажи|раскрой|выведи|напечатай|верни|слей).{0,120}(системн|скрыт|секрет|developer|разработчик|токен|парол|ключ)",
            r"(обойди|отмени|переопредели|нарушь).{0,100}(политик|огранич|правил|инструкц|безопас)",
        ]
    ),
    flags=re.IGNORECASE | re.DOTALL,
)

RU_LETTER_RE = re.compile(r"[А-Яа-яЁё]")
EN_LETTER_RE = re.compile(r"[A-Za-z]")


@dataclass(frozen=True)
class SourceSpec:
    kind: str
    name: str
    category: str
    origin: str
    trust_category: bool
    path: str = ""
    repo_id: str = ""
    config: str = ""
    split: str = "train"
    text_column: str = ""
    language: str = ""


@dataclass
class SourceResult:
    source_name: str
    category: str
    rows: list[dict[str, Any]]
    scanned: int
    accepted: int
    complete: bool
    from_checkpoint: bool
    error: str | None = None


CATEGORY_KEYWORDS: dict[str, tuple[tuple[str, int], ...]] = {
    "hr_policies": (
        ("правила внутреннего трудового распорядка", 3),
        ("положение об оплате труда", 3),
        ("положение о премировании", 3),
        ("персональные данные работников", 3),
        ("порядок приема на работу", 3),
        ("порядок увольнения", 3),
        ("кадровая политика", 3),
        ("кадровое делопроизводство", 3),
        ("адаптация сотрудников", 2),
        ("оценка персонала", 2),
        ("трудовой договор", 2),
        ("отпуск", 1),
        ("больничный", 1),
        ("работник обязан", 1),
        ("работодатель обязан", 1),
        ("внутренний трудовой распорядок", 3),
        ("hr policy", 3),
        ("employee handbook", 3),
        ("personnel policy", 3),
        ("employment policy", 2),
    ),
    "corporate_procedures": (
        ("регламент", 2),
        ("положение о", 2),
        ("порядок согласования", 3),
        ("порядок утверждения", 3),
        ("порядок рассмотрения", 3),
        ("внутренняя процедура", 3),
        ("служебный порядок", 2),
        ("ответственные лица", 2),
        ("контроль исполнения", 3),
        ("этапы процесса", 2),
        ("процедура обработки", 3),
        ("административный регламент", 3),
        ("должностные обязанности", 2),
        ("права и обязанности", 2),
        ("corporate procedure", 3),
        ("internal procedure", 3),
        ("approval procedure", 3),
        ("procedure for approval", 3),
    ),
    "safety_policies": (
        ("охрана труда", 3),
        ("техника безопасности", 3),
        ("инструкция по охране труда", 4),
        ("пожарная безопасность", 3),
        ("безопасные условия труда", 3),
        ("инструктаж по охране труда", 4),
        ("требования безопасности", 3),
        ("средства индивидуальной защиты", 3),
        ("санпин", 3),
        ("гост", 2),
        ("производственная безопасность", 3),
        ("несчастный случай", 2),
        ("профессиональный риск", 2),
        ("occupational safety", 4),
        ("workplace safety", 3),
        ("safety policy", 3),
        ("safety instruction", 3),
    ),
    "support_documentation": (
        ("служба поддержки", 3),
        ("обращение в поддержку", 3),
        ("центр поддержки", 3),
        ("решение проблемы", 2),
        ("инструкция пользователя", 2),
        ("часто задаваемые вопросы", 3),
        ("не удается войти", 2),
        ("восстановление доступа", 2),
        ("создать заявку", 2),
        ("настройка сервиса", 2),
        ("как настроить", 2),
        ("пошаговая инструкция", 2),
        ("устранение неполадок", 3),
        ("support documentation", 3),
        ("help center", 3),
        ("troubleshooting", 3),
        ("how to configure", 2),
        ("user guide", 2),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a targeted V18 benign enrichment corpus for underfilled HR/procedure/safety/support categories."
    )
    parser.add_argument("--output-jsonl", default="v18-targeted-benign-corpus.jsonl")
    parser.add_argument("--report-json", default="v18-targeted-benign-corpus-report.json")
    parser.add_argument("--tokenizer-id", default="microsoft/mdeberta-v3-base")
    parser.add_argument("--source-spec", action="append", default=[], help="Local source spec: path|category|source_name[|origin][|trust].")
    parser.add_argument("--hf-source", action="append", default=[], help="HF source spec: repo|config|split|category|source_name[|text_column][|trust]. Use '-' for empty config.")
    parser.add_argument("--include-hf-instruction-sources", action="store_true")
    parser.add_argument("--target-hr-policies", type=int, default=DEFAULT_TARGETS["hr_policies"])
    parser.add_argument("--target-corporate-procedures", type=int, default=DEFAULT_TARGETS["corporate_procedures"])
    parser.add_argument("--target-safety-policies", type=int, default=DEFAULT_TARGETS["safety_policies"])
    parser.add_argument("--target-support-documentation", type=int, default=DEFAULT_TARGETS["support_documentation"])
    parser.add_argument("--min-category-score", type=int, default=2)
    parser.add_argument("--min-document-chars", type=int, default=80)
    parser.add_argument("--max-document-chars", type=int, default=100_000)
    parser.add_argument("--max-scan-per-source", type=int, default=500_000)
    parser.add_argument("--source-workers", type=int, default=2)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--checkpoint-scan-interval", type=int, default=2000)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--source-pool", default="external_mining_only", choices=("external_mining_only", "train", "internal_validation"))
    parser.add_argument("--allow-source-errors", action="store_true")
    parser.add_argument("--allow-underfilled", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = {
        "hr_policies": args.target_hr_policies,
        "corporate_procedures": args.target_corporate_procedures,
        "safety_policies": args.target_safety_policies,
        "support_documentation": args.target_support_documentation,
    }
    sources = parse_sources(args)
    if args.dry_run:
        print(json.dumps({"targets": targets, "sources": [source.__dict__ for source in sources]}, ensure_ascii=False, indent=2))
        return
    if not sources:
        raise ValueError("No sources configured. Use --source-spec, --hf-source, or --include-hf-instruction-sources.")

    results = list(collect_sources(args, sources))
    errors = [result.error for result in results if result.error]
    rows_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_reports = []
    for result in results:
        source_reports.append(source_result_report(result))
        if result.error:
            continue
        for row in result.rows:
            rows_by_category[row["category"]].append(row)

    selected = select_targeted_rows(rows_by_category, targets, args.allow_underfilled)
    selected = sorted(selected, key=lambda row: row["document_id"])
    write_jsonl(Path(args.output_jsonl), selected)
    report = make_report(args, targets, selected, rows_by_category, source_reports, errors)
    Path(args.report_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_jsonl": args.output_jsonl, "report_json": args.report_json, **summarize_rows(selected), "errors": errors}, ensure_ascii=False, indent=2))


def parse_sources(args: argparse.Namespace) -> list[SourceSpec]:
    sources: list[SourceSpec] = []
    for value in args.source_spec:
        parts = value.split("|")
        if len(parts) < 3:
            raise ValueError(f"Bad --source-spec: {value}")
        path, category, source_name = parts[:3]
        origin = parts[3] if len(parts) >= 4 and parts[3] else path
        trust = parse_bool(parts[4]) if len(parts) >= 5 else False
        validate_category(category)
        sources.append(SourceSpec(kind="local", name=source_name, category=category, origin=origin, trust_category=trust, path=path))
    for value in args.hf_source:
        parts = value.split("|")
        if len(parts) < 5:
            raise ValueError(f"Bad --hf-source: {value}")
        repo_id, config, split, category, source_name = parts[:5]
        text_column = parts[5] if len(parts) >= 6 else ""
        trust = parse_bool(parts[6]) if len(parts) >= 7 else False
        config = "" if config == "-" else config
        validate_category(category)
        if category != "support_documentation":
            raise ValueError("--hf-source is restricted to support_documentation. Export non-support HF/source material to local JSONL and pass it with --source-spec after review.")
        sources.append(
            SourceSpec(
                kind="hf",
                name=source_name,
                category=category,
                origin=repo_id,
                trust_category=trust,
                repo_id=repo_id,
                config=config,
                split=split,
                text_column=text_column,
            )
        )
    if args.include_hf_instruction_sources:
        sources.extend(
            [
                SourceSpec(
                    kind="hf",
                    name="hf_russian_instructions_2",
                    category="support_documentation",
                    origin="Den4ikAI/russian_instructions_2",
                    trust_category=False,
                    repo_id="Den4ikAI/russian_instructions_2",
                    split="train",
                ),
                SourceSpec(
                    kind="hf",
                    name="hf_russian_easy_instructions",
                    category="support_documentation",
                    origin="attn-signs/russian-easy-instructions",
                    trust_category=False,
                    repo_id="attn-signs/russian-easy-instructions",
                    split="train",
                ),
            ]
        )
    return sources


def validate_category(category: str) -> None:
    if category not in TARGET_CATEGORIES:
        raise ValueError(f"Unsupported category {category}. Expected one of {TARGET_CATEGORIES}.")


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "trust", "trusted"}


def collect_sources(args: argparse.Namespace, sources: list[SourceSpec]) -> Iterable[SourceResult]:
    workers = max(1, min(int(args.source_workers or 1), len(sources)))
    if workers == 1:
        for source in sources:
            try:
                yield collect_source(args, source)
            except Exception as exc:
                if not args.allow_source_errors:
                    raise
                yield SourceResult(source.name, source.category, [], 0, 0, False, False, f"{source.name}: {type(exc).__name__}: {exc}")
        return
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_source = {executor.submit(collect_source, args, source): source for source in sources}
        for future in as_completed(future_to_source):
            source = future_to_source[future]
            try:
                yield future.result()
            except Exception as exc:
                if not args.allow_source_errors:
                    raise
                yield SourceResult(source.name, source.category, [], 0, 0, False, False, f"{source.name}: {type(exc).__name__}: {exc}")


def collect_source(args: argparse.Namespace, source: SourceSpec) -> SourceResult:
    row_path, state_path = checkpoint_paths(args, source)
    existing_rows: list[dict[str, Any]] = []
    state: dict[str, Any] = {}
    compatible_checkpoint = False
    if args.resume and row_path and state_path and row_path.exists() and state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        compatible_checkpoint = checkpoint_state_compatible(state, args, source)
        if compatible_checkpoint:
            existing_rows = read_jsonl(row_path)
            if state.get("complete") and checkpoint_satisfies_request(state, args):
                return SourceResult(source.name, source.category, existing_rows, int(state.get("scanned", 0) or 0), len(existing_rows), True, True)
    if row_path and not compatible_checkpoint:
        row_path.parent.mkdir(parents=True, exist_ok=True)
        row_path.write_text("", encoding="utf-8")
        existing_rows = []

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)
    rows = list(existing_rows)
    seen_hashes = {row["text_hash"] for row in rows if row.get("text_hash")}
    resume_scanned = int(state.get("scanned", 0) or 0) if compatible_checkpoint else 0
    scanned = resume_scanned
    last_state_scan = resume_scanned
    stop_reason = "unknown"
    handle = row_path.open("a", encoding="utf-8") if row_path else None
    write_checkpoint_state(state_path, args, source, scanned, len(rows), False, "started")
    try:
        iterable = iter_source_rows(source)
        progress = tqdm(iterable, desc=f"Targeted {source.name}", unit="row", total=args.max_scan_per_source, disable=args.no_progress)
        for index, raw_row in enumerate(progress, start=1):
            if index <= resume_scanned:
                continue
            scanned = index
            if scanned > args.max_scan_per_source:
                stop_reason = "max_scan"
                break
            row_out = build_row(args, source, raw_row, tokenizer, seen_hashes)
            if row_out:
                rows.append(row_out)
                seen_hashes.add(row_out["text_hash"])
                if handle:
                    handle.write(json.dumps(row_out, ensure_ascii=False) + "\n")
                    handle.flush()
                progress.set_postfix(accepted=len(rows))
            if scanned - last_state_scan >= max(1, args.checkpoint_scan_interval):
                write_checkpoint_state(state_path, args, source, scanned, len(rows), False, "scan")
                last_state_scan = scanned
        else:
            stop_reason = "source_exhausted"
    finally:
        if handle:
            handle.close()
    complete = stop_reason in {"max_scan", "source_exhausted"}
    write_checkpoint_state(state_path, args, source, scanned, len(rows), complete, stop_reason)
    return SourceResult(source.name, source.category, rows, scanned, len(rows), complete, bool(existing_rows))


def iter_source_rows(source: SourceSpec) -> Iterator[Any]:
    if source.kind == "hf":
        kwargs: dict[str, Any] = {"split": source.split, "streaming": True}
        if source.config:
            yield from load_dataset(source.repo_id, source.config, **kwargs)
        else:
            yield from load_dataset(source.repo_id, **kwargs)
        return
    path = Path(source.path)
    if path.is_dir():
        for file_path in sorted(path.rglob("*")):
            if file_path.is_file() and file_path.suffix.lower() in {".txt", ".md", ".html", ".htm", ".json", ".jsonl"}:
                yield from iter_local_file(file_path)
        return
    yield from iter_local_file(path)


def iter_local_file(path: Path) -> Iterator[Any]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    yield {"text": line, "source_path": str(path)}
        return
    if suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, list):
            yield from value
        else:
            yield value
        return
    yield {"text": path.read_text(encoding="utf-8", errors="ignore"), "source_path": str(path)}


def build_row(
    args: argparse.Namespace,
    source: SourceSpec,
    raw_row: Any,
    tokenizer: AutoTokenizer,
    seen_hashes: set[str],
) -> dict[str, Any] | None:
    text = extract_text(raw_row, source.text_column)
    prepared = prepare_text(text, args)
    if not prepared or PROMPT_INJECTION_RE.search(prepared):
        return None
    text_hash = stable_hash(prepared)
    if text_hash in seen_hashes:
        return None
    category, score, hits = choose_category(prepared, source)
    if not category:
        return None
    if score < args.min_category_score and not source.trust_category:
        return None
    token_count = token_length(prepared, tokenizer)
    window_count = production_window_count(token_count)
    language = source.language or infer_language(prepared)
    source_row_id = extract_source_row_id(raw_row)
    document_id = f"targeted:{category}:{source.name}:{source_row_id or text_hash}"
    return {
        "document_id": document_id,
        "document_label": "not_prompt_injection",
        "label": "not_prompt_injection",
        "category": category,
        "source_name": source.name,
        "source_origin": source.origin,
        "source_path": source.path or source.repo_id,
        "source_pool": args.source_pool,
        "source_pool_assignment": args.source_pool,
        "source_row_id": source_row_id,
        "language": language,
        "text_length": len(prepared),
        "text_token_length": token_count,
        "production_window_count": window_count,
        "length_bucket": length_bucket_for_window_count(window_count),
        "category_score": score,
        "keyword_hits": hits,
        "text_hash": text_hash,
        "normalized_text_hash": stable_hash(normalize_text(prepared).lower()),
        "text": prepared,
    }


def choose_category(text: str, source: SourceSpec) -> tuple[str | None, int, list[str]]:
    if source.category:
        score, hits = category_score(text, source.category)
        return source.category, score, hits
    scored = []
    for category in TARGET_CATEGORIES:
        score, hits = category_score(text, category)
        scored.append((score, category, hits))
    scored.sort(reverse=True)
    score, category, hits = scored[0]
    return (category, score, hits) if score else (None, 0, [])


def category_score(text: str, category: str) -> tuple[int, list[str]]:
    lower = normalize_text(text).lower()
    score = 0
    hits: list[str] = []
    for keyword, weight in CATEGORY_KEYWORDS[category]:
        if keyword.lower() in lower:
            score += weight
            hits.append(keyword)
    return score, hits[:12]


def select_targeted_rows(
    rows_by_category: dict[str, list[dict[str, Any]]],
    targets: dict[str, int],
    allow_underfilled: bool,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_hashes: set[str] = set()
    for category, target in targets.items():
        rows = rows_by_category.get(category, [])
        by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_bucket[row["length_bucket"]].append(row)
        category_selected: list[dict[str, Any]] = []
        category_hashes: set[str] = set()
        quotas = scaled_targets(DEFAULT_LENGTH_BUCKET_WEIGHTS, target)
        for bucket, quota in quotas.items():
            for row in by_bucket.get(bucket, []):
                row_hash = row["text_hash"]
                if row_hash in selected_hashes or row_hash in category_hashes:
                    continue
                category_selected.append(row)
                category_hashes.add(row_hash)
                if len([item for item in category_selected if item["length_bucket"] == bucket]) >= quota:
                    break
        if len(category_selected) < target:
            for row in rows:
                row_hash = row["text_hash"]
                if row_hash in selected_hashes or row_hash in category_hashes:
                    continue
                category_selected.append(row)
                category_hashes.add(row_hash)
                if len(category_selected) >= target:
                    break
        if len(category_selected) < target and not allow_underfilled:
            raise ValueError(f"{category} underfilled: {len(category_selected):,}/{target:,}.")
        chosen = category_selected[:target]
        selected.extend(chosen)
        selected_hashes.update(row["text_hash"] for row in chosen)
    return selected


def scaled_targets(weights: dict[str, float], total: int) -> dict[str, int]:
    exact = {key: total * value for key, value in weights.items()}
    targets = {key: int(value) for key, value in exact.items()}
    remainder = total - sum(targets.values())
    for key, _ in sorted(exact.items(), key=lambda item: item[1] - int(item[1]), reverse=True)[:remainder]:
        targets[key] += 1
    return targets


def prepare_text(text: str, args: argparse.Namespace) -> str:
    text = strip_markup(normalize_text(text))
    if len(text) < args.min_document_chars:
        return ""
    if args.max_document_chars and len(text) > args.max_document_chars:
        return ""
    if looks_corrupted(text):
        return ""
    return text


def strip_markup(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"<script\b.*?</script>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<style\b.*?</style>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    return normalize_text(text)


def extract_text(row: Any, text_column: str = "") -> str:
    if isinstance(row, str):
        return normalize_text(row)
    if isinstance(row, list):
        return normalize_text(" ".join(extract_text(item, text_column) for item in row))
    if not isinstance(row, dict):
        return ""
    if text_column and text_column in row:
        return normalize_text(row[text_column])
    for key in TEXT_COLUMNS:
        if key in row:
            text = normalize_text(row[key])
            if text:
                return text
    values = []
    for value in row.values():
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, dict):
            nested = extract_text(value, text_column)
            if nested:
                values.append(nested)
    return normalize_text(" ".join(values))


def extract_source_row_id(row: Any) -> str:
    if not isinstance(row, dict):
        return ""
    for key in ("id", "doc_id", "document_id", "url", "uri", "source", "path", "sha"):
        value = normalize_text(row.get(key))
        if value:
            return stable_hash(value)
    return ""


def checkpoint_dir_for_args(args: argparse.Namespace) -> Path:
    if args.checkpoint_dir:
        return Path(args.checkpoint_dir)
    output = Path(args.output_jsonl)
    return output.with_suffix("").parent / f"{output.with_suffix('').name}-checkpoints"


def checkpoint_paths(args: argparse.Namespace, source: SourceSpec) -> tuple[Path | None, Path | None]:
    if args.no_checkpoint:
        return None, None
    directory = checkpoint_dir_for_args(args)
    name = f"source-{safe_filename(source.name)}-{source_identity_hash(source)}"
    return directory / f"{name}.jsonl", directory / f"{name}.state.json"


def checkpoint_state_compatible(state: dict[str, Any], args: argparse.Namespace, source: SourceSpec) -> bool:
    return (
        state.get("name") == source.name
        and state.get("kind") == source.kind
        and state.get("category") == source.category
        and state.get("origin") == source.origin
        and state.get("path") == source.path
        and state.get("repo_id") == source.repo_id
        and state.get("config") == source.config
        and state.get("split") == source.split
        and state.get("text_column") == source.text_column
        and bool(state.get("trust_category")) == source.trust_category
        and state.get("source_identity_hash") == source_identity_hash(source)
        and state.get("tokenizer_id") == args.tokenizer_id
        and int(state.get("min_document_chars", -1)) == args.min_document_chars
        and int(state.get("max_document_chars", -1)) == args.max_document_chars
        and int(state.get("min_category_score", -1)) == args.min_category_score
        and state.get("source_pool") == args.source_pool
    )


def checkpoint_satisfies_request(state: dict[str, Any], args: argparse.Namespace) -> bool:
    stop_reason = str(state.get("stop_reason") or "")
    previous_max_scan = int(state.get("max_scan_per_source", 0) or 0)
    return stop_reason in {"max_scan", "source_exhausted"} and previous_max_scan >= args.max_scan_per_source


def write_checkpoint_state(
    path: Path | None,
    args: argparse.Namespace,
    source: SourceSpec,
    scanned: int,
    accepted: int,
    complete: bool,
    stop_reason: str,
) -> None:
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "name": source.name,
        "kind": source.kind,
        "category": source.category,
        "origin": source.origin,
        "path": source.path,
        "repo_id": source.repo_id,
        "config": source.config,
        "split": source.split,
        "text_column": source.text_column,
        "trust_category": source.trust_category,
        "source_identity_hash": source_identity_hash(source),
        "tokenizer_id": args.tokenizer_id,
        "min_document_chars": args.min_document_chars,
        "max_document_chars": args.max_document_chars,
        "min_category_score": args.min_category_score,
        "source_pool": args.source_pool,
        "max_scan_per_source": args.max_scan_per_source,
        "scanned": scanned,
        "accepted": accepted,
        "complete": complete,
        "stop_reason": stop_reason,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_report(
    args: argparse.Namespace,
    targets: dict[str, int],
    selected: list[dict[str, Any]],
    rows_by_category: dict[str, list[dict[str, Any]]],
    source_reports: list[dict[str, Any]],
    errors: list[str | None],
) -> dict[str, Any]:
    return {
        "config": vars(args),
        "checkpoint_dir": "" if args.no_checkpoint else str(checkpoint_dir_for_args(args)),
        "targets": targets,
        "selected": summarize_rows(selected),
        "candidate_counts": {category: len(rows) for category, rows in sorted(rows_by_category.items())},
        "source_reports": source_reports,
        "errors": [error for error in errors if error],
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    token_total = sum(int(row["text_token_length"]) for row in rows)
    window_total = sum(int(row["production_window_count"]) for row in rows)
    return {
        "documents": len(rows),
        "categories": count_dict(Counter(row["category"] for row in rows)),
        "sources": count_dict(Counter(row["source_name"] for row in rows)),
        "languages": count_dict(Counter(row["language"] for row in rows)),
        "length_buckets": count_dict(Counter(row["length_bucket"] for row in rows)),
        "token_count": {"total": token_total, "avg": round(token_total / len(rows), 2) if rows else 0},
        "window_count": {"total": window_total, "avg": round(window_total / len(rows), 2) if rows else 0},
    }


def source_result_report(result: SourceResult) -> dict[str, Any]:
    return {
        "source_name": result.source_name,
        "category": result.category,
        "rows": len(result.rows),
        "scanned": result.scanned,
        "accepted": result.accepted,
        "complete": result.complete,
        "from_checkpoint": result.from_checkpoint,
        "error": result.error,
    }


def count_dict(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in counter.most_common()}


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
    has_ru = bool(RU_LETTER_RE.search(text))
    has_en = bool(EN_LETTER_RE.search(text))
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
    text = str(value).replace(chr(0), " ")
    return re.sub(r"\s+", " ", text).strip()


def looks_corrupted(text: str) -> bool:
    bad_chars = (chr(0xFFFD), chr(0x00D0), chr(0x00D1), chr(0x00E2))
    return sum(text.count(char) for char in bad_chars) >= 4


def stable_hash(value: str) -> str:
    return hashlib.sha1(normalize_text(value).lower().encode("utf-8")).hexdigest()[:20]


def safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return cleaned or "source"


def source_identity_hash(source: SourceSpec) -> str:
    payload = {
        "kind": source.kind,
        "name": source.name,
        "category": source.category,
        "origin": source.origin,
        "trust_category": source.trust_category,
        "path": source.path,
        "repo_id": source.repo_id,
        "config": source.config,
        "split": source.split,
        "text_column": source.text_column,
        "language": source.language,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:12]


if __name__ == "__main__":
    main()
