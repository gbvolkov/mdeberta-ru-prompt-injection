# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

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

SKIP_URL_SUFFIXES = (
    ".7z",
    ".avi",
    ".bmp",
    ".css",
    ".csv",
    ".doc",
    ".docx",
    ".gif",
    ".gz",
    ".ico",
    ".jpeg",
    ".jpg",
    ".js",
    ".mp3",
    ".mp4",
    ".pdf",
    ".png",
    ".ppt",
    ".pptx",
    ".rar",
    ".rss",
    ".svg",
    ".tar",
    ".tgz",
    ".webp",
    ".xls",
    ".xlsx",
    ".xml",
    ".zip",
)

RU_LETTER_RE = re.compile(r"[А-Яа-яЁё]")
EN_LETTER_RE = re.compile(r"[A-Za-z]")
MOJIBAKE_MARKERS = tuple(chr(value) for value in (0x00D0, 0x00D1, 0x00C2, 0x00E2))

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

CATEGORY_KEYWORDS: dict[str, tuple[tuple[str, int], ...]] = {
    "hr_policies": (
        ("правила внутреннего трудового распорядка", 4),
        ("кадровая политика", 4),
        ("кадровое делопроизводство", 4),
        ("персональные данные работников", 4),
        ("положение об оплате труда", 4),
        ("положение о премировании", 4),
        ("порядок приема на работу", 4),
        ("порядок увольнения", 4),
        ("трудовой договор", 3),
        ("оценка персонала", 3),
        ("адаптация сотрудников", 3),
        ("отпуск", 2),
        ("больничный", 2),
        ("работник обязан", 2),
        ("работодатель обязан", 2),
        ("employee handbook", 4),
        ("hr policy", 4),
        ("personnel policy", 4),
        ("employment policy", 3),
    ),
    "corporate_procedures": (
        ("регламент", 3),
        ("административный регламент", 4),
        ("положение о", 2),
        ("порядок согласования", 4),
        ("порядок утверждения", 4),
        ("порядок рассмотрения", 4),
        ("внутренняя процедура", 4),
        ("служебный порядок", 3),
        ("ответственные лица", 3),
        ("контроль исполнения", 4),
        ("этапы процесса", 3),
        ("процедура обработки", 4),
        ("права и обязанности", 3),
        ("corporate procedure", 4),
        ("internal procedure", 4),
        ("approval procedure", 4),
        ("procedure for approval", 4),
    ),
    "safety_policies": (
        ("охрана труда", 4),
        ("техника безопасности", 4),
        ("инструкция по охране труда", 5),
        ("пожарная безопасность", 4),
        ("безопасные условия труда", 4),
        ("инструктаж по охране труда", 5),
        ("требования безопасности", 4),
        ("средства индивидуальной защиты", 4),
        ("производственная безопасность", 4),
        ("несчастный случай", 3),
        ("профессиональный риск", 3),
        ("санпин", 3),
        ("гост", 2),
        ("occupational safety", 4),
        ("workplace safety", 4),
        ("safety policy", 4),
        ("safety instruction", 4),
    ),
    "support_documentation": (
        ("служба поддержки", 4),
        ("обращение в поддержку", 4),
        ("центр поддержки", 4),
        ("решение проблемы", 3),
        ("инструкция пользователя", 3),
        ("часто задаваемые вопросы", 4),
        ("не удается войти", 3),
        ("восстановление доступа", 3),
        ("создать заявку", 3),
        ("настройка сервиса", 3),
        ("как настроить", 3),
        ("пошаговая инструкция", 3),
        ("устранение неполадок", 4),
        ("support documentation", 4),
        ("help center", 4),
        ("troubleshooting", 4),
        ("how to configure", 3),
        ("user guide", 3),
    ),
}


@dataclass(frozen=True)
class SourceSpec:
    name: str
    kind: str
    categories: tuple[str, ...]
    seeds: tuple[str, ...] = ()
    allowed_domains: tuple[str, ...] = ()
    allowed_path_prefixes: tuple[str, ...] = ()
    trust_category: bool = False
    repo_id: str = ""
    config: str = ""
    split: str = "train"
    text_column: str = ""
    hh_queries: tuple[str, ...] = ()


@dataclass
class FetchResult:
    source_name: str
    kind: str
    rows: list[dict[str, Any]]
    scanned: int
    fetched: int
    accepted: int
    rejected: Counter[str]
    rejected_samples: dict[str, list[str]]
    status_counts: Counter[str]
    sample_urls: list[str]
    errors: list[str]


DEFAULT_SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec(
        name="geolibrary",
        kind="web",
        categories=("safety_policies",),
        seeds=(
            "https://www.geolibrary.org/",
            "http://www.geolibrary.org/",
            "https://geolibrary.org/",
            "http://geolibrary.org/",
        ),
        allowed_domains=("www.geolibrary.org", "geolibrary.org"),
        trust_category=True,
    ),
    SourceSpec(
        name="russian_legislative_corpus_arxiv",
        kind="web",
        categories=("corporate_procedures", "safety_policies"),
        seeds=(
            "https://arxiv.org/abs/2406.04855",
            "https://arxiv.org/html/2406.04855",
        ),
        allowed_domains=("arxiv.org",),
        allowed_path_prefixes=("/abs/2406.04855", "/html/2406.04855"),
        trust_category=False,
    ),
    SourceSpec(
        name="russian_legislative_corpus_ruslawod",
        kind="hf",
        categories=("hr_policies", "corporate_procedures", "safety_policies"),
        repo_id="irlspbru/RusLawOD",
        split="train",
        trust_category=False,
    ),
    SourceSpec(
        name="profstandards_registry",
        kind="web",
        categories=("hr_policies", "corporate_procedures"),
        seeds=("https://profstandart.rosmintrud.ru/",),
        allowed_domains=("profstandart.rosmintrud.ru",),
        trust_category=False,
    ),
    SourceSpec(
        name="yandex_cloud_docs",
        kind="web",
        categories=("support_documentation",),
        seeds=(
            "https://yandex.cloud/ru/docs/",
            "https://yandex.cloud/ru/docs/compute/",
            "https://yandex.cloud/ru/docs/iam/concepts/access-control/roles",
        ),
        allowed_domains=("yandex.cloud",),
        allowed_path_prefixes=("/ru/docs",),
        trust_category=True,
    ),
    SourceSpec(
        name="bitrix24_helpdesk",
        kind="web",
        categories=("support_documentation", "corporate_procedures"),
        seeds=("https://helpdesk.bitrix24.ru/",),
        allowed_domains=("helpdesk.bitrix24.ru",),
        trust_category=True,
    ),
    SourceSpec(
        name="microsoft_learn_ru",
        kind="web",
        categories=("support_documentation", "safety_policies"),
        seeds=(
            "https://learn.microsoft.com/ru-ru/",
            "https://learn.microsoft.com/ru-ru/azure/security/fundamentals/overview",
            "https://learn.microsoft.com/ru-ru/azure/architecture/framework/security/overview",
        ),
        allowed_domains=("learn.microsoft.com",),
        allowed_path_prefixes=("/ru-ru/",),
        trust_category=True,
    ),
    SourceSpec(
        name="hf_russian_instructions_2",
        kind="hf",
        categories=("support_documentation",),
        repo_id="Den4ikAI/russian_instructions_2",
        split="train",
        trust_category=False,
    ),
    SourceSpec(
        name="hf_russian_easy_instructions",
        kind="hf",
        categories=("support_documentation",),
        repo_id="attn-signs/russian-easy-instructions",
        split="train",
        trust_category=False,
    ),
    SourceSpec(
        name="hh_api_vacancies",
        kind="hh_api",
        categories=("hr_policies",),
        hh_queries=(
            "кадровая политика",
            "специалист по кадрам",
            "менеджер по персоналу",
            "охрана труда",
            "специалист по охране труда",
        ),
        trust_category=False,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch V18 targeted benign source documents directly from every configured source family."
    )
    parser.add_argument("--output-dir", default="v18-direct-source-documents")
    parser.add_argument("--combined-jsonl", default=None)
    parser.add_argument("--report-json", default=None)
    parser.add_argument("--tokenizer-id", default="microsoft/mdeberta-v3-base")
    parser.add_argument("--source", action="append", default=[], help="Source name to include. Defaults to all sources.")
    parser.add_argument("--exclude-source", action="append", default=[])
    parser.add_argument("--max-pages-per-source", type=int, default=250)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-documents-per-source", type=int, default=5000)
    parser.add_argument("--max-hf-scan-per-source", type=int, default=50_000)
    parser.add_argument("--early-stop-empty-hf-scan", type=int, default=10_000)
    parser.add_argument("--hh-pages-per-query", type=int, default=3)
    parser.add_argument("--target-hr-policies", type=int, default=0)
    parser.add_argument("--target-corporate-procedures", type=int, default=0)
    parser.add_argument("--target-safety-policies", type=int, default=0)
    parser.add_argument("--target-support-documentation", type=int, default=0)
    parser.add_argument("--target-unit", choices=("documents", "windows"), default="windows")
    parser.add_argument("--min-document-chars", type=int, default=400)
    parser.add_argument("--max-document-chars", type=int, default=100_000)
    parser.add_argument("--min-category-score", type=int, default=2)
    parser.add_argument("--source-workers", type=int, default=4)
    parser.add_argument("--request-timeout", type=int, default=25)
    parser.add_argument("--request-delay-seconds", type=float, default=0.2)
    parser.add_argument("--user-agent", default="mdeberta-v18-source-prep/1.0 (+dataset preparation)")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument(
        "--source-pool",
        default="external_mining_only",
        choices=("external_mining_only", "train", "internal_validation"),
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--allow-source-errors", action="store_true")
    parser.add_argument("--allow-underfilled", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    configure_stdout()
    args = parse_args()
    output_dir = Path(args.output_dir)
    combined_jsonl = Path(args.combined_jsonl) if args.combined_jsonl else output_dir / "v18-direct-source-documents.jsonl"
    report_json = Path(args.report_json) if args.report_json else output_dir / "v18-direct-source-documents-report.json"

    sources = selected_sources(args)
    if args.dry_run:
        print(json.dumps({"sources": [source_to_dict(source) for source in sources]}, ensure_ascii=False, indent=2))
        return

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[FetchResult] = []
    pending_sources: list[SourceSpec] = []
    if args.resume:
        for source in sources:
            cached = load_source_checkpoint(args, source)
            if cached:
                results.append(cached)
            else:
                pending_sources.append(source)
    else:
        pending_sources = sources
    with ThreadPoolExecutor(max_workers=max(1, args.source_workers)) as executor:
        futures = {
            executor.submit(collect_source, args, tokenizer, source): source
            for source in pending_sources
        }
        iterator = as_completed(futures)
        if not args.no_progress:
            iterator = tqdm(iterator, total=len(futures), desc="Collecting direct sources")
        for future in iterator:
            source = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                if not args.allow_source_errors:
                    raise
                result = FetchResult(
                    source_name=source.name,
                    kind=source.kind,
                    rows=[],
                    scanned=0,
                    fetched=0,
                    accepted=0,
                    rejected=Counter(),
                    rejected_samples={},
                    status_counts=Counter(),
                    sample_urls=[],
                    errors=[f"{type(exc).__name__}: {exc}"],
                )
            write_source_checkpoint(args, source, result)
            results.append(result)

    all_rows = dedupe_rows([row for result in results for row in result.rows])
    targets = category_targets(args)
    selected_rows, target_report = select_target_rows(all_rows, targets, args.target_unit)
    if targets:
        output_rows = selected_rows
    else:
        output_rows = all_rows
    write_outputs(output_dir, combined_jsonl, output_rows)

    report = make_report(args, sources, results, output_rows, all_rows, target_report)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    if target_report.get("underfilled") and not args.allow_underfilled:
        raise ValueError(f"Underfilled requested target buckets: {target_report['underfilled']}")


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass


def selected_sources(args: argparse.Namespace) -> list[SourceSpec]:
    included = set(args.source)
    excluded = set(args.exclude_source)
    sources = []
    known = {source.name for source in DEFAULT_SOURCES}
    unknown = (included | excluded) - known
    if unknown:
        raise ValueError(f"Unknown source name(s): {sorted(unknown)}. Known sources: {sorted(known)}")
    for source in DEFAULT_SOURCES:
        if included and source.name not in included:
            continue
        if source.name in excluded:
            continue
        sources.append(source)
    if not sources:
        raise ValueError("No sources selected.")
    return sources


def collect_source(args: argparse.Namespace, tokenizer: Any, source: SourceSpec) -> FetchResult:
    if source.kind == "web":
        return collect_web_source(args, tokenizer, source)
    if source.kind == "hf":
        return collect_hf_source(args, tokenizer, source)
    if source.kind == "hh_api":
        return collect_hh_api_source(args, tokenizer, source)
    raise ValueError(f"Unsupported source kind: {source.kind}")


def collect_web_source(args: argparse.Namespace, tokenizer: Any, source: SourceSpec) -> FetchResult:
    rows: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    rejected_samples: dict[str, list[str]] = {}
    status_counts: Counter[str] = Counter()
    errors: list[str] = []
    sample_urls: list[str] = []
    visited: set[str] = set()
    queue: deque[tuple[str, int]] = deque((url, 0) for url in source.seeds)
    scanned = 0
    fetched = 0

    while queue and scanned < args.max_pages_per_source and len(rows) < args.max_documents_per_source:
        url, depth = queue.popleft()
        url = canonical_url(url)
        if not url or url in visited or should_skip_url(url):
            continue
        if not allowed_url(url, source.allowed_domains, source.allowed_path_prefixes):
            continue
        visited.add(url)
        scanned += 1
        time.sleep(max(0.0, args.request_delay_seconds))

        try:
            body, status, final_url = fetch_url(url, args)
        except Exception as exc:
            errors.append(f"{url}: {type(exc).__name__}: {exc}")
            status_counts["error"] += 1
            continue

        fetched += 1
        status_counts[str(status)] += 1
        final_url = canonical_url(final_url or url)
        if len(sample_urls) < 10:
            sample_urls.append(final_url)
        if not allowed_url(final_url, source.allowed_domains, source.allowed_path_prefixes):
            record_rejection(rejected, rejected_samples, "redirected_outside_allowed_scope", f"{url} -> {final_url}")
            continue
        title = extract_title(body)
        if looks_blocked(body, title):
            record_rejection(rejected, rejected_samples, "blocked_or_antibot", final_url)
        else:
            text = html_to_text(body)
            row = build_document_row(args, tokenizer, source, text, final_url, title, scanned)
            if row:
                rows.append(row)
            else:
                record_rejection(rejected, rejected_samples, "not_accepted", final_url)

        if depth < args.max_depth:
            for link in extract_links(final_url, body):
                link = canonical_url(link)
                if (
                    link
                    and link not in visited
                    and allowed_url(link, source.allowed_domains, source.allowed_path_prefixes)
                    and not should_skip_url(link)
                ):
                    queue.append((link, depth + 1))

    return FetchResult(
        source_name=source.name,
        kind=source.kind,
        rows=rows,
        scanned=scanned,
        fetched=fetched,
        accepted=len(rows),
        rejected=rejected,
        rejected_samples=rejected_samples,
        status_counts=status_counts,
        sample_urls=sample_urls,
        errors=errors[:25],
    )


def collect_hf_source(args: argparse.Namespace, tokenizer: Any, source: SourceSpec) -> FetchResult:
    rows: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    rejected_samples: dict[str, list[str]] = {}
    errors: list[str] = []
    scanned = 0
    dataset = load_dataset(source.repo_id, source.config or None, split=source.split, streaming=True)
    iterator: Iterable[Any] = dataset
    for raw in iterator:
        scanned += 1
        if scanned > args.max_hf_scan_per_source or len(rows) >= args.max_documents_per_source:
            break
        if args.early_stop_empty_hf_scan and scanned > args.early_stop_empty_hf_scan and not rows:
            break
        text = extract_text(raw, source.text_column)
        source_id = extract_source_row_id(raw) or str(scanned)
        row = build_document_row(args, tokenizer, source, text, f"hf://{source.repo_id}/{source.split}/{source_id}", "", scanned)
        if row:
            rows.append(row)
        else:
            if len(rejected_samples.setdefault("not_accepted", [])) < 10:
                rejected_samples["not_accepted"].append(f"hf://{source.repo_id}/{source.split}/{source_id}")
            rejected["not_accepted"] += 1
    return FetchResult(
        source_name=source.name,
        kind=source.kind,
        rows=rows,
        scanned=scanned,
        fetched=scanned,
        accepted=len(rows),
        rejected=rejected,
        rejected_samples=rejected_samples,
        status_counts=Counter({"streamed": scanned}),
        sample_urls=[f"hf://{source.repo_id}/{source.split}"],
        errors=errors,
    )


def collect_hh_api_source(args: argparse.Namespace, tokenizer: Any, source: SourceSpec) -> FetchResult:
    rows: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    rejected_samples: dict[str, list[str]] = {}
    status_counts: Counter[str] = Counter()
    errors: list[str] = []
    sample_urls: list[str] = []
    scanned = 0
    fetched = 0
    for query in source.hh_queries:
        for page in range(max(0, args.hh_pages_per_query)):
            params = urllib.parse.urlencode({"text": query, "per_page": 100, "page": page})
            url = f"https://api.hh.ru/vacancies?{params}"
            try:
                body, status, _ = fetch_url(url, args, accept_json=True)
            except Exception as exc:
                errors.append(f"{url}: {type(exc).__name__}: {exc}")
                status_counts["error"] += 1
                continue
            fetched += 1
            status_counts[str(status)] += 1
            if len(sample_urls) < 10:
                sample_urls.append(url)
            try:
                payload = json.loads(body)
            except json.JSONDecodeError as exc:
                errors.append(f"{url}: JSONDecodeError: {exc}")
                continue
            for item in payload.get("items", []):
                scanned += 1
                text = hh_item_text(item)
                row = build_document_row(args, tokenizer, source, text, item.get("alternate_url") or url, item.get("name") or "", scanned)
                if row:
                    rows.append(row)
                    if len(rows) >= args.max_documents_per_source:
                        break
                else:
                    record_rejection(rejected, rejected_samples, "not_accepted", item.get("alternate_url") or url)
            if len(rows) >= args.max_documents_per_source:
                break
        if len(rows) >= args.max_documents_per_source:
            break

    return FetchResult(
        source_name=source.name,
        kind=source.kind,
        rows=rows,
        scanned=scanned,
        fetched=fetched,
        accepted=len(rows),
        rejected=rejected,
        rejected_samples=rejected_samples,
        status_counts=status_counts,
        sample_urls=sample_urls,
        errors=errors[:25],
    )


def fetch_url(url: str, args: argparse.Namespace, accept_json: bool = False) -> tuple[str, int, str]:
    headers = {
        "User-Agent": args.user_agent,
        "Accept": "application/json" if accept_json else "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ru,en;q=0.8",
    }
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=args.request_timeout) as response:
        data = response.read()
        charset = response.headers.get_content_charset()
        text = decode_bytes(data, charset)
        status = int(getattr(response, "status", 200) or 200)
        final_url = response.geturl()
    return text, status, final_url


def decode_bytes(data: bytes, charset: str | None) -> str:
    encodings = [sniff_html_charset(data), "utf-8", charset, "windows-1251", "cp1251", "latin-1"]
    tried: set[str] = set()
    for encoding in encodings:
        if not encoding or encoding.lower() in tried:
            continue
        tried.add(encoding.lower())
        try:
            return repair_mojibake(data.decode(encoding))
        except UnicodeDecodeError:
            continue
    return repair_mojibake(data.decode("utf-8", errors="replace"))


def sniff_html_charset(data: bytes) -> str:
    head = data[:4096].decode("ascii", errors="ignore")
    match = re.search(r"charset\s*=\s*['\"]?([A-Za-z0-9_.-]+)", head, flags=re.IGNORECASE)
    return match.group(1) if match else ""


def repair_mojibake(text: str) -> str:
    if mojibake_score(text) < 4:
        return text
    candidates = [text]
    try:
        candidates.append(text.encode("latin-1").decode("utf-8"))
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    return min(candidates, key=lambda value: (mojibake_score(value), -cyrillic_count(value)))


def mojibake_score(text: str) -> int:
    return sum(text.count(marker) for marker in MOJIBAKE_MARKERS)


def cyrillic_count(text: str) -> int:
    return sum(("А" <= char <= "я") or char in "Ёё" for char in text)


def build_document_row(
    args: argparse.Namespace,
    tokenizer: Any,
    source: SourceSpec,
    raw_text: str,
    source_url: str,
    title: str,
    ordinal: int,
) -> dict[str, Any] | None:
    text = prepare_text(raw_text, args)
    if not text:
        return None
    if PROMPT_INJECTION_RE.search(text):
        return None
    category, score = choose_category(text, source, args.min_category_score)
    if not category:
        return None
    token_count = token_length(text, tokenizer)
    window_count = production_window_count(token_count)
    document_id = f"v18_direct_{source.name}_{stable_hash(source_url + '|' + str(ordinal) + '|' + text)[:16]}"
    normalized_hash = normalized_text_hash(text)
    row = {
        "document_id": document_id,
        "source_document_id": document_id,
        "document_label": "benign",
        "label": "not_prompt_injection",
        "category": category,
        "category_score": score,
        "source_name": source.name,
        "source_origin": source_url,
        "source_path": source_url,
        "source_pool": args.source_pool,
        "source_pool_assignment": args.source_pool,
        "language": infer_language(text),
        "title": normalize_text(title),
        "text": text,
        "text_hash": stable_hash(text),
        "normalized_text_hash": normalized_hash,
        "dedupe_cluster_id": normalized_hash[:16],
        "text_token_length": token_count,
        "production_window_count": window_count,
        "length_bucket": length_bucket_for_window_count(window_count),
    }
    return row


def choose_category(text: str, source: SourceSpec, min_category_score: int) -> tuple[str, int]:
    scores = {
        category: category_score(text, category)
        for category in source.categories
    }
    category, score = max(scores.items(), key=lambda item: item[1])
    if score >= min_category_score:
        return category, score
    if source.trust_category and source.categories:
        return source.categories[0], score
    return "", score


def category_score(text: str, category: str) -> int:
    lowered = text.lower()
    score = 0
    for keyword, weight in CATEGORY_KEYWORDS[category]:
        if keyword.lower() in lowered:
            score += weight
    return score


def prepare_text(text: str, args: argparse.Namespace) -> str:
    text = normalize_text(text)
    if len(text) < args.min_document_chars:
        return ""
    if args.max_document_chars and len(text) > args.max_document_chars:
        text = text[: args.max_document_chars]
    if looks_corrupted(text):
        return ""
    return text


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


def hh_item_text(item: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("name", "description"):
        if item.get(key):
            parts.append(html_to_text(str(item[key])))
    snippet = item.get("snippet") or {}
    if isinstance(snippet, dict):
        for key in ("requirement", "responsibility"):
            if snippet.get(key):
                parts.append(html_to_text(str(snippet[key])))
    employer = item.get("employer") or {}
    if isinstance(employer, dict) and employer.get("name"):
        parts.append(str(employer["name"]))
    return normalize_text(" ".join(parts))


def html_to_text(value: str) -> str:
    text = html.unescape(value)
    text = re.sub(r"<script\b.*?</script>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<style\b.*?</style>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<noscript\b.*?</noscript>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<svg\b.*?</svg>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<(br|p|li|tr|td|th|h[1-6])\b[^>]*>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    return normalize_text(text)


def extract_title(value: str) -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", value, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    return normalize_text(html_to_text(match.group(1)))


def extract_links(base_url: str, value: str) -> list[str]:
    links = []
    for match in re.finditer(r"""href\s*=\s*["']([^"']+)["']""", value, flags=re.IGNORECASE):
        href = html.unescape(match.group(1).strip())
        if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue
        links.append(urllib.parse.urljoin(base_url, href))
    return links


def looks_blocked(body: str, title: str) -> bool:
    sample = f"{title} {body[:4000]}".lower()
    markers = (
        "are you not a robot",
        "captcha",
        "access denied",
        "доступ ограничен",
        "проверка безопасности",
        "подтвердите, что вы не робот",
    )
    return any(marker in sample for marker in markers)


def should_skip_url(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return True
    lowered_path = parsed.path.lower()
    if any(lowered_path.endswith(suffix) for suffix in SKIP_URL_SUFFIXES):
        return True
    if any(part in lowered_path for part in ("/login", "/signin", "/auth", "/register", "/cart")):
        return True
    return False


def allowed_url(url: str, domains: tuple[str, ...], path_prefixes: tuple[str, ...] = ()) -> bool:
    if not domains:
        return True
    parsed = urllib.parse.urlparse(url)
    host = parsed.netloc.lower()
    host = host.split("@")[-1].split(":")[0]
    if not any(host == domain or host.endswith("." + domain) for domain in domains):
        return False
    if path_prefixes:
        path = parsed.path
        return any(path == prefix or path.startswith(prefix.rstrip("/") + "/") for prefix in path_prefixes)
    return True


def canonical_url(url: str) -> str:
    if not url:
        return ""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return ""
    query = parsed.query
    blocked_query_keys = ("utm_", "fbclid", "gclid", "yclid")
    if query:
        pairs = urllib.parse.parse_qsl(query, keep_blank_values=True)
        pairs = [(key, value) for key, value in pairs if not any(key.lower().startswith(prefix) for prefix in blocked_query_keys)]
        query = urllib.parse.urlencode(pairs)
    normalized = parsed._replace(fragment="", query=query)
    return urllib.parse.urlunparse(normalized)


def dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    seen: set[str] = set()
    for row in rows:
        key = row["normalized_text_hash"]
        if key in seen:
            continue
        seen.add(key)
        selected.append(row)
    return selected


def category_targets(args: argparse.Namespace) -> dict[str, int]:
    return {
        category: value
        for category, value in {
            "hr_policies": args.target_hr_policies,
            "corporate_procedures": args.target_corporate_procedures,
            "safety_policies": args.target_safety_policies,
            "support_documentation": args.target_support_documentation,
        }.items()
        if value > 0
    }


def select_target_rows(
    rows: list[dict[str, Any]],
    targets: dict[str, int],
    target_unit: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not targets:
        return rows, {"targets": {}, "unit": target_unit, "underfilled": {}}

    selected: list[dict[str, Any]] = []
    selected_hashes: set[str] = set()
    achieved: dict[str, int] = {}
    available: dict[str, int] = {}
    underfilled: dict[str, dict[str, int]] = {}

    rows_by_category: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_category.setdefault(str(row["category"]), []).append(row)

    for category, target in targets.items():
        category_rows = rows_by_category.get(category, [])
        available[category] = sum(row_unit(row, target_unit) for row in category_rows)
        current = 0
        for row in sorted(category_rows, key=target_sort_key):
            row_hash = row["normalized_text_hash"]
            if row_hash in selected_hashes:
                continue
            selected.append(row)
            selected_hashes.add(row_hash)
            current += row_unit(row, target_unit)
            if current >= target:
                break
        achieved[category] = current
        if current < target:
            underfilled[category] = {"actual": current, "target": target, "available": available[category]}

    return selected, {
        "targets": targets,
        "unit": target_unit,
        "available": available,
        "achieved": achieved,
        "underfilled": underfilled,
    }


def row_unit(row: dict[str, Any], target_unit: str) -> int:
    if target_unit == "documents":
        return 1
    return int(row.get("production_window_count", 1) or 1)


def target_sort_key(row: dict[str, Any]) -> tuple[int, int, str]:
    # Prefer long documents first for window targets, then stable source order.
    return (-int(row.get("production_window_count", 1) or 1), -int(row.get("category_score", 0) or 0), str(row["document_id"]))


def write_outputs(output_dir: Path, combined_jsonl: Path, rows: list[dict[str, Any]]) -> None:
    by_source: dict[str, list[dict[str, Any]]] = {}
    by_category: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_source.setdefault(row["source_name"], []).append(row)
        by_category.setdefault(row["category"], []).append(row)

    write_jsonl(combined_jsonl, rows)
    for source_name, source_rows in sorted(by_source.items()):
        write_jsonl(output_dir / "by_source" / f"{safe_filename(source_name)}.jsonl", source_rows)
    for category, category_rows in sorted(by_category.items()):
        write_jsonl(output_dir / "by_category" / f"{safe_filename(category)}.jsonl", category_rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def checkpoint_directory(args: argparse.Namespace) -> Path:
    if args.checkpoint_dir:
        return Path(args.checkpoint_dir)
    return Path(args.output_dir) / "source-checkpoints"


def checkpoint_paths(args: argparse.Namespace, source: SourceSpec) -> tuple[Path, Path]:
    directory = checkpoint_directory(args)
    stem = f"{safe_filename(source.name)}-{source_identity_hash(source)}"
    return directory / f"{stem}.jsonl", directory / f"{stem}.state.json"


def load_source_checkpoint(args: argparse.Namespace, source: SourceSpec) -> FetchResult | None:
    if args.no_checkpoint:
        return None
    rows_path, state_path = checkpoint_paths(args, source)
    if not rows_path.exists() or not state_path.exists():
        return None
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not checkpoint_state_compatible(args, source, state):
        return None
    rows = read_jsonl(rows_path)
    return FetchResult(
        source_name=source.name,
        kind=source.kind,
        rows=rows,
        scanned=int(state.get("scanned", 0) or 0),
        fetched=int(state.get("fetched", 0) or 0),
        accepted=len(rows),
        rejected=Counter(state.get("rejected", {})),
        rejected_samples=state.get("rejected_samples", {}),
        status_counts=Counter(state.get("status_counts", {})),
        sample_urls=list(state.get("sample_urls", [])),
        errors=list(state.get("errors", [])),
    )


def write_source_checkpoint(args: argparse.Namespace, source: SourceSpec, result: FetchResult) -> None:
    if args.no_checkpoint:
        return
    rows_path, state_path = checkpoint_paths(args, source)
    write_jsonl(rows_path, result.rows)
    state = {
        "source_identity_hash": source_identity_hash(source),
        "source": source_to_dict(source),
        "tokenizer_id": args.tokenizer_id,
        "min_document_chars": args.min_document_chars,
        "max_document_chars": args.max_document_chars,
        "min_category_score": args.min_category_score,
        "source_pool": args.source_pool,
        "scanned": result.scanned,
        "fetched": result.fetched,
        "accepted": result.accepted,
        "rejected": count_dict(result.rejected),
        "rejected_samples": result.rejected_samples,
        "status_counts": count_dict(result.status_counts),
        "sample_urls": result.sample_urls,
        "errors": result.errors,
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_path.with_suffix(state_path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(state_path)


def checkpoint_state_compatible(args: argparse.Namespace, source: SourceSpec, state: dict[str, Any]) -> bool:
    return (
        state.get("source_identity_hash") == source_identity_hash(source)
        and state.get("source") == source_to_dict(source)
        and state.get("tokenizer_id") == args.tokenizer_id
        and int(state.get("min_document_chars", -1)) == args.min_document_chars
        and int(state.get("max_document_chars", -1)) == args.max_document_chars
        and int(state.get("min_category_score", -1)) == args.min_category_score
        and state.get("source_pool") == args.source_pool
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def make_report(
    args: argparse.Namespace,
    sources: list[SourceSpec],
    results: list[FetchResult],
    rows: list[dict[str, Any]],
    all_rows: list[dict[str, Any]],
    target_report: dict[str, Any],
) -> dict[str, Any]:
    source_reports = []
    for result in sorted(results, key=lambda item: item.source_name):
        source_reports.append(
            {
                "source_name": result.source_name,
                "kind": result.kind,
                "source_status": source_status(result),
                "scanned": result.scanned,
                "fetched": result.fetched,
                "accepted": result.accepted,
                "status_counts": count_dict(result.status_counts),
                "rejected": count_dict(result.rejected),
                "rejected_samples": result.rejected_samples,
                "sample_urls": result.sample_urls,
                "errors": result.errors,
            }
        )
    return {
        "config": vars(args),
        "configured_sources": [source_to_dict(source) for source in sources],
        "summary": summarize_rows(rows),
        "candidate_summary_before_target_selection": summarize_rows(all_rows),
        "target_report": target_report,
        "source_reports": source_reports,
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


def source_status(result: FetchResult) -> str:
    if result.accepted > 0:
        return "ok"
    if result.rejected.get("blocked_or_antibot") or result.rejected.get("redirected_outside_allowed_scope"):
        return "blocked"
    if result.status_counts.get("error") and result.fetched == 0:
        return "blocked_or_unreachable"
    if result.fetched > 0 or result.scanned > 0:
        return "attempted_no_rows"
    return "not_attempted"


def source_to_dict(source: SourceSpec) -> dict[str, Any]:
    return {
        "name": source.name,
        "kind": source.kind,
        "categories": list(source.categories),
        "seeds": list(source.seeds),
        "allowed_domains": list(source.allowed_domains),
        "allowed_path_prefixes": list(source.allowed_path_prefixes),
        "trust_category": source.trust_category,
        "repo_id": source.repo_id,
        "config": source.config,
        "split": source.split,
        "text_column": source.text_column,
        "hh_queries": list(source.hh_queries),
    }


def count_dict(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in counter.most_common()}


def record_rejection(counter: Counter[str], samples: dict[str, list[str]], reason: str, sample: Any) -> None:
    counter[reason] += 1
    bucket = samples.setdefault(reason, [])
    if len(bucket) < 10:
        bucket.append(normalize_text(sample))


def token_length(text: str, tokenizer: Any) -> int:
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
    return text.count(chr(0xFFFD)) >= 3


def stable_hash(value: str) -> str:
    return hashlib.sha256(normalize_text(value).encode("utf-8")).hexdigest()


def normalized_text_hash(value: str) -> str:
    folded = normalize_text(value).lower()
    folded = re.sub(r"[^\wА-Яа-яЁё]+", " ", folded)
    return stable_hash(folded)


def safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return cleaned or "source"


def source_identity_hash(source: SourceSpec) -> str:
    payload = source_to_dict(source)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:12]


if __name__ == "__main__":
    main()
