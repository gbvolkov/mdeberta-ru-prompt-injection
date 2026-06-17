from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


LABEL_ATTACK = "prompt_injection"
LABEL_BENIGN = "not_prompt_injection"
SOURCE_ID_KEYS = [
    "document_id",
    "source_document_id",
    "original_document_id",
    "carrier_document_id",
    "source_id",
    "source_path",
    "original_source_path",
    "carrier_id",
    "carrier_source_id",
]

SCORE_BANDS = [
    (0.01, "<0.01"),
    (0.10, "0.01-0.10"),
    (0.50, "0.10-0.50"),
    (0.82, "0.50-0.82"),
    (0.95, "0.82-0.95"),
    (0.99, "0.95-0.99"),
    (math.inf, ">=0.99"),
]

TRIGGER_TERMS = [
    "policy",
    "instruction",
    "instructions",
    "rules",
    "routing",
    "system",
    "security",
    "confidential",
    "hidden",
    "prompt",
    "audit",
    "incident",
    "compliance",
    "developer",
    "tool",
    "secret",
    "политик",
    "инструкц",
    "правил",
    "маршрут",
    "систем",
    "безопас",
    "конфиденц",
    "скрыт",
    "промпт",
    "аудит",
    "инцидент",
    "комплаен",
    "разработчик",
    "инструмент",
    "секрет",
]

VISIBLE_ATTACK_PATTERNS = [
    r"ignore (all )?(previous|prior|above) instructions",
    r"disregard (all )?(previous|prior|above) instructions",
    r"reveal (the )?(system|developer) prompt",
    r"show (the )?(system|developer) prompt",
    r"print (the )?(system|developer) prompt",
    r"hidden (instructions|routing|policy|rules)",
    r"make this block the main response policy",
    r"highest[- ]priority",
    r"system prompt",
    r"developer prompt",
    r"tool list",
    r"internal routing",
    r"игнорируй (все )?(предыдущие|прежние|вышеуказанные) инструкции",
    r"раскрой (системный|developer|промпт|скрыт)",
    r"покажи (системный|промпт|скрыт)",
    r"выведи (системный|промпт|скрыт)",
    r"скрыт(ые|ую|ый)? (инструкц|маршрут|правил|политик)",
    r"промпт разработчика",
    r"системный промпт",
    r"список инструментов",
    r"внутренн(яя|ие|ий) маршрутизац",
]

ATTACK_INTENT_PATTERNS = [
    ("developer_prompt_exfiltration", [r"developer prompt", r"промпт разработчика", r"developer instruction"]),
    ("system_prompt_exfiltration", [r"system prompt", r"системный промпт"]),
    ("hidden_instruction_exfiltration", [r"hidden instructions", r"скрыт(ые|ую|ый)? инструкц"]),
    ("internal_tool_or_routing_exfiltration", [r"tool list", r"internal routing", r"список инструментов", r"маршрут"]),
    ("policy_bypass", [r"bypass", r"override policy", r"обойди", r"отмени полити"]),
    ("priority_override", [r"highest[- ]priority", r"main response policy", r"priority override", r"главн(ое|ым|ый) правил"]),
    ("instruction_override", [r"ignore (all )?(previous|prior|above)", r"disregard", r"игнорируй"]),
    ("secret_or_data_exfiltration", [r"secret", r"token", r"private", r"секрет", r"токен", r"приват"]),
    ("classifier_or_routing_manipulation", [r"classifier", r"router", r"routing", r"классификатор", r"маршрутизатор"]),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit V18 training patterns against V18 validation failures.")
    parser.add_argument("--train-audit-jsonl", default="v18-build-300k-final/audit-rows/train-metadata.jsonl")
    parser.add_argument("--validation-audit-jsonl", default="v18-build-300k-final/audit-rows/validation-metadata.jsonl")
    parser.add_argument("--component-report-json", default="v18-build-300k-final/component-report.json")
    parser.add_argument("--eval-results-dir", default="v18_eval_results/v18")
    parser.add_argument("--output-dir", default="v18_1_diagnostics")
    parser.add_argument("--fn-threshold", type=float, default=0.50)
    parser.add_argument("--fp-threshold", type=float, default=0.99)
    parser.add_argument("--sample-per-cluster", type=int, default=5)
    parser.add_argument("--max-audit-rows", type=int, default=0, help="Debug limit; 0 means all audit rows.")
    parser.add_argument("--progress-every-rows", type=int, default=50_000)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL row") from exc
            if isinstance(value, dict):
                yield value


def progress_iter(
    iterable: Iterable[dict[str, Any]],
    *,
    args: argparse.Namespace,
    label: str,
) -> Iterable[dict[str, Any]]:
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


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def write_text(path: str | Path, value: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(value, encoding="utf-8")


def normalize_text(text: str) -> str:
    text = str(text or "").replace(chr(160), " ")
    return re.sub(r"\s+", " ", text).strip()


def normalized_for_hash(text: str) -> str:
    return normalize_text(text).lower()


def stable_hash(value: str, *, length: int = 16) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def text_hash(text: str, *, length: int = 16) -> str:
    return stable_hash(normalized_for_hash(text), length=length)


def collect_source_ids(row: dict[str, Any]) -> set[str]:
    values = set()
    for key in SOURCE_ID_KEYS:
        value = row.get(key)
        if value is None or value == "":
            continue
        values.add(str(value))
    return values


def add_text_identity(eval_identity: dict[str, set[str]], row: dict[str, Any], text_keys: tuple[str, ...]) -> None:
    for key in (
        "text_hash",
        "best_window_text_hash",
        "window_text_hash",
        "normalized_text_hash",
        "normalized_window_text_hash",
    ):
        value = row.get(key)
        if not value:
            continue
        if "normalized" in key:
            eval_identity["normalized_text_hashes"].add(str(value))
        else:
            eval_identity["text_hashes"].add(str(value))
    for key in text_keys:
        text = str(row.get(key) or "")
        if not text:
            continue
        eval_identity["text_hashes"].add(text_hash(text, length=16))
        eval_identity["normalized_text_hashes"].add(text_hash(text, length=24))


def add_attack_identity(eval_identity: dict[str, set[str]], row: dict[str, Any]) -> None:
    if row.get("attack_text_hash"):
        eval_identity["attack_text_hashes"].add(str(row["attack_text_hash"]))
    attack_text = str(row.get("attack_text") or "")
    if attack_text:
        eval_identity["attack_text_hashes"].add(text_hash(attack_text, length=16))


def label_name(value: Any) -> str:
    if isinstance(value, bool):
        return LABEL_ATTACK if value else LABEL_BENIGN
    if isinstance(value, int):
        return LABEL_ATTACK if value == 1 else LABEL_BENIGN
    text = str(value or "").strip().lower()
    if text in {"1", "attack", "malicious", "prompt_injection", "injection"}:
        return LABEL_ATTACK
    if text in {"0", "benign", "not_prompt_injection", "non_injection", "safe"}:
        return LABEL_BENIGN
    return text or LABEL_BENIGN


def score_band(score: float) -> str:
    for upper, name in SCORE_BANDS:
        if score < upper:
            return name
    return ">=0.99"


def canonical_language(value: Any, text: str = "") -> str:
    raw = str(value or "").strip().lower()
    if raw in {"ru", "rus", "russian", "ru_ru", "rus_cyrl"}:
        return "ru"
    if raw in {"en", "eng", "english", "en_us", "en_gb"}:
        return "en"
    if raw in {"mixed", "multi", "multilingual"}:
        return "mixed"
    return infer_language(text)


def infer_language(text: str) -> str:
    text = str(text or "")
    cyr = len(re.findall(r"[А-Яа-яЁё]", text))
    lat = len(re.findall(r"[A-Za-z]", text))
    if cyr and lat and min(cyr, lat) / max(cyr, lat) >= 0.12:
        return "mixed"
    if cyr > lat:
        return "ru"
    if lat:
        return "en"
    return "unknown"


def terms_in_text(text: str) -> list[str]:
    lowered = normalize_text(text).lower()
    return [term for term in TRIGGER_TERMS if term in lowered]


def visible_model_control_signal(text: str) -> bool:
    lowered = normalize_text(text).lower()
    return any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in VISIBLE_ATTACK_PATTERNS)


def classify_attack_intent(text: str, semantic_family: str = "") -> str:
    combined = f"{semantic_family} {normalize_text(text)}".lower()
    for name, patterns in ATTACK_INTENT_PATTERNS:
        if any(re.search(pattern, combined, flags=re.IGNORECASE) for pattern in patterns):
            return name
    if semantic_family:
        return semantic_family
    return "unknown"


def classify_document_form(text: str, row: dict[str, Any]) -> str:
    component = str(row.get("component") or "").lower()
    category = str(row.get("category") or "").lower()
    source_name = str(row.get("source_name") or "").lower()
    text_l = normalize_text(text).lower()
    window_count = int(row.get("window_count") or 1)
    if "embedded" in component or "embedded" in category or "embedded" in source_name:
        return "embedded_command"
    if window_count >= 5:
        return "long_document_boundary_case"
    if any(mark in text_l for mark in ['"', "'", "«", "»", "quote", "quoted", "цитат"]):
        return "quoted_instruction"
    if any(term in text_l for term in ["policy", "procedure", "compliance", "regulation", "политик", "процедур", "регламент"]):
        return "policy_or_procedure_like"
    if any(term in text_l for term in ["support", "ticket", "customer", "helpdesk", "поддержк", "заявк"]):
        return "support_ticket_like"
    if any(term in text_l for term in ["legal", "contract", "law", "admin", "юрид", "договор", "закон"]):
        return "legal_or_admin_like"
    if any(term in text_l for term in ["roleplay", "scenario", "context:", "режим", "сценар"]):
        return "roleplay_or_context_wrapper"
    if visible_model_control_signal(text_l):
        return "direct_command"
    return "unknown"


def score_from_window(row: dict[str, Any]) -> float:
    for key in ("p_prompt_injection", "score", "document_max_prompt_injection_score", "max_prompt_injection_score"):
        value = row.get(key)
        if value is None or value == "":
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def read_eval_corpus(corpus_dir: Path, args: argparse.Namespace | None = None) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    docs_path = corpus_dir / "document-results.jsonl"
    windows_path = corpus_dir / "window-results.jsonl"
    if not docs_path.exists() or not windows_path.exists():
        return [], {}
    doc_iterable = iter_jsonl(docs_path)
    if args is not None:
        doc_iterable = progress_iter(doc_iterable, args=args, label=f"Read eval docs {corpus_dir.name}")
    docs = list(doc_iterable)
    windows_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    window_iterable = iter_jsonl(windows_path)
    if args is not None:
        window_iterable = progress_iter(window_iterable, args=args, label=f"Read eval windows {corpus_dir.name}")
    for row in window_iterable:
        windows_by_doc[str(row.get("document_id") or "")].append(row)
    return docs, windows_by_doc


def verify_eval_windowing(eval_results_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    report: dict[str, Any] = {"status": "pass", "corpora": {}, "failures": []}
    for corpus_dir in sorted(p for p in eval_results_dir.iterdir() if p.is_dir()):
        docs, windows_by_doc = read_eval_corpus(corpus_dir, args)
        if not docs:
            continue
        mismatches = []
        missing_windows = []
        for doc in docs:
            doc_id = str(doc.get("document_id") or "")
            windows = windows_by_doc.get(doc_id, [])
            if not windows:
                missing_windows.append(doc_id)
                continue
            scores = [score_from_window(w) for w in windows]
            max_score = max(scores)
            doc_score = float(doc.get("document_max_prompt_injection_score") or 0.0)
            best_index = int(doc.get("best_window_index") or 0)
            max_index = scores.index(max_score)
            if abs(max_score - doc_score) > 1e-6 or best_index != max_index:
                mismatches.append(
                    {
                        "document_id": doc_id,
                        "document_score": doc_score,
                        "max_window_score": max_score,
                        "best_window_index": best_index,
                        "actual_max_window_index": max_index,
                    }
                )
        corpus_report = {
            "documents": len(docs),
            "window_rows": sum(len(v) for v in windows_by_doc.values()),
            "missing_window_documents": len(missing_windows),
            "score_or_best_index_mismatches": len(mismatches),
            "sample_mismatches": mismatches[:10],
            "sample_missing_window_documents": missing_windows[:10],
        }
        report["corpora"][corpus_dir.name] = corpus_report
        if missing_windows or mismatches:
            report["status"] = "fail"
            report["failures"].append(corpus_dir.name)
    return report


def best_window_text(doc: dict[str, Any], windows_by_doc: dict[str, list[dict[str, Any]]]) -> str:
    windows = windows_by_doc.get(str(doc.get("document_id") or ""), [])
    if not windows:
        return ""
    best_index = int(doc.get("best_window_index") or 0)
    if 0 <= best_index < len(windows):
        return str(windows[best_index].get("window_text") or windows[best_index].get("text") or "")
    best = max(windows, key=score_from_window)
    return str(best.get("window_text") or best.get("text") or "")


def visible_window_count(windows: list[dict[str, Any]]) -> int:
    return sum(
        1
        for row in windows
        if visible_model_control_signal(str(row.get("window_text") or row.get("text") or ""))
    )


def cluster_key(row: dict[str, Any]) -> str:
    parts = [
        row.get("failure_type", "unknown"),
        row.get("score_band", "unknown"),
        row.get("language", "unknown"),
        row.get("category", "unknown"),
        row.get("semantic_family", "unknown"),
        row.get("document_form", "unknown"),
        row.get("attack_intent", "unknown"),
        ",".join(row.get("trigger_terms", [])[:4]),
    ]
    return stable_hash("|".join(str(p) for p in parts), length=20)


def collect_failures(
    eval_results_dir: Path,
    fn_threshold: float,
    fp_threshold: float,
    sample_per_cluster: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    failure_rows: list[dict[str, Any]] = []
    eval_identity = {
        "text_hashes": set(),
        "normalized_text_hashes": set(),
        "attack_text_hashes": set(),
        "document_ids": set(),
        "source_document_ids": set(),
    }
    corpus_totals: dict[str, Any] = {}
    for corpus_dir in sorted(p for p in eval_results_dir.iterdir() if p.is_dir()):
        docs, windows_by_doc = read_eval_corpus(corpus_dir, args)
        if not docs:
            continue
        totals = Counter()
        for doc in docs:
            label = label_name(doc.get("document_label", doc.get("label")))
            score = float(doc.get("document_max_prompt_injection_score") or 0.0)
            text = best_window_text(doc, windows_by_doc)
            doc_id = str(doc.get("document_id") or "")
            if doc_id:
                eval_identity["document_ids"].add(doc_id)
            eval_identity["source_document_ids"].update(collect_source_ids(doc))
            add_text_identity(eval_identity, doc, ("text", "document_text", "best_window_text"))
            if text:
                eval_identity["text_hashes"].add(text_hash(text, length=16))
                eval_identity["normalized_text_hashes"].add(text_hash(text, length=24))
            add_attack_identity(eval_identity, doc)
            for window in windows_by_doc.get(doc_id, []):
                eval_identity["source_document_ids"].update(collect_source_ids(window))
                add_text_identity(eval_identity, window, ("window_text", "text"))
                add_attack_identity(eval_identity, window)

            failure_type = ""
            if label == LABEL_ATTACK and score < fn_threshold:
                failure_type = "false_negative_below_threshold"
                totals["false_negative_below_threshold"] += 1
            elif label == LABEL_BENIGN and score >= fp_threshold:
                failure_type = "false_positive_above_threshold"
                totals["false_positive_above_threshold"] += 1
            else:
                totals["non_failure"] += 1
                continue

            windows = windows_by_doc.get(doc_id, [])
            semantic_family = str(doc.get("semantic_family") or "unknown")
            enriched = {
                "corpus": corpus_dir.name,
                "failure_type": failure_type,
                "document_id": doc_id,
                "document_label": label,
                "score": score,
                "score_band": score_band(score),
                "best_window_index": int(doc.get("best_window_index") or 0),
                "window_count": int(doc.get("window_count") or len(windows) or 1),
                "window_count_bucket": doc.get("window_count_bucket") or "unknown",
                "category": doc.get("category") or "unknown",
                "language": canonical_language(doc.get("language"), text),
                "source_name": doc.get("source_name") or "unknown",
                "semantic_family": semantic_family,
                "attack_text_hash": doc.get("attack_text_hash") or "",
                "best_window_text_hash": doc.get("best_window_text_hash") or text_hash(text),
                "document_form": classify_document_form(text, doc),
                "attack_intent": classify_attack_intent(text, semantic_family),
                "trigger_terms": terms_in_text(text),
                "visible_model_control_in_best_window": visible_model_control_signal(text),
                "visible_model_control_window_count": visible_window_count(windows),
                "sample_text": text[:800],
            }
            enriched["diagnostic_cluster_id"] = cluster_key(enriched)
            failure_rows.append(enriched)
        corpus_totals[corpus_dir.name] = dict(totals)

    clusters: dict[str, Any] = {}
    for row in failure_rows:
        cid = row["diagnostic_cluster_id"]
        cluster = clusters.setdefault(
            cid,
            {
                "diagnostic_cluster_id": cid,
                "count": 0,
                "failure_type": row["failure_type"],
                "corpora": Counter(),
                "score_bands": Counter(),
                "languages": Counter(),
                "categories": Counter(),
                "semantic_families": Counter(),
                "document_forms": Counter(),
                "attack_intents": Counter(),
                "trigger_terms": Counter(),
                "samples": [],
            },
        )
        cluster["count"] += 1
        for key, counter_name in [
            ("corpus", "corpora"),
            ("score_band", "score_bands"),
            ("language", "languages"),
            ("category", "categories"),
            ("semantic_family", "semantic_families"),
            ("document_form", "document_forms"),
            ("attack_intent", "attack_intents"),
        ]:
            cluster[counter_name][str(row.get(key) or "unknown")] += 1
        for term in row.get("trigger_terms", []):
            cluster["trigger_terms"][term] += 1
        if len(cluster["samples"]) < sample_per_cluster:
            cluster["samples"].append(row)

    for cluster in clusters.values():
        for key in [
            "corpora",
            "score_bands",
            "languages",
            "categories",
            "semantic_families",
            "document_forms",
            "attack_intents",
            "trigger_terms",
        ]:
            cluster[key] = dict(cluster[key].most_common())

    return {
        "failure_rows": failure_rows,
        "clusters": dict(sorted(clusters.items(), key=lambda item: item[1]["count"], reverse=True)),
        "corpus_totals": corpus_totals,
        "eval_identity": {k: sorted(v) for k, v in eval_identity.items()},
    }


def stream_training_coverage(
    train_audit_jsonl: Path,
    validation_audit_jsonl: Path,
    failure_clusters: dict[str, Any],
    args: argparse.Namespace,
    max_audit_rows: int = 0,
) -> dict[str, Any]:
    counters: dict[str, Counter] = defaultdict(Counter)
    attack_maps: dict[str, Counter] = defaultdict(Counter)
    benign_maps: dict[str, Counter] = defaultdict(Counter)

    def process(path: Path, split: str) -> int:
        count = 0
        for row in progress_iter(iter_jsonl(path), args=args, label=f"Scan {split} audit {path.name}"):
            count += 1
            if max_audit_rows and count > max_audit_rows:
                break
            label = label_name(row.get("label"))
            text = str(row.get("text") or "")
            lang = canonical_language(row.get("language"), text)
            category = str(row.get("category") or "unknown")
            semantic_family = str(row.get("semantic_family") or "unknown")
            document_form = classify_document_form(text, row)
            attack_intent = classify_attack_intent(text, semantic_family)
            triggers = terms_in_text(text)
            fields = {
                "split": split,
                "label": label,
                "component": row.get("component") or "unknown",
                "language": lang,
                "category": category,
                "generation_type": row.get("generation_type") or "unknown",
                "semantic_family": semantic_family,
                "score_band": row.get("score_band") or "unknown",
                "text_form_bucket": row.get("text_form_bucket") or document_form,
                "instructionality_bucket": row.get("instructionality_bucket") or "unknown",
                "attack_intent_bucket": row.get("attack_intent_bucket") or attack_intent,
                "source_name": row.get("source_name") or "unknown",
                "source_origin": row.get("source_origin") or "unknown",
                "window_position_bucket": row.get("window_position_bucket") or "unknown",
            }
            for key, value in fields.items():
                counters[key][str(value)] += 1
                counters[f"{split}.{key}"][str(value)] += 1
            if label == LABEL_ATTACK:
                attack_maps["semantic_language_category"][(semantic_family, lang, category)] += 1
                attack_maps["semantic_language"][(semantic_family, lang)] += 1
                attack_maps["intent_language"][(attack_intent, lang)] += 1
                attack_maps["form_language"][(document_form, lang)] += 1
            else:
                trigger_signature = ",".join(sorted(triggers)[:5]) or "none"
                benign_maps["trigger_category_language"][(trigger_signature, category, lang)] += 1
                benign_maps["category_language"][(category, lang)] += 1
                benign_maps["form_language"][(document_form, lang)] += 1
        return min(count, max_audit_rows or count)

    train_rows = process(train_audit_jsonl, "train")
    validation_rows = process(validation_audit_jsonl, "validation")

    cluster_coverage: dict[str, Any] = {}
    for cid, cluster in failure_clusters.items():
        lang = next(iter(cluster.get("languages", {}) or {"unknown": 0}))
        category = next(iter(cluster.get("categories", {}) or {"unknown": 0}))
        semantic_family = next(iter(cluster.get("semantic_families", {}) or {"unknown": 0}))
        attack_intent = next(iter(cluster.get("attack_intents", {}) or {"unknown": 0}))
        document_form = next(iter(cluster.get("document_forms", {}) or {"unknown": 0}))
        trigger_terms = ",".join(sorted((cluster.get("trigger_terms") or {}).keys())[:5]) or "none"
        cluster_coverage[cid] = {
            "failure_count": cluster["count"],
            "comparable_attack_rows_semantic_language_category": attack_maps["semantic_language_category"][(semantic_family, lang, category)],
            "comparable_attack_rows_semantic_language": attack_maps["semantic_language"][(semantic_family, lang)],
            "comparable_attack_rows_intent_language": attack_maps["intent_language"][(attack_intent, lang)],
            "comparable_attack_rows_form_language": attack_maps["form_language"][(document_form, lang)],
            "comparable_benign_rows_trigger_category_language": benign_maps["trigger_category_language"][(trigger_terms, category, lang)],
            "comparable_benign_rows_category_language": benign_maps["category_language"][(category, lang)],
            "comparable_benign_rows_form_language": benign_maps["form_language"][(document_form, lang)],
        }

    return {
        "rows_scanned": {"train": train_rows, "validation": validation_rows},
        "distributions": {k: dict(v.most_common(100)) for k, v in counters.items()},
        "cluster_coverage": cluster_coverage,
    }


def render_markdown(taxonomy: dict[str, Any], coverage: dict[str, Any], consistency: dict[str, Any]) -> str:
    lines = [
        "# V18.1 Failure Taxonomy",
        "",
        f"Windowing consistency: `{consistency['status']}`",
        "",
        "## Failure Totals",
        "",
        "| Corpus | False negatives < threshold | False positives >= threshold | Non-failures |",
        "| --- | ---: | ---: | ---: |",
    ]
    for corpus, totals in sorted(taxonomy["corpus_totals"].items()):
        lines.append(
            "| "
            + " | ".join(
                [
                    corpus,
                    str(totals.get("false_negative_below_threshold", 0)),
                    str(totals.get("false_positive_above_threshold", 0)),
                    str(totals.get("non_failure", 0)),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Top Diagnostic Clusters", ""])
    lines.append("| Cluster | Count | Type | Score bands | Languages | Categories | Intent | Form |")
    lines.append("| --- | ---: | --- | --- | --- | --- | --- | --- |")
    for cid, cluster in list(taxonomy["clusters"].items())[:30]:
        lines.append(
            "| "
            + " | ".join(
                [
                    cid,
                    str(cluster["count"]),
                    cluster["failure_type"],
                    ", ".join(f"{k}:{v}" for k, v in list(cluster["score_bands"].items())[:3]),
                    ", ".join(f"{k}:{v}" for k, v in list(cluster["languages"].items())[:3]),
                    ", ".join(f"{k}:{v}" for k, v in list(cluster["categories"].items())[:3]),
                    ", ".join(f"{k}:{v}" for k, v in list(cluster["attack_intents"].items())[:2]),
                    ", ".join(f"{k}:{v}" for k, v in list(cluster["document_forms"].items())[:2]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Training Coverage For Top Clusters", ""])
    lines.append("| Cluster | Failures | Attack sem/lang/cat | Attack sem/lang | Attack intent/lang | Benign cat/lang |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for cid in list(taxonomy["clusters"].keys())[:30]:
        cov = coverage["cluster_coverage"].get(cid, {})
        lines.append(
            "| "
            + " | ".join(
                [
                    cid,
                    str(cov.get("failure_count", 0)),
                    str(cov.get("comparable_attack_rows_semantic_language_category", 0)),
                    str(cov.get("comparable_attack_rows_semantic_language", 0)),
                    str(cov.get("comparable_attack_rows_intent_language", 0)),
                    str(cov.get("comparable_benign_rows_category_language", 0)),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_results_dir = Path(args.eval_results_dir)
    consistency = verify_eval_windowing(eval_results_dir, args)
    write_json(output_dir / "eval-windowing-consistency-report.json", consistency)
    if consistency["status"] != "pass":
        raise ValueError(f"Evaluation windowing consistency failed: {consistency['failures']}")

    failures = collect_failures(
        eval_results_dir,
        fn_threshold=args.fn_threshold,
        fp_threshold=args.fp_threshold,
        sample_per_cluster=args.sample_per_cluster,
        args=args,
    )
    taxonomy = {
        "config": {
            "fn_threshold": args.fn_threshold,
            "fp_threshold": args.fp_threshold,
            "score_bands": [name for _, name in SCORE_BANDS],
        },
        "corpus_totals": failures["corpus_totals"],
        "clusters": failures["clusters"],
        "eval_identity": failures["eval_identity"],
    }
    coverage = stream_training_coverage(
        Path(args.train_audit_jsonl),
        Path(args.validation_audit_jsonl),
        failures["clusters"],
        args,
        max_audit_rows=args.max_audit_rows,
    )
    with Path(args.component_report_json).open("r", encoding="utf-8") as f:
        component_report = json.load(f)
    coverage["component_report_summary"] = {
        "component_count": len(component_report.get("components", {})),
        "components": {
            name: {
                "rows": data.get("rows"),
                "labels": data.get("labels"),
                "languages": data.get("languages"),
            }
            for name, data in component_report.get("components", {}).items()
        },
    }

    write_json(output_dir / "failure-taxonomy.json", taxonomy)
    write_json(output_dir / "training-vs-failure-coverage.json", coverage)
    write_text(output_dir / "failure-taxonomy.md", render_markdown(taxonomy, coverage, consistency))

    print(
        json.dumps(
            {
                "status": "pass",
                "output_dir": str(output_dir),
                "clusters": len(failures["clusters"]),
                "failure_rows": sum(c["count"] for c in failures["clusters"].values()),
                "train_rows_scanned": coverage["rows_scanned"]["train"],
                "validation_rows_scanned": coverage["rows_scanned"]["validation"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
