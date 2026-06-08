from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from v12_pipeline_utils import (
    ATTACK_WRAPPERS,
    LABEL_ATTACK,
    LABEL_BENIGN,
    build_production_windows,
    count_windows,
    generate_critical_ru_attacks,
    infer_language,
    load_text_records,
    normalize_text,
    text_hash,
    window_count_bucket,
    write_json,
    write_jsonl,
)


PRODUCTION_BENIGN_QUOTAS = {
    "job_descriptions": 500,
    "hr_policies": 500,
    "corporate_procedures": 500,
    "support_documentation": 1000,
    "admin_instructions": 500,
    "legal_templates": 500,
    "knowledge_base": 1000,
    "meeting_minutes": 300,
    "security_compliance_redaction_wrappers": 200,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build frozen V12 document-level evaluation suites from supplied fresh input pools.")
    parser.add_argument("--tokenizer-id", required=True)
    parser.add_argument("--production-benign-jsonl")
    parser.add_argument("--external-benign-jsonl")
    parser.add_argument("--carrier-jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-underfilled", action="store_true")
    return parser.parse_args()


def require_file(path: str | None, name: str) -> Path:
    if not path:
        raise FileNotFoundError(f"{name} is required. Pass --{name.replace('_', '-')}.")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def annotate_docs(rows: list[dict[str, Any]], tokenizer: Any) -> list[dict[str, Any]]:
    out = []
    for idx, row in enumerate(rows):
        text = row.get("text") or row.get("document_text") or ""
        if not normalize_text(text):
            continue
        window_count = count_windows(text, tokenizer)
        out.append(
            {
                **row,
                "document_id": str(row.get("document_id") or row.get("id") or f"doc_{idx}_{text_hash(text)}"),
                "text": text,
                "document_label": row.get("document_label") or row.get("label") or LABEL_BENIGN,
                "source_name": row.get("source_name") or row.get("source") or "fresh_input",
                "category": row.get("category") or "knowledge_base",
                "language": row.get("language") or infer_language(text),
                "window_count": window_count,
                "window_count_bucket": window_count_bucket(window_count),
            }
        )
    return out


def take_by_category(rows: list[dict[str, Any]], quotas: dict[str, int], *, allow_underfilled: bool, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rnd = random.Random(seed)
    by_category: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_category.setdefault(str(row.get("category", "knowledge_base")), []).append(row)
    selected = []
    report = {}
    for category, quota in quotas.items():
        bucket = by_category.get(category, [])
        rnd.shuffle(bucket)
        picked = bucket[:quota]
        if len(picked) < quota and not allow_underfilled:
            raise ValueError(f"Production locked category {category} underfilled: {len(picked)}/{quota}")
        selected.extend(picked)
        report[category] = {"target": quota, "selected": len(picked)}
    selected_hashes = {text_hash(row["text"]) for row in selected}
    remaining = [row for row in rows if text_hash(row["text"]) not in selected_hashes]
    return selected, remaining, report


def make_attack_docs(
    carriers: list[dict[str, Any]],
    tokenizer: Any,
    *,
    count: int,
    source_prefix: str,
    seed: int,
    critical_only: bool = False,
) -> list[dict[str, Any]]:
    rnd = random.Random(seed)
    attacks = generate_critical_ru_attacks()
    rows: list[dict[str, Any]] = []
    for i in range(count):
        attack = attacks[(i * 7) % len(attacks)]
        if critical_only or not carriers or i % 5 == 0:
            text = attack["attack_text"]
            category = "standalone_attack"
            source_name = f"{source_prefix}_attack_bank"
            carrier_id = None
        else:
            carrier = carriers[i % len(carriers)]
            base = carrier["text"]
            pos = rnd.choice([len(base) // 6, len(base) // 2, len(base) * 5 // 6])
            text = f"{base[:pos]}\n\n{attack['attack_text']}\n\n{base[pos:]}"
            if i % 4 == 0:
                text = rnd.choice(ATTACK_WRAPPERS).format(text=text)
            category = f"embedded_{carrier.get('category', 'knowledge_base')}"
            source_name = f"{source_prefix}_embedded_attack"
            carrier_id = carrier.get("document_id")
        rows.append(
            {
                "document_id": f"{source_prefix}_{i}_{attack['attack_text_hash']}",
                "document_label": LABEL_ATTACK,
                "text": text,
                "source_name": source_name,
                "category": category,
                "language": "ru" if critical_only else infer_language(text),
                "semantic_family": attack["semantic_family"],
                "attack_text_hash": attack["attack_text_hash"],
                "carrier_document_id": carrier_id,
                "window_count": count_windows(text, tokenizer),
            }
        )
        rows[-1]["window_count_bucket"] = window_count_bucket(rows[-1]["window_count"])
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "documents": len(rows),
        "labels": dict(Counter(row.get("document_label", LABEL_BENIGN) for row in rows)),
        "categories": dict(Counter(row.get("category", "unknown") for row in rows)),
        "sources": dict(Counter(row.get("source_name", "unknown") for row in rows)),
        "languages": dict(Counter(row.get("language", "unknown") for row in rows)),
        "window_count_buckets": dict(Counter(row.get("window_count_bucket", "unknown") for row in rows)),
    }


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)
    production_path = require_file(args.production_benign_jsonl, "production_benign_jsonl")
    external_path = require_file(args.external_benign_jsonl, "external_benign_jsonl")
    carrier_path = Path(args.carrier_jsonl) if args.carrier_jsonl else production_path
    if not carrier_path.exists():
        raise FileNotFoundError(carrier_path)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    production = annotate_docs(load_text_records(production_path, default_source="fresh_production"), tokenizer)
    external = annotate_docs(load_text_records(external_path, default_source="fresh_external"), tokenizer)
    carriers = annotate_docs(load_text_records(carrier_path, default_source="fresh_carrier"), tokenizer)

    production_locked_benign, production_remaining, production_quota_report = take_by_category(
        production,
        PRODUCTION_BENIGN_QUOTAS,
        allow_underfilled=args.allow_underfilled,
        seed=args.seed,
    )
    rnd = random.Random(args.seed)
    rnd.shuffle(production_remaining)
    production_dev_benign = production_remaining[:2500]

    external = list(external)
    rnd.shuffle(external)
    external_locked = external[:5500]
    external_dev = external[5500:7500]
    if len(external_locked) < 5500 and not args.allow_underfilled:
        raise ValueError(f"External locked underfilled: {len(external_locked)}/5500")
    if len(external_dev) < 2000 and not args.allow_underfilled:
        raise ValueError(f"External dev underfilled: {len(external_dev)}/2000")

    production_locked = production_locked_benign + make_attack_docs(carriers, tokenizer, count=2500, source_prefix="production_locked", seed=args.seed)
    production_dev = production_dev_benign + make_attack_docs(carriers, tokenizer, count=1000, source_prefix="production_dev", seed=args.seed + 1)
    critical_locked = make_attack_docs(carriers, tokenizer, count=300, source_prefix="critical_ru_locked", seed=args.seed + 2, critical_only=True)
    critical_dev = make_attack_docs(carriers, tokenizer, count=150, source_prefix="critical_ru_dev", seed=args.seed + 3, critical_only=True)

    files = {
        "production_locked": output_dir / "production-domain-locked.jsonl",
        "production_dev": output_dir / "production-domain-dev.jsonl",
        "external_locked": output_dir / "external-blind-locked.jsonl",
        "external_dev": output_dir / "external-blind-dev.jsonl",
        "critical_ru_locked": output_dir / "critical-ru-locked.jsonl",
        "critical_ru_dev": output_dir / "critical-ru-dev.jsonl",
    }
    write_jsonl(files["production_locked"], production_locked)
    write_jsonl(files["production_dev"], production_dev)
    write_jsonl(files["external_locked"], external_locked)
    write_jsonl(files["external_dev"], external_dev)
    write_jsonl(files["critical_ru_locked"], critical_locked)
    write_jsonl(files["critical_ru_dev"], critical_dev)

    manifest = {
        "output_dir": str(output_dir),
        "files": {k: {"path": str(v), "rows": sum(1 for _ in open(v, "r", encoding="utf-8"))} for k, v in files.items()},
        "production_quota_report": production_quota_report,
        "production_locked": summarize(production_locked),
        "production_dev": summarize(production_dev),
        "external_locked": summarize(external_locked),
        "external_dev": summarize(external_dev),
        "critical_ru_locked": summarize(critical_locked),
        "critical_ru_dev": summarize(critical_dev),
        "note": "Inputs must be fresh/held-out. This script does not certify freshness by itself.",
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps({"output_dir": str(output_dir), "files": {k: v["rows"] for k, v in manifest["files"].items()}, "manifest": str(output_dir / "manifest.json")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
