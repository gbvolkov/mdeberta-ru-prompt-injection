from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from v12_pipeline_utils import count_windows, infer_language, load_text_records, normalize_text, text_hash, window_count_bucket, write_json, write_jsonl


PRODUCTION_CATEGORIES = {
    "job_descriptions",
    "hr_policies",
    "corporate_procedures",
    "support_documentation",
    "admin_instructions",
    "legal_templates",
    "knowledge_base",
    "meeting_minutes",
    "security_compliance_redaction_wrappers",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare fresh input pools for V12 locked-suite building from explicit JSONL inputs. "
            "This script does not certify freshness; use sources not used in prior training/mining."
        )
    )
    parser.add_argument("--tokenizer-id", required=True)
    parser.add_argument("--input-jsonl", action="append", required=True)
    parser.add_argument("--output-production-jsonl", required=True)
    parser.add_argument("--output-external-jsonl", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--production-target", type=int, default=10000)
    parser.add_argument("--external-target", type=int, default=7500)
    parser.add_argument("--min-chars", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def annotate(rows, tokenizer):
    out = []
    seen = set()
    for i, row in enumerate(rows):
        text = row.get("text", "")
        if len(normalize_text(text)) < 400:
            continue
        h = text_hash(text)
        if h in seen:
            continue
        seen.add(h)
        windows = count_windows(text, tokenizer)
        out.append(
            {
                **row,
                "document_id": row.get("document_id") or f"fresh_{i}_{h}",
                "text": text,
                "document_label": "not_prompt_injection",
                "source_name": row.get("source_name") or row.get("source") or "fresh_jsonl",
                "category": row.get("category") or "knowledge_base",
                "language": row.get("language") or infer_language(text),
                "window_count": windows,
                "window_count_bucket": window_count_bucket(windows),
            }
        )
    return out


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)
    rows = []
    for path in args.input_jsonl:
        rows.extend(load_text_records(path, default_source=Path(path).stem))
    rows = annotate(rows, tokenizer)
    rnd = random.Random(args.seed)
    rnd.shuffle(rows)

    production = [row for row in rows if row.get("category") in PRODUCTION_CATEGORIES]
    external = [row for row in rows if row.get("category") not in PRODUCTION_CATEGORIES]
    if len(external) < args.external_target:
        production_hashes = {text_hash(row["text"]) for row in production[: args.production_target]}
        external.extend([row for row in rows if text_hash(row["text"]) not in production_hashes])

    production = production[: args.production_target]
    external = external[: args.external_target]
    write_jsonl(args.output_production_jsonl, production)
    write_jsonl(args.output_external_jsonl, external)
    report = {
        "input_jsonl": args.input_jsonl,
        "production_jsonl": args.output_production_jsonl,
        "production_rows": len(production),
        "external_jsonl": args.output_external_jsonl,
        "external_rows": len(external),
        "production_categories": dict(Counter(row.get("category", "unknown") for row in production)),
        "external_sources": dict(Counter(row.get("source_name", "unknown") for row in external)),
        "note": "Freshness depends on the supplied inputs. Do not pass prior training/mining corpora for locked acceptance.",
    }
    write_json(args.report_json, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
