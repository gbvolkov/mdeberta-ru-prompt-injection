# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm
from transformers import AutoTokenizer

from v12_pipeline_utils import extract_text, iter_jsonl, normalize_text, stable_hash, text_hash, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split very large V18 fresh source documents into builder-sized source chunks.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--tokenizer-id", default="microsoft/mdeberta-v3-base")
    parser.add_argument("--chunk-token-length", type=int, default=2048)
    parser.add_argument("--chunk-token-stride", type=int, default=1536)
    parser.add_argument("--min-chars", type=int, default=400)
    parser.add_argument("--max-chunks-per-document", type=int, default=1000)
    return parser.parse_args()


def safe_str(row: dict[str, Any], key: str, default: str = "") -> str:
    value = row.get(key)
    if value is None or value == "":
        return default
    return str(value)


def main() -> None:
    args = parse_args()
    if args.chunk_token_stride <= 0:
        raise ValueError("--chunk-token-stride must be positive")
    if args.chunk_token_length <= 0:
        raise ValueError("--chunk-token-length must be positive")
    if args.chunk_token_stride > args.chunk_token_length:
        raise ValueError("--chunk-token-stride must be <= --chunk-token-length")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)
    output_rows: list[dict[str, Any]] = []
    rejected = Counter()
    source_docs = 0
    source_tokens = 0
    chunks_by_source = Counter()

    for row in tqdm(iter_jsonl(args.input_jsonl), desc="Chunking source documents", unit="doc", dynamic_ncols=True, ascii=True):
        source_docs += 1
        text = normalize_text(extract_text(row))
        if len(text) < args.min_chars:
            rejected["source_too_short"] += 1
            continue
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        source_tokens += len(token_ids)
        if not token_ids:
            rejected["source_empty_tokens"] += 1
            continue
        source_document_id = safe_str(row, "document_id", text_hash(text))
        source_name = safe_str(row, "source_name", "unknown_source")
        source_origin = safe_str(row, "source_origin", safe_str(row, "source_path", source_name))
        category = safe_str(row, "category", "knowledge_base")
        language = safe_str(row, "language", "")
        source_pool = safe_str(row, "source_pool", safe_str(row, "source_pool_assignment", "external_mining_only"))
        chunk_index = 0
        start = 0
        while start < len(token_ids) and chunk_index < args.max_chunks_per_document:
            end = min(len(token_ids), start + args.chunk_token_length)
            chunk_tokens = token_ids[start:end]
            chunk_text = normalize_text(tokenizer.decode(chunk_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True))
            if len(chunk_text) >= args.min_chars:
                chunk_hash = text_hash(chunk_text)
                output_rows.append(
                    {
                        "document_id": f"{source_document_id}:chunk:{chunk_index}:{chunk_hash}",
                        "original_document_id": source_document_id,
                        "source_document_id": source_document_id,
                        "source_name": f"{source_name}:doc:{stable_hash(source_document_id, length=12)}",
                        "source_family": source_name,
                        "source_origin": f"{source_origin}#chunk={chunk_index}",
                        "source_pool": source_pool,
                        "category": category,
                        "language": language,
                        "text": chunk_text,
                        "chunk_index": chunk_index,
                        "chunk_token_start": start,
                        "chunk_token_end": end,
                        "chunk_token_length": end - start,
                        "original_text_hash": safe_str(row, "text_hash", text_hash(text)),
                        "text_hash": chunk_hash,
                    }
                )
                chunks_by_source[source_name] += 1
            else:
                rejected["chunk_too_short"] += 1
            chunk_index += 1
            if end >= len(token_ids):
                break
            start += args.chunk_token_stride

    write_jsonl(args.output_jsonl, output_rows)
    report = {
        "input_jsonl": args.input_jsonl,
        "output_jsonl": args.output_jsonl,
        "source_documents": source_docs,
        "source_tokens": source_tokens,
        "written_chunks": len(output_rows),
        "chunk_token_length": args.chunk_token_length,
        "chunk_token_stride": args.chunk_token_stride,
        "categories": dict(Counter(row["category"] for row in output_rows).most_common()),
        "languages": dict(Counter(row["language"] for row in output_rows).most_common()),
        "source_families": dict(Counter(row["source_family"] for row in output_rows).most_common()),
        "chunks_by_original_source_name": dict(chunks_by_source.most_common()),
        "rejected": dict(rejected),
        "status": "pass" if output_rows else "fail",
    }
    write_json(args.report_json, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
