# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from json import JSONDecodeError
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "gpt-4.1-nano"
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"

MODEL_PRICES_PER_1M = {
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-5-nano": (0.05, 0.40),
    "gpt-5.4-nano": (0.20, 1.25),
}

SYSTEM_PROMPT = """You label rows for a binary prompt-injection detector.

Use exactly one label:
- prompt_injection: The text tries to override, ignore, replace, or bypass system/developer/tool instructions; asks to reveal hidden/system/developer/tool instructions; asks to reveal model/tool secrets; role-switches into an unrestricted/jailbreak persona; or contains indirect injection inside a document/email/RAG chunk.
- unsafe_non_injection: Harmful, illegal, toxic, credential-theft, command-abuse, privacy-invasive, or policy-violating content that does NOT target model/system/developer/tool instructions.
- benign: Normal text, factual/document text, ordinary question, security discussion, translation/classification of suspicious phrases, or quoted text that is not asking the model to follow the quoted attack.
- ambiguous: Insufficient context or too unclear to decide.

Important:
- A generic harmful request is not prompt injection.
- A shell command, password-reset question, or toxic content is not prompt injection unless it targets model/tool/system/developer instructions or hidden secrets.
- Return short evidence_span from the input. If no evidence exists, use an empty string.
- Prefer ambiguous over guessing.
"""


JUDGMENT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "judgments": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "candidate_id": {"type": "string"},
                    "label": {
                        "type": "string",
                        "enum": ["prompt_injection", "unsafe_non_injection", "benign", "ambiguous"],
                    },
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "evidence_span": {"type": "string"},
                    "rationale_short": {"type": "string"},
                },
                "required": ["candidate_id", "label", "confidence", "evidence_span", "rationale_short"],
            },
        }
    },
    "required": ["judgments"],
}


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\x00", " ")
    text = text.replace("\u200b", " ").replace("\ufeff", " ")
    return re.sub(r"\s+", " ", text).strip()


def candidate_id(source_name: str, text: str) -> str:
    normalized = normalize_text(text).lower()
    return hashlib.sha256(f"{source_name}\n{normalized}".encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL in {path} at line {line_number}") from exc
    return rows


def load_existing_judgment_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    for row in read_jsonl(path):
        if row.get("candidate_id"):
            ids.add(str(row["candidate_id"]))
    return ids


def compact_judgments_file(path: Path) -> None:
    if not path.exists():
        return
    rows = read_jsonl(path)
    order: list[str] = []
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        candidate = str(row.get("candidate_id", "")).strip()
        if not candidate:
            continue
        if candidate not in by_id:
            order.append(candidate)
        by_id[candidate] = row
    if len(by_id) == len(rows):
        return
    with path.open("w", encoding="utf-8") as out:
        for candidate in order:
            out.write(json.dumps(by_id[candidate], ensure_ascii=False) + "\n")


def parse_sources(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def prepare_audit_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared = []
    for row in rows:
        text = normalize_text(row.get("text"))
        source = normalize_text(row.get("source_name"))
        if not text or not source:
            continue
        clean = dict(row)
        clean["text"] = text
        clean["source_name"] = source
        clean["candidate_id"] = normalize_text(row.get("candidate_id")) or candidate_id(source, text)
        clean["metadata_score"] = int(row.get("metadata_score") or 0)
        clean["score"] = int(row.get("score") or 0)
        prepared.append(clean)
    return prepared


def select_rows(
    rows: list[dict[str, Any]],
    *,
    max_rows: int,
    seed: int,
    existing_ids: set[str],
    sources: set[str],
) -> list[dict[str, Any]]:
    eligible = [row for row in rows if row["candidate_id"] not in existing_ids]
    if sources:
        eligible = [row for row in eligible if row["source_name"] in sources]
    if max_rows <= 0 or len(eligible) <= max_rows:
        return eligible

    rng = random.Random(seed)
    for row in eligible:
        row["_sort_noise"] = rng.random()

    source_priority = {
        "jackhhao/jailbreak-classification": 0,
        "dmtrdr/russian_prompt_injections": 1,
        "OpenSafetyLab/Salad-Data": 2,
    }
    eligible.sort(
        key=lambda row: (
            -int(row.get("metadata_score", 0)),
            source_priority.get(str(row.get("source_name")), 9),
            str(row.get("reason", "")),
            row["_sort_noise"],
        )
    )
    selected = eligible[:max_rows]
    for row in eligible:
        row.pop("_sort_noise", None)
    return selected


def estimate_cost(rows: list[dict[str, Any]], model: str) -> dict[str, float]:
    chars = sum(len(row["text"]) for row in rows)
    input_tokens = max(1, int(chars / 4) + 500 * max(1, (len(rows) + 9) // 10))
    output_tokens = max(1, len(rows) * 70)
    input_price, output_price = MODEL_PRICES_PER_1M.get(model, MODEL_PRICES_PER_1M[DEFAULT_MODEL])
    return {
        "approx_input_tokens": float(input_tokens),
        "approx_output_tokens": float(output_tokens),
        "approx_cost_usd": input_tokens * input_price / 1_000_000 + output_tokens * output_price / 1_000_000,
    }


def make_user_payload(rows: list[dict[str, Any]]) -> str:
    items = [
        {
            "candidate_id": row["candidate_id"],
            "source_name": row.get("source_name", ""),
            "source_reason": row.get("reason", ""),
            "regex_score": row.get("score", 0),
            "metadata_score": row.get("metadata_score", 0),
            "metadata_evidence": row.get("metadata_evidence", []),
            "text": row["text"],
        }
        for row in rows
    ]
    return "Classify these rows. Return JSON only.\n\n" + json.dumps(items, ensure_ascii=False, indent=2)


def extract_response_text(response: dict[str, Any]) -> str:
    if isinstance(response.get("output_text"), str):
        return response["output_text"]
    parts = []
    for item in response.get("output", []):
        for content in item.get("content", []):
            if isinstance(content, dict):
                text = content.get("text")
                if isinstance(text, str):
                    parts.append(text)
    return "\n".join(parts)


def parse_judgments(text: str) -> list[dict[str, Any]]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    data = json.loads(cleaned)
    judgments = data.get("judgments")
    if not isinstance(judgments, list):
        raise ValueError("Response JSON did not contain a judgments array.")
    return [dict(item) for item in judgments if isinstance(item, dict)]


def call_openai_responses(
    *,
    api_key: str,
    url: str,
    model: str,
    rows: list[dict[str, Any]],
    timeout: int,
    reasoning_effort: str,
    max_output_tokens_per_row: int,
) -> list[dict[str, Any]]:
    payload: dict[str, Any] = {
        "model": model,
        "input": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": make_user_payload(rows)},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "label_judgments",
                "schema": JUDGMENT_SCHEMA,
                "strict": True,
            }
        },
        "max_output_tokens": max(2000, max_output_tokens_per_row * len(rows)),
    }
    if reasoning_effort and model.startswith("gpt-5"):
        payload["reasoning"] = {"effort": reasoning_effort}

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API error {exc.code}: {body}") from exc

    response_text = extract_response_text(data)
    try:
        return parse_judgments(response_text)
    except JSONDecodeError as exc:
        snippet = response_text[max(0, exc.pos - 120) : exc.pos + 120]
        raise ValueError(
            f"Could not parse model JSON response for {len(rows)} row(s): {exc}. "
            f"Nearby response text: {snippet!r}"
        ) from exc


def batched(rows: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [rows[idx : idx + batch_size] for idx in range(0, len(rows), batch_size)]


def write_judgments(
    out,
    judgments: list[dict[str, Any]],
    batch: list[dict[str, Any]],
    written_ids: set[str],
) -> set[str]:
    batch_ids = {row["candidate_id"] for row in batch}
    batch_written: set[str] = set()
    for judgment in judgments:
        candidate = str(judgment.get("candidate_id", ""))
        if candidate not in batch_ids or candidate in written_ids:
            continue
        out.write(json.dumps(judgment, ensure_ascii=False) + "\n")
        batch_written.add(candidate)
        written_ids.add(candidate)
    out.flush()
    return batch_written


def judge_batch_with_split_retry(
    *,
    api_key: str,
    url: str,
    model: str,
    batch: list[dict[str, Any]],
    timeout: int,
    reasoning_effort: str,
    max_output_tokens_per_row: int,
    min_split_batch_size: int,
) -> list[dict[str, Any]]:
    try:
        return call_openai_responses(
            api_key=api_key,
            url=url,
            model=model,
            rows=batch,
            timeout=timeout,
            reasoning_effort=reasoning_effort,
            max_output_tokens_per_row=max_output_tokens_per_row,
        )
    except (ValueError, RuntimeError) as exc:
        if len(batch) <= min_split_batch_size:
            raise
        midpoint = len(batch) // 2
        print(f"batch parse/API failure for {len(batch)} rows; retrying as {midpoint}+{len(batch) - midpoint}. Reason: {exc}")
        return [
            *judge_batch_with_split_retry(
                api_key=api_key,
                url=url,
                model=model,
                batch=batch[:midpoint],
                timeout=timeout,
                reasoning_effort=reasoning_effort,
                max_output_tokens_per_row=max_output_tokens_per_row,
                min_split_batch_size=min_split_batch_size,
            ),
            *judge_batch_with_split_retry(
                api_key=api_key,
                url=url,
                model=model,
                batch=batch[midpoint:],
                timeout=timeout,
                reasoning_effort=reasoning_effort,
                max_output_tokens_per_row=max_output_tokens_per_row,
                min_split_batch_size=min_split_batch_size,
            ),
        ]


def judge_batch_until_complete(
    *,
    api_key: str,
    url: str,
    model: str,
    batch: list[dict[str, Any]],
    timeout: int,
    reasoning_effort: str,
    max_output_tokens_per_row: int,
    min_split_batch_size: int,
    missing_retries: int,
) -> list[dict[str, Any]]:
    judgments_by_id: dict[str, dict[str, Any]] = {}
    pending = list(batch)
    for attempt in range(missing_retries + 1):
        judgments = judge_batch_with_split_retry(
            api_key=api_key,
            url=url,
            model=model,
            batch=pending,
            timeout=timeout,
            reasoning_effort=reasoning_effort,
            max_output_tokens_per_row=max_output_tokens_per_row,
            min_split_batch_size=min_split_batch_size,
        )
        pending_ids = {row["candidate_id"] for row in pending}
        for judgment in judgments:
            candidate = str(judgment.get("candidate_id", ""))
            if candidate in pending_ids:
                judgments_by_id[candidate] = judgment

        missing = [row for row in pending if row["candidate_id"] not in judgments_by_id]
        if not missing:
            return list(judgments_by_id.values())
        if attempt < missing_retries:
            print(f"model omitted {len(missing)} row(s); retrying missing rows only")
            pending = missing

    return list(judgments_by_id.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM-judge selected attack-curation audit rows and write reusable JSONL judgments."
    )
    parser.add_argument("--input-audit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-rows", type=int, default=1000, help="0 means all remaining rows.")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sources", default="", help="Optional comma-separated source filter.")
    parser.add_argument("--env-file", default=".env", help="Path to .env file containing OPENAI_API_KEY.")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--url", default=OPENAI_RESPONSES_URL)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--reasoning-effort", default="minimal")
    parser.add_argument(
        "--max-output-tokens-per-row",
        type=int,
        default=450,
        help="Output budget per row. Increase if the model returns truncated JSON.",
    )
    parser.add_argument(
        "--min-split-batch-size",
        type=int,
        default=1,
        help="Smallest retry batch size after parse/API failures.",
    )
    parser.add_argument(
        "--missing-retries",
        type=int,
        default=2,
        help="Retry rows omitted from otherwise valid model responses.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    configure_stdout()
    args = parse_args()
    load_env_file(Path(args.env_file))

    input_path = Path(args.input_audit)
    output_path = Path(args.output)
    compact_judgments_file(output_path)
    rows = prepare_audit_rows(read_jsonl(input_path))
    existing_ids = load_existing_judgment_ids(output_path)
    selected = select_rows(
        rows,
        max_rows=args.max_rows,
        seed=args.seed,
        existing_ids=existing_ids,
        sources=parse_sources(args.sources),
    )

    print(f"audit rows        : {len(rows):,}")
    print(f"existing judgments: {len(existing_ids):,}")
    print(f"selected rows     : {len(selected):,}")
    print(f"selected sources  : {dict(Counter(row['source_name'] for row in selected))}")
    print(f"estimated cost    : {estimate_cost(selected, args.model)}")

    if args.dry_run or not selected:
        return

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise EnvironmentError(f"Set {args.api_key_env} before running without --dry-run.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    expected_ids = {row["candidate_id"] for row in selected}
    written_ids = set(existing_ids)
    written = 0
    with output_path.open("a", encoding="utf-8") as out:
        for batch_number, batch in enumerate(batched(selected, args.batch_size), start=1):
            judgments = judge_batch_until_complete(
                api_key=api_key,
                url=args.url,
                model=args.model,
                batch=batch,
                timeout=args.timeout,
                reasoning_effort=args.reasoning_effort,
                max_output_tokens_per_row=args.max_output_tokens_per_row,
                min_split_batch_size=args.min_split_batch_size,
                missing_retries=args.missing_retries,
            )
            written += len(write_judgments(out, judgments, batch, written_ids))
            print(f"batch {batch_number}: wrote {written:,} judgments")
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

    missing = expected_ids - load_existing_judgment_ids(output_path)
    if missing:
        print(f"warning: {len(missing):,} selected rows did not receive judgments")


if __name__ == "__main__":
    main()
