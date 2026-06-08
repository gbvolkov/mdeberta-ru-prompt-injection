from __future__ import annotations

import json
from pathlib import Path


ERROR_JSONL = Path("v16-error-inspection-critical-ru-fn-at-0.95.jsonl")
CORPUS_JSONL = Path("v13-critical-ru-validation-corpus.jsonl")
OUTPUT_MD = Path("v16-critical-ru-fn-27-examples.md")


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> None:
    error_rows = list(iter_jsonl(ERROR_JSONL))
    error_by_id = {row["document_id"]: row for row in error_rows}

    corpus_by_id = {}
    for row in iter_jsonl(CORPUS_JSONL):
        document_id = row.get("document_id")
        if document_id in error_by_id:
            corpus_by_id[document_id] = row

    rows = sorted(error_rows, key=lambda row: float(row.get("score") or 0.0))

    lines: list[str] = [
        "# V16 Critical RU False Negatives at Threshold 0.95",
        "",
        (
            "This file contains all 27 V16 critical-RU false negatives at threshold `0.95`, "
            "sorted from lowest score to closest miss. Scores and metadata come from "
            "`validation-comparison-v13-v16-core/v16/v13_critical_ru-documents.jsonl`; "
            "full text comes from `v13-critical-ru-validation-corpus.jsonl`."
        ),
        "",
        "## Summary",
        "",
        "- Total examples: `27`",
        "- Threshold: `0.95`",
        "- All examples are labelled `prompt_injection`.",
        "- These are diagnostic examples, not blind acceptance data.",
        "",
    ]

    for index, row in enumerate(rows, 1):
        corpus_row = corpus_by_id.get(row["document_id"], {})
        text = str(corpus_row.get("text") or row.get("best_window_text") or "").strip()
        text = text.replace("```", "` ` `")
        tags = ", ".join(row.get("tags") or [])
        attack_template_id = corpus_row.get("attack_template_id") or row.get("attack_template_id") or ""
        attack_text_hash = corpus_row.get("attack_text_hash") or row.get("attack_text_hash") or ""

        lines.extend(
            [
                f"## {index}. `{row['document_id']}`",
                "",
                f"- V16 score: `{float(row.get('score') or 0.0):.12f}`",
                f"- Category: `{row.get('category')}`",
                f"- Source: `{row.get('source_name')}`",
                f"- Language: `{row.get('language')}`",
                f"- Semantic family: `{row.get('semantic_family')}`",
                f"- Attack template ID: `{attack_template_id}`",
                f"- Attack text hash: `{attack_text_hash}`",
                f"- Window count bucket: `{row.get('window_count_bucket')}`",
                f"- Tags: `{tags}`",
                "",
                "```text",
                text,
                "```",
                "",
            ]
        )

    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUTPUT_MD} with {len(rows)} examples.")


if __name__ == "__main__":
    main()
