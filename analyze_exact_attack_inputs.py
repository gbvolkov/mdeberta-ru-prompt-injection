from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


INPUT_JSONL = Path("v16-exact-attack-inputs-train.jsonl")
OUTPUT_JSON = Path("v16-exact-attack-inputs-analysis.json")
OUTPUT_MD = Path("v16-exact-attack-inputs-analysis.md")


CUE_PATTERNS = {
    "disclosure_target_system_developer_tools_routing": re.compile(
        r"системн|промпт|developer|tool names|tools|hidden routing|routing|маршрут|инструмент",
        re.IGNORECASE,
    ),
    "hidden_internal_private_config": re.compile(
        r"скрыт|невидим|закрыт|приват|private|hidden|internal|внутренн|служебн|конфигурац|контекст|policy|rules|route",
        re.IGNORECASE,
    ),
    "imperative_disclosure_verb": re.compile(
        r"покажи|выведи|раскрой|перечисли|напечатай|сообщи|верни|составь|сформируй|show|print|reveal|list|output|dump|export",
        re.IGNORECASE,
    ),
    "override_audit_role_frame": re.compile(
        r"override|internal_review|внутренн(?:ий|его)? оператор|аудит|диагностик|служебн(?:ая|ый)|административн|режим|internal auditor|forensic|parser|контроль",
        re.IGNORECASE,
    ),
    "wrapper_or_structured_marker": re.compile(
        r"BEGIN INTERNAL|END INTERNAL|BEGIN OF DOCUMENT|END OF DOCUMENT|```|\{\"|\[metadata\]|redacted|masked-id|internal_review",
        re.IGNORECASE,
    ),
    "embedded_context_marker": re.compile(
        r"Контекст:|context:|========|BEGIN OF DOCUMENT|END OF DOCUMENT",
        re.IGNORECASE,
    ),
}


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def compact_text(text: str, limit: int = 700) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text if len(text) <= limit else text[:limit].rstrip() + " ..."


def cue_hits(text: str) -> dict[str, bool]:
    return {name: bool(pattern.search(text or "")) for name, pattern in CUE_PATTERNS.items()}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_component: dict[str, Counter[str]] = defaultdict(Counter)
    examples_by_component: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        component = str(row.get("component"))
        hits = cue_hits(str(row.get("text") or ""))
        by_component[component]["rows"] += 1
        for name, hit in hits.items():
            if hit:
                by_component[component][name] += 1
        if all(hits.values()):
            by_component[component]["all_cue_groups"] += 1
        if not any(hits.values()):
            by_component[component]["no_cue_groups"] += 1
        if len(examples_by_component[component]) < 4:
            examples_by_component[component].append(
                {
                    "document_id": row.get("document_id"),
                    "source_name": row.get("source_name"),
                    "category": row.get("category"),
                    "language": row.get("language"),
                    "generation_type": row.get("generation_type"),
                    "score_band": row.get("score_band"),
                    "cue_hits": hits,
                    "text": compact_text(str(row.get("text") or "")),
                }
            )

    return {
        "rows": len(rows),
        "by_component": {component: dict(counter) for component, counter in sorted(by_component.items())},
        "examples_by_component": dict(examples_by_component),
    }


def write_md(report: dict[str, Any]) -> None:
    lines = [
        "# Exact V16 Attack Training Inputs Analysis",
        "",
        "This analyzes exact `text` values that were passed to the model as `label == 1` in `training-dataset-v16-critical-recall-restoration-windowed/train`.",
        "",
        "No examples are reconstructed here.",
        "",
        "## Component Summary",
        "",
        "| Component | Rows | Main reason these were labelled as attack |",
        "| --- | ---: | --- |",
    ]

    reasons = {
        "critical_ru_near_anchor_positive": "Anchored variants around previous critical-RU misses; usually explicit requests to show hidden system/developer/tool/routing context plus carrier context.",
        "critical_ru_embedded_positive": "Attack instruction embedded into production-like benign document text.",
        "critical_ru_v15_regression_guard_positive": "Minimal/light mutations of cases V15 still missed; used as regression guards.",
        "critical_ru_semantic_hard_positive": "Standalone semantic paraphrases asking for hidden/internal prompt stack, tools, routing, private context, or internal policy.",
        "critical_ru_wrapper_positive": "Same malicious intent inside document wrappers, JSON/metadata, redaction or BEGIN/END markers.",
        "critical_ru_exact_v13_miss": "Exact or near-exact V13 missed attack-labelled rows preserved as difficult positives.",
    }

    for component, counts in report["by_component"].items():
        lines.append(f"| `{component}` | {counts['rows']} | {reasons.get(component, '')} |")

    lines.extend(["", "## Cue Counts By Component", ""])
    for component, counts in report["by_component"].items():
        rows = counts["rows"]
        lines.extend([f"### `{component}`", ""])
        for key in CUE_PATTERNS:
            value = counts.get(key, 0)
            lines.append(f"- `{key}`: `{value}/{rows}`")
        lines.append(f"- `all_cue_groups`: `{counts.get('all_cue_groups', 0)}/{rows}`")
        lines.append(f"- `no_cue_groups`: `{counts.get('no_cue_groups', 0)}/{rows}`")
        lines.append("")

    lines.extend(["## Representative Exact Samples", ""])
    for component, examples in report["examples_by_component"].items():
        lines.extend([f"### `{component}`", ""])
        for example in examples:
            lines.extend(
                [
                    f"#### `{example['document_id']}`",
                    "",
                    f"- Source: `{example['source_name']}`",
                    f"- Category: `{example['category']}`",
                    f"- Language: `{example['language']}`",
                    f"- Generation type: `{example['generation_type']}`",
                    f"- Score band: `{example['score_band']}`",
                    "",
                    "```text",
                    example["text"].replace("```", "` ` `"),
                    "```",
                    "",
                ]
            )

    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = list(iter_jsonl(INPUT_JSONL))
    report = summarize(rows)
    OUTPUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_md(report)
    print(json.dumps({"input": str(INPUT_JSONL), "rows": len(rows), "output_json": str(OUTPUT_JSON), "output_md": str(OUTPUT_MD)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
