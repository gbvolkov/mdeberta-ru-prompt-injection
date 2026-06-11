from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.ipc as ipc


def compile_phrase_re(values: list[str]) -> re.Pattern[str]:
    return re.compile("|".join(re.escape(value) for value in values), re.IGNORECASE)


ATTACK_TARGET_RE = compile_phrase_re([
    "system prompt", "developer prompt", "prompt stack", "hidden routing",
    "tool names", "tool registry", "route selection", "private context",
    "internal configuration", "developer instructions", "system_message",
    "developer_policy", "available_tools", "routing_rules", "system layer",
    "developer layer", "tool schema", "routing metadata",
    "системн", "промпт", "разработчик",
    "инструмент", "скрыт", "маршрут", "служебн",
    "конфигурац", "внутренн", "правил", "приват",
])

ATTACK_VERB_RE = compile_phrase_re([
    "show", "print", "reveal", "list", "output", "dump", "export", "return",
    "покажи", "выведи", "раскрой", "перечисли",
    "напечатай", "сообщи", "верни", "составь", "сформируй",
])

ATTACK_FRAME_RE = compile_phrase_re([
    "override", "internal_review", "BEGIN INTERNAL", "END INTERNAL",
    "internal auditor", "forensic", "parser", "diagnostic", "audit",
    "внутренний оператор", "внутреннего оператора",
    "аудит", "диагностик", "служебная", "служебный",
    "административная команда", "режим", "контроль",
])

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a strict exact-window dataset by dropping attack-labelled rows without visible attack text."
    )
    parser.add_argument("--input-dir", default="training-dataset-v16-critical-recall-restoration-windowed")
    parser.add_argument("--output-dir", default="training-dataset-v16-strict-exact-windowed")
    parser.add_argument("--validation-output-dir", default="training-dataset-v16-strict-exact-windowed-validation")
    parser.add_argument("--report-json", default="training-dataset-v16-strict-exact-windowed-report.json")
    parser.add_argument("--dropped-jsonl", default="training-dataset-v16-strict-exact-windowed-dropped-attack-rows.jsonl")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_arrow_split(split_dir: Path) -> tuple[pa.Schema, list[dict[str, Any]]]:
    paths = sorted(split_dir.glob("data-*.arrow"))
    if not paths:
        raise FileNotFoundError(f"No Arrow data files found in {split_dir}")
    tables = []
    for path in paths:
        with path.open("rb") as f:
            tables.append(ipc.open_stream(f).read_all())
    table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
    return table.schema, table.to_pylist()


def visible_attack(text: str) -> bool:
    text = str(text or "")
    if not text.strip():
        return False
    has_target = bool(ATTACK_TARGET_RE.search(text))
    has_verb = bool(ATTACK_VERB_RE.search(text))
    has_frame = bool(ATTACK_FRAME_RE.search(text))
    if has_target and (has_verb or has_frame):
        return True
    # Accept structured requests that name internal output fields even when the verb is implicit.
    lower = text.lower()
    structured_targets = sum(
        token in lower
        for token in ("system", "developer", "tools", "routing", "private_context", "hidden_policies")
    )
    return structured_targets >= 3 and ("json" in lower or "yaml" in lower or "internal" in lower)


def filter_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for row in rows:
        label = int(row.get("label") or 0)
        if label == 1 and not visible_attack(str(row.get("text") or "")):
            dropped.append(row)
            continue
        kept.append(row)
    return kept, dropped


def fingerprint(rows: list[dict[str, Any]]) -> str:
    h = hashlib.sha256()
    for row in rows:
        h.update(str(row.get("text_hash") or row.get("text") or "").encode("utf-8", errors="replace"))
        h.update(str(row.get("label") or "").encode("ascii", errors="ignore"))
    return h.hexdigest()[:16]


def write_arrow_dataset(split_dir: Path, schema: pa.Schema, rows: list[dict[str, Any]], source_info: Path | None) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    clean_rows = [{name: row.get(name) for name in schema.names} for row in rows]
    table = pa.Table.from_pylist(clean_rows, schema=schema)
    with (split_dir / "data-00000-of-00001.arrow").open("wb") as f:
        with ipc.new_stream(f, table.schema) as writer:
            writer.write_table(table)
    if source_info and source_info.exists():
        shutil.copy2(source_info, split_dir / "dataset_info.json")
    state = {
        "_data_files": [{"filename": "data-00000-of-00001.arrow"}],
        "_fingerprint": fingerprint(rows),
        "_format_columns": None,
        "_format_kwargs": {},
        "_format_type": None,
        "_output_all_columns": False,
        "_split": None,
    }
    (split_dir / "state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "labels": dict(Counter("attack" if int(row.get("label") or 0) == 1 else "benign" for row in rows)),
        "components": dict(Counter(str(row.get("component")) for row in rows).most_common()),
        "categories": dict(Counter(str(row.get("category")) for row in rows).most_common()),
        "sources": dict(Counter(str(row.get("source_name")) for row in rows).most_common(50)),
        "languages": dict(Counter(str(row.get("language")) for row in rows).most_common()),
    }


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} already exists. Use --overwrite to replace it.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    validation_output_dir = Path(args.validation_output_dir)

    train_schema, train_rows_raw = read_arrow_split(input_dir / "train")
    validation_schema, validation_rows_raw = read_arrow_split(input_dir / "validation")
    if train_schema.names != validation_schema.names:
        raise ValueError("Train and validation schemas differ.")

    train_rows, dropped_train = filter_rows(train_rows_raw)
    validation_rows, dropped_validation = filter_rows(validation_rows_raw)

    prepare_output(output_dir, args.overwrite)
    write_arrow_dataset(output_dir / "train", train_schema, train_rows, input_dir / "train" / "dataset_info.json")
    write_arrow_dataset(output_dir / "validation", validation_schema, validation_rows, input_dir / "validation" / "dataset_info.json")
    (output_dir / "dataset_dict.json").write_text(json.dumps({"splits": ["train", "validation"]}) + "\n", encoding="utf-8")

    prepare_output(validation_output_dir, args.overwrite)
    write_arrow_dataset(validation_output_dir, validation_schema, validation_rows, input_dir / "validation" / "dataset_info.json")

    dropped = [{**row, "dropped_reason": "attack_label_without_visible_attack_text"} for row in dropped_train + dropped_validation]
    write_jsonl(Path(args.dropped_jsonl), dropped)

    report = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "validation_output_dir": str(validation_output_dir),
        "policy": (
            "Rows are exact existing windows. Benign rows are kept unchanged. "
            "Attack-labelled rows are kept only when the exact text contains visible attack intent "
            "targeting hidden/system/developer/tool/routing/internal content."
        ),
        "train_before": summarize(train_rows_raw),
        "train_after": summarize(train_rows),
        "validation_before": summarize(validation_rows_raw),
        "validation_after": summarize(validation_rows),
        "dropped_attack_rows": {
            "train": len(dropped_train),
            "validation": len(dropped_validation),
            "total": len(dropped),
            "jsonl": args.dropped_jsonl,
            "by_component": dict(Counter(str(row.get("component")) for row in dropped).most_common()),
            "by_source": dict(Counter(str(row.get("source_name")) for row in dropped).most_common(50)),
        },
    }
    Path(args.report_json).write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
