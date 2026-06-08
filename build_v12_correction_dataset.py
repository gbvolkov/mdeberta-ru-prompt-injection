from __future__ import annotations

import argparse
import glob
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoTokenizer

from v12_pipeline_utils import (
    ATTACK_WRAPPERS,
    LABEL_ATTACK,
    LABEL_BENIGN,
    build_production_windows,
    generate_critical_ru_attacks,
    infer_language,
    iter_jsonl,
    label_id,
    label_name,
    normalize_text,
    read_jsonl,
    split_train_validation,
    stable_hash,
    text_hash,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a small V12 Russian critical correction dataset.")
    parser.add_argument("--tokenizer-id", required=True)
    parser.add_argument("--v11-dataset-dir", required=True)
    parser.add_argument("--carrier-jsonl", required=True)
    parser.add_argument("--locked-corpus-glob", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--validation-output-dir")
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--target-train-rows", type=int, default=35000)
    parser.add_argument("--target-validation-rows", type=int, default=5000)
    parser.add_argument("--max-train-rows", type=int, default=55000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-underfilled", action="store_true")
    return parser.parse_args()


def build_locked_index(patterns: list[str]) -> dict[str, set[str]]:
    text_hashes: set[str] = set()
    doc_ids: set[str] = set()
    attack_hashes: set[str] = set()
    paths: set[str] = set()
    for pattern in patterns or []:
        for path_text in glob.glob(pattern):
            path = Path(path_text)
            if not path.exists() or path.is_dir():
                continue
            paths.add(str(path))
            for row in iter_jsonl(path):
                text = row.get("text") or row.get("document_text") or row.get("window_text") or ""
                if normalize_text(text):
                    text_hashes.add(text_hash(text))
                doc_id = row.get("document_id") or row.get("id")
                if doc_id is not None:
                    doc_ids.add(str(doc_id))
                attack_hash = row.get("attack_text_hash")
                if attack_hash:
                    attack_hashes.add(str(attack_hash))
    return {
        "text_hashes": text_hashes,
        "document_ids": doc_ids,
        "attack_text_hashes": attack_hashes,
        "paths": paths,
    }


def load_carriers(path: str | Path, tokenizer: Any, locked_index: dict[str, set[str]], *, max_rows: int = 20000) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, row in enumerate(iter_jsonl(path)):
        if len(rows) >= max_rows:
            break
        text = row.get("text") or row.get("document_text") or ""
        if len(normalize_text(text)) < 400:
            continue
        doc_id = str(row.get("document_id") or row.get("id") or f"carrier_{i}")
        if doc_id in locked_index.get("document_ids", set()):
            continue
        if text_hash(text) in locked_index.get("text_hashes", set()):
            continue
        windows = build_production_windows(text, tokenizer)
        if not windows:
            continue
        rows.append(
            {
                "document_id": doc_id,
                "text": text,
                "source_name": row.get("source_name") or row.get("source") or "carrier_jsonl",
                "category": row.get("category") or "knowledge_base",
                "language": row.get("language") or infer_language(text),
                "window_count": len(windows),
            }
        )
    return rows


def load_v11_replay(v11_dataset_dir: str | Path, locked_index: dict[str, set[str]], *, seed: int, benign_target: int, attack_target: int) -> list[dict[str, Any]]:
    dataset = load_from_disk(str(v11_dataset_dir))
    train = dataset["train"] if "train" in dataset else next(iter(dataset.values()))
    benign: list[dict[str, Any]] = []
    attack: list[dict[str, Any]] = []
    for row in train:
        text = row.get("text", "")
        if not normalize_text(text) or text_hash(text) in locked_index.get("text_hashes", set()):
            continue
        label = label_name(row.get("label", row.get("labels", LABEL_BENIGN)))
        out = {
            "text": text,
            "label": label_id(label),
            "source_name": f"v11_replay_{row.get('source_name', 'unknown')}",
            "component": "v11_attack_replay" if label == LABEL_ATTACK else "v11_benign_replay",
        }
        if label == LABEL_ATTACK:
            attack.append(out)
        else:
            benign.append(out)
    rnd = random.Random(seed)
    rnd.shuffle(benign)
    rnd.shuffle(attack)
    return benign[:benign_target] + attack[:attack_target]


def make_positive_rows(carriers: list[dict[str, Any]], tokenizer: Any, locked_index: dict[str, set[str]], *, seed: int) -> list[dict[str, Any]]:
    rnd = random.Random(seed)
    attacks = generate_critical_ru_attacks()
    rows: list[dict[str, Any]] = []
    for i in range(9000):
        attack = attacks[i % len(attacks)]
        text = attack["attack_text"]
        if attack["attack_text_hash"] in locked_index.get("attack_text_hashes", set()):
            continue
        if text_hash(text) in locked_index.get("text_hashes", set()):
            continue
        rows.append(
            {
                "text": text,
                "label": 1,
                "source_name": "v12_ru_critical_standalone",
                "component": "critical_ru_standalone_positive",
                "attack_text_hash": attack["attack_text_hash"],
                "semantic_family": attack["semantic_family"],
            }
        )
    for i in range(12000):
        if not carriers:
            break
        carrier = carriers[i % len(carriers)]
        attack = attacks[(i * 7) % len(attacks)]
        base = carrier["text"]
        cut = rnd.randint(max(1, len(base) // 5), max(2, len(base) * 4 // 5))
        merged = f"{base[:cut]}\n\n{attack['attack_text']}\n\n{base[cut:]}"
        if i % 3 == 0:
            merged = rnd.choice(ATTACK_WRAPPERS).format(text=merged)
        for window in build_production_windows(merged, tokenizer):
            if attack["attack_text"] not in window["text"]:
                continue
            if text_hash(window["text"]) in locked_index.get("text_hashes", set()):
                continue
            rows.append(
                {
                    "text": window["text"],
                    "label": 1,
                    "source_name": f"v12_ru_critical_embedded_{carrier['category']}",
                    "component": "critical_ru_embedded_positive",
                    "attack_text_hash": attack["attack_text_hash"],
                    "semantic_family": attack["semantic_family"],
                    "carrier_document_id": carrier["document_id"],
                }
            )
            break
    return rows


def make_benign_hard_negatives(carriers: list[dict[str, Any]], tokenizer: Any, locked_index: dict[str, set[str]], *, target: int, seed: int) -> list[dict[str, Any]]:
    rnd = random.Random(seed)
    rows: list[dict[str, Any]] = []
    keywords = ("секрет", "скрыт", "конфиденц", "расслед", "инцидент", "security", "secret", "hidden", "investigation")
    shuffled = list(carriers)
    rnd.shuffle(shuffled)
    for carrier in shuffled:
        for window in build_production_windows(carrier["text"], tokenizer):
            text = window["text"]
            if not any(k in text.lower() for k in keywords):
                continue
            if text_hash(text) in locked_index.get("text_hashes", set()):
                continue
            rows.append(
                {
                    "text": text,
                    "label": 0,
                    "source_name": f"v12_benign_hard_negative_{carrier['category']}",
                    "component": "benign_news_security_hard_negative",
                    "carrier_document_id": carrier["document_id"],
                }
            )
            if len(rows) >= target:
                return rows
    return rows


def dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        h = text_hash(row.get("text", ""))
        if h in seen:
            continue
        seen.add(h)
        out.append(row)
    return out


def save_dataset(train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]], output_dir: str | Path, validation_output_dir: str | Path | None) -> None:
    train = Dataset.from_list(train_rows)
    validation = Dataset.from_list(val_rows)
    DatasetDict({"train": train, "validation": validation}).save_to_disk(str(output_dir))
    if validation_output_dir:
        DatasetDict({"validation": validation}).save_to_disk(str(validation_output_dir))


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)
    locked_index = build_locked_index(args.locked_corpus_glob)
    carriers = load_carriers(args.carrier_jsonl, tokenizer, locked_index)

    print("Building critical Russian positive windows...")
    rows = make_positive_rows(carriers, tokenizer, locked_index, seed=args.seed)

    print("Loading reviewed/diagnostic benign hard negatives...")
    rows.extend(make_benign_hard_negatives(carriers, tokenizer, locked_index, target=5000, seed=args.seed + 1))

    print("Adding selected V11 replay...")
    rows.extend(load_v11_replay(args.v11_dataset_dir, locked_index, seed=args.seed, benign_target=12000, attack_target=20000))

    rows = dedupe_rows(rows)
    rnd = random.Random(args.seed)
    rnd.shuffle(rows)
    if len(rows) > args.max_train_rows + args.target_validation_rows:
        rows = rows[: args.max_train_rows + args.target_validation_rows]
    if len(rows) < args.target_train_rows and not args.allow_underfilled:
        raise ValueError(f"Only built {len(rows):,} rows. Use --allow-underfilled to inspect partial output.")

    validation_size = min(args.target_validation_rows, max(0, len(rows) - args.target_train_rows))
    train_rows, val_rows = split_train_validation(rows, validation_size, seed=args.seed)

    for row in train_rows + val_rows:
        row["label"] = int(row["label"])

    save_dataset(train_rows, val_rows, args.output_dir, args.validation_output_dir)

    report = {
        "output_dir": args.output_dir,
        "train_rows": len(train_rows),
        "validation_rows": len(val_rows),
        "train_labels": Counter("prompt_injection" if r["label"] else "benign" for r in train_rows),
        "validation_labels": Counter("prompt_injection" if r["label"] else "benign" for r in val_rows),
        "components": Counter(r.get("component", "unknown") for r in train_rows),
        "locked_corpus_paths": sorted(locked_index.get("paths", set())),
    }
    report["train_labels"] = dict(report["train_labels"])
    report["validation_labels"] = dict(report["validation_labels"])
    report["components"] = dict(report["components"])
    write_json(args.report_json, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
