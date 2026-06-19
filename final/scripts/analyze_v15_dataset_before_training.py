from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from v12_pipeline_utils import LABEL_ATTACK, LABEL_BENIGN, label_id, normalize_text, text_hash, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze V15 anchored correction dataset before training.")
    parser.add_argument("--dataset-dir", default="training-dataset-v15-anchored-critical-correction-windowed")
    parser.add_argument("--report-json", default="training-dataset-v15-anchored-critical-correction-windowed-analysis.json")
    parser.add_argument("--scored-jsonl", default="")
    parser.add_argument("--tokenizer-id", default="mdeberta-ru-prompt-injection-v13-critical-correction-ft")
    parser.add_argument("--model-id", default="", help="Optional V13 model id for pretraining scoring.")
    parser.add_argument("--score-split", choices=["train", "validation", "all"], default="validation")
    parser.add_argument("--thresholds", default="0.82,0.95,0.99,0.995,0.999")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--limit-rows", type=int)
    return parser.parse_args()


def parse_thresholds(value: str) -> list[float]:
    return sorted({float(part.strip()) for part in value.split(",") if part.strip()})


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def rows_for_split(data: Dataset | DatasetDict, split: str) -> list[dict[str, Any]]:
    if isinstance(data, DatasetDict):
        if split == "all":
            rows = []
            for split_name in data:
                for row in data[split_name]:
                    rows.append({**row, "_split": split_name})
            return rows
        return [{**row, "_split": split} for row in data[split]]
    return [{**row, "_split": split} for row in data]


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    text_hashes = [text_hash(row.get("text", "")) for row in rows]
    duplicate_count = len(text_hashes) - len(set(text_hashes))
    lengths = [len(normalize_text(row.get("text", ""))) for row in rows]
    return {
        "rows": len(rows),
        "labels": dict(Counter(LABEL_ATTACK if label_id(row.get("label")) else LABEL_BENIGN for row in rows)),
        "components": dict(Counter(row.get("component", "unknown") for row in rows)),
        "generation_types": dict(Counter(row.get("generation_type", "unknown") for row in rows)),
        "score_bands": dict(Counter(row.get("score_band", "") for row in rows if row.get("score_band"))),
        "categories_top50": dict(Counter(row.get("category", "unknown") for row in rows).most_common(50)),
        "sources_top50": dict(Counter(row.get("source_name", "unknown") for row in rows).most_common(50)),
        "duplicate_text_rows": duplicate_count,
        "text_length_chars": {
            "min": min(lengths) if lengths else 0,
            "median": statistics.median(lengths) if lengths else 0,
            "mean": statistics.mean(lengths) if lengths else 0,
            "max": max(lengths) if lengths else 0,
        },
    }


def score_rows(rows: list[dict[str, Any]], args: argparse.Namespace, thresholds: list[float]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    device = choose_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_id).to(device)
    model.eval()
    scored = []
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            batch_rows = rows[start : start + args.batch_size]
            texts = [row.get("text", "") for row in batch_rows]
            enc = tokenizer(texts, truncation=True, max_length=256, padding=True, return_tensors="pt")
            enc = {key: value.to(device) for key, value in enc.items()}
            probs = torch.softmax(model(**enc).logits, dim=-1)[:, 1].detach().cpu().tolist()
            for row, score in zip(batch_rows, probs):
                scored.append({**row, "v13_validation_score": float(score)})

    by_threshold: dict[str, Any] = {}
    for threshold in thresholds:
        tp = fp = tn = fn = 0
        by_component: dict[str, Counter] = defaultdict(Counter)
        for row in scored:
            gold = label_id(row.get("label")) == 1
            pred = float(row["v13_validation_score"]) >= threshold
            component = row.get("component", "unknown")
            if pred and gold:
                tp += 1
                by_component[component]["tp"] += 1
            elif pred and not gold:
                fp += 1
                by_component[component]["fp"] += 1
            elif not pred and gold:
                fn += 1
                by_component[component]["fn"] += 1
            else:
                tn += 1
                by_component[component]["tn"] += 1
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        benign_fpr = fp / (fp + tn) if fp + tn else 0.0
        by_threshold[str(threshold)] = {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "benign_fp_rate": benign_fpr,
            "by_component": {component: dict(counts) for component, counts in sorted(by_component.items())},
        }

    positives = [row["v13_validation_score"] for row in scored if label_id(row.get("label")) == 1]
    benign = [row["v13_validation_score"] for row in scored if label_id(row.get("label")) == 0]
    score_summary = {
        "rows_scored": len(scored),
        "device": str(device),
        "positive_score_distribution": describe_scores(positives),
        "benign_score_distribution": describe_scores(benign),
        "threshold_metrics": by_threshold,
    }
    return scored, score_summary


def describe_scores(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    def q(p: float) -> float:
        idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * p)))
        return ordered[idx]
    return {
        "min": ordered[0],
        "p05": q(0.05),
        "p25": q(0.25),
        "median": q(0.5),
        "p75": q(0.75),
        "p95": q(0.95),
        "max": ordered[-1],
    }


def main() -> None:
    args = parse_args()
    thresholds = parse_thresholds(args.thresholds)
    data = load_from_disk(args.dataset_dir)

    split_summaries = {}
    if isinstance(data, DatasetDict):
        for split_name in data:
            split_summaries[split_name] = summarize_rows([{**row, "_split": split_name} for row in data[split_name]])
    else:
        split_summaries["dataset"] = summarize_rows([{**row, "_split": "dataset"} for row in data])

    rows_to_score = rows_for_split(data, args.score_split)
    if args.limit_rows:
        rows_to_score = rows_to_score[: args.limit_rows]

    scoring_summary = None
    if args.model_id:
        scored, scoring_summary = score_rows(rows_to_score, args, thresholds)
        if args.scored_jsonl:
            write_jsonl(args.scored_jsonl, scored)

    report = {
        "dataset_dir": args.dataset_dir,
        "thresholds": thresholds,
        "split_summaries": split_summaries,
        "scored_split": args.score_split if args.model_id else None,
        "model_id": args.model_id or None,
        "scoring_summary": scoring_summary,
        "notes": [
            "High positive recall by V13 means those positive rows may be relearned replay rather than new correction signal.",
            "Benign rows scoring high under V13 are intended hard negatives; benign replay should mostly score low.",
            "This is a pretraining diagnostic, not an acceptance evaluation.",
        ],
    }
    write_json(args.report_json, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
