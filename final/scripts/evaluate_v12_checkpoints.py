from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from transformers import AutoTokenizer

from run_blind_broad_eval import ExclusionIndex, evaluate_documents, iter_corpus, parse_thresholds
from v12_pipeline_utils import LABEL_ATTACK, LABEL_BENIGN, label_name, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate all retained V12/V13 checkpoints on document-level dev corpora.")
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--corpus-jsonl", action="append", required=True, help="Name/path pair: name=path.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--thresholds", default="0.82,0.90,0.95,0.99,0.999,0.9995")
    parser.add_argument("--primary-threshold", type=float, default=0.95)
    parser.add_argument("--window-batch-size", type=int, default=128)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--include-final", action="store_true", default=True)
    parser.add_argument("--limit-documents", type=int)
    return parser.parse_args()


def checkpoint_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"checkpoint-(\d+)$", path.name)
    if match:
        return (int(match.group(1)), path.name)
    return (10**12, path.name)


def discover_models(root: str | Path, include_final: bool) -> list[tuple[str, Path]]:
    root = Path(root)
    models: list[tuple[str, Path]] = []
    for path in sorted(root.glob("checkpoint-*"), key=checkpoint_sort_key):
        if (path / "config.json").exists():
            models.append((path.name, path))
    if include_final and (root / "config.json").exists():
        models.append(("final", root))
    if not models:
        raise FileNotFoundError(f"No checkpoint-* directories or final model found under {root}")
    return models


def parse_corpora(values: list[str]) -> list[tuple[str, Path]]:
    corpora = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"--corpus-jsonl must be name=path, got: {value}")
        name, path = value.split("=", 1)
        corpora.append((name.strip(), Path(path.strip())))
    return corpora


def compact_threshold_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for threshold, metrics in summary.get("threshold_metrics", {}).items():
        out[threshold] = {
            "f1": metrics.get("f1"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "benign_fp_rate": metrics.get("benign_fp_rate"),
            "tp": metrics.get("tp"),
            "fp": metrics.get("fp"),
            "tn": metrics.get("tn"),
            "fn": metrics.get("fn"),
        }
    return out


def choose_candidates(results: list[dict[str, Any]], thresholds: list[float]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for item in results:
        aggregate: dict[str, dict[str, float]] = {str(t): {"tp": 0, "fp": 0, "tn": 0, "fn": 0} for t in thresholds}
        for corpus in item["corpora"].values():
            for threshold, metrics in corpus["threshold_metrics"].items():
                if threshold not in aggregate:
                    continue
                for key in ("tp", "fp", "tn", "fn"):
                    aggregate[threshold][key] += int(metrics.get(key, 0))
        for threshold, counts in aggregate.items():
            tp, fp, tn, fn = counts["tp"], counts["fp"], counts["tn"], counts["fn"]
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            benign_fp_rate = fp / (fp + tn) if fp + tn else 0.0
            candidates.append(
                {
                    "model_name": item["model_name"],
                    "model_path": item["model_path"],
                    "threshold": float(threshold),
                    "f1": f1,
                    "precision": precision,
                    "recall": recall,
                    "benign_fp_rate": benign_fp_rate,
                    "tp": tp,
                    "fp": fp,
                    "tn": tn,
                    "fn": fn,
                }
            )
    return sorted(candidates, key=lambda r: (r["recall"] >= 0.99, -r["benign_fp_rate"], r["f1"]), reverse=True)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = parse_thresholds(args.thresholds)
    if args.primary_threshold not in thresholds:
        thresholds = sorted(set(thresholds + [args.primary_threshold]))
    corpora = parse_corpora(args.corpus_jsonl)
    models = discover_models(args.checkpoint_root, args.include_final)

    all_results: list[dict[str, Any]] = []
    for model_name, model_path in models:
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        model_record = {"model_name": model_name, "model_path": str(model_path), "corpora": {}}
        for corpus_name, corpus_path in corpora:
            docs = list(iter_corpus(corpus_path))
            if args.limit_documents:
                docs = docs[: args.limit_documents]
            document_results = output_dir / f"{model_name}-{corpus_name}-documents.jsonl"
            window_results = output_dir / f"{model_name}-{corpus_name}-windows.jsonl"
            summary_path = output_dir / f"{model_name}-{corpus_name}-summary.json"
            eval_args = SimpleNamespace(
                model_id=str(model_path),
                device=args.device,
                window_batch_size=args.window_batch_size,
                primary_threshold=args.primary_threshold,
            )
            summary = evaluate_documents(
                eval_args,
                docs,
                thresholds,
                tokenizer,
                document_results,
                window_results,
                summary_path,
                ExclusionIndex(set(), set(), {"mode": "checkpoint_dev_eval"}),
                corpus_path,
            )
            model_record["corpora"][corpus_name] = {
                "summary_json": str(summary_path),
                "document_results": str(document_results),
                "window_results": str(window_results),
                "threshold_metrics": compact_threshold_metrics(summary),
                "labels": summary.get("labels", {}),
                "documents": summary.get("documents"),
            }
        all_results.append(model_record)

    candidates = choose_candidates(all_results, thresholds)
    report = {
        "checkpoint_root": args.checkpoint_root,
        "corpora": {name: str(path) for name, path in corpora},
        "thresholds": thresholds,
        "results": all_results,
        "candidate_ranking": candidates,
        "best_candidate": candidates[0] if candidates else None,
        "gate_note": "Use this only on dev/calibration corpora. Locked acceptance must be run once after checkpoint and threshold are fixed.",
    }
    write_json(output_dir / "checkpoint-selection-summary.json", report)
    print(json.dumps(report["best_candidate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
