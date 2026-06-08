from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from v12_pipeline_utils import (
    LABEL_ATTACK,
    LABEL_BENIGN,
    binomial_wilson_interval,
    build_production_windows,
    infer_language,
    iter_jsonl,
    label_name,
    normalize_text,
    text_hash,
    window_count_bucket,
    write_json,
    write_jsonl,
)


@dataclass
class ExclusionIndex:
    text_hashes: set[str]
    source_names: set[str]
    artifacts: dict[str, Any]


def parse_thresholds(value: str | None) -> list[float]:
    if not value:
        return [0.82, 0.90, 0.95, 0.99, 0.999, 0.9995]
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a prompt-injection classifier on full documents with production windowing.")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--window-output-jsonl")
    parser.add_argument("--thresholds", default="0.82,0.90,0.95,0.99,0.999,0.9995")
    parser.add_argument("--primary-threshold", type=float, default=0.95)
    parser.add_argument("--window-batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--limit-documents", type=int)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="document_label")
    parser.add_argument("--id-column", default="document_id")
    parser.add_argument("--progress-every-docs", type=int, default=25)
    parser.add_argument("--progress-every-windows", type=int, default=1000)
    return parser.parse_args()


def iter_corpus(path: str | Path, *, text_column: str = "text", label_column: str = "document_label", id_column: str = "document_id") -> Iterable[dict[str, Any]]:
    for idx, row in enumerate(iter_jsonl(path)):
        text = row.get(text_column) or row.get("document_text") or row.get("content") or row.get("window_text") or ""
        if not normalize_text(text):
            continue
        label = label_name(row.get(label_column, row.get("label", LABEL_BENIGN)))
        yield {
            **row,
            "document_id": str(row.get(id_column) or row.get("id") or f"doc_{idx}"),
            "text": text,
            "document_label": label,
            "source_name": row.get("source_name") or row.get("source") or "unknown",
            "category": row.get("category") or "unknown",
            "language": row.get("language") or infer_language(text),
        }


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def score_windows(model: Any, tokenizer: Any, texts: list[str], *, device: torch.device, batch_size: int) -> list[float]:
    scores: list[float] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = tokenizer(batch, truncation=True, max_length=256, padding=True, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)[:, 1]
            scores.extend(float(v) for v in probs.detach().cpu())
    return scores


def threshold_metrics(results: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    tp = fp = tn = fn = 0
    for row in results:
        gold = label_name(row["document_label"]) == LABEL_ATTACK
        pred = float(row["document_max_prompt_injection_score"]) >= threshold
        if pred and gold:
            tp += 1
        elif pred and not gold:
            fp += 1
        elif not pred and gold:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    benign_total = fp + tn
    attack_total = tp + fn
    fp_rate = fp / benign_total if benign_total else 0.0
    fp_ci = binomial_wilson_interval(fp, benign_total)
    recall_ci = binomial_wilson_interval(tp, attack_total)
    return {
        "threshold": threshold,
        "documents": len(results),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "benign_fp_rate": fp_rate,
        "benign_fp_rate_ci95": fp_ci,
        "malicious_recall_ci95": recall_ci,
    }


def grouped_metrics(results: list[dict[str, Any]], threshold: float, key: str, *, attack_only: bool | None = None) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        is_attack = label_name(row["document_label"]) == LABEL_ATTACK
        if attack_only is True and not is_attack:
            continue
        if attack_only is False and is_attack:
            continue
        groups.setdefault(str(row.get(key, "unknown")), []).append(row)
    return {name: threshold_metrics(rows, threshold) for name, rows in sorted(groups.items())}


def build_eval_summary(results: list[dict[str, Any]], thresholds: list[float], primary_threshold: float, *, model_id: str, corpus_path: str | Path) -> dict[str, Any]:
    threshold_results = {str(t): threshold_metrics(results, t) for t in thresholds}
    primary = threshold_results[str(primary_threshold)] if str(primary_threshold) in threshold_results else threshold_metrics(results, primary_threshold)
    return {
        "model_id": model_id,
        "corpus_path": str(corpus_path),
        "documents": len(results),
        "labels": {
            LABEL_BENIGN: sum(1 for r in results if label_name(r["document_label"]) == LABEL_BENIGN),
            LABEL_ATTACK: sum(1 for r in results if label_name(r["document_label"]) == LABEL_ATTACK),
        },
        "windows": sum(int(r.get("window_count", 0)) for r in results),
        "primary_threshold": primary_threshold,
        "primary_metrics": primary,
        "threshold_metrics": threshold_results,
        "by_language": grouped_metrics(results, primary_threshold, "language"),
        "by_category": grouped_metrics(results, primary_threshold, "category"),
        "by_window_count_bucket": grouped_metrics(results, primary_threshold, "window_count_bucket"),
        "malicious_by_semantic_family": grouped_metrics(results, primary_threshold, "semantic_family", attack_only=True),
    }


def evaluate_documents(
    args: argparse.Namespace,
    docs: list[dict[str, Any]],
    thresholds: list[float],
    tokenizer: Any,
    document_results_path: str | Path,
    window_results_path: str | Path | None,
    summary_path: str | Path,
    exclusion: ExclusionIndex | None,
    corpus_path: str | Path,
) -> dict[str, Any]:
    device = choose_device(getattr(args, "device", "auto"))
    started = time.time()
    print(
        f"[eval] model={args.model_id} docs={len(docs):,} device={device} "
        f"window_batch_size={args.window_batch_size}",
        flush=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(args.model_id).to(device)
    print("[eval] model loaded; scoring documents", flush=True)

    doc_results: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    processed_windows = 0
    next_window_report = int(getattr(args, "progress_every_windows", 1000) or 0)
    progress_every_docs = int(getattr(args, "progress_every_docs", 25) or 0)
    for doc_idx, doc in enumerate(docs, 1):
        windows = build_production_windows(doc["text"], tokenizer)
        if not windows:
            continue
        if len(windows) >= 100:
            print(
                f"[eval] scoring long document {doc_idx:,}/{len(docs):,}: "
                f"document_id={doc['document_id']} windows={len(windows):,}",
                flush=True,
            )
        scores = score_windows(model, tokenizer, [w["text"] for w in windows], device=device, batch_size=args.window_batch_size)
        best_idx, best_score = max(enumerate(scores), key=lambda item: item[1])
        window_count = len(windows)
        processed_windows += window_count
        doc_result = {
            "document_id": doc["document_id"],
            "document_label": doc["document_label"],
            "source_name": doc.get("source_name", "unknown"),
            "category": doc.get("category", "unknown"),
            "language": doc.get("language", "unknown"),
            "semantic_family": doc.get("semantic_family", "unknown"),
            "attack_text_hash": doc.get("attack_text_hash"),
            "window_count": window_count,
            "window_count_bucket": window_count_bucket(window_count),
            "document_max_prompt_injection_score": best_score,
            "best_window_index": best_idx,
            "best_window_text_hash": text_hash(windows[best_idx]["text"]),
        }
        doc_results.append(doc_result)
        if window_results_path:
            for window, score in zip(windows, scores):
                window_rows.append(
                    {
                        "document_id": doc["document_id"],
                        "document_label": doc["document_label"],
                        "source_name": doc.get("source_name", "unknown"),
                        "category": doc.get("category", "unknown"),
                        "language": doc.get("language", "unknown"),
                        "window_index": window["window_index"],
                        "window_count": window_count,
                        "p_prompt_injection": score,
                        "window_text_hash": text_hash(window["text"]),
                        "window_text": window["text"],
                    }
                )

        should_report_docs = progress_every_docs and (doc_idx == 1 or doc_idx % progress_every_docs == 0 or doc_idx == len(docs))
        should_report_windows = next_window_report and processed_windows >= next_window_report
        if should_report_docs or should_report_windows:
            elapsed = max(0.001, time.time() - started)
            print(
                f"[eval] progress docs={doc_idx:,}/{len(docs):,} "
                f"windows={processed_windows:,} elapsed={elapsed/60:.1f}m "
                f"docs_per_min={doc_idx/elapsed*60:.2f} "
                f"last_score={best_score:.6f}",
                flush=True,
            )
            while next_window_report and processed_windows >= next_window_report:
                next_window_report += int(getattr(args, "progress_every_windows", 1000) or 1000)

    write_jsonl(document_results_path, doc_results)
    if window_results_path:
        write_jsonl(window_results_path, window_rows)
    primary = getattr(args, "primary_threshold", thresholds[0] if thresholds else 0.95)
    summary = build_eval_summary(doc_results, thresholds, primary, model_id=args.model_id, corpus_path=corpus_path)
    if exclusion:
        summary["exclusion"] = exclusion.artifacts
    write_json(summary_path, summary)
    print(json.dumps(summary["primary_metrics"], ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    args = parse_args()
    thresholds = parse_thresholds(args.thresholds)
    if args.primary_threshold not in thresholds:
        thresholds.append(args.primary_threshold)
        thresholds = sorted(set(thresholds))
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    docs = list(iter_corpus(args.input_jsonl, text_column=args.text_column, label_column=args.label_column, id_column=args.id_column))
    if args.limit_documents:
        docs = docs[: args.limit_documents]
    evaluate_documents(
        args,
        docs,
        thresholds,
        tokenizer,
        args.output_jsonl,
        args.window_output_jsonl,
        args.summary_json,
        ExclusionIndex(set(), set(), {"mode": "eval_only"}),
        args.input_jsonl,
    )


if __name__ == "__main__":
    main()
