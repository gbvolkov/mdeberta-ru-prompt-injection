from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


LABEL_BENIGN = "not_prompt_injection"
LABEL_ATTACK = "prompt_injection"
WINDOW_TOKEN_LENGTH = 254
WINDOW_TOKEN_STRIDE = 128

HARD_ATTACK_ANCHOR_PATTERNS = [
    r"ignore (all )?(previous|prior|above|earlier) instructions",
    r"disregard (all )?(previous|prior|above|earlier) instructions",
    r"(reveal|show|print|output|dump|leak|exfiltrate).{0,100}(system prompt|developer prompt|hidden instructions|internal policy|private context|tool list|routing)",
    r"(system prompt|developer prompt|hidden instructions|internal policy|private context|tool list|routing).{0,100}(reveal|show|print|output|dump|leak|exfiltrate)",
    r"(do not refuse|never refuse|bypass policy|override policy|highest priority|new system instruction)",
    r"(secret token|api key|private key|credentials).{0,80}(reveal|show|print|output|dump|leak|exfiltrate)",
    r"(раскрой|покажи|выведи|напечатай|сообщи).{0,100}(системн|промпт|разработчик|скрыт|служебн|маршрутизац|инструмент|политик)",
    r"(игнорируй|проигнорируй|обойди|переопредели).{0,100}(инструкц|правил|контекст|политик|ограничен)",
]

HARD_ATTACK_ANCHOR_RE = re.compile(
    "|".join(f"(?:{pattern})" for pattern in HARD_ATTACK_ANCHOR_PATTERNS),
    re.IGNORECASE | re.DOTALL,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate frozen V16 plus second-stage reviewer cascade.")
    parser.add_argument("--stage1-model-id", default="gbv/mdeberta-ru-prompt-injection")
    parser.add_argument("--reviewer-model-id", required=True)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--window-output-jsonl")
    parser.add_argument("--stage1-thresholds", default="0.5,0.82,0.90,0.95,0.99")
    parser.add_argument("--reviewer-thresholds", default="0.50,0.70,0.82,0.90,0.95,0.99")
    parser.add_argument("--primary-stage1-threshold", type=float, default=0.82)
    parser.add_argument("--primary-reviewer-threshold", type=float, default=0.82)
    parser.add_argument("--window-batch-size", type=int, default=128)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="document_label")
    parser.add_argument("--id-column", default="document_id")
    parser.add_argument("--limit-documents", type=int)
    parser.add_argument("--disable-hard-attack-anchor-veto", action="store_true")
    parser.add_argument("--progress-every-docs", type=int, default=25)
    return parser.parse_args()


def parse_thresholds(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def label_name(value: Any) -> str:
    if isinstance(value, bool):
        return LABEL_ATTACK if value else LABEL_BENIGN
    if isinstance(value, int):
        return LABEL_ATTACK if value == 1 else LABEL_BENIGN
    text = str(value or "").strip().lower()
    if text in {"1", "attack", "malicious", "prompt_injection", "injection"}:
        return LABEL_ATTACK
    if text in {"0", "benign", "not_prompt_injection", "non_injection", "safe"}:
        return LABEL_BENIGN
    return text or LABEL_BENIGN


def infer_language(text: str) -> str:
    cyr = len(re.findall(r"[А-Яа-яЁё]", text or ""))
    lat = len(re.findall(r"[A-Za-z]", text or ""))
    if cyr and lat and min(cyr, lat) / max(cyr, lat) >= 0.12:
        return "mixed"
    if cyr > lat:
        return "ru"
    if lat:
        return "en"
    return "unknown"


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL") from exc
            if isinstance(row, dict):
                yield row


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_production_windows(text: str, tokenizer: Any) -> list[dict[str, Any]]:
    token_ids = tokenizer.encode(text or "", add_special_tokens=False)
    if not token_ids:
        return []
    windows: list[dict[str, Any]] = []
    start = 0
    index = 0
    while start < len(token_ids):
        chunk = token_ids[start : start + WINDOW_TOKEN_LENGTH]
        window_text = tokenizer.decode(chunk, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        windows.append(
            {
                "window_index": index,
                "token_start": start,
                "token_end": min(start + len(chunk), len(token_ids)),
                "token_length": len(chunk),
                "text": window_text,
            }
        )
        if start + WINDOW_TOKEN_LENGTH >= len(token_ids):
            break
        start += WINDOW_TOKEN_STRIDE
        index += 1
    return windows


def window_count_bucket(count: int) -> str:
    if count <= 1:
        return "1"
    if count <= 4:
        return "2-4"
    if count <= 20:
        return "5-20"
    if count <= 50:
        return "21-50"
    if count <= 100:
        return "51-100"
    return "101+"


def iter_corpus(args: argparse.Namespace) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for idx, row in enumerate(iter_jsonl(args.input_jsonl)):
        text = row.get(args.text_column) or row.get("document_text") or row.get("content") or row.get("window_text") or ""
        if not normalize_text(text):
            continue
        docs.append(
            {
                **row,
                "document_id": str(row.get(args.id_column) or row.get("id") or f"doc_{idx}"),
                "text": text,
                "document_label": label_name(row.get(args.label_column, row.get("label", LABEL_BENIGN))),
                "source_name": row.get("source_name") or row.get("source") or "unknown",
                "category": row.get("category") or "unknown",
                "language": row.get("language") or infer_language(text),
                "semantic_family": row.get("semantic_family") or "unknown",
            }
        )
        if args.limit_documents and len(docs) >= args.limit_documents:
            break
    return docs


def score_texts(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    *,
    device: torch.device,
    batch_size: int,
) -> list[float]:
    scores: list[float] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = tokenizer(batch, truncation=True, max_length=256, padding=True, return_tensors="pt")
            enc = {key: value.to(device) for key, value in enc.items()}
            logits = model(**enc).logits
            if logits.shape[-1] == 1:
                probs = torch.sigmoid(logits[:, 0])
            else:
                probs = torch.softmax(logits, dim=-1)[:, 1]
            scores.extend(float(v) for v in probs.detach().cpu())
    return scores


def hard_attack_anchor_present(text: str) -> bool:
    return bool(HARD_ATTACK_ANCHOR_RE.search(text or ""))


def threshold_metrics(results: list[dict[str, Any]], key: str) -> dict[str, Any]:
    tp = fp = tn = fn = 0
    for row in results:
        gold = label_name(row["document_label"]) == LABEL_ATTACK
        pred = bool(row[key])
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
    return {
        "documents": len(results),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "benign_fp_rate": fp / benign_total if benign_total else 0.0,
    }


def threshold_metrics_with_score(results: list[dict[str, Any]], score_key: str, threshold: float) -> dict[str, Any]:
    metric_key = f"__score_positive__{score_key}__{threshold}"
    projected: list[dict[str, Any]] = []
    for row in results:
        projected.append(
            {
                **row,
                metric_key: float(row.get(score_key) or 0.0) >= threshold,
            }
        )
    return threshold_metrics(projected, metric_key)


def grouped_metrics(results: list[dict[str, Any]], key: str, group_key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        groups.setdefault(str(row.get(group_key, "unknown")), []).append(row)
    return {name: threshold_metrics(rows, key) for name, rows in sorted(groups.items())}


def compact_sample(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "document_id": row.get("document_id"),
        "document_label": row.get("document_label"),
        "source_name": row.get("source_name"),
        "category": row.get("category"),
        "language": row.get("language"),
        "semantic_family": row.get("semantic_family"),
        "stage1_max_score": row.get("stage1_max_score"),
        "reviewer_max_score_on_stage1_positive": row.get("reviewer_max_score_on_stage1_positive"),
        "hard_attack_anchor_window_count": row.get("hard_attack_anchor_window_count"),
        "cascade_primary_decision_reason": row.get("cascade_primary_decision_reason"),
        "cascade_decision_reasons": row.get("cascade_decision_reasons"),
    }


def diagnostic_samples(
    results: list[dict[str, Any]],
    primary_key: str,
    *,
    limit: int = 20,
) -> dict[str, list[dict[str, Any]]]:
    false_negatives = [
        row
        for row in results
        if label_name(row["document_label"]) == LABEL_ATTACK and not bool(row.get(primary_key))
    ]
    false_positives = [
        row
        for row in results
        if label_name(row["document_label"]) == LABEL_BENIGN and bool(row.get(primary_key))
    ]
    reviewer_vetoed_attacks = [
        row
        for row in false_negatives
        if float(row.get("stage1_max_score") or 0.0) >= 0.0
        and "reviewer_benign_veto" in set(row.get("cascade_decision_reasons") or [])
    ]
    hard_anchor_benign_fps = [
        row
        for row in false_positives
        if row.get("cascade_primary_decision_reason") == "hard_attack_anchor"
        or "hard_attack_anchor" in set(row.get("cascade_decision_reasons") or [])
    ]
    return {
        "top_cascade_false_negatives": [
            compact_sample(row)
            for row in sorted(false_negatives, key=lambda r: float(r.get("stage1_max_score") or 0.0), reverse=True)[:limit]
        ],
        "top_cascade_false_positives": [
            compact_sample(row)
            for row in sorted(false_positives, key=lambda r: float(r.get("stage1_max_score") or 0.0), reverse=True)[:limit]
        ],
        "top_reviewer_vetoed_attacks": [
            compact_sample(row)
            for row in sorted(
                reviewer_vetoed_attacks,
                key=lambda r: float(r.get("stage1_max_score") or 0.0),
                reverse=True,
            )[:limit]
        ],
        "top_hard_anchor_benign_false_positives": [
            compact_sample(row)
            for row in sorted(
                hard_anchor_benign_fps,
                key=lambda r: float(r.get("stage1_max_score") or 0.0),
                reverse=True,
            )[:limit]
        ],
    }


def decide_window(
    *,
    stage1_score: float,
    reviewer_score: float | None,
    hard_anchor: bool,
    stage1_threshold: float,
    reviewer_threshold: float,
    hard_anchor_veto_enabled: bool,
) -> tuple[bool, str]:
    if stage1_score < stage1_threshold:
        return False, "stage1_negative"
    if hard_anchor_veto_enabled and hard_anchor:
        return True, "hard_attack_anchor"
    if reviewer_score is None:
        return False, "reviewer_not_scored"
    if reviewer_score >= reviewer_threshold:
        return True, "reviewer_real_attack"
    return False, "reviewer_benign_veto"


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    stage1_thresholds = parse_thresholds(args.stage1_thresholds)
    reviewer_thresholds = parse_thresholds(args.reviewer_thresholds)
    min_stage1_threshold = min(stage1_thresholds)
    hard_anchor_veto_enabled = not args.disable_hard_attack_anchor_veto

    device = choose_device(args.device)
    docs = iter_corpus(args)
    started = time.time()
    print(
        f"[cascade] docs={len(docs):,} device={device} stage1={args.stage1_model_id} reviewer={args.reviewer_model_id}",
        flush=True,
    )

    stage1_tokenizer = AutoTokenizer.from_pretrained(args.stage1_model_id, use_fast=True)
    stage1_model = AutoModelForSequenceClassification.from_pretrained(args.stage1_model_id).to(device)
    reviewer_tokenizer = AutoTokenizer.from_pretrained(args.reviewer_model_id, use_fast=True)
    reviewer_model = AutoModelForSequenceClassification.from_pretrained(args.reviewer_model_id).to(device)

    document_results: list[dict[str, Any]] = []
    window_results: list[dict[str, Any]] = []

    for doc_idx, doc in enumerate(docs, 1):
        windows = build_production_windows(doc["text"], stage1_tokenizer)
        if not windows:
            continue
        window_texts = [window["text"] for window in windows]
        stage1_scores = score_texts(
            stage1_model,
            stage1_tokenizer,
            window_texts,
            device=device,
            batch_size=args.window_batch_size,
        )
        reviewer_candidate_indices = [
            idx for idx, value in enumerate(stage1_scores) if value >= min_stage1_threshold
        ]
        reviewer_scores_by_index: dict[int, float] = {}
        if reviewer_candidate_indices:
            reviewer_texts = [window_texts[idx] for idx in reviewer_candidate_indices]
            reviewer_scores = score_texts(
                reviewer_model,
                reviewer_tokenizer,
                reviewer_texts,
                device=device,
                batch_size=args.window_batch_size,
            )
            reviewer_scores_by_index = dict(zip(reviewer_candidate_indices, reviewer_scores))

        hard_anchors = [hard_attack_anchor_present(text) for text in window_texts]
        primary_window_decisions: list[bool] = []
        primary_reasons: list[str] = []
        grid_doc_decisions: dict[str, bool] = {}

        for stage1_threshold in stage1_thresholds:
            for reviewer_threshold in reviewer_thresholds:
                key = f"s1={stage1_threshold}:reviewer={reviewer_threshold}"
                doc_positive = False
                for idx, stage1_score in enumerate(stage1_scores):
                    pred, _reason = decide_window(
                        stage1_score=stage1_score,
                        reviewer_score=reviewer_scores_by_index.get(idx),
                        hard_anchor=hard_anchors[idx],
                        stage1_threshold=stage1_threshold,
                        reviewer_threshold=reviewer_threshold,
                        hard_anchor_veto_enabled=hard_anchor_veto_enabled,
                    )
                    if pred:
                        doc_positive = True
                        break
                grid_doc_decisions[key] = doc_positive

        for idx, window in enumerate(windows):
            pred, reason = decide_window(
                stage1_score=stage1_scores[idx],
                reviewer_score=reviewer_scores_by_index.get(idx),
                hard_anchor=hard_anchors[idx],
                stage1_threshold=args.primary_stage1_threshold,
                reviewer_threshold=args.primary_reviewer_threshold,
                hard_anchor_veto_enabled=hard_anchor_veto_enabled,
            )
            primary_window_decisions.append(pred)
            primary_reasons.append(reason)
            if args.window_output_jsonl:
                window_results.append(
                    {
                        "document_id": doc["document_id"],
                        "document_label": doc["document_label"],
                        "source_name": doc.get("source_name", "unknown"),
                        "category": doc.get("category", "unknown"),
                        "language": doc.get("language", "unknown"),
                        "semantic_family": doc.get("semantic_family", "unknown"),
                        "window_index": window["window_index"],
                        "window_count": len(windows),
                        "window_text": window["text"],
                        "stage1_score": stage1_scores[idx],
                        "reviewer_score": reviewer_scores_by_index.get(idx),
                        "hard_attack_anchor_present": hard_anchors[idx],
                        "cascade_positive": pred,
                        "cascade_decision_reason": reason,
                    }
                )

        best_stage1_idx, best_stage1_score = max(enumerate(stage1_scores), key=lambda item: item[1])
        reviewer_scores_present = list(reviewer_scores_by_index.values())
        primary_key = f"s1={args.primary_stage1_threshold}:reviewer={args.primary_reviewer_threshold}"
        primary_positive_reasons = [reason for pred, reason in zip(primary_window_decisions, primary_reasons) if pred]
        if primary_positive_reasons:
            if "hard_attack_anchor" in primary_positive_reasons:
                primary_decision_reason = "hard_attack_anchor"
            elif "reviewer_real_attack" in primary_positive_reasons:
                primary_decision_reason = "reviewer_real_attack"
            else:
                primary_decision_reason = sorted(primary_positive_reasons)[0]
        elif "reviewer_benign_veto" in primary_reasons:
            primary_decision_reason = "reviewer_benign_veto"
        elif "reviewer_not_scored" in primary_reasons:
            primary_decision_reason = "reviewer_not_scored"
        else:
            primary_decision_reason = "stage1_negative"
        document_results.append(
            {
                "document_id": doc["document_id"],
                "document_label": doc["document_label"],
                "source_name": doc.get("source_name", "unknown"),
                "category": doc.get("category", "unknown"),
                "language": doc.get("language", "unknown"),
                "semantic_family": doc.get("semantic_family", "unknown"),
                "stage1_max_score": best_stage1_score,
                "stage1_best_window_index": best_stage1_idx,
                "reviewer_max_score_on_stage1_positive": max(reviewer_scores_present)
                if reviewer_scores_present
                else None,
                "window_count": len(windows),
                "window_count_bucket": window_count_bucket(len(windows)),
                "hard_attack_anchor_window_count": sum(1 for value in hard_anchors if value),
                "cascade_positive": grid_doc_decisions[primary_key],
                "cascade_primary_key": primary_key,
                "cascade_primary_decision_reason": primary_decision_reason,
                "cascade_positive_window_count": sum(1 for value in primary_window_decisions if value),
                "cascade_decision_reasons": sorted(set(primary_reasons)),
                **{
                    f"stage1_positive__s1={stage1_threshold}": best_stage1_score >= stage1_threshold
                    for stage1_threshold in stage1_thresholds
                },
                **{f"cascade_positive__{key}": value for key, value in grid_doc_decisions.items()},
            }
        )

        if args.progress_every_docs and doc_idx % args.progress_every_docs == 0:
            elapsed = time.time() - started
            print(
                f"[cascade] docs={doc_idx:,}/{len(docs):,} windows={sum(r['window_count'] for r in document_results):,} elapsed={elapsed:.1f}s",
                flush=True,
            )

    primary_key = f"cascade_positive__s1={args.primary_stage1_threshold}:reviewer={args.primary_reviewer_threshold}"
    threshold_grid_metrics = {
        key: threshold_metrics(document_results, f"cascade_positive__{key}")
        for key in sorted(
            f"s1={stage1_threshold}:reviewer={reviewer_threshold}"
            for stage1_threshold in stage1_thresholds
            for reviewer_threshold in reviewer_thresholds
        )
    }
    stage1_only_metrics = {
        f"s1={stage1_threshold}": threshold_metrics_with_score(
            document_results,
            "stage1_max_score",
            stage1_threshold,
        )
        for stage1_threshold in stage1_thresholds
    }
    decision_reason_counts = {
        str(key): int(value)
        for key, value in Counter(
            str(row.get("cascade_primary_decision_reason") or "unknown") for row in document_results
        ).most_common()
    }
    window_decision_reason_counts = {
        str(key): int(value)
        for key, value in Counter(
            reason
            for row in document_results
            for reason in (row.get("cascade_decision_reasons") or ["unknown"])
        ).most_common()
    }
    summary = {
        "stage1_model_id": args.stage1_model_id,
        "reviewer_model_id": args.reviewer_model_id,
        "input_jsonl": args.input_jsonl,
        "documents": len(document_results),
        "windows": sum(int(row.get("window_count", 0)) for row in document_results),
        "hard_anchor_veto_enabled": hard_anchor_veto_enabled,
        "primary_stage1_threshold": args.primary_stage1_threshold,
        "primary_reviewer_threshold": args.primary_reviewer_threshold,
        "primary_metrics": threshold_metrics(document_results, primary_key),
        "stage1_only_metrics": stage1_only_metrics,
        "threshold_grid_metrics": threshold_grid_metrics,
        "decision_reason_counts": decision_reason_counts,
        "window_decision_reason_counts": window_decision_reason_counts,
        "diagnostic_samples": diagnostic_samples(document_results, primary_key),
        "by_language": grouped_metrics(document_results, primary_key, "language"),
        "by_category": grouped_metrics(document_results, primary_key, "category"),
        "by_window_count_bucket": grouped_metrics(document_results, primary_key, "window_count_bucket"),
    }

    write_jsonl(args.output_jsonl, document_results)
    if args.window_output_jsonl:
        write_jsonl(args.window_output_jsonl, window_results)
    write_json(args.summary_json, summary)
    print(json.dumps(summary["primary_metrics"], ensure_ascii=False, indent=2, sort_keys=True))
    return summary


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()
