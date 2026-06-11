# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch
from datasets import Dataset, DatasetDict, load_from_disk
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_MODEL_ID = "./mdeberta-ru-prompt-injection-v10-benign-scratch"
DEFAULT_THRESHOLD = 0.82
MODEL_MAX_LENGTH = 256
WINDOW_TOKEN_LENGTH = MODEL_MAX_LENGTH - 2
WINDOW_TOKEN_STRIDE = 128
DEFAULT_EXTENSIONS = (".txt", ".md", ".rst", ".html", ".htm", ".csv", ".json")
BENIGN_LABELS = {"benign", "not_prompt_injection", "not_pi", "safe", "0", 0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run production-style sliding-window prompt-injection scoring over a benign review corpus "
            "and save window-level false-positive review rows."
        )
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input-dir", help="Directory with benign documents as text-like files.")
    input_group.add_argument("--input-jsonl", help="JSONL file with one document per line.")
    input_group.add_argument("--dataset-dir", help="Hugging Face Dataset or DatasetDict saved with save_to_disk().")

    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--output-jsonl", default="false-positive-review-windows.jsonl")
    parser.add_argument("--summary-json", default="false-positive-review-summary.json")
    parser.add_argument("--split", default="train", help="Split to read when --dataset-dir is a DatasetDict.")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--id-column", default="document_id")
    parser.add_argument("--label-column", default="document_label")
    parser.add_argument("--category-column", default="category")
    parser.add_argument("--source-column", default="source_name")
    parser.add_argument("--default-label", default="not_prompt_injection")
    parser.add_argument("--default-category", default="uncategorized")
    parser.add_argument("--extensions", default=",".join(DEFAULT_EXTENSIONS))
    parser.add_argument("--limit-documents", type=int, default=None)
    parser.add_argument("--window-batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--only-flagged-documents", action="store_true")
    parser.add_argument("--include-newlines", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    tokenizer, model = load_model(args.model_id, device)
    benign_id, injection_id = resolve_label_ids(model)

    docs = list(iter_documents(args))
    if args.limit_documents is not None:
        docs = docs[: args.limit_documents]
    if not docs:
        raise ValueError("No documents found for false-positive review.")

    summary = {
        "model_id": args.model_id,
        "threshold": args.threshold,
        "model_max_length": MODEL_MAX_LENGTH,
        "window_token_length": WINDOW_TOKEN_LENGTH,
        "window_token_stride": WINDOW_TOKEN_STRIDE,
        "documents": 0,
        "windows": 0,
        "flagged_documents": 0,
        "false_positive_documents": 0,
        "by_category": defaultdict(lambda: {"documents": 0, "flagged_documents": 0, "windows": 0}),
    }

    output_path = Path(args.output_jsonl)
    with output_path.open("w", encoding="utf-8") as f:
        for doc in tqdm(docs, desc="Reviewing documents", unit="doc"):
            review = score_document(
                doc=doc,
                tokenizer=tokenizer,
                model=model,
                device=device,
                threshold=args.threshold,
                benign_id=benign_id,
                injection_id=injection_id,
                window_batch_size=args.window_batch_size,
                include_newlines=args.include_newlines,
            )
            update_summary(summary, review)
            if args.only_flagged_documents and not review["document_false_flagged"]:
                continue
            for window in review["windows"]:
                f.write(json.dumps(window, ensure_ascii=False) + "\n")

    summary["by_category"] = dict(sorted(summary["by_category"].items()))
    Path(args.summary_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary_brief(summary, args.output_jsonl, args.summary_json), ensure_ascii=False, indent=2))


def load_model(model_id: str, device: torch.device) -> tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.eval()
    model.to(device)
    return tokenizer, model


def resolve_device(value: str) -> torch.device:
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if value == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_label_ids(model: AutoModelForSequenceClassification) -> tuple[int, int]:
    label2id = dict(model.config.label2id)
    normalized = {str(label).lower(): int(idx) for label, idx in label2id.items()}
    benign_candidates = ["benign", "safe", "legitimate", "not_injection", "not_prompt_injection", "label_0"]
    injection_candidates = ["prompt_injection", "injection", "jailbreak", "malicious", "unsafe", "label_1"]
    benign_id = next((normalized[label] for label in benign_candidates if label in normalized), 0)
    injection_id = next((normalized[label] for label in injection_candidates if label in normalized), 1)
    if injection_id == benign_id and len(label2id) >= 2:
        injection_id = 1 if benign_id == 0 else 0
    return benign_id, injection_id


def iter_documents(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    if args.input_dir:
        yield from iter_directory_documents(Path(args.input_dir), args)
        return
    if args.input_jsonl:
        yield from iter_jsonl_documents(Path(args.input_jsonl), args)
        return
    if args.dataset_dir:
        yield from iter_dataset_documents(Path(args.dataset_dir), args)
        return


def iter_directory_documents(root: Path, args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    if not root.exists():
        raise FileNotFoundError(root)
    extensions = {ext.strip().lower() for ext in args.extensions.split(",") if ext.strip()}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        text = read_text_file(path)
        if not text:
            continue
        rel = path.relative_to(root)
        category = rel.parts[0] if len(rel.parts) > 1 else args.default_category
        yield {
            "document_id": str(rel).replace("\\", "/"),
            "document_label": args.default_label,
            "category": category,
            "source_name": "local_review_corpus",
            "source_path": str(path),
            "text": text,
        }


def iter_jsonl_documents(path: Path, args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = normalize_text(row.get(args.text_column))
            if not text:
                continue
            yield {
                "document_id": str(row.get(args.id_column) or f"{path.stem}:{idx}"),
                "document_label": str(row.get(args.label_column) or args.default_label),
                "category": str(row.get(args.category_column) or args.default_category),
                "source_name": str(row.get(args.source_column) or path.name),
                "source_path": str(path),
                "source_origin": str(row.get("source_origin") or row.get("source_path") or path),
                "source_pool": str(row.get("source_pool") or row.get("source_pool_assignment") or ""),
                "source_document_id": str(row.get("source_document_id") or row.get("document_id") or f"{path.stem}:{idx}"),
                "original_document_id": str(row.get("original_document_id") or row.get("source_document_id") or row.get("document_id") or f"{path.stem}:{idx}"),
                "text_hash": str(row.get("text_hash") or ""),
                "normalized_text_hash": str(row.get("normalized_text_hash") or ""),
                "dedupe_cluster_id": str(row.get("dedupe_cluster_id") or ""),
                "text": text,
            }


def iter_dataset_documents(path: Path, args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    loaded = load_from_disk(str(path))
    if isinstance(loaded, DatasetDict):
        if args.split not in loaded:
            raise ValueError(f"Split {args.split!r} not found. Available: {list(loaded.keys())}")
        ds = loaded[args.split]
    elif isinstance(loaded, Dataset):
        ds = loaded
    else:
        raise TypeError(f"Unsupported dataset type: {type(loaded).__name__}")

    for idx, row in enumerate(ds):
        text = normalize_text(row.get(args.text_column))
        if not text:
            continue
        yield {
            "document_id": str(row.get(args.id_column) or row.get("source_doc_id") or row.get("parent_id") or idx),
            "document_label": normalize_label(row.get(args.label_column, row.get("label", args.default_label))),
            "category": str(row.get(args.category_column) or row.get("bucket") or args.default_category),
            "source_name": str(row.get(args.source_column) or row.get("source_name") or path.name),
            "source_path": str(path),
            "source_origin": str(row.get("source_origin") or row.get("source_path") or path),
            "source_pool": str(row.get("source_pool") or row.get("source_pool_assignment") or ""),
            "source_document_id": str(row.get("source_document_id") or row.get("document_id") or row.get("source_doc_id") or row.get("parent_id") or idx),
            "original_document_id": str(row.get("original_document_id") or row.get("source_document_id") or row.get("document_id") or row.get("source_doc_id") or row.get("parent_id") or idx),
            "text_hash": str(row.get("text_hash") or ""),
            "normalized_text_hash": str(row.get("normalized_text_hash") or ""),
            "dedupe_cluster_id": str(row.get("dedupe_cluster_id") or ""),
            "text": text,
        }


def read_text_file(path: Path) -> str:
    try:
        return normalize_text(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return normalize_text(path.read_text(encoding="utf-8-sig", errors="ignore"))


def score_document(
    *,
    doc: dict[str, Any],
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    device: torch.device,
    threshold: float,
    benign_id: int,
    injection_id: int,
    window_batch_size: int,
    include_newlines: bool,
) -> dict[str, Any]:
    text = str(doc["text"])
    input_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    window_entries = build_window_entries(text, tokenizer, input_ids)
    window_texts = [str(entry["text"]) for entry in window_entries]
    benign_scores: list[float] = []
    injection_scores: list[float] = []

    for start in range(0, len(window_texts), window_batch_size):
        batch = window_texts[start : start + window_batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=MODEL_MAX_LENGTH, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
            probabilities = torch.softmax(logits, dim=-1)
        benign_scores.extend(probabilities[:, benign_id].detach().cpu().tolist())
        injection_scores.extend(probabilities[:, injection_id].detach().cpu().tolist())

    best_index = max(range(len(injection_scores)), key=lambda idx: injection_scores[idx])
    best_score = float(injection_scores[best_index])
    document_predicted_label = "prompt_injection" if best_score >= threshold else "benign"
    document_false_flagged = is_benign_label(doc["document_label"]) and document_predicted_label == "prompt_injection"

    base = {
        "document_id": doc["document_id"],
        "document_label": normalize_label(doc["document_label"]),
        "category": doc["category"],
        "source_name": doc["source_name"],
        "source_path": doc["source_path"],
        "document_text_length": len(text),
        "document_token_length": len(input_ids),
        "window_count": len(window_entries),
        "threshold": threshold,
        "document_predicted_label": document_predicted_label,
        "document_max_prompt_injection_score": best_score,
        "document_best_window_index": best_index,
        "document_false_flagged": document_false_flagged,
    }
    for key in (
        "source_origin",
        "source_pool",
        "source_pool_assignment",
        "source_document_id",
        "original_document_id",
        "text_hash",
        "normalized_text_hash",
        "dedupe_cluster_id",
    ):
        value = doc.get(key)
        if value is not None and value != "":
            base[key] = value

    windows = []
    for idx, entry in enumerate(window_entries):
        window_text = str(entry["text"])
        if not include_newlines:
            window_text = normalize_text(window_text)
        windows.append(
            {
                **base,
                "window_index": idx,
                "is_best_window": idx == best_index,
                "token_start": int(entry["token_start"]),
                "token_end": int(entry["token_end"]),
                "window_text": window_text,
                "window_text_length": len(window_text),
                "window_token_length": int(entry["token_end"]) - int(entry["token_start"]),
                "p_benign": float(benign_scores[idx]),
                "p_prompt_injection": float(injection_scores[idx]),
                "window_predicted_label": "prompt_injection" if injection_scores[idx] >= threshold else "benign",
            }
        )
    return {**base, "windows": windows}


def build_window_entries(
    text: str,
    tokenizer: AutoTokenizer,
    input_ids: list[int],
) -> list[dict[str, object]]:
    if len(input_ids) <= WINDOW_TOKEN_LENGTH:
        return [{"token_start": 0, "token_end": len(input_ids), "text": text}]

    windows: list[dict[str, object]] = []
    start = 0
    last_start = max(0, len(input_ids) - WINDOW_TOKEN_LENGTH)
    while start <= last_start:
        chunk_ids = input_ids[start : start + WINDOW_TOKEN_LENGTH]
        windows.append(
            {
                "token_start": start,
                "token_end": start + len(chunk_ids),
                "text": tokenizer.decode(chunk_ids, skip_special_tokens=True),
            }
        )
        if start == last_start:
            break
        start = min(start + WINDOW_TOKEN_STRIDE, last_start)
    return windows


def update_summary(summary: dict[str, Any], review: dict[str, Any]) -> None:
    category = str(review["category"])
    summary["documents"] += 1
    summary["windows"] += len(review["windows"])
    flagged = review["document_predicted_label"] == "prompt_injection"
    false_flagged = bool(review["document_false_flagged"])
    if flagged:
        summary["flagged_documents"] += 1
    if false_flagged:
        summary["false_positive_documents"] += 1
    category_info = summary["by_category"][category]
    category_info["documents"] += 1
    category_info["windows"] += len(review["windows"])
    if flagged:
        category_info["flagged_documents"] += 1


def summary_brief(summary: dict[str, Any], output_jsonl: str, summary_json: str) -> dict[str, Any]:
    documents = int(summary["documents"])
    flagged = int(summary["flagged_documents"])
    false_positive = int(summary["false_positive_documents"])
    return {
        "documents": documents,
        "windows": summary["windows"],
        "flagged_documents": flagged,
        "false_positive_documents": false_positive,
        "false_positive_rate": false_positive / documents if documents else 0.0,
        "output_jsonl": output_jsonl,
        "summary_json": summary_json,
    }


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\x00", " ")
    return re.sub(r"\s+", " ", text).strip()


def normalize_label(value: Any) -> str:
    if isinstance(value, int):
        return "not_prompt_injection" if value == 0 else "prompt_injection"
    text = normalize_text(value).lower()
    if text in {"0", "benign", "safe", "not_pi", "not_prompt_injection"}:
        return "not_prompt_injection"
    if text in {"1", "prompt_injection", "injection", "attack", "malicious"}:
        return "prompt_injection"
    return text or "not_prompt_injection"


def is_benign_label(value: Any) -> bool:
    normalized = normalize_label(value)
    return normalized in {"not_prompt_injection", "benign", "safe", "not_pi"}


if __name__ == "__main__":
    main()
