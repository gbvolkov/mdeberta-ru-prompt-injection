from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


MODELS = {
    "v10": "mdeberta-ru-prompt-injection-v10-benign-scratch",
    "v11": "mdeberta-ru-prompt-injection-v11-fp-correction-ft",
    "v12": "mdeberta-ru-prompt-injection-v12-critical-correction-ft",
    "v13": "mdeberta-ru-prompt-injection-v13-critical-correction-ft",
    "v14": "mdeberta-ru-prompt-injection-v14-boundary-correction-ft",
    "v15": "mdeberta-ru-prompt-injection-v15-anchored-critical-correction-ft",
    "v16": "mdeberta-ru-prompt-injection-v16-critical-recall-restoration-ft",
}


CORPORA = {
    "v13_critical_ru": {
        "path": "v13-critical-ru-validation-corpus.jsonl",
        "suite": ["quick", "core", "full"],
        "kind": "critical_attack_regression",
        "note": "Critical Russian prompt/system/developer/tool exfiltration windows.",
    },
    "v13_benign_windows": {
        "path": "v13-benign-validation-corpus.jsonl",
        "suite": ["quick", "core", "full"],
        "kind": "benign_window_regression",
        "note": "V13 benign validation windows.",
    },
    "malicious_dev": {
        "path": "malicious-document-dev.jsonl",
        "suite": ["core", "full"],
        "kind": "malicious_long_document_dev",
        "note": "Existing malicious document development set.",
    },
    "benign_prod_dev": {
        "path": "benign-prod-calibration-dev.jsonl",
        "suite": ["core", "full"],
        "kind": "benign_long_document_dev",
        "note": "Existing benign production calibration set.",
    },
    "malicious_test": {
        "path": "malicious-document-test.jsonl",
        "suite": ["full"],
        "kind": "malicious_long_document_test",
        "note": "Existing malicious document test set.",
    },
    "benign_prod_locked_test": {
        "path": "benign-prod-locked-test.jsonl",
        "suite": ["full"],
        "kind": "benign_long_document_test",
        "note": "Existing benign production locked-test style set. Treat as diagnostic if previously inspected.",
    },
    "false_positive_corpus": {
        "path": "false-positive-corpus-documents.jsonl",
        "suite": ["full"],
        "kind": "large_benign_fp_regression",
        "note": "Large V10 false-positive corpus; diagnostic/mining corpus, not final blind acceptance.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare V10-V16 models on the shared V10-V13 diagnostic validation suite.")
    parser.add_argument("--suite", choices=["quick", "core", "full"], default="core")
    parser.add_argument("--output-dir", default="validation-comparison-v10-v13")
    parser.add_argument("--thresholds", default="0.82,0.90,0.95,0.99,0.999,0.9995")
    parser.add_argument("--primary-threshold", type=float, default=0.95)
    parser.add_argument("--window-batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--limit-documents", type=int, help="Apply the same document limit to every corpus.")
    parser.add_argument("--progress-every-docs", type=int, default=25)
    parser.add_argument("--progress-every-windows", type=int, default=1000)
    parser.add_argument("--models", default="v10,v11,v12,v13", help="Comma-separated model keys.")
    parser.add_argument("--force", action="store_true", help="Re-run evaluations even if summary files already exist.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands and write manifest only.")
    return parser.parse_args()


def selected_models(value: str) -> list[tuple[str, Path]]:
    out = []
    for key in [part.strip() for part in value.split(",") if part.strip()]:
        if key not in MODELS:
            raise ValueError(f"Unknown model key {key!r}. Available: {', '.join(MODELS)}")
        path = Path(MODELS[key])
        if not (path / "config.json").exists():
            raise FileNotFoundError(f"Missing model config for {key}: {path}")
        if not (path / "model.safetensors").exists():
            raise FileNotFoundError(f"Missing model weights for {key}: {path / 'model.safetensors'}")
        out.append((key, path))
    return out


def selected_corpora(suite: str) -> list[tuple[str, dict[str, Any]]]:
    out = []
    for name, spec in CORPORA.items():
        if suite in spec["suite"]:
            path = Path(spec["path"])
            if not path.exists():
                raise FileNotFoundError(f"Missing corpus {name}: {path}")
            out.append((name, spec))
    return out


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def write_manifest(path: Path, suite: str, models: list[tuple[str, Path]], corpora: list[tuple[str, dict[str, Any]]], args: argparse.Namespace) -> None:
    manifest = {
        "suite": suite,
        "models": {name: str(model_path) for name, model_path in models},
        "corpora": {
            name: {
                **spec,
                "rows": count_jsonl(Path(spec["path"])),
            }
            for name, spec in corpora
        },
        "thresholds": args.thresholds,
        "primary_threshold": args.primary_threshold,
        "window_batch_size": args.window_batch_size,
        "device": args.device,
        "limit_documents": args.limit_documents,
        "note": "This is a comparative diagnostic suite. Do not relabel it as blind acceptance if any corpus was used or inspected in earlier training/mining.",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def run_eval(model_name: str, model_path: Path, corpus_name: str, corpus_path: Path, output_dir: Path, args: argparse.Namespace) -> Path:
    model_out = output_dir / model_name
    model_out.mkdir(parents=True, exist_ok=True)
    summary_path = model_out / f"{corpus_name}-summary.json"
    result_path = model_out / f"{corpus_name}-documents.jsonl"
    if summary_path.exists() and result_path.exists() and not args.force:
        print(f"[compare] skip existing model={model_name} corpus={corpus_name} summary={summary_path}", flush=True)
        return summary_path
    command = [
        sys.executable,
        "-u",
        "run_blind_broad_eval.py",
        "--model-id",
        str(model_path),
        "--input-jsonl",
        str(corpus_path),
        "--thresholds",
        args.thresholds,
        "--primary-threshold",
        str(args.primary_threshold),
        "--window-batch-size",
        str(args.window_batch_size),
        "--output-jsonl",
        str(result_path),
        "--summary-json",
        str(summary_path),
        "--device",
        args.device,
        "--progress-every-docs",
        str(args.progress_every_docs),
        "--progress-every-windows",
        str(args.progress_every_windows),
    ]
    if args.limit_documents:
        command.extend(["--limit-documents", str(args.limit_documents)])
    print(f"[compare] start model={model_name} corpus={corpus_name}", flush=True)
    print(" ".join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, check=True)
        print(f"[compare] done model={model_name} corpus={corpus_name}", flush=True)
    return summary_path


def flatten_summary(model_name: str, corpus_name: str, summary_path: Path) -> list[dict[str, Any]]:
    if not summary_path.exists():
        return []
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    rows = []
    for threshold, metrics in summary.get("threshold_metrics", {}).items():
        rows.append(
            {
                "model": model_name,
                "corpus": corpus_name,
                "threshold": threshold,
                "documents": metrics.get("documents"),
                "tp": metrics.get("tp"),
                "fp": metrics.get("fp"),
                "tn": metrics.get("tn"),
                "fn": metrics.get("fn"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1": metrics.get("f1"),
                "benign_fp_rate": metrics.get("benign_fp_rate"),
                "windows": summary.get("windows"),
                "labels": json.dumps(summary.get("labels", {}), ensure_ascii=False, sort_keys=True),
                "summary_json": str(summary_path),
            }
        )
    return rows


def write_comparison(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    json_path = output_dir / "comparison-summary.json"
    csv_path = output_dir / "comparison-summary.csv"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    fields = [
        "model",
        "corpus",
        "threshold",
        "documents",
        "windows",
        "tp",
        "fp",
        "tn",
        "fn",
        "precision",
        "recall",
        "f1",
        "benign_fp_rate",
        "labels",
        "summary_json",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models = selected_models(args.models)
    corpora = selected_corpora(args.suite)
    write_manifest(output_dir / "validation-suite-manifest.json", args.suite, models, corpora, args)

    summary_paths: list[tuple[str, str, Path]] = []
    for model_name, model_path in models:
        for corpus_name, corpus_spec in corpora:
            summary_paths.append(
                (
                    model_name,
                    corpus_name,
                    run_eval(model_name, model_path, corpus_name, Path(corpus_spec["path"]), output_dir, args),
                )
            )

    rows: list[dict[str, Any]] = []
    if not args.dry_run:
        for model_name, corpus_name, summary_path in summary_paths:
            rows.extend(flatten_summary(model_name, corpus_name, summary_path))
        write_comparison(output_dir, rows)
        print(json.dumps({"output_dir": str(output_dir), "rows": len(rows), "csv": str(output_dir / "comparison-summary.csv")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
