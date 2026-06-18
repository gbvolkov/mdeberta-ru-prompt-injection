#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the V13/V16 core diagnostic comparison.")
    parser.add_argument("--v13-model", type=Path, required=True)
    parser.add_argument("--v16-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("work/validation-v13-v16-core"))
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cuda")
    parser.add_argument("--window-batch-size", type=int, default=64)
    parser.add_argument("--thresholds", default="0.82,0.90,0.95,0.99,0.999,0.9995")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    evaluator = root / "scripts/run_blind_broad_eval.py"
    models = {
        "v13": args.v13_model.resolve(),
        "v16": args.v16_model.resolve(),
    }
    corpora = {
        "v13_critical_ru": root / "datasets/raw/v13-critical-ru-validation-corpus.jsonl",
        "v13_benign_windows": root / "datasets/raw/v13-benign-validation-corpus.jsonl",
        "malicious_dev": root / "datasets/validation/malicious-document-dev.jsonl",
        "benign_prod_dev": root / "datasets/raw/benign-prod-calibration-dev.jsonl",
    }
    for model_name, model_path in models.items():
        for corpus_name, corpus_path in corpora.items():
            output_dir = args.output_dir / model_name
            output_dir.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                "-u",
                str(evaluator),
                "--model-id",
                str(model_path),
                "--input-jsonl",
                str(corpus_path),
                "--thresholds",
                args.thresholds,
                "--primary-threshold",
                "0.82",
                "--window-batch-size",
                str(args.window_batch_size),
                "--output-jsonl",
                str(output_dir / f"{corpus_name}-documents.jsonl"),
                "--summary-json",
                str(output_dir / f"{corpus_name}-summary.json"),
                "--device",
                args.device,
                "--progress-every-docs",
                "25",
                "--progress-every-windows",
                "1000",
            ]
            print(f"[validation] model={model_name} corpus={corpus_name}", flush=True)
            subprocess.run(command, check=True, cwd=root)


if __name__ == "__main__":
    main()
