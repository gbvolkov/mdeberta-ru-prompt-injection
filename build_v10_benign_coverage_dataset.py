from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from build_v9_coverage_dataset import (
    DEFAULT_BASE_PHRASE_BANK,
    extend_benign_templates,
    extend_developer_message_family,
    find_family,
    label_judgment_files_exist,
    load_json,
    write_json,
)


DEFAULT_V10_PHRASE_BANK = "short_exfiltration_phrase_variants_v3.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a v10 dataset candidate. It keeps the v9 attack/short-exfiltration coverage "
            "and adds a large benign pool of English/Russian role prompts, tool policies, "
            "web-search rules, Markdown formatting rules, and normal application instructions."
        )
    )
    parser.add_argument("--base-phrase-bank", default=DEFAULT_BASE_PHRASE_BANK)
    parser.add_argument("--output-phrase-bank", default=DEFAULT_V10_PHRASE_BANK)
    parser.add_argument("--output-dir", default="training-dataset-v10-benign-coverage")
    parser.add_argument("--validation-output-dir", default="training-dataset-v10-benign-coverage-validation")
    parser.add_argument("--report-path", default="training-dataset-v10-benign-coverage-report.json")
    parser.add_argument(
        "--attack-curation-audit-path",
        default="training-dataset-v10-benign-coverage-attack-curation-audit.jsonl",
    )
    parser.add_argument(
        "--label-judgments-path",
        default="training-dataset-v4-label-judgments.jsonl,training-dataset-v5-balanced-llm-audit.jsonl",
    )
    parser.add_argument("--max-deep-embedded-prompt-injection-attacks", type=int, default=6000)
    parser.add_argument("--max-short-embedded-exfiltration-attacks", type=int, default=12000)
    parser.add_argument("--max-short-exfiltration-hard-negatives", type=int, default=12000)
    parser.add_argument("--max-embedded-prompt-injection-attacks", type=int, default=6000)
    parser.add_argument("--max-embedded-quoted-hard-negatives", type=int, default=8000)
    parser.add_argument("--max-manual-hard-negatives", type=int, default=5000)
    parser.add_argument("--max-manual-benign-document-fragments", type=int, default=40000)
    parser.add_argument("--max-manual-benign-instruction-policies", type=int, default=60000)
    parser.add_argument("--benign-to-attack-ratio", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--write-phrase-bank-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_path = Path(args.base_phrase_bank)
    output_bank_path = Path(args.output_phrase_bank)

    phrase_bank = load_json(base_path)
    extend_developer_message_family(phrase_bank)
    extend_benign_templates(phrase_bank)
    phrase_bank.setdefault("recommended_generation", {})
    phrase_bank["recommended_generation"]["positive_train_rows"] = args.max_short_embedded_exfiltration_attacks
    phrase_bank["recommended_generation"]["negative_train_rows"] = args.max_short_exfiltration_hard_negatives
    phrase_bank["recommended_generation"][
        "benign_instruction_policy_rows"
    ] = args.max_manual_benign_instruction_policies
    write_json(output_bank_path, phrase_bank)

    dev_family = find_family(phrase_bank, "developer_message_exfiltration")
    print(
        "Wrote v10 phrase bank:",
        output_bank_path,
        f"(developer_message_exfiltration phrases={len(dev_family['base_phrases'])})",
    )
    if args.write_phrase_bank_only:
        return

    command = build_dataset_command(args, output_bank_path)
    if args.dry_run:
        print(" ".join(command))
        return

    subprocess.run(command, check=True)


def build_dataset_command(args: argparse.Namespace, phrase_bank_path: Path) -> list[str]:
    command = [
        sys.executable,
        "build_training_dataset.py",
        "--output-dir",
        args.output_dir,
        "--validation-output-dir",
        args.validation_output_dir,
        "--report-path",
        args.report_path,
        "--seed",
        str(args.seed),
        "--benign-to-attack-ratio",
        str(args.benign_to_attack_ratio),
        "--max-manual-hard-negatives",
        str(args.max_manual_hard_negatives),
        "--max-manual-benign-document-fragments",
        str(args.max_manual_benign_document_fragments),
        "--max-manual-benign-instruction-policies",
        str(args.max_manual_benign_instruction_policies),
        "--max-embedded-prompt-injection-attacks",
        str(args.max_embedded_prompt_injection_attacks),
        "--max-deep-embedded-prompt-injection-attacks",
        str(args.max_deep_embedded_prompt_injection_attacks),
        "--max-embedded-quoted-hard-negatives",
        str(args.max_embedded_quoted_hard_negatives),
        "--max-short-embedded-exfiltration-attacks",
        str(args.max_short_embedded_exfiltration_attacks),
        "--max-short-exfiltration-hard-negatives",
        str(args.max_short_exfiltration_hard_negatives),
        "--short-exfiltration-phrase-bank-path",
        str(phrase_bank_path),
        "--attack-curation-audit-path",
        args.attack_curation_audit_path,
    ]
    if args.label_judgments_path and label_judgment_files_exist(args.label_judgments_path):
        command.extend(["--label-judgments-path", args.label_judgments_path])
    return command


if __name__ == "__main__":
    main()
