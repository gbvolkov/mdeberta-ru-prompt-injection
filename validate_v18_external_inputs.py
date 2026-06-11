# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


BASE_TARGET_TOTAL_ROWS = 500_000
BASE_TARGETS = {
    "benign_mined_high_score_windows": 30_000,
    "benign_reviewed_attack_lexicon_context_windows": 10_000,
    "attack_hard_fn_visible": 20_000,
}

BENIGN_LABELS = {"not_prompt_injection", "benign", "safe", "0", "false"}
ATTACK_LABELS = {"prompt_injection", "attack", "malicious", "1", "true"}
ALLOWED_TRAIN_SOURCE_POOLS = {"train", "external_mining_only"}

MODEL_CONTROL_RE = re.compile(
    r"("
    r"system\s+prompt|developer\s+prompt|hidden\s+instruction|internal\s+tool|hidden\s+routing|"
    r"ignore\s+(all\s+){0,1}(previous|prior)\s+instruction[s]{0,1}|jailbreak|prompt\s+injection|"
    r"reveal\s+(the\s+){0,1}(system|developer)|show\s+(the\s+){0,1}(system|developer)|"
    r"систем\w*\s+промпт|"
    r"промпт\s+разработчик\w*|"
    r"скрыт\w*\s+инструкц|"
    r"скрыт\w*\s+маршрут|"
    r"внутрен\w*\s+инструмент|"
    r"игнор\w*\s+.*инструкц|"
    r"раскр\w*\s+.*промпт"
    r")",
    re.IGNORECASE,
)

BENIGN_CONTEXT_RE = re.compile(
    r"("
    r"false\s+positive|security\s+policy|compliance|incident\s+report|training|awareness|"
    r"documentation|guideline|procedure|policy\s+document|audit|redaction|"
    r"политик\w*\s+безопас|"
    r"комплаенс|аудит|"
    r"инцидент|инструкци\w*\s+по|"
    r"регламент|документаци"
    r")",
    re.IGNORECASE,
)

FILLER_ANCHOR_RE = re.compile(
    r"("
    r"while\s+summari[sz]ing|inside\s+a\s+support\s+ticket|before\s+answering|"
    r"compliance\s+audit|use\s+compact\s+list|application\s+context|"
    r"support\s+ticket\s+response|surrounding\s+document"
    r")",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight-check V18 external input files before 20K/50K/500K builds.")
    parser.add_argument("--mined-benign-jsonl", action="append", default=[])
    parser.add_argument("--hard-fn-jsonl", action="append", default=[])
    parser.add_argument("--attack-bank-jsonl", action="append", default=[])
    parser.add_argument("--target-total-rows", type=int, default=20_000)
    parser.add_argument("--report-json", default="v18-external-inputs-preflight-report.json")
    parser.add_argument("--mined-score-threshold", type=float, default=0.82)
    parser.add_argument("--reviewed-near-boundary-capacity-factor", type=float, default=1.20)
    parser.add_argument("--min-attack-bank-rows-20k", type=int, default=40_000)
    parser.add_argument("--min-attack-bank-unique-hashes-20k", type=int, default=10_000)
    parser.add_argument("--min-attack-bank-anchor-share-20k", type=float, default=0.80)
    parser.add_argument("--min-attack-bank-anchor-share-full", type=float, default=0.90)
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if isinstance(value, dict):
                yield value


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def text_hash(value: str) -> str:
    return hashlib.sha256(normalize_text(value).encode("utf-8")).hexdigest()


def normalized_text_hash(value: str) -> str:
    folded = normalize_text(value).lower()
    folded = re.sub(r"[^\wА-Яа-яЁё]+", " ", folded)
    return hashlib.sha256(normalize_text(folded).encode("utf-8")).hexdigest()


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "prompt_injection", "attack", "malicious"}


def nested_get(row: dict[str, Any], *path: str) -> Any:
    value: Any = row
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def extract_text(row: dict[str, Any]) -> str:
    for key in ("window_text", "text", "best_window_text", "reviewed_window_text", "manual_reviewed_window_text"):
        text = normalize_text(row.get(key))
        if text:
            return text
    return ""


def extract_attack_text(row: dict[str, Any]) -> str:
    return normalize_text(row.get("attack_text") or row.get("source_attack_text") or row.get("text") or "")


def extract_score(row: dict[str, Any]) -> float:
    candidates = [
        row.get("model_score"),
        row.get("p_prompt_injection"),
        row.get("document_max_prompt_injection_score"),
        row.get("max_prompt_injection_score"),
        row.get("score"),
        nested_get(row, "student", "p_prompt_injection"),
        nested_get(row, "student", "document_max_prompt_injection_score"),
        nested_get(row, "student", "max_prompt_injection_score"),
    ]
    for value in candidates:
        if value is None or value == "":
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def label_value(row: dict[str, Any]) -> str:
    return str(row.get("document_label") or row.get("label") or row.get("gold_label") or "").strip().lower()


def source_pool_value(row: dict[str, Any]) -> str:
    return str(row.get("source_pool") or row.get("source_pool_assignment") or "").strip().lower()


def reviewed_benign(row: dict[str, Any]) -> bool:
    return any(
        truthy(row.get(key))
        for key in (
            "manual_reviewed_benign",
            "reviewed_benign",
            "confirmed_benign",
        )
    )


def visible_anchor(text: str, anchor: str) -> bool:
    text_norm = normalize_text(text).lower()
    anchor_norm = normalize_text(anchor).lower()
    return bool(anchor_norm and anchor_norm in text_norm)


def good_attack_anchor(row: dict[str, Any]) -> bool:
    anchor = normalize_text(row.get("attack_anchor_text") or row.get("anchor_text") or row.get("attack_span_text") or row.get("model_control_span"))
    attack_text = extract_attack_text(row)
    if not anchor:
        return False
    if FILLER_ANCHOR_RE.search(anchor):
        return False
    if not MODEL_CONTROL_RE.search(anchor):
        return False
    return visible_anchor(attack_text, anchor)


def language_counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(row.get("language") or "unknown").strip().lower() or "unknown" for row in rows).most_common())


def expected_targets(target_total_rows: int, capacity_factor: float) -> dict[str, int]:
    scale = target_total_rows / BASE_TARGET_TOTAL_ROWS
    high_score = int(round(BASE_TARGETS["benign_mined_high_score_windows"] * scale))
    reviewed_target = int(round(BASE_TARGETS["benign_reviewed_attack_lexicon_context_windows"] * scale))
    hard_fn = int(round(BASE_TARGETS["attack_hard_fn_visible"] * scale))
    return {
        "benign_mined_high_score_windows": high_score,
        "benign_reviewed_attack_lexicon_context_windows": reviewed_target,
        "benign_reviewed_attack_lexicon_context_capacity": int(round(reviewed_target * capacity_factor)),
        "attack_hard_fn_visible": hard_fn,
        "attack_bank_size": 40_000 if target_total_rows <= 20_000 else 100_000 if target_total_rows <= 50_000 else 200_000,
    }


def validate_mined(paths: list[str], args: argparse.Namespace, targets: dict[str, int]) -> dict[str, Any]:
    counters = Counter()
    rejected = Counter()
    samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows_for_language: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for path_value in paths:
        path = Path(path_value)
        if not path.exists():
            raise FileNotFoundError(path)
        for row in iter_jsonl(path):
            counters["rows"] += 1
            text = extract_text(row)
            label = label_value(row)
            score = extract_score(row)
            pool = source_pool_value(row)
            if label not in BENIGN_LABELS:
                rejected["not_benign_label"] += 1
                continue
            if score < args.mined_score_threshold:
                rejected["below_score_threshold"] += 1
                continue
            if not text:
                rejected["missing_text"] += 1
                continue
            if pool not in ALLOWED_TRAIN_SOURCE_POOLS:
                rejected["missing_or_disallowed_source_pool"] += 1
                if len(samples["missing_or_disallowed_source_pool"]) < 10:
                    samples["missing_or_disallowed_source_pool"].append({"document_id": row.get("document_id"), "source_pool": pool})
                continue
            th = text_hash(text)
            if th in seen_hashes:
                rejected["duplicate_text_hash"] += 1
                continue
            seen_hashes.add(th)
            near_boundary = bool(MODEL_CONTROL_RE.search(text))
            if near_boundary:
                if not reviewed_benign(row):
                    rejected["near_boundary_without_review"] += 1
                    continue
                if not BENIGN_CONTEXT_RE.search(text):
                    rejected["near_boundary_without_context"] += 1
                    continue
                counters["usable_reviewed_near_boundary_benign"] += 1
            else:
                counters["usable_high_score_benign"] += 1
            rows_for_language.append(row)
    failures = []
    if counters["usable_high_score_benign"] < targets["benign_mined_high_score_windows"]:
        failures.append("mined_high_score_benign_underfilled")
    if counters["usable_reviewed_near_boundary_benign"] < targets["benign_reviewed_attack_lexicon_context_capacity"]:
        failures.append("reviewed_near_boundary_benign_capacity_underfilled")
    return {
        "paths": paths,
        "targets": {
            "usable_high_score_benign_min": targets["benign_mined_high_score_windows"],
            "usable_reviewed_near_boundary_capacity_min": targets["benign_reviewed_attack_lexicon_context_capacity"],
        },
        "counts": dict(counters),
        "rejected": dict(rejected),
        "languages": language_counts(rows_for_language),
        "samples": dict(samples),
        "failures": failures,
        "status": "pass" if not failures else "fail",
    }


def validate_hard_fn(paths: list[str], targets: dict[str, int]) -> dict[str, Any]:
    counters = Counter()
    rejected = Counter()
    samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows_for_language: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for path_value in paths:
        path = Path(path_value)
        if not path.exists():
            raise FileNotFoundError(path)
        for row in iter_jsonl(path):
            counters["rows"] += 1
            text = extract_text(row)
            label = label_value(row)
            anchor = normalize_text(row.get("attack_anchor_text") or row.get("anchor_text") or row.get("attack_span_text"))
            pool = source_pool_value(row)
            visible_flag = truthy(row.get("manual_reviewed_visible_attack")) or truthy(row.get("attack_visible_in_window")) or truthy(row.get("attack_visible")) or truthy(row.get("contains_attack"))
            if label not in ATTACK_LABELS:
                rejected["not_attack_label"] += 1
                continue
            if not text:
                rejected["missing_window_text"] += 1
                continue
            if pool not in ALLOWED_TRAIN_SOURCE_POOLS:
                rejected["missing_or_disallowed_source_pool"] += 1
                continue
            if not anchor:
                rejected["missing_attack_anchor_text"] += 1
                continue
            if FILLER_ANCHOR_RE.search(anchor):
                rejected["filler_attack_anchor_text"] += 1
                continue
            if not MODEL_CONTROL_RE.search(anchor):
                rejected["anchor_without_model_control_signal"] += 1
                if len(samples["anchor_without_model_control_signal"]) < 10:
                    samples["anchor_without_model_control_signal"].append({"document_id": row.get("document_id"), "anchor": anchor, "text_excerpt": text[:300]})
                continue
            if not visible_anchor(text, anchor):
                rejected["anchor_not_visible_in_window"] += 1
                if len(samples["anchor_not_visible_in_window"]) < 10:
                    samples["anchor_not_visible_in_window"].append({"document_id": row.get("document_id"), "anchor": anchor, "text_excerpt": text[:300]})
                continue
            if not visible_flag:
                rejected["missing_review_or_visible_flag"] += 1
                continue
            th = text_hash(text)
            if th in seen_hashes:
                rejected["duplicate_text_hash"] += 1
                continue
            seen_hashes.add(th)
            counters["usable_visible_attack_windows"] += 1
            rows_for_language.append(row)
    failures = []
    if counters["usable_visible_attack_windows"] < targets["attack_hard_fn_visible"]:
        failures.append("hard_fn_visible_attack_windows_underfilled")
    return {
        "paths": paths,
        "targets": {"usable_visible_attack_windows_min": targets["attack_hard_fn_visible"]},
        "counts": dict(counters),
        "rejected": dict(rejected),
        "languages": language_counts(rows_for_language),
        "samples": dict(samples),
        "failures": failures,
        "status": "pass" if not failures else "fail",
    }


def validate_attack_bank(paths: list[str], args: argparse.Namespace, targets: dict[str, int]) -> dict[str, Any]:
    counters = Counter()
    rejected = Counter()
    families = Counter()
    languages = Counter()
    source_names = Counter()
    unique_hashes: set[str] = set()
    unique_templates: set[str] = set()
    samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path_value in paths:
        path = Path(path_value)
        if not path.exists():
            raise FileNotFoundError(path)
        for row in iter_jsonl(path):
            counters["rows"] += 1
            attack_text = extract_attack_text(row)
            if not attack_text:
                rejected["missing_attack_text"] += 1
                continue
            unique_hashes.add(text_hash(attack_text))
            template = str(row.get("attack_template_id") or row.get("template_id") or "")
            if template:
                unique_templates.add(template)
            family = str(row.get("semantic_family") or row.get("attack_family") or "unknown")
            families[family] += 1
            languages[str(row.get("language") or "unknown").lower()] += 1
            source_names[str(row.get("source_name") or "unknown")] += 1
            generation_type = str(row.get("generation_type") or "").strip().lower()
            source_family = str(row.get("source_family") or "").strip().lower()
            if generation_type in {"generated_external_seed", "generated_seed", "generated_seed_input", "synthetic_seed"} or source_family == "generated_seed_attack_bank":
                rejected["generated_seed_not_reviewed_external"] += 1
                continue
            if good_attack_anchor(row):
                counters["good_attack_anchor"] += 1
            else:
                rejected["bad_or_missing_attack_anchor"] += 1
                if len(samples["bad_or_missing_attack_anchor"]) < 10:
                    samples["bad_or_missing_attack_anchor"].append(
                        {
                            "attack_template_id": row.get("attack_template_id"),
                            "anchor": row.get("attack_anchor_text") or row.get("anchor_text"),
                            "attack_text_excerpt": attack_text[:400],
                        }
                    )
            if label_value(row) in ATTACK_LABELS and not (
                truthy(row.get("manual_reviewed_attack"))
                or truthy(row.get("trusted_attack"))
                or truthy(row.get("attack_visible_in_window"))
                or normalize_text(row.get("attack_anchor_text"))
                or MODEL_CONTROL_RE.search(attack_text)
            ):
                rejected["unsupported_label_only"] += 1
    row_count = counters["rows"]
    anchor_share = counters["good_attack_anchor"] / max(1, row_count)
    source_top_share = source_names.most_common(1)[0][1] / max(1, row_count) if source_names else 0.0
    family_top_share = families.most_common(1)[0][1] / max(1, row_count) if families else 0.0
    required_anchor_share = args.min_attack_bank_anchor_share_full if args.target_total_rows >= 240_000 else args.min_attack_bank_anchor_share_20k
    failures = []
    if args.target_total_rows <= 20_000 and row_count < args.min_attack_bank_rows_20k:
        failures.append("attack_bank_rows_below_20k_recommended_min")
    if args.target_total_rows <= 20_000 and len(unique_hashes) < args.min_attack_bank_unique_hashes_20k:
        failures.append("unique_attack_text_hashes_below_20k_min")
    if row_count < targets["attack_bank_size"]:
        failures.append("attack_bank_rows_below_stage_default_size")
    if anchor_share < required_anchor_share:
        failures.append("good_attack_anchor_share_below_gate")
    if rejected["unsupported_label_only"]:
        failures.append("unsupported_label_only_rows_present")
    if rejected["generated_seed_not_reviewed_external"]:
        failures.append("generated_seed_rows_present_in_reviewed_external_attack_bank")
    if source_top_share > 0.35:
        failures.append("source_name_dominance_above_35pct")
    if family_top_share > 0.35:
        failures.append("semantic_family_dominance_above_35pct")
    return {
        "paths": paths,
        "targets": {
            "stage_default_attack_bank_size": targets["attack_bank_size"],
            "rows_20k_recommended_min": args.min_attack_bank_rows_20k,
            "unique_hashes_20k_min": args.min_attack_bank_unique_hashes_20k,
            "good_anchor_share_min": required_anchor_share,
        },
        "counts": dict(counters),
        "unique_attack_text_hashes": len(unique_hashes),
        "unique_attack_template_ids": len(unique_templates),
        "good_attack_anchor_share": anchor_share,
        "top_source_name_share": source_top_share,
        "top_semantic_family_share": family_top_share,
        "languages": dict(languages.most_common()),
        "semantic_families": dict(families.most_common(30)),
        "source_names": dict(source_names.most_common(30)),
        "rejected": dict(rejected),
        "samples": dict(samples),
        "failures": failures,
        "status": "pass" if not failures else "fail",
    }


def main() -> None:
    args = parse_args()
    targets = expected_targets(args.target_total_rows, args.reviewed_near_boundary_capacity_factor)
    failures = []
    missing_inputs = []
    if not args.mined_benign_jsonl:
        missing_inputs.append("--mined-benign-jsonl")
    if not args.hard_fn_jsonl:
        missing_inputs.append("--hard-fn-jsonl")
    if not args.attack_bank_jsonl:
        missing_inputs.append("--attack-bank-jsonl")
    report: dict[str, Any] = {
        "target_total_rows": args.target_total_rows,
        "scaled_targets": targets,
        "missing_inputs": missing_inputs,
    }
    if missing_inputs:
        report["status"] = "fail"
        report["failures"] = ["missing_required_inputs"]
        write_json(args.report_json, report)
        raise SystemExit(2)
    report["mined_benign"] = validate_mined(args.mined_benign_jsonl, args, targets)
    report["hard_fn"] = validate_hard_fn(args.hard_fn_jsonl, targets)
    report["attack_bank"] = validate_attack_bank(args.attack_bank_jsonl, args, targets)
    for section_name in ("mined_benign", "hard_fn", "attack_bank"):
        if report[section_name]["status"] != "pass":
            failures.extend(f"{section_name}:{failure}" for failure in report[section_name].get("failures", []))
    report["failures"] = failures
    report["status"] = "pass" if not failures else "fail"
    write_json(args.report_json, report)
    print(json.dumps({"report_json": args.report_json, "status": report["status"], "failures": failures}, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
