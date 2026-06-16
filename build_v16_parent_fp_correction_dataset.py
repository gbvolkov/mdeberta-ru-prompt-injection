from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from datasets import Dataset, DatasetDict


BASE_TARGETS = {
    "benign_v18_1_corporate_high_score": 9_500,
    "benign_v18_1_corporate_mid_score": 7_500,
    "benign_v18_reviewed_attack_lexicon_context": 6_500,
    "benign_v18_1_policy_security_audit": 4_000,
    "benign_v18_mined_high_score": 10_000,
    "benign_v18_external_process_instruction": 7_500,
    "benign_v18_carrier_contrast": 5_500,
    "benign_v18_broad_production": 4_500,
    "benign_v18_long_wrapper": 3_500,
    "attack_v18_embedded_visible": 11_000,
    "attack_v18_direct_standalone": 8_000,
    "attack_v18_critical_multilingual": 8_000,
    "attack_v18_wrapper_boundary": 5_000,
    "attack_v18_semantic_paraphrase": 4_000,
    "attack_v18_1_recall_analog": 4_000,
    "attack_v18_1_classifier_or_routing": 1_000,
    "attack_v18_1_hard_critical_rehearsal": 1_000,
}


V18_COMPONENT_MAP = {
    "benign_mined_high_score_windows": "benign_v18_mined_high_score",
    "benign_reviewed_attack_lexicon_context_windows": "benign_v18_reviewed_attack_lexicon_context",
    "benign_external_process_instruction_windows": "benign_v18_external_process_instruction",
    "benign_matched_carrier_contrast_windows": "benign_v18_carrier_contrast",
    "benign_random_broad_production_windows": "benign_v18_broad_production",
    "benign_random_long_document_windows": "benign_v18_long_wrapper",
    "benign_wrapper_url_redaction_metadata_windows": "benign_v18_long_wrapper",
    "attack_embedded_visible_random_carriers": "attack_v18_embedded_visible",
    "attack_direct_standalone": "attack_v18_direct_standalone",
    "attack_critical_ru_multilingual_model_control": "attack_v18_critical_multilingual",
    "attack_wrapper_url_boundary": "attack_v18_wrapper_boundary",
    "attack_semantic_paraphrase_variants": "attack_v18_semantic_paraphrase",
}


V18_1_COMPONENT_MAP = {
    "benign_corporate_high_score_analog": "benign_v18_1_corporate_high_score",
    "benign_corporate_procedure_security_mid_score_analog": "benign_v18_1_corporate_mid_score",
    "benign_policy_security_audit_attack_lexicon": "benign_v18_1_policy_security_audit",
    "attack_classifier_or_routing_manipulation": "attack_v18_1_classifier_or_routing",
}


V18_1_RECALL_ANALOG_COMPONENTS = {
    "attack_fn_analog_critical_ru_or_mixed",
    "attack_fn_analog_malicious_dev",
    "attack_fn_analog_blind_malicious",
    "attack_policy_like_malicious",
    "attack_long_document_embedded",
    "attack_noncanonical_exfiltration",
}


TEXT_LABEL_COLUMNS = ["text", "label"]


@dataclass
class CandidateRow:
    text: str
    label: int
    component: str
    source_component: str
    source_file: str
    text_hash: str
    normalized_text_hash: str
    split_group_id: str
    template_family_id: str
    source_pattern_id: str
    attack_generation_recipe_id: str
    benign_generation_recipe_id: str
    source_document_id: str
    source_name: str
    language: str
    category: str
    semantic_family: str
    audit: dict[str, Any]


def stable_hash(value: str, length: int = 16) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()[:length]


def normalize_text(value: str) -> str:
    return " ".join(str(value or "").split()).strip()


def to_label(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    raw = str(value).strip().lower()
    if raw in {"1", "prompt_injection", "attack", "malicious"}:
        return 1
    if raw in {"0", "benign", "not_prompt_injection"}:
        return 0
    raise ValueError(f"Unknown label value: {value!r}")


def scaled_targets(target_total_rows: int) -> dict[str, int]:
    base_total = sum(BASE_TARGETS.values())
    raw = {name: target_total_rows * value / base_total for name, value in BASE_TARGETS.items()}
    targets = {name: int(value) for name, value in raw.items()}
    remainder = target_total_rows - sum(targets.values())
    fractional = sorted(raw, key=lambda name: raw[name] - int(raw[name]), reverse=True)
    for name in fractional[:remainder]:
        targets[name] += 1
    return targets


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def positive_visible(row: dict[str, Any]) -> bool:
    if boolish(row.get("attack_visible_in_window")):
        return True
    if boolish(row.get("attack_anchor_visible_in_window")):
        return True
    if boolish(row.get("direct_attack_signal")):
        return True
    try:
        if float(row.get("attack_overlap_ratio", 0.0) or 0.0) >= 0.20:
            return True
    except (TypeError, ValueError):
        pass
    text = normalize_text(str(row.get("text") or ""))
    attack_text = normalize_text(str(row.get("attack_text") or ""))
    return bool(attack_text and attack_text in text)


def benign_has_visible_attack(row: dict[str, Any]) -> bool:
    if boolish(row.get("attack_visible_in_window")):
        return True
    if boolish(row.get("attack_anchor_visible_in_window")):
        return True
    if boolish(row.get("direct_attack_signal")):
        return True
    return False


def target_component_for_row(row: dict[str, Any], source_kind: str) -> str | None:
    component = str(row.get("component") or "")
    if source_kind == "v18":
        return V18_COMPONENT_MAP.get(component)

    if component in V18_1_COMPONENT_MAP:
        return V18_1_COMPONENT_MAP[component]

    if component in V18_1_RECALL_ANALOG_COMPONENTS:
        if (
            component == "attack_fn_analog_critical_ru_or_mixed"
            and str(row.get("source_eval_corpus") or "") == "proper_critical_attack_windows"
        ):
            return "attack_v18_1_hard_critical_rehearsal"
        return "attack_v18_1_recall_analog"

    return None


def make_candidate(row: dict[str, Any], target_component: str, source_file: str) -> CandidateRow | None:
    text = normalize_text(str(row.get("text") or ""))
    if len(text) < 20:
        return None
    label = to_label(row.get("label"))
    source_component = str(row.get("component") or "")
    text_hash = str(row.get("text_hash") or stable_hash(text))
    normalized_text_hash = str(row.get("normalized_text_hash") or stable_hash(normalize_text(text), 24))
    source_document_id = str(row.get("source_document_id") or row.get("document_id") or "")
    base_split_group_id = str(
        row.get("split_group_id")
        or row.get("source_pattern_id")
        or row.get("template_family_id")
        or source_document_id
        or text_hash
    )
    template_family_id = str(row.get("template_family_id") or "")
    source_pattern_id = str(row.get("source_pattern_id") or base_split_group_id)
    attack_generation_recipe_id = str(row.get("attack_generation_recipe_id") or "")
    benign_generation_recipe_id = str(row.get("benign_generation_recipe_id") or "")
    split_group_id = "|".join(
        part
        for part in [
            target_component,
            base_split_group_id,
            template_family_id,
            source_pattern_id,
            attack_generation_recipe_id,
            benign_generation_recipe_id,
        ]
        if part
    )
    audit = dict(row)
    audit["v16_parent_component"] = target_component
    audit["v16_parent_source_component"] = source_component
    audit["v16_parent_source_file"] = source_file

    return CandidateRow(
        text=text,
        label=label,
        component=target_component,
        source_component=source_component,
        source_file=source_file,
        text_hash=text_hash,
        normalized_text_hash=normalized_text_hash,
        split_group_id=split_group_id,
        template_family_id=template_family_id,
        source_pattern_id=source_pattern_id,
        attack_generation_recipe_id=attack_generation_recipe_id,
        benign_generation_recipe_id=benign_generation_recipe_id,
        source_document_id=source_document_id,
        source_name=str(row.get("source_name") or ""),
        language=str(row.get("language") or ""),
        category=str(row.get("category") or ""),
        semantic_family=str(row.get("semantic_family") or ""),
        audit=audit,
    )


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def collect_candidates(
    *,
    path: Path,
    source_kind: str,
    targets: dict[str, int],
    seed: int,
) -> tuple[dict[str, list[CandidateRow]], dict[str, Any]]:
    rnd = random.Random(seed)
    caps = {component: max(target * 3, target + 500) for component, target in targets.items()}
    candidates: dict[str, list[CandidateRow]] = defaultdict(list)
    seen = Counter()
    rejected = Counter()

    for row in iter_jsonl(path):
        seen["rows"] += 1
        if source_kind == "v18_1" and str(row.get("split") or "train") != "train":
            rejected["v18_1_non_train_split"] += 1
            continue
        target_component = target_component_for_row(row, source_kind)
        if not target_component:
            rejected["unmapped_component"] += 1
            continue
        if target_component not in targets:
            rejected["component_not_targeted"] += 1
            continue

        candidate = make_candidate(row, target_component, str(path))
        if not candidate:
            rejected["invalid_candidate"] += 1
            continue
        if candidate.label == 1 and not positive_visible(row):
            rejected["positive_without_visible_attack"] += 1
            continue
        if candidate.label == 0 and benign_has_visible_attack(row):
            rejected["benign_with_visible_attack"] += 1
            continue

        bucket = candidates[target_component]
        if len(bucket) < caps[target_component]:
            bucket.append(candidate)
        else:
            index = rnd.randint(0, seen["rows"] - 1)
            if index < len(bucket):
                bucket[index] = candidate

    report = {
        "path": str(path),
        "source_kind": source_kind,
        "seen": dict(seen),
        "rejected": dict(rejected),
        "candidate_counts": {component: len(rows) for component, rows in candidates.items()},
    }
    return candidates, report


def merge_candidate_maps(*maps: dict[str, list[CandidateRow]]) -> dict[str, list[CandidateRow]]:
    merged: dict[str, list[CandidateRow]] = defaultdict(list)
    for candidate_map in maps:
        for component, rows in candidate_map.items():
            merged[component].extend(rows)
    return merged


def select_rows(
    candidates: dict[str, list[CandidateRow]],
    targets: dict[str, int],
    seed: int,
) -> tuple[list[CandidateRow], dict[str, Any]]:
    rnd = random.Random(seed)
    selected: list[CandidateRow] = []
    seen_text_hash: dict[str, int] = {}
    duplicate_conflicts: list[dict[str, Any]] = []
    dropped = Counter()
    component_counts = Counter()

    for component, target in targets.items():
        rows = list(candidates.get(component, []))
        rnd.shuffle(rows)
        for row in rows:
            existing_label = seen_text_hash.get(row.normalized_text_hash)
            if existing_label is not None:
                if existing_label != row.label:
                    duplicate_conflicts.append(
                        {
                            "normalized_text_hash": row.normalized_text_hash,
                            "existing_label": existing_label,
                            "new_label": row.label,
                            "component": component,
                        }
                    )
                dropped["duplicate_normalized_text"] += 1
                continue
            seen_text_hash[row.normalized_text_hash] = row.label
            selected.append(row)
            component_counts[component] += 1
            if component_counts[component] >= target:
                break

    underfilled = {
        component: {"target": target, "actual": component_counts.get(component, 0), "minimum": int(target * 0.8)}
        for component, target in targets.items()
        if component_counts.get(component, 0) < int(target * 0.8)
    }
    return selected, {
        "component_counts": dict(component_counts),
        "underfilled_components": underfilled,
        "duplicate_conflicts": duplicate_conflicts[:100],
        "duplicate_conflict_count": len(duplicate_conflicts),
        "dropped": dict(dropped),
    }


def split_rows(rows: list[CandidateRow], validation_rows: int, seed: int) -> tuple[list[CandidateRow], list[CandidateRow], dict[str, Any]]:
    rnd = random.Random(seed)
    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    protected_keys = [
        "split_group_id",
        "template_family_id",
        "source_pattern_id",
        "attack_generation_recipe_id",
        "benign_generation_recipe_id",
    ]
    first_seen: dict[tuple[str, str], int] = {}
    for index, row in enumerate(rows):
        for key_name in protected_keys:
            value = getattr(row, key_name)
            if not value:
                continue
            key = (key_name, value)
            previous = first_seen.get(key)
            if previous is None:
                first_seen[key] = index
            else:
                union(previous, index)

    groups: dict[str, list[CandidateRow]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(find(index))].append(row)

    group_items = list(groups.items())
    rnd.shuffle(group_items)

    validation_group_ids: set[str] = set()
    validation_components: set[str] = set()
    all_components = {row.component for row in rows}

    for component in sorted(all_components):
        matching = [(gid, group_rows) for gid, group_rows in group_items if any(row.component == component for row in group_rows)]
        if not matching:
            continue
        gid, _ = min(matching, key=lambda item: len(item[1]))
        validation_group_ids.add(gid)
        validation_components.add(component)

    current = sum(len(groups[gid]) for gid in validation_group_ids)
    for gid, group_rows in group_items:
        if current >= validation_rows:
            break
        if gid in validation_group_ids:
            continue
        validation_group_ids.add(gid)
        current += len(group_rows)

    train: list[CandidateRow] = []
    validation: list[CandidateRow] = []
    for gid, group_rows in groups.items():
        if gid in validation_group_ids:
            validation.extend(group_rows)
        else:
            train.extend(group_rows)

    missing_validation_components = sorted(all_components.difference({row.component for row in validation}))

    overlap_report = {}
    for key_name in [
        "split_group_id",
        "template_family_id",
        "source_pattern_id",
        "attack_generation_recipe_id",
        "benign_generation_recipe_id",
        "source_document_id",
    ]:
        train_values = {getattr(row, key_name) for row in train if getattr(row, key_name)}
        validation_values = {getattr(row, key_name) for row in validation if getattr(row, key_name)}
        overlap = sorted(train_values.intersection(validation_values))[:25]
        overlap_report[key_name] = {"count": len(train_values.intersection(validation_values)), "samples": overlap}

    return train, validation, {
        "groups": len(groups),
        "validation_group_count": len(validation_group_ids),
        "missing_validation_components": missing_validation_components,
        "pattern_overlap_report": overlap_report,
    }


def dataset_rows(rows: list[CandidateRow]) -> list[dict[str, Any]]:
    return [{"text": row.text, "label": row.label} for row in rows]


def write_audit(path: Path, rows: list[CandidateRow], split: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if path.exists() else "w"
    with path.open(mode, encoding="utf-8") as f:
        for row in rows:
            audit = dict(row.audit)
            audit["v16_parent_split"] = split
            audit["v16_parent_component"] = row.component
            audit["v16_parent_text_hash"] = row.text_hash
            audit["v16_parent_normalized_text_hash"] = row.normalized_text_hash
            f.write(json.dumps(audit, ensure_ascii=False, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a V16-parent FP-correction dataset from V18/V18.1 train-side audit rows.")
    parser.add_argument("--v18-train-audit-jsonl", default="./v18-build-300k-final/audit-rows/train-metadata.jsonl")
    parser.add_argument("--v18-1-audit-jsonl", default="./v18_1_build/audit-rows.jsonl")
    parser.add_argument("--diagnostic-json", default="./v18_2_parent_strategy_diagnostics/v16-parent-strategy-diagnostic.json")
    parser.add_argument("--output-dir", default="./training-dataset-v16-parent-fp-correction")
    parser.add_argument("--validation-output-dir", default="./training-dataset-v16-parent-fp-correction-validation")
    parser.add_argument("--report-json", default="./training-dataset-v16-parent-fp-correction-report.json")
    parser.add_argument("--audit-jsonl", default="./v16-parent-fp-correction-build/audit-rows.jsonl")
    parser.add_argument("--target-total-rows", type=int, default=100_000)
    parser.add_argument("--validation-rows", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--allow-underfilled", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = scaled_targets(args.target_total_rows)

    v18_path = Path(args.v18_train_audit_jsonl)
    v18_1_path = Path(args.v18_1_audit_jsonl)
    diagnostic_path = Path(args.diagnostic_json)

    if not v18_path.exists():
        raise FileNotFoundError(v18_path)
    if not v18_1_path.exists():
        raise FileNotFoundError(v18_1_path)
    if not diagnostic_path.exists():
        raise FileNotFoundError(diagnostic_path)

    v18_candidates, v18_report = collect_candidates(path=v18_path, source_kind="v18", targets=targets, seed=args.seed)
    v18_1_candidates, v18_1_report = collect_candidates(path=v18_1_path, source_kind="v18_1", targets=targets, seed=args.seed + 1)
    candidates = merge_candidate_maps(v18_candidates, v18_1_candidates)
    rows, selection_report = select_rows(candidates, targets, args.seed + 2)
    train_rows, validation_rows, split_report = split_rows(rows, args.validation_rows, args.seed + 3)

    report = {
        "status": "pass",
        "target_total_rows": args.target_total_rows,
        "actual_total_rows": len(rows),
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "targets": targets,
        "labels": dict(Counter(row.label for row in rows)),
        "component_counts": selection_report["component_counts"],
        "underfilled_components": selection_report["underfilled_components"],
        "duplicate_conflict_count": selection_report["duplicate_conflict_count"],
        "duplicate_conflicts": selection_report["duplicate_conflicts"],
        "missing_validation_components": split_report["missing_validation_components"],
        "pattern_overlap_report": split_report["pattern_overlap_report"],
        "source_reports": {
            "v18": v18_report,
            "v18_1": v18_1_report,
        },
        "diagnostic_source": json.loads(diagnostic_path.read_text(encoding="utf-8")).get("totals", {}),
        "output_dir": args.output_dir,
        "validation_output_dir": args.validation_output_dir,
        "audit_jsonl": args.audit_jsonl,
    }

    failures: list[str] = []
    if report["underfilled_components"] and not args.allow_underfilled:
        failures.append("component_underfilled")
    if report["duplicate_conflict_count"]:
        failures.append("conflicting_duplicate_labels")
    if report["missing_validation_components"]:
        failures.append("missing_validation_components")
    overlap_failures = {
        key: value["count"]
        for key, value in split_report["pattern_overlap_report"].items()
        if key != "source_document_id" and value["count"]
    }
    if overlap_failures:
        failures.append("train_validation_pattern_overlap")
    report["failures"] = failures
    if failures:
        report["status"] = "fail"

    output_dir = Path(args.output_dir)
    validation_output_dir = Path(args.validation_output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    validation_output_dir.parent.mkdir(parents=True, exist_ok=True)

    ds = DatasetDict(
        {
            "train": Dataset.from_list(dataset_rows(train_rows)),
            "validation": Dataset.from_list(dataset_rows(validation_rows)),
        }
    )
    ds.save_to_disk(str(output_dir))
    Dataset.from_list(dataset_rows(validation_rows)).save_to_disk(str(validation_output_dir))

    audit_path = Path(args.audit_jsonl)
    if audit_path.exists():
        audit_path.unlink()
    write_audit(audit_path, train_rows, "train")
    write_audit(audit_path, validation_rows, "validation")

    Path(args.report_json).write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    if failures:
        raise ValueError(f"V16-parent FP-correction dataset gates failed: {failures}")

    print(
        json.dumps(
            {
                "status": "pass",
                "rows": len(rows),
                "train_rows": len(train_rows),
                "validation_rows": len(validation_rows),
                "report_json": args.report_json,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
