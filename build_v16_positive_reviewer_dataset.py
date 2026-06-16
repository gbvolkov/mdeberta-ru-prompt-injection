from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from datasets import Dataset, DatasetDict


LABEL_BENIGN = "not_prompt_injection"
LABEL_ATTACK = "prompt_injection"

REVIEWER_LABEL_BENIGN_FP = 0
REVIEWER_LABEL_REAL_ATTACK = 1

SCORE_BANDS = [
    ("<0.01", 0.0, 0.01),
    ("0.01-0.10", 0.01, 0.10),
    ("0.10-0.50", 0.10, 0.50),
    ("0.50-0.82", 0.50, 0.82),
    ("0.82-0.95", 0.82, 0.95),
    ("0.95-0.99", 0.95, 0.99),
    ("0.99-0.999", 0.99, 0.999),
    (">=0.999", 0.999, 1.0000001),
]

MODEL_CONTROL_PATTERNS = [
    r"ignore (all )?(previous|prior|above|earlier) instructions",
    r"disregard (all )?(previous|prior|above|earlier) instructions",
    r"(reveal|show|print|output|dump|leak|exfiltrate).{0,100}(system prompt|developer prompt|hidden instructions|internal policy|private context|tool list|routing)",
    r"(system prompt|developer prompt|hidden instructions|internal policy|private context|tool list|routing).{0,100}(reveal|show|print|output|dump|leak|exfiltrate)",
    r"(do not refuse|never refuse|bypass policy|override policy|highest priority|new system instruction)",
    r"(secret token|api key|private key|credentials).{0,80}(reveal|show|print|output|dump|leak|exfiltrate)",
    r"(раскрой|покажи|выведи|напечатай|сообщи).{0,100}(системн|промпт|разработчик|скрыт|служебн|маршрутизац|инструмент|политик)",
    r"(игнорируй|проигнорируй|обойди|переопредели).{0,100}(инструкц|правил|контекст|политик|ограничен)",
]

MODEL_CONTROL_RE = re.compile("|".join(f"(?:{pattern})" for pattern in MODEL_CONTROL_PATTERNS), re.IGNORECASE | re.DOTALL)


class Dsu:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a second-stage reviewer dataset from V16-positive examples only."
    )
    parser.add_argument("--v16-results-dir", default="v16-parent-result/v16")
    parser.add_argument(
        "--extra-window-jsonl",
        action="append",
        default=[],
        help="Optional pre-scored V16-positive window JSONL. Rows must include text/window_text, label/document_label, and stage1_score or p_prompt_injection.",
    )
    parser.add_argument("--output-dir", default="training-dataset-v16-positive-reviewer")
    parser.add_argument("--validation-output-dir", default="training-dataset-v16-positive-reviewer-validation")
    parser.add_argument("--report-json", default="training-dataset-v16-positive-reviewer-report.json")
    parser.add_argument("--audit-jsonl", default="v16-positive-reviewer-build/audit-rows.jsonl")
    parser.add_argument("--dropped-jsonl", default="v16-positive-reviewer-build/dropped.jsonl")
    parser.add_argument("--stage1-threshold", type=float, default=0.82)
    parser.add_argument("--target-total-rows", type=int, default=0)
    parser.add_argument("--validation-rows", type=int, default=0)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--research-only", action="store_true")
    parser.add_argument("--allow-underfilled", action="store_true")
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def normalized_for_hash(text: str) -> str:
    return normalize_text(text).casefold()


def stable_hash(text: str, length: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def text_hash(text: str, length: int = 16) -> str:
    return stable_hash(normalized_for_hash(text), length=length)


def score_band(value: float) -> str:
    for name, low, high in SCORE_BANDS:
        if low <= value < high:
            return name
    if value >= 1.0:
        return ">=0.999"
    return "unknown"


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL row") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


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


def resolve_results_dir(path: str | Path) -> Path:
    root = Path(path)
    if (root / "v16").is_dir():
        return root / "v16"
    return root


def score_from_row(row: dict[str, Any]) -> float:
    for key in (
        "stage1_score",
        "p_prompt_injection",
        "document_max_prompt_injection_score",
        "max_prompt_injection_score",
    ):
        value = row.get(key)
        if value is None or value == "":
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def visible_model_control_signal(text: str) -> bool:
    return bool(MODEL_CONTROL_RE.search(text or ""))


def component_for(corpus: str, reviewer_label: int) -> str:
    prefix = "real_attack" if reviewer_label == REVIEWER_LABEL_REAL_ATTACK else "benign_false_positive"
    return f"{prefix}_v16_positive_{corpus}"


def template_family_id(row: dict[str, Any], reviewer_label: int, band: str) -> str:
    if reviewer_label == REVIEWER_LABEL_REAL_ATTACK:
        attack_hash = str(row.get("attack_text_hash") or "")
        family = str(row.get("semantic_family") or "unknown")
        category = str(row.get("category") or "unknown")
        if attack_hash:
            return f"attack:{attack_hash}:{family}:{category}"
        return f"attack:{family}:{category}:{band}"
    category = str(row.get("category") or "unknown")
    source = str(row.get("source_name") or "unknown")
    return f"benign:{source}:{category}:{band}"


def source_pattern_id(row: dict[str, Any], band: str) -> str:
    parts = [
        str(row.get("source_name") or "unknown"),
        str(row.get("category") or "unknown"),
        str(row.get("semantic_family") or "unknown"),
        str(row.get("window_count_bucket") or "unknown"),
        band,
    ]
    return ":".join(parts)


def candidate_from_raw_row(
    args: argparse.Namespace,
    row: dict[str, Any],
    *,
    corpus: str,
    source_is_eval_result: bool,
    dropped: list[dict[str, Any]],
) -> dict[str, Any] | None:
    stage1_score = score_from_row(row)
    if stage1_score < args.stage1_threshold:
        dropped.append(
            {
                "reason": "not_v16_positive",
                "corpus": corpus,
                "document_id": row.get("document_id"),
                "window_index": row.get("window_index"),
                "stage1_score": stage1_score,
            }
        )
        return None
    text = str(row.get("window_text") or row.get("text") or "")
    if not normalize_text(text):
        dropped.append(
            {
                "reason": "empty_text",
                "corpus": corpus,
                "document_id": row.get("document_id"),
                "window_index": row.get("window_index"),
            }
        )
        return None

    source_label = label_name(row.get("document_label", row.get("label")))
    reviewer_label = REVIEWER_LABEL_REAL_ATTACK if source_label == LABEL_ATTACK else REVIEWER_LABEL_BENIGN_FP
    visible_attack = visible_model_control_signal(text)
    trusted_malicious_provenance = bool(
        source_label == LABEL_ATTACK
        and (
            row.get("trusted_malicious_provenance")
            or row.get("trusted_attack")
            or row.get("manual_reviewed_attack")
            or row.get("attack_visible_in_window")
            or row.get("attack_anchor_visible_in_window")
            or row.get("direct_attack_signal")
        )
    )

    if reviewer_label == REVIEWER_LABEL_REAL_ATTACK and not (visible_attack or trusted_malicious_provenance):
        dropped.append(
            {
                "reason": "positive_without_visible_attack_or_trusted_provenance",
                "corpus": corpus,
                "document_id": row.get("document_id"),
                "window_index": row.get("window_index"),
                "stage1_score": stage1_score,
                "source_is_eval_result": source_is_eval_result,
            }
        )
        return None
    if reviewer_label == REVIEWER_LABEL_BENIGN_FP and visible_attack:
        dropped.append(
            {
                "reason": "benign_with_visible_model_control_signal",
                "corpus": corpus,
                "document_id": row.get("document_id"),
                "window_index": row.get("window_index"),
                "stage1_score": stage1_score,
                "source_is_eval_result": source_is_eval_result,
            }
        )
        return None

    band = score_band(stage1_score)
    h = text_hash(text)
    component = str(row.get("component") or component_for(corpus, reviewer_label))
    row_out = {
                "text": text,
                "label": reviewer_label,
                "reviewer_label_name": "real_attack"
                if reviewer_label == REVIEWER_LABEL_REAL_ATTACK
                else "benign_false_positive",
                "component": component,
                "corpus": corpus,
                "document_id": str(row.get("document_id") or ""),
                "source_name": str(row.get("source_name") or "unknown"),
                "category": str(row.get("category") or "unknown"),
                "language": str(row.get("language") or "unknown"),
                "semantic_family": str(row.get("semantic_family") or "unknown"),
                "attack_text_hash": row.get("attack_text_hash"),
                "window_index": row.get("window_index"),
                "window_count": row.get("window_count"),
                "window_text_hash": row.get("window_text_hash") or h,
                "text_hash": h,
                "stage1_score": stage1_score,
                "stage1_score_band": band,
                "stage1_threshold": args.stage1_threshold,
                "v16_positive": True,
                "visible_model_control_signal": visible_attack,
                "trusted_malicious_provenance": trusted_malicious_provenance,
                "research_only": args.research_only,
                "source_is_eval_result": source_is_eval_result,
                "template_family_id": template_family_id(row, reviewer_label, band),
                "source_pattern_id": source_pattern_id(row, band),
                "attack_generation_recipe_id": "",
                "benign_generation_recipe_id": "",
                "eval_derived_cluster_id": str(row.get("document_id") or h),
    }
    if reviewer_label == REVIEWER_LABEL_REAL_ATTACK:
        row_out["attack_generation_recipe_id"] = str(
            row.get("attack_generation_recipe_id")
            or f"{row_out['semantic_family']}:{row_out['category']}:{row_out['stage1_score_band']}"
        )
    else:
        row_out["benign_generation_recipe_id"] = str(
            row.get("benign_generation_recipe_id")
            or f"{row_out['source_name']}:{row_out['category']}:{row_out['stage1_score_band']}"
        )
    return row_out


def build_candidate_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []

    if args.research_only:
        results_dir = resolve_results_dir(args.v16_results_dir)
        if not results_dir.exists():
            raise FileNotFoundError(results_dir)

        for corpus_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
            window_path = corpus_dir / "window-results.jsonl"
            if not window_path.exists():
                continue
            corpus = corpus_dir.name
            for row in read_jsonl(window_path):
                candidate = candidate_from_raw_row(
                    args,
                    row,
                    corpus=corpus,
                    source_is_eval_result=True,
                    dropped=dropped,
                )
                if candidate is not None:
                    rows.append(candidate)
    elif not args.extra_window_jsonl:
        raise ValueError(
            "Non-research reviewer dataset build requires --extra-window-jsonl with fresh/non-eval V16-positive rows."
        )

    for extra_path_value in args.extra_window_jsonl:
        extra_path = Path(extra_path_value)
        if not extra_path.exists():
            raise FileNotFoundError(extra_path)
        corpus = extra_path.stem
        for row in read_jsonl(extra_path):
            candidate = candidate_from_raw_row(
                args,
                row,
                corpus=str(row.get("corpus") or corpus),
                source_is_eval_result=False,
                dropped=dropped,
            )
            if candidate is not None:
                rows.append(candidate)

    return rows, dropped


def dedupe_rows(rows: list[dict[str, Any]], dropped: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    by_hash: dict[str, dict[str, Any]] = {}
    conflicts: set[str] = set()
    for row in rows:
        h = str(row["text_hash"])
        existing = by_hash.get(h)
        if existing is None:
            by_hash[h] = row
            continue
        if existing["label"] != row["label"]:
            conflicts.add(h)
            dropped.append(
                {
                    "reason": "conflicting_duplicate_label",
                    "text_hash": h,
                    "existing_label": existing["label"],
                    "new_label": row["label"],
                    "existing_component": existing.get("component"),
                    "new_component": row.get("component"),
                }
            )
            continue
        if h in conflicts:
            dropped.append(
                {
                    "reason": "duplicate_of_conflicting_hash",
                    "text_hash": h,
                    "component": row.get("component"),
                }
            )
            continue
        if float(row.get("stage1_score", 0.0)) > float(existing.get("stage1_score", 0.0)):
            dropped.append(
                {
                    "reason": "duplicate_same_label_lower_score",
                    "text_hash": h,
                    "kept_component": row.get("component"),
                    "dropped_component": existing.get("component"),
                }
            )
            by_hash[h] = row
        else:
            dropped.append(
                {
                    "reason": "duplicate_same_label_lower_score",
                    "text_hash": h,
                    "kept_component": existing.get("component"),
                    "dropped_component": row.get("component"),
                }
            )
    for h in sorted(conflicts):
        existing = by_hash.pop(h, None)
        if existing is not None:
            dropped.append(
                {
                    "reason": "removed_conflicting_duplicate_hash",
                    "text_hash": h,
                    "component": existing.get("component"),
                    "label": existing.get("label"),
                }
            )
    return list(by_hash.values()), sorted(conflicts)


def select_rows(rows: list[dict[str, Any]], target_total: int, seed: int) -> list[dict[str, Any]]:
    rnd = random.Random(seed)
    if target_total <= 0 or len(rows) <= target_total:
        selected = list(rows)
        rnd.shuffle(selected)
        return selected

    by_label: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_label[int(row["label"])].append(row)
    for label_rows in by_label.values():
        rnd.shuffle(label_rows)

    half = target_total // 2
    selected = by_label[REVIEWER_LABEL_REAL_ATTACK][:half] + by_label[REVIEWER_LABEL_BENIGN_FP][:half]
    remaining = [row for row in rows if row not in selected]
    rnd.shuffle(remaining)
    selected.extend(remaining[: max(0, target_total - len(selected))])
    rnd.shuffle(selected)
    return selected[:target_total]


def connected_split(
    rows: list[dict[str, Any]],
    *,
    validation_rows: int,
    validation_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    if not rows:
        return [], [], {}

    protected_keys = [
        "document_id",
        "template_family_id",
        "source_pattern_id",
        "attack_generation_recipe_id",
        "benign_generation_recipe_id",
        "eval_derived_cluster_id",
    ]
    dsu = Dsu(len(rows))
    seen: dict[tuple[str, str], int] = {}
    for idx, row in enumerate(rows):
        for key in protected_keys:
            value = str(row.get(key) or "")
            if not value:
                continue
            lookup = (key, value)
            prev = seen.get(lookup)
            if prev is None:
                seen[lookup] = idx
            else:
                dsu.union(prev, idx)

    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for idx, row in enumerate(rows):
        groups[dsu.find(idx)].append(row)

    group_list = list(groups.values())
    rnd = random.Random(seed)
    rnd.shuffle(group_list)

    if validation_rows <= 0:
        validation_rows = int(round(len(rows) * validation_fraction))
    validation_rows = max(1, min(validation_rows, len(rows) - 1))

    validation: list[dict[str, Any]] = []
    train: list[dict[str, Any]] = []
    label_need = {REVIEWER_LABEL_REAL_ATTACK, REVIEWER_LABEL_BENIGN_FP}
    component_need = {str(row["component"]) for row in rows}

    for group in sorted(group_list, key=lambda g: len(g), reverse=True):
        group_labels = {int(row["label"]) for row in group}
        group_components = {str(row["component"]) for row in group}
        improves_coverage = bool(group_labels & label_need) or bool(group_components & component_need)
        if len(validation) + len(group) <= validation_rows or (not validation and len(group) < len(rows)) or improves_coverage:
            if len(validation) + len(group) <= max(validation_rows * 2, validation_rows + 100):
                validation.extend(group)
                label_need -= group_labels
                component_need -= group_components
                continue
        train.extend(group)

    if not train:
        train, validation = validation[len(validation) // 2 :], validation[: len(validation) // 2]

    overlap_report = protected_overlap_report(train, validation, protected_keys)
    return train, validation, overlap_report


def protected_overlap_report(
    train: list[dict[str, Any]], validation: list[dict[str, Any]], keys: list[str]
) -> dict[str, int]:
    report: dict[str, int] = {}
    for key in keys:
        train_values = {str(row.get(key) or "") for row in train if str(row.get(key) or "")}
        val_values = {str(row.get(key) or "") for row in validation if str(row.get(key) or "")}
        report[key] = len(train_values & val_values)
    return report


def counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return {str(k): int(v) for k, v in Counter(str(row.get(key) or "unknown") for row in rows).most_common()}


def final_dataset_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"text": row["text"], "label": int(row["label"])} for row in rows]


def save_dataset(train: list[dict[str, Any]], validation: list[dict[str, Any]], output_dir: str, validation_output_dir: str) -> None:
    ds = DatasetDict(
        {
            "train": Dataset.from_list(final_dataset_rows(train)),
            "validation": Dataset.from_list(final_dataset_rows(validation)),
        }
    )
    ds.save_to_disk(output_dir)
    Dataset.from_list(final_dataset_rows(validation)).save_to_disk(validation_output_dir)


def build_report(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    train: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    dropped: list[dict[str, Any]],
    conflicts: list[str],
    overlap_report: dict[str, int],
) -> dict[str, Any]:
    failures: list[str] = []
    final_label_by_hash: dict[str, int] = {}
    final_conflicts = 0
    for row in selected:
        h = str(row["text_hash"])
        label = int(row["label"])
        if h in final_label_by_hash and final_label_by_hash[h] != label:
            final_conflicts += 1
        final_label_by_hash[h] = label
    if final_conflicts:
        failures.append("conflicting_duplicate_labels")
    if any(overlap_report.values()):
        failures.append("protected_train_validation_overlap")
    if {int(row["label"]) for row in train} != {0, 1}:
        failures.append("train_missing_label")
    if {int(row["label"]) for row in validation} != {0, 1}:
        failures.append("validation_missing_label")
    if not selected:
        failures.append("empty_dataset")
    if args.target_total_rows > 0 and len(selected) < int(args.target_total_rows * 0.8):
        failures.append("selected_rows_below_80_percent_target")
    if failures and args.allow_underfilled:
        failures = [name for name in failures if name not in {"selected_rows_below_80_percent_target"}]

    return {
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "research_only": args.research_only,
        "extra_window_jsonl": list(args.extra_window_jsonl),
        "stage1_threshold": args.stage1_threshold,
        "input_rows_after_v16_positive_and_visibility_gates": len(rows),
        "selected_rows": len(selected),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "dropped_rows": len(dropped),
        "quarantined_conflicting_duplicate_text_hashes": len(conflicts),
        "final_conflicting_duplicate_labels": final_conflicts,
        "protected_overlap": overlap_report,
        "label_counts_all": counts(selected, "reviewer_label_name"),
        "label_counts_train": counts(train, "reviewer_label_name"),
        "label_counts_validation": counts(validation, "reviewer_label_name"),
        "components_all": counts(selected, "component"),
        "components_train": counts(train, "component"),
        "components_validation": counts(validation, "component"),
        "languages_all": counts(selected, "language"),
        "categories_all": counts(selected, "category"),
        "stage1_score_bands_all": counts(selected, "stage1_score_band"),
        "dropped_by_reason": {str(k): int(v) for k, v in Counter(row.get("reason") for row in dropped).most_common()},
        "output_dir": args.output_dir,
        "validation_output_dir": args.validation_output_dir,
        "audit_jsonl": args.audit_jsonl,
        "dropped_jsonl": args.dropped_jsonl,
    }


def main() -> None:
    args = parse_args()
    candidate_rows, dropped = build_candidate_rows(args)
    deduped_rows, conflicts = dedupe_rows(candidate_rows, dropped)
    selected = select_rows(deduped_rows, args.target_total_rows, args.seed)
    train, validation, overlap_report = connected_split(
        selected,
        validation_rows=args.validation_rows,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    report = build_report(args, deduped_rows, selected, train, validation, dropped, conflicts, overlap_report)

    write_jsonl(args.audit_jsonl, train + validation)
    write_jsonl(args.dropped_jsonl, dropped)
    write_json(args.report_json, report)

    if report["failures"]:
        raise ValueError(f"Reviewer dataset gates failed: {report['failures']}")

    save_dataset(train, validation, args.output_dir, args.validation_output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
