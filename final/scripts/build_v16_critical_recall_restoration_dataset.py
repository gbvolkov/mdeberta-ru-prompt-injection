from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

import build_v15_anchored_critical_correction_dataset as v15
from v12_pipeline_utils import (
    LABEL_ATTACK,
    LABEL_BENIGN,
    build_production_windows,
    infer_language,
    iter_jsonl,
    normalize_text,
    text_hash,
    write_json,
    write_jsonl,
)


COMPONENT_RENAMES = {
    "critical_ru_anchored_positive": "critical_ru_near_anchor_positive",
    "critical_ru_semantic_paraphrase_positive": "critical_ru_semantic_hard_positive",
}

TRAINING_DERIVED_MARKERS = (
    "training-dataset",
    "replay",
    "v10_attack_replay",
    "v10_replay",
    "v11_replay",
    "v12_critical",
    "v12_replay",
    "v13_replay",
    "v14_replay",
    "v15_replay",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build V16 recall-restoration dataset from V13 diagnostics. "
            "V16 is intended to train from V13, not V15."
        )
    )
    parser.add_argument("--tokenizer-id", default="mdeberta-ru-prompt-injection-v13-critical-correction-ft")
    parser.add_argument("--critical-corpus-jsonl", default="v13-critical-ru-validation-corpus.jsonl")
    parser.add_argument("--critical-results-jsonl", default="validation-comparison-v10-v13-core/v13/v13_critical_ru-documents.jsonl")
    parser.add_argument("--v15-critical-results-jsonl", default="validation-comparison-v10-v13-core/v15/v13_critical_ru-documents.jsonl")
    parser.add_argument("--benign-prod-jsonl", default="benign-prod-calibration-dev.jsonl")
    parser.add_argument("--benign-prod-results-jsonl", default="validation-comparison-v10-v13-core/v13/benign_prod_dev-documents.jsonl")
    parser.add_argument("--benign-window-jsonl", default="v13-benign-validation-corpus.jsonl")
    parser.add_argument("--benign-window-results-jsonl", default="validation-comparison-v10-v13-core/v13/v13_benign_windows-documents.jsonl")
    parser.add_argument("--carrier-jsonl", default="benign-prod-calibration-dev.jsonl")
    parser.add_argument(
        "--base-replay-dataset-dir",
        default=None,
        help=(
            "Optional prepared training dataset replay directory. Ignored unless "
            "--include-prior-training-replay is passed."
        ),
    )
    parser.add_argument(
        "--extra-replay-dataset-dir",
        action="append",
        default=None,
        help=(
            "Optional additional prepared training dataset replay directory. May be passed "
            "multiple times, and is ignored unless --include-prior-training-replay is passed."
        ),
    )
    parser.add_argument("--output-dir", default="training-dataset-v16-critical-recall-restoration-windowed")
    parser.add_argument("--validation-output-dir", default="training-dataset-v16-critical-recall-restoration-windowed-validation")
    parser.add_argument("--report-json", default="training-dataset-v16-critical-recall-restoration-windowed-report.json")
    parser.add_argument("--diagnostic-errors-jsonl", default="v16-diagnostic-errors-used-for-training.jsonl")
    parser.add_argument("--thresholds", default="0.95,0.99,0.999")
    parser.add_argument("--include-prior-training-replay", action="store_true")
    parser.add_argument("--target-total-rows", type=int, default=45000)
    parser.add_argument("--validation-rows", type=int, default=6000)
    parser.add_argument("--anchored-critical-target", type=int, default=13000)
    parser.add_argument("--semantic-critical-target", type=int, default=3500)
    parser.add_argument("--embedded-critical-target", type=int, default=5000)
    parser.add_argument("--wrapper-critical-target", type=int, default=1500)
    parser.add_argument("--v15-regression-guard-target", type=int, default=4500)
    parser.add_argument("--benign-fp-hard-target", type=int, default=1000)
    parser.add_argument("--benign-keyword-target", type=int, default=0)
    parser.add_argument("--benign-context-target", type=int, default=16000)
    parser.add_argument("--benign-replay-target", type=int, default=0)
    parser.add_argument("--attack-replay-target", type=int, default=0)
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument("--allow-underfilled", action="store_true")
    return parser.parse_args()


def retag(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    component = str(out.get("component", ""))
    out["component"] = COMPONENT_RENAMES.get(component, component)
    for key in ("source_name", "document_id"):
        value = str(out.get(key, ""))
        if value.startswith("v15_"):
            out[key] = "v16_" + value[4:]
        else:
            out[key] = value
    out["review_status"] = out.get("review_status") or "diagnostic_mined"
    return out


def is_training_derived_record(row: dict[str, Any]) -> bool:
    values = [
        row.get("source_name"),
        row.get("source"),
        row.get("category"),
        row.get("component"),
        row.get("semantic_family"),
        row.get("document_id"),
        row.get("source_document_id"),
    ]
    joined = " ".join(str(value).lower() for value in values if value is not None)
    return any(marker in joined for marker in TRAINING_DERIVED_MARKERS)


def filter_training_derived_pairs(
    rows: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept_rows: list[dict[str, Any]] = []
    kept_diagnostics: list[dict[str, Any]] = []
    for row, diagnostic in zip(rows, diagnostics):
        if is_training_derived_record(row) or is_training_derived_record(diagnostic):
            continue
        kept_rows.append(row)
        kept_diagnostics.append(diagnostic)
    return kept_rows, kept_diagnostics


def dedupe(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [retag(row) for row in v15.dedupe(rows)]


def load_by_id(path: str | Path) -> dict[str, dict[str, Any]]:
    rows = {}
    for row in iter_jsonl(path):
        doc_id = row.get("document_id") or row.get("id")
        if doc_id:
            rows[str(doc_id)] = row
    return rows


def miss_band(score: float) -> str:
    if score < 0.95:
        return "v15_miss_below_0.95"
    if score < 0.99:
        return "v15_miss_below_0.99"
    if score < 0.999:
        return "v15_miss_below_0.999"
    return "v15_not_missed"


def build_v15_regression_guard_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not Path(args.v15_critical_results_jsonl).exists():
        return [], []
    rnd = random.Random(args.seed + 160)
    corpus = load_by_id(args.critical_corpus_jsonl)
    anchors: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for result in iter_jsonl(args.v15_critical_results_jsonl):
        score = float(result.get("document_max_prompt_injection_score", 0.0))
        if score >= 0.999:
            continue
        doc_id = str(result.get("document_id"))
        source = corpus.get(doc_id)
        if not source:
            continue
        if is_training_derived_record(source) or is_training_derived_record(result):
            continue
        text = normalize_text(source.get("text", ""))
        if not text:
            continue
        band = miss_band(score)
        anchor = {
            "text": text,
            "category": source.get("category", result.get("category", "critical_ru")),
            "language": source.get("language") or result.get("language") or infer_language(text),
            "document_id": doc_id,
            "v15_score": score,
            "score_band": band,
            "semantic_family": source.get("semantic_family", result.get("semantic_family", "critical_ru_override_hidden_prompt_exfiltration")),
            "attack_text_hash": source.get("attack_text_hash") or result.get("attack_text_hash") or text_hash(text, length=64),
        }
        anchors.append(anchor)
        diagnostics.append({**result, "used_as": "v15_regression_guard_anchor", "score_band": band, "source_text_hash": text_hash(text)})
    if not anchors:
        return [], diagnostics

    anchors = sorted(anchors, key=lambda row: float(row.get("v15_score") or 0.0))
    weighted: list[dict[str, Any]] = []
    for anchor in anchors:
        score = float(anchor.get("v15_score") or 0.0)
        if score < 0.95:
            weight = 12
        elif score < 0.99:
            weight = 8
        else:
            weight = 5
        weighted.extend([anchor] * weight)
    rnd.shuffle(weighted)

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    attempt = 0
    while len(rows) < args.v15_regression_guard_target and attempt < args.v15_regression_guard_target * 40:
        anchor = weighted[attempt % len(weighted)]
        fragment = v15.extract_attack_fragment(anchor["text"])
        if attempt % 5 == 0:
            text = fragment
            generation_type = "v15_regression_exact_or_fragment"
        elif attempt % 3 == 0:
            text = v15.lightly_transform_anchor(fragment, rnd, attempt)
            generation_type = "v15_regression_light_anchor"
        else:
            text = v15.mutate_fragment(fragment, rnd, attempt)
            generation_type = "v15_regression_minimal_mutation"
        h = text_hash(text)
        if h in seen:
            attempt += 1
            continue
        seen.add(h)
        rows.append(
            v15.make_row(
                text=text,
                label=1,
                source_name=f"v16_regression_guard_{anchor['score_band']}",
                component="critical_ru_v15_regression_guard_positive",
                category=anchor.get("category", "critical_ru"),
                language=anchor.get("language") or infer_language(text),
                document_id=f"v16_guard_{attempt}_{h}",
                v13_score=float(anchor.get("v15_score") or 0.0),
                score_band_value=anchor.get("score_band", ""),
                semantic_family=anchor.get("semantic_family", "critical_ru_override_hidden_prompt_exfiltration"),
                attack_text_hash=anchor.get("attack_text_hash") or text_hash(fragment, length=64),
                generation_type=generation_type,
                anchor_document_id=anchor.get("document_id", ""),
                anchor_text_hash=text_hash(anchor.get("text", "")),
            )
        )
        attempt += 1
    return rows, diagnostics


def select_benign_hard_negatives(rows: list[dict[str, Any]], target: int, seed: int) -> list[dict[str, Any]]:
    rnd = random.Random(seed + 170)
    candidates = dedupe(rows)
    rnd.shuffle(candidates)
    candidates.sort(key=lambda row: float(row.get("v13_score") or 0.0), reverse=True)
    selected: list[dict[str, Any]] = []
    per_doc: Counter[str] = Counter()
    per_category: Counter[str] = Counter()
    per_doc_cap = 10 if target >= 750 else 4 if target >= 500 else 2
    category_cap = max(250, target // 2)
    for row in candidates:
        doc_id = str(row.get("source_document_id") or row.get("anchor_document_id") or row.get("document_id"))
        category = str(row.get("category", "unknown"))
        if per_doc[doc_id] >= per_doc_cap:
            continue
        if per_category[category] >= category_cap:
            continue
        row = retag(row)
        row["generation_type"] = "v13_false_positive_window_exact_capped"
        selected.append(row)
        per_doc[doc_id] += 1
        per_category[category] += 1
        if len(selected) >= target:
            break
    return selected


def extract_expanded_benign_fp_rows(
    args: argparse.Namespace,
    tokenizer: Any,
    corpus_path: str | Path,
    results_path: str | Path,
    *,
    prefix: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    corpus = load_by_id(corpus_path)
    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for result in iter_jsonl(results_path):
        score = float(result.get("document_max_prompt_injection_score", 0.0))
        if score < 0.95:
            continue
        doc_id = str(result.get("document_id"))
        source = corpus.get(doc_id)
        if not source:
            continue
        if is_training_derived_record(source) or is_training_derived_record(result):
            continue
        text = normalize_text(source.get("text", ""))
        if not text:
            continue
        windows = build_production_windows(text, tokenizer)
        if not windows:
            continue
        best_idx = max(0, min(len(windows) - 1, int(result.get("best_window_index", 0))))
        candidate_indices = set(range(max(0, best_idx - 6), min(len(windows), best_idx + 7)))
        probe_indices = [0, len(windows) // 4, len(windows) // 2, len(windows) * 3 // 4, len(windows) - 1]
        if len(windows) >= 12:
            probe_indices.extend(range(0, len(windows), max(1, len(windows) // 8)))
        for idx in probe_indices:
            candidate_indices.add(max(0, min(len(windows) - 1, idx)))
        band = "fp_at_0.999" if score >= 0.999 else "fp_at_0.99" if score >= 0.99 else "fp_at_0.95"
        category = source.get("category", result.get("category", "unknown"))
        for window_index in sorted(candidate_indices):
            window = windows[window_index]
            generation_type = (
                "v13_false_positive_actual_or_neighbor_window"
                if abs(window_index - best_idx) <= 2
                else "v13_false_positive_document_context_window"
            )
            rows.append(
                v15.make_row(
                    text=window["text"],
                    label=0,
                    source_name=f"v16_{prefix}_{band}_{category}",
                    component="benign_actual_fp_hard_negative",
                    category=category,
                    language=source.get("language", result.get("language")) or infer_language(window["text"]),
                    document_id=f"v16_{prefix}_{doc_id}_{window_index}",
                    source_document_id=doc_id,
                    window_index=window_index,
                    window_count=len(windows),
                    v13_score=score,
                    score_band_value=band,
                    semantic_family="benign_false_positive_actual_document_window",
                    generation_type=generation_type,
                    anchor_document_id=doc_id,
                    anchor_text_hash=text_hash(text),
                )
            )
        diagnostics.append({**result, "used_as": "benign_fp_anchor", "score_band": band})
    return rows, diagnostics


def build_benign_context_rows(args: argparse.Namespace, tokenizer: Any) -> list[dict[str, Any]]:
    rnd = random.Random(args.seed + 175)
    source_paths = [args.benign_window_jsonl, args.benign_prod_jsonl]
    candidates: list[dict[str, Any]] = []
    seen_docs: set[str] = set()
    for source_path in source_paths:
        path_stem = Path(source_path).stem
        for source in iter_jsonl(source_path):
            text = source.get("text") or source.get("document_text") or ""
            if not normalize_text(text):
                continue
            label = source.get("document_label", source.get("label", LABEL_BENIGN))
            if str(label).lower() in {"1", LABEL_ATTACK, "prompt_injection", "attack", "malicious"}:
                continue
            category = source.get("category") or "unknown"
            if category in {"toxic_chat", "unsafe_non_injection"}:
                continue
            if is_training_derived_record(source):
                continue
            doc_id = str(source.get("document_id") or source.get("id") or text_hash(text))
            if doc_id in seen_docs:
                continue
            seen_docs.add(doc_id)
            windows = build_production_windows(text, tokenizer)
            if not windows:
                continue
            preferred_indices = [0, len(windows) // 2, len(windows) - 1]
            if len(windows) > 4:
                preferred_indices.extend([len(windows) // 4, len(windows) * 3 // 4])
            if len(windows) > 6:
                preferred_indices.extend([len(windows) // 3, len(windows) * 2 // 3])
            if len(windows) > 12:
                preferred_indices.extend(
                    [
                        len(windows) // 5,
                        len(windows) * 2 // 5,
                        len(windows) * 3 // 5,
                        len(windows) * 4 // 5,
                    ]
                )
            for window_index in sorted(set(max(0, min(len(windows) - 1, idx)) for idx in preferred_indices)):
                window = windows[window_index]
                candidates.append(
                    v15.make_row(
                        text=window["text"],
                        label=0,
                        source_name=f"v16_context_benign_{path_stem}_{category}",
                        component="benign_context_non_training_source",
                        category=category,
                        language=source.get("language") or infer_language(window["text"]),
                        document_id=f"v16_context_{path_stem}_{doc_id}_{window_index}",
                        source_document_id=doc_id,
                        window_index=int(window["window_index"]),
                        window_count=len(windows),
                        semantic_family="benign_context_non_training_source",
                        generation_type="benign_context_window",
                    )
                )
            if len(candidates) >= args.benign_context_target * 8:
                break
    rnd.shuffle(candidates)
    selected: list[dict[str, Any]] = []
    per_doc: Counter[str] = Counter()
    per_category: Counter[str] = Counter()
    category_cap = max(250, args.benign_context_target // 2)
    for row in dedupe(candidates):
        doc_id = str(row.get("source_document_id") or row.get("document_id"))
        category = str(row.get("category", "unknown"))
        if per_doc[doc_id] >= 8:
            continue
        if per_category[category] >= category_cap:
            continue
        selected.append(row)
        per_doc[doc_id] += 1
        per_category[category] += 1
        if len(selected) >= args.benign_context_target:
            break
    return selected


def split_rows(rows: list[dict[str, Any]], validation_rows: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return v15.stratified_split(rows, validation_rows, seed)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "labels": dict(Counter(LABEL_ATTACK if int(row["label"]) == 1 else LABEL_BENIGN for row in rows)),
        "components": dict(Counter(row.get("component", "unknown") for row in rows)),
        "score_bands": dict(Counter(row.get("score_band", "") for row in rows if row.get("score_band"))),
        "categories": dict(Counter(row.get("category", "unknown") for row in rows).most_common(50)),
        "sources": dict(Counter(row.get("source_name", "unknown") for row in rows).most_common(50)),
        "generation_types": dict(Counter(row.get("generation_type", "unknown") for row in rows)),
    }


def save_dataset(rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows, validation_rows = split_rows(rows, args.validation_rows, args.seed)
    DatasetDict({"train": Dataset.from_list(train_rows), "validation": Dataset.from_list(validation_rows)}).save_to_disk(args.output_dir)
    Dataset.from_list(validation_rows).save_to_disk(args.validation_output_dir)
    return train_rows, validation_rows


def main() -> None:
    args = parse_args()
    thresholds = v15.parse_thresholds(args.thresholds)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)

    exact_critical, critical_diagnostics = v15.extract_critical_misses(args, thresholds)
    exact_critical, critical_diagnostics = filter_training_derived_pairs(exact_critical, critical_diagnostics)
    anchored = [retag(row) for row in v15.build_anchored_critical_rows(args, exact_critical)]
    semantic = [retag(row) for row in v15.build_semantic_critical_rows(args, exact_critical)]
    carriers = [
        row
        for row in v15.load_carriers(args.carrier_jsonl, seed=args.seed, target=max(args.embedded_critical_target, 2000))
        if not is_training_derived_record(row)
    ]
    embedded_and_wrapped = [retag(row) for row in v15.build_embedded_and_wrapper_rows(args, tokenizer, exact_critical, carriers)]
    v15_guard, v15_guard_diagnostics = build_v15_regression_guard_rows(args)

    benign_prod_fp, benign_prod_diagnostics = extract_expanded_benign_fp_rows(
        args, tokenizer, args.benign_prod_jsonl, args.benign_prod_results_jsonl, prefix="benign_prod_fp"
    )
    benign_window_fp, benign_window_diagnostics = extract_expanded_benign_fp_rows(
        args, tokenizer, args.benign_window_jsonl, args.benign_window_results_jsonl, prefix="benign_window_fp"
    )
    benign_prod_diagnostics = [row for row in benign_prod_diagnostics if not is_training_derived_record(row)]
    benign_window_diagnostics = [row for row in benign_window_diagnostics if not is_training_derived_record(row)]
    benign_fp = select_benign_hard_negatives(
        [row for row in benign_prod_fp + benign_window_fp if not is_training_derived_record(row)],
        args.benign_fp_hard_target,
        args.seed,
    )
    keyword_benign = (
        [retag(row) for row in v15.build_keyword_benign_rows(args, tokenizer)[: args.benign_keyword_target]]
        if args.benign_keyword_target > 0
        else []
    )
    benign_context = build_benign_context_rows(args, tokenizer)
    if args.include_prior_training_replay:
        benign_replay, attack_replay = v15.load_replay(args)
        benign_replay = [retag(row) for row in benign_replay]
        attack_replay = [retag(row) for row in attack_replay]
    else:
        benign_replay, attack_replay = [], []

    component_groups = {
        "critical_ru_exact_v13_miss": [retag(row) for row in exact_critical],
        "critical_ru_near_anchor_positive": anchored[: args.anchored_critical_target],
        "critical_ru_semantic_hard_positive": semantic[: args.semantic_critical_target],
        "critical_ru_embedded_positive": [row for row in embedded_and_wrapped if row.get("component") == "critical_ru_embedded_positive"][: args.embedded_critical_target],
        "critical_ru_wrapper_positive": [row for row in embedded_and_wrapped if row.get("component") == "critical_ru_wrapper_positive"][: args.wrapper_critical_target],
        "critical_ru_v15_regression_guard_positive": v15_guard[: args.v15_regression_guard_target],
        "benign_actual_fp_hard_negative": benign_fp[: args.benign_fp_hard_target],
        "benign_keyword_hard_negative": keyword_benign[: args.benign_keyword_target],
        "benign_context_non_training_source": benign_context[: args.benign_context_target],
        "benign_replay": benign_replay[: args.benign_replay_target],
        "attack_replay": attack_replay[: args.attack_replay_target],
    }
    rows = dedupe(row for group in component_groups.values() for row in group)
    training_derived_rows = [row for row in rows if is_training_derived_record(row)]
    if training_derived_rows:
        samples = [
            {
                "component": row.get("component"),
                "source_name": row.get("source_name"),
                "category": row.get("category"),
                "document_id": row.get("document_id"),
            }
            for row in training_derived_rows[:10]
        ]
        raise ValueError(
            "V16 strict build found prior-training-derived rows after filtering: "
            f"{samples}"
        )

    if len(rows) > args.target_total_rows:
        rnd = random.Random(args.seed + 180)
        by_component: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_component[str(row.get("component", "unknown"))].append(row)
        for values in by_component.values():
            rnd.shuffle(values)
        ordered_components = [
            "critical_ru_exact_v13_miss",
            "critical_ru_v15_regression_guard_positive",
            "critical_ru_near_anchor_positive",
            "critical_ru_embedded_positive",
            "critical_ru_semantic_hard_positive",
            "critical_ru_wrapper_positive",
            "attack_replay",
            "benign_context_non_training_source",
            "benign_actual_fp_hard_negative",
            "benign_keyword_hard_negative",
            "benign_replay",
        ]
        trimmed: list[dict[str, Any]] = []
        for component in ordered_components:
            trimmed.extend(by_component.get(component, []))
        rows = trimmed[: args.target_total_rows]

    component_counts = Counter(row.get("component", "unknown") for row in rows)
    required_minimums = {
        "critical_ru_near_anchor_positive": int(args.anchored_critical_target * 0.85),
        "critical_ru_embedded_positive": int(args.embedded_critical_target * 0.85),
        "critical_ru_v15_regression_guard_positive": int(args.v15_regression_guard_target * 0.70),
        "benign_context_non_training_source": int(args.benign_context_target * 0.75),
    }
    if args.include_prior_training_replay:
        required_minimums["benign_replay"] = int(args.benign_replay_target * 0.75)
        required_minimums["attack_replay"] = int(args.attack_replay_target * 0.75)
    underfilled = {
        component: {"actual": component_counts.get(component, 0), "minimum": minimum}
        for component, minimum in required_minimums.items()
        if component_counts.get(component, 0) < minimum
    }
    if (len(rows) < int(args.target_total_rows * 0.90) or underfilled) and not args.allow_underfilled:
        raise ValueError(f"V16 dataset underfilled: rows={len(rows):,}, underfilled={underfilled}. Use --allow-underfilled to inspect.")

    train_rows, validation_rows = save_dataset(rows, args)
    diagnostics = [
        row
        for row in critical_diagnostics + v15_guard_diagnostics + benign_prod_diagnostics + benign_window_diagnostics
        if not is_training_derived_record(row)
    ]
    write_jsonl(args.diagnostic_errors_jsonl, diagnostics)

    report = {
        "output_dir": args.output_dir,
        "validation_output_dir": args.validation_output_dir,
        "report_json": args.report_json,
        "diagnostic_errors_jsonl": args.diagnostic_errors_jsonl,
        "parent_model": "mdeberta-ru-prompt-injection-v13-critical-correction-ft",
        "do_not_train_from": "mdeberta-ru-prompt-injection-v15-anchored-critical-correction-ft",
        "thresholds": thresholds,
        "targets": {
            "target_total_rows": args.target_total_rows,
            "validation_rows": args.validation_rows,
            "anchored_critical_target": args.anchored_critical_target,
            "semantic_critical_target": args.semantic_critical_target,
            "embedded_critical_target": args.embedded_critical_target,
            "wrapper_critical_target": args.wrapper_critical_target,
            "v15_regression_guard_target": args.v15_regression_guard_target,
            "benign_fp_hard_target": args.benign_fp_hard_target,
            "benign_keyword_target": args.benign_keyword_target,
            "benign_context_target": args.benign_context_target,
            "benign_replay_target": args.benign_replay_target,
            "attack_replay_target": args.attack_replay_target,
        },
        "prior_training_replay": {
            "included": bool(args.include_prior_training_replay),
            "base_replay_dataset_dir": args.base_replay_dataset_dir if args.include_prior_training_replay else None,
            "extra_replay_dataset_dir": list(args.extra_replay_dataset_dir or []) if args.include_prior_training_replay else [],
            "note": (
                "No prepared training-dataset-* replay directories are read unless "
                "--include-prior-training-replay is passed."
            ),
        },
        "diagnostic_errors_used": {
            "v13_critical_false_negative_docs": len(critical_diagnostics),
            "v15_critical_false_negative_docs": len(v15_guard_diagnostics),
            "v13_benign_prod_false_positive_docs": len(benign_prod_diagnostics),
            "v13_benign_window_false_positive_docs": len(benign_window_diagnostics),
        },
        "underfilled_components": underfilled,
        "all_rows": summarize(rows),
        "train": summarize(train_rows),
        "validation": summarize(validation_rows),
        "note": (
            "V16 is a V13-parent recall-restoration dataset. By default it does not read prepared "
            "training-dataset-* replay directories. Diagnostic corpora used here are not blind acceptance data."
        ),
    }
    write_json(args.report_json, report)
    print(__import__("json").dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
