from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ATTACK_LABELS = {"prompt_injection", "attack", "malicious", 1, "1", True}

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare V16-parent-FP false negatives against original V16 scores."
    )
    parser.add_argument("--results-dir", default="v16-parent-result")
    parser.add_argument("--baseline-model", default="v16")
    parser.add_argument("--candidate-model", default="v16_parent_fp")
    parser.add_argument("--threshold", type=float, default=0.82)
    parser.add_argument("--output-dir", default="v16-parent-fp-regression-diagnostics")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def score(row: dict[str, Any]) -> float:
    value = row.get("document_max_prompt_injection_score", 0.0)
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def is_attack(row: dict[str, Any]) -> bool:
    return row.get("document_label") in ATTACK_LABELS


def score_band(value: float) -> str:
    for name, low, high in SCORE_BANDS:
        if low <= value < high:
            return name
    if value >= 1.0:
        return ">=0.999"
    return "unknown"


def key_for(row: dict[str, Any]) -> tuple[str, str]:
    document_id = str(row.get("document_id") or "")
    if document_id:
        return ("document_id", document_id)
    attack_hash = str(row.get("attack_text_hash") or "")
    if attack_hash:
        return ("attack_text_hash", attack_hash)
    return ("best_window_text_hash", str(row.get("best_window_text_hash") or ""))


def model_corpus_dirs(results_dir: Path, model: str) -> list[Path]:
    root = results_dir / model
    if not root.exists():
        raise FileNotFoundError(f"Missing model results directory: {root}")
    return sorted(p for p in root.iterdir() if (p / "document-results.jsonl").exists())


def load_model_rows(results_dir: Path, model: str) -> dict[str, dict[tuple[str, str], dict[str, Any]]]:
    by_corpus: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    for corpus_dir in model_corpus_dirs(results_dir, model):
        rows = read_jsonl(corpus_dir / "document-results.jsonl")
        by_corpus[corpus_dir.name] = {key_for(row): row for row in rows}
    return by_corpus


def counter_dict(counter: Counter[Any]) -> dict[str, int]:
    return {str(k): int(v) for k, v in counter.most_common()}


def pct(part: int, total: int) -> float:
    return 0.0 if total <= 0 else part / total


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    baseline_ge_099 = sum(1 for r in records if r["baseline_score"] >= 0.99)
    baseline_ge_0999 = sum(1 for r in records if r["baseline_score"] >= 0.999)
    candidate_lt_001 = sum(1 for r in records if r["candidate_score"] < 0.01)
    baseline_ge_099_candidate_lt_001 = sum(
        1 for r in records if r["baseline_score"] >= 0.99 and r["candidate_score"] < 0.01
    )

    return {
        "total_candidate_false_negatives": total,
        "baseline_score_ge_0_99": baseline_ge_099,
        "baseline_score_ge_0_99_share": pct(baseline_ge_099, total),
        "baseline_score_ge_0_999": baseline_ge_0999,
        "baseline_score_ge_0_999_share": pct(baseline_ge_0999, total),
        "candidate_score_lt_0_01": candidate_lt_001,
        "candidate_score_lt_0_01_share": pct(candidate_lt_001, total),
        "baseline_ge_0_99_and_candidate_lt_0_01": baseline_ge_099_candidate_lt_001,
        "baseline_ge_0_99_and_candidate_lt_0_01_share": pct(
            baseline_ge_099_candidate_lt_001, total
        ),
        "by_corpus": counter_dict(Counter(r["corpus"] for r in records)),
        "by_language": counter_dict(Counter(r["language"] for r in records)),
        "by_category": counter_dict(Counter(r["category"] for r in records)),
        "by_semantic_family": counter_dict(Counter(r["semantic_family"] for r in records)),
        "by_source_name": counter_dict(Counter(r["source_name"] for r in records)),
        "by_baseline_score_band": counter_dict(Counter(r["baseline_score_band"] for r in records)),
        "by_candidate_score_band": counter_dict(Counter(r["candidate_score_band"] for r in records)),
    }


def write_md(path: Path, summary: dict[str, Any], records: list[dict[str, Any]], threshold: float) -> None:
    lines = [
        "# V16-Parent-FP Regression Diagnostic",
        "",
        f"Candidate false-negative threshold: `{threshold}`",
        "",
        "## Verdict",
        "",
    ]

    total = summary["total_candidate_false_negatives"]
    baseline_ge_099 = summary["baseline_score_ge_0_99"]
    candidate_lt_001 = summary["candidate_score_lt_0_01"]
    both = summary["baseline_ge_0_99_and_candidate_lt_0_01"]

    if total and baseline_ge_099 / total >= 0.8:
        lines.append("This strongly supports catastrophic forgetting / overcorrection.")
    else:
        lines.append("This shows recall regression, but not a clean all-high-confidence V16 pattern.")

    lines += [
        "",
        "## Headline",
        "",
        f"- Candidate false negatives: `{total}`",
        f"- Of those, original V16 score >= 0.99: `{baseline_ge_099}` ({baseline_ge_099 / total:.2%})"
        if total
        else "- Of those, original V16 score >= 0.99: `0`",
        f"- Of those, candidate score < 0.01: `{candidate_lt_001}` ({candidate_lt_001 / total:.2%})"
        if total
        else "- Of those, candidate score < 0.01: `0`",
        f"- Original V16 >= 0.99 and candidate < 0.01: `{both}` ({both / total:.2%})"
        if total
        else "- Original V16 >= 0.99 and candidate < 0.01: `0`",
        "",
        "## By Corpus",
        "",
        "| Corpus | False negatives |",
        "| --- | ---: |",
    ]
    for key, value in summary["by_corpus"].items():
        lines.append(f"| {key} | {value} |")

    lines += [
        "",
        "## Candidate Score Bands",
        "",
        "| Candidate score band | Count |",
        "| --- | ---: |",
    ]
    for key, value in summary["by_candidate_score_band"].items():
        lines.append(f"| {key} | {value} |")

    lines += [
        "",
        "## Original V16 Score Bands",
        "",
        "| V16 score band | Count |",
        "| --- | ---: |",
    ]
    for key, value in summary["by_baseline_score_band"].items():
        lines.append(f"| {key} | {value} |")

    lines += [
        "",
        "## Top Semantic Families",
        "",
        "| Semantic family | Count |",
        "| --- | ---: |",
    ]
    for key, value in list(summary["by_semantic_family"].items())[:20]:
        lines.append(f"| {key} | {value} |")

    lines += [
        "",
        "## Lowest Candidate Scores",
        "",
        "| Corpus | Document | Candidate | V16 | Language | Category | Family |",
        "| --- | --- | ---: | ---: | --- | --- | --- |",
    ]
    for row in sorted(records, key=lambda r: r["candidate_score"])[:30]:
        lines.append(
            "| {corpus} | {document_id} | {candidate_score:.6g} | {baseline_score:.6g} | "
            "{language} | {category} | {semantic_family} |".format(**row)
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline = load_model_rows(results_dir, args.baseline_model)
    candidate = load_model_rows(results_dir, args.candidate_model)

    records: list[dict[str, Any]] = []
    missing_baseline = 0

    for corpus, candidate_rows_by_key in candidate.items():
        baseline_rows_by_key = baseline.get(corpus, {})
        for key, candidate_row in candidate_rows_by_key.items():
            if not is_attack(candidate_row):
                continue
            candidate_score = score(candidate_row)
            if candidate_score >= args.threshold:
                continue
            baseline_row = baseline_rows_by_key.get(key)
            if baseline_row is None:
                missing_baseline += 1
                continue
            baseline_score = score(baseline_row)
            records.append(
                {
                    "corpus": corpus,
                    "document_id": candidate_row.get("document_id"),
                    "document_label": candidate_row.get("document_label"),
                    "candidate_score": candidate_score,
                    "baseline_score": baseline_score,
                    "score_delta": candidate_score - baseline_score,
                    "candidate_score_band": score_band(candidate_score),
                    "baseline_score_band": score_band(baseline_score),
                    "language": candidate_row.get("language") or "unknown",
                    "category": candidate_row.get("category") or "unknown",
                    "semantic_family": candidate_row.get("semantic_family") or "unknown",
                    "source_name": candidate_row.get("source_name") or "unknown",
                    "attack_text_hash": candidate_row.get("attack_text_hash"),
                    "best_window_index": candidate_row.get("best_window_index"),
                    "best_window_text_hash": candidate_row.get("best_window_text_hash"),
                    "window_count": candidate_row.get("window_count"),
                    "window_count_bucket": candidate_row.get("window_count_bucket"),
                    "baseline_model_id": baseline_row.get("model_id"),
                    "candidate_model_id": candidate_row.get("model_id"),
                }
            )

    summary = summarize(records)
    summary["threshold"] = args.threshold
    summary["results_dir"] = str(results_dir)
    summary["baseline_model"] = args.baseline_model
    summary["candidate_model"] = args.candidate_model
    summary["missing_baseline_matches"] = missing_baseline

    json_path = output_dir / "v16-parent-fp-fn-regression-summary.json"
    jsonl_path = output_dir / "v16-parent-fp-fn-regression-records.jsonl"
    md_path = output_dir / "v16-parent-fp-fn-regression-summary.md"

    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with jsonl_path.open("w", encoding="utf-8") as f:
        for record in sorted(records, key=lambda r: (r["corpus"], r["candidate_score"])):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    write_md(md_path, summary, records, args.threshold)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(md_path)
    print(json_path)
    print(jsonl_path)


if __name__ == "__main__":
    main()
