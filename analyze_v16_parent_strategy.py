from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


MALICIOUS_CORPORA = [
    "proper_critical_attack_windows",
    "proper_malicious_dev_documents",
    "blind_acceptance_malicious_documents",
    "blind_acceptance_critical_ru_windows",
]

BENIGN_CORPORA = [
    "proper_benign_prod_dev_documents",
    "proper_benign_windows",
    "blind_acceptance_benign_documents",
    "blind_acceptance_corporate_benign_windows",
    "benign_stress_windows",
]

SCORE_BANDS = [
    ("lt_0_01", 0.0, 0.01),
    ("0_01_to_0_10", 0.01, 0.10),
    ("0_10_to_0_50", 0.10, 0.50),
    ("0_50_to_0_82", 0.50, 0.82),
    ("0_82_to_0_95", 0.82, 0.95),
    ("0_95_to_0_99", 0.95, 0.99),
    ("ge_0_99", 0.99, 2.0),
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def score(row: dict[str, Any]) -> float:
    return float(row.get("document_max_prompt_injection_score", 0.0) or 0.0)


def band(value: float) -> str:
    for name, low, high in SCORE_BANDS:
        if low <= value < high:
            return name
    return "unknown"


def read_model_rows(root: Path, corpus: str) -> dict[str, dict[str, Any]]:
    path = root / corpus / "document-results.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    rows = load_jsonl(path)
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        document_id = str(row.get("document_id") or "")
        if not document_id:
            raise ValueError(f"Missing document_id in {path}")
        by_id[document_id] = row
    return by_id


def top_counts(counter: Counter[str], limit: int = 20) -> dict[str, int]:
    return dict(counter.most_common(limit))


def summarize_bucket(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_language: Counter[str] = Counter()
    by_category: Counter[str] = Counter()
    by_source: Counter[str] = Counter()
    by_family: Counter[str] = Counter()
    by_v18_1_score_band: Counter[str] = Counter()
    by_v18_score_band: Counter[str] = Counter()
    by_v16_score_band: Counter[str] = Counter()

    for row in rows:
        base = row["metadata"]
        by_language[str(base.get("language") or "unknown")] += 1
        by_category[str(base.get("category") or "unknown")] += 1
        by_source[str(base.get("source_name") or "unknown")] += 1
        by_family[str(base.get("semantic_family") or "unknown")] += 1
        by_v18_1_score_band[band(row["scores"]["v18_1"])] += 1
        by_v18_score_band[band(row["scores"]["v18"])] += 1
        by_v16_score_band[band(row["scores"]["v16"])] += 1

    return {
        "count": len(rows),
        "by_language": top_counts(by_language),
        "by_category": top_counts(by_category),
        "by_source_name": top_counts(by_source),
        "by_semantic_family": top_counts(by_family),
        "by_v16_score_band": dict(by_v16_score_band),
        "by_v18_score_band": dict(by_v18_score_band),
        "by_v18_1_score_band": dict(by_v18_1_score_band),
    }


def classify_malicious(
    *,
    corpus: str,
    v16_rows: dict[str, dict[str, Any]],
    v18_rows: dict[str, dict[str, Any]],
    v18_1_rows: dict[str, dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    shared_ids = set(v16_rows).intersection(v18_rows).intersection(v18_1_rows)
    if not shared_ids:
        raise ValueError(f"No shared document ids for {corpus}")

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    mismatch = {
        "v16_missing": len(set(v18_1_rows).difference(v16_rows)),
        "v18_missing": len(set(v18_1_rows).difference(v18_rows)),
        "v18_1_missing": len(set(v16_rows).difference(v18_1_rows)),
    }

    for document_id in sorted(shared_ids):
        base = v18_1_rows[document_id]
        v16_score = score(v16_rows[document_id])
        v18_score = score(v18_rows[document_id])
        v18_1_score = score(base)

        v16_detected = v16_score >= threshold
        v18_detected = v18_score >= threshold
        v18_1_detected = v18_1_score >= threshold

        if not v18_detected and not v18_1_detected and v16_detected:
            bucket = "missed_by_v18_and_v18_1_detected_by_v16"
        elif not v18_detected and not v18_1_detected and not v16_detected:
            bucket = "missed_by_all"
        elif v18_detected and not v18_1_detected:
            bucket = "detected_by_v18_missed_by_v18_1"
        elif not v18_detected and v18_1_detected:
            bucket = "missed_by_v18_detected_by_v18_1"
        elif v16_detected and v18_detected and v18_1_detected:
            bucket = "detected_by_all"
        else:
            bucket = "other"

        buckets[bucket].append(
            {
                "document_id": document_id,
                "corpus": corpus,
                "scores": {
                    "v16": v16_score,
                    "v18": v18_score,
                    "v18_1": v18_1_score,
                },
                "detected": {
                    "v16": v16_detected,
                    "v18": v18_detected,
                    "v18_1": v18_1_detected,
                },
                "metadata": {
                    "language": base.get("language"),
                    "category": base.get("category"),
                    "source_name": base.get("source_name"),
                    "semantic_family": base.get("semantic_family"),
                    "attack_text_hash": base.get("attack_text_hash"),
                    "best_window_index": base.get("best_window_index"),
                    "best_window_text_hash": base.get("best_window_text_hash"),
                    "window_count": base.get("window_count"),
                    "window_count_bucket": base.get("window_count_bucket"),
                },
            }
        )

    summaries = {name: summarize_bucket(rows) for name, rows in sorted(buckets.items())}
    v18_1_fn = [row for rows in buckets.values() for row in rows if not row["detected"]["v18_1"]]
    v18_v18_1_shared_fn = [
        row
        for rows in buckets.values()
        for row in rows
        if not row["detected"]["v18"] and not row["detected"]["v18_1"]
    ]
    return {
        "corpus": corpus,
        "threshold": threshold,
        "documents": len(shared_ids),
        "join_mismatch": mismatch,
        "bucket_summaries": summaries,
        "v18_1_false_negatives": summarize_bucket(v18_1_fn),
        "v18_and_v18_1_shared_false_negatives": summarize_bucket(v18_v18_1_shared_fn),
        "examples": {
            name: sorted(rows, key=lambda row: row["scores"]["v18_1"])[:25]
            for name, rows in sorted(buckets.items())
            if name != "detected_by_all"
        },
    }


def classify_benign(
    *,
    corpus: str,
    v16_rows: dict[str, dict[str, Any]],
    v18_rows: dict[str, dict[str, Any]],
    v18_1_rows: dict[str, dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    shared_ids = set(v16_rows).intersection(v18_rows).intersection(v18_1_rows)
    rows = []
    for document_id in sorted(shared_ids):
        rows.append(
            {
                "document_id": document_id,
                "scores": {
                    "v16": score(v16_rows[document_id]),
                    "v18": score(v18_rows[document_id]),
                    "v18_1": score(v18_1_rows[document_id]),
                },
                "metadata": {
                    "language": v18_1_rows[document_id].get("language"),
                    "category": v18_1_rows[document_id].get("category"),
                    "source_name": v18_1_rows[document_id].get("source_name"),
                    "semantic_family": v18_1_rows[document_id].get("semantic_family"),
                },
            }
        )
    return {
        "corpus": corpus,
        "threshold": threshold,
        "documents": len(rows),
        "false_positive_counts": {
            "v16": sum(1 for row in rows if row["scores"]["v16"] >= threshold),
            "v18": sum(1 for row in rows if row["scores"]["v18"] >= threshold),
            "v18_1": sum(1 for row in rows if row["scores"]["v18_1"] >= threshold),
        },
        "high_score_v16_false_positives": summarize_bucket(
            [
                {"scores": row["scores"], "metadata": row["metadata"]}
                for row in rows
                if row["scores"]["v16"] >= threshold
            ]
        ),
        "high_score_v18_1_false_positives": summarize_bucket(
            [
                {"scores": row["scores"], "metadata": row["metadata"]}
                for row in rows
                if row["scores"]["v18_1"] >= threshold
            ]
        ),
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# V16 Parent Strategy Diagnostic",
        "",
        f"Threshold: `{report['threshold']}`",
        "",
        "## Malicious Failure Overlap",
        "",
        "| corpus | docs | V18.1 FN | shared V18+V18.1 FN | shared FN detected by V16 | shared FN missed by all | detected by V18 missed by V18.1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in report["malicious"].values():
        buckets = item["bucket_summaries"]
        lines.append(
            "| {corpus} | {docs} | {v18_1_fn} | {shared_fn} | {a} | {b} | {c} |".format(
                corpus=item["corpus"],
                docs=item["documents"],
                v18_1_fn=item["v18_1_false_negatives"]["count"],
                shared_fn=item["v18_and_v18_1_shared_false_negatives"]["count"],
                a=buckets.get("missed_by_v18_and_v18_1_detected_by_v16", {}).get("count", 0),
                b=buckets.get("missed_by_all", {}).get("count", 0),
                c=buckets.get("detected_by_v18_missed_by_v18_1", {}).get("count", 0),
            )
        )

    lines.extend(
        [
            "",
            "## Benign False Positives",
            "",
            "| corpus | docs | V16 FP | V18 FP | V18.1 FP |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in report["benign"].values():
        fp = item["false_positive_counts"]
        lines.append(
            f"| {item['corpus']} | {item['documents']} | {fp['v16']} | {fp['v18']} | {fp['v18_1']} |"
        )

    lines.extend(
        [
            "",
            "## Decision",
            "",
            report["decision"],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v16-results-dir", default="v18-1-eval-results/v16")
    parser.add_argument("--v18-results-dir", default="v18_eval_results/v18")
    parser.add_argument("--v18-1-results-dir", default="v18-1-eval-results/v18_1")
    parser.add_argument("--threshold", type=float, default=0.82)
    parser.add_argument("--output-dir", default="v18_2_parent_strategy_diagnostics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    v16_root = Path(args.v16_results_dir)
    v18_root = Path(args.v18_results_dir)
    v18_1_root = Path(args.v18_1_results_dir)

    malicious = {}
    for corpus in MALICIOUS_CORPORA:
        malicious[corpus] = classify_malicious(
            corpus=corpus,
            v16_rows=read_model_rows(v16_root, corpus),
            v18_rows=read_model_rows(v18_root, corpus),
            v18_1_rows=read_model_rows(v18_1_root, corpus),
            threshold=args.threshold,
        )

    benign = {}
    for corpus in BENIGN_CORPORA:
        benign[corpus] = classify_benign(
            corpus=corpus,
            v16_rows=read_model_rows(v16_root, corpus),
            v18_rows=read_model_rows(v18_root, corpus),
            v18_1_rows=read_model_rows(v18_1_root, corpus),
            threshold=args.threshold,
        )

    shared_fn_detected_by_v16 = sum(
        item["bucket_summaries"].get("missed_by_v18_and_v18_1_detected_by_v16", {}).get("count", 0)
        for item in malicious.values()
    )
    shared_fn_missed_by_all = sum(
        item["bucket_summaries"].get("missed_by_all", {}).get("count", 0)
        for item in malicious.values()
    )
    v18_1_fn_total = sum(item["v18_1_false_negatives"]["count"] for item in malicious.values())

    if shared_fn_detected_by_v16 > shared_fn_missed_by_all:
        decision = (
            "Most shared V18/V18.1 malicious misses are detected by V16. "
            "This supports switching to a V16-parent FP-correction run."
        )
    else:
        decision = (
            "A large share of shared V18/V18.1 malicious misses are also missed by V16. "
            "This requires broader fresh attack expansion before or alongside a V16-parent run."
        )

    report = {
        "threshold": args.threshold,
        "inputs": {
            "v16_results_dir": str(v16_root),
            "v18_results_dir": str(v18_root),
            "v18_1_results_dir": str(v18_1_root),
        },
        "totals": {
            "v18_1_false_negatives": v18_1_fn_total,
            "shared_v18_v18_1_false_negatives_detected_by_v16": shared_fn_detected_by_v16,
            "shared_v18_v18_1_false_negatives_missed_by_all": shared_fn_missed_by_all,
        },
        "decision": decision,
        "malicious": malicious,
        "benign": benign,
    }

    json_path = out_dir / "v16-parent-strategy-diagnostic.json"
    md_path = out_dir / "v16-parent-strategy-diagnostic.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, md_path)

    print(json.dumps({"status": "pass", "json": str(json_path), "md": str(md_path), "totals": report["totals"]}, indent=2))


if __name__ == "__main__":
    main()
