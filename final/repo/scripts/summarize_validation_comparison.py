from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_CORPORA = {
    "benign_prod_dev": {
        "metric": "benign_fp_rate",
        "op": "lt",
        "target": 0.01,
        "label": "benign_prod_dev FPR < 1%",
        "format": "percent",
    },
    "v13_benign_windows": {
        "metric": "benign_fp_rate",
        "op": "lt",
        "target": 0.01,
        "label": "v13_benign_windows FPR < 1%",
        "format": "percent",
    },
    "malicious_dev": {
        "metric": "recall",
        "op": "gte",
        "target": 0.99,
        "label": "malicious_dev recall >= 99%",
        "format": "percent",
    },
    "v13_critical_ru": {
        "metric": "recall",
        "op": "gte",
        "target": 0.99,
        "label": "v13_critical_ru recall >= 99%",
        "format": "percent",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize V10-V14 diagnostic validation gates.")
    parser.add_argument("--comparison-json", required=True, help="comparison-summary.json from compare_v10_v13_validation_suite.py")
    parser.add_argument("--model", default="v14")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def format_metric(value: float | None) -> str:
    if value is None:
        return "missing"
    return f"{value * 100:.2f}%"


def gate_passes(value: float | None, spec: dict[str, Any]) -> bool:
    if value is None:
        return False
    if spec["op"] == "lt":
        return value < spec["target"]
    if spec["op"] == "gte":
        return value >= spec["target"]
    raise ValueError(f"Unsupported gate op: {spec['op']}")


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def summarize(rows: list[dict[str, Any]], model: str) -> dict[str, Any]:
    model_rows = [row for row in rows if row.get("model") == model]
    thresholds = sorted({float(row["threshold"]) for row in model_rows})
    by_threshold: dict[str, dict[str, Any]] = {}
    passing_thresholds: list[str] = []

    for threshold in thresholds:
        threshold_key = f"{threshold:g}"
        corpus_rows = {
            row["corpus"]: row
            for row in model_rows
            if float(row["threshold"]) == threshold
        }
        checks = {}
        passes_all = True
        for corpus, spec in REQUIRED_CORPORA.items():
            row = corpus_rows.get(corpus)
            value = as_float(row.get(spec["metric"])) if row else None
            passed = gate_passes(value, spec)
            checks[corpus] = {
                "label": spec["label"],
                "metric": spec["metric"],
                "value": value,
                "formatted_value": format_metric(value),
                "target": spec["target"],
                "passed": passed,
            }
            passes_all = passes_all and passed
        by_threshold[threshold_key] = {
            "threshold": threshold,
            "passes_all_required_gates": passes_all,
            "checks": checks,
        }
        if passes_all:
            passing_thresholds.append(threshold_key)

    return {
        "model": model,
        "required_gates": {name: spec["label"] for name, spec in REQUIRED_CORPORA.items()},
        "thresholds": by_threshold,
        "passing_thresholds": passing_thresholds,
        "recommendation": (
            f"Candidate thresholds satisfying all gates: {', '.join(passing_thresholds)}"
            if passing_thresholds
            else "No evaluated threshold satisfies all required V14 diagnostic gates."
        ),
    }


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        f"# Diagnostic Gate Summary: {summary['model']}",
        "",
        summary["recommendation"],
        "",
        "| Threshold | benign_prod_dev FPR | v13_benign_windows FPR | malicious_dev recall | v13_critical_ru recall | Pass |",
        "|---:|---:|---:|---:|---:|:---:|",
    ]
    for threshold, threshold_summary in summary["thresholds"].items():
        checks = threshold_summary["checks"]
        lines.append(
            "| "
            + " | ".join(
                [
                    threshold,
                    checks["benign_prod_dev"]["formatted_value"],
                    checks["v13_benign_windows"]["formatted_value"],
                    checks["malicious_dev"]["formatted_value"],
                    checks["v13_critical_ru"]["formatted_value"],
                    "yes" if threshold_summary["passes_all_required_gates"] else "no",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Required gates:",
            "",
            "- `benign_prod_dev` false-positive rate < 1%",
            "- `v13_benign_windows` false-positive rate < 1%",
            "- `malicious_dev` recall >= 99%",
            "- `v13_critical_ru` recall >= 99%",
            "",
            "This is a diagnostic summary over previously used or inspected corpora. It is not a blind acceptance report.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    comparison_path = Path(args.comparison_json)
    summary = summarize(load_rows(comparison_path), args.model)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(summary, output_md)
    print(json.dumps({"output_json": str(output_json), "output_md": str(output_md), "recommendation": summary["recommendation"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
