from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


TARGETS = [
    "md",
    "vfi",
    "td_pointwise",
    "md_fast",
    "md_fast_no_p_cut",
    "td_pointwise_no_p_cut",
]

EXPECTED_POSITIVES = {
    "md": 18,
    "vfi": 19,
    "td_pointwise": 18,
    "md_fast": 4,
    "md_fast_no_p_cut": 6,
    "td_pointwise_no_p_cut": 60,
}

METHODS = [
    ("rnflt_logreg", "RNFLT logistic", "RNFLT"),
    ("clinical_logreg", "Clinical/TDS logistic", "Clinical + TDS"),
    ("bscan_logreg", "B-scan logistic", "OCT B-scan"),
    ("rnflt_clinical_logreg", "RNFLT + clinical logistic", "RNFLT + clinical/TDS"),
    ("bscan_clinical_logreg", "B-scan + clinical logistic", "OCT B-scan + clinical/TDS"),
    ("all_logreg", "All-feature logistic", "RNFLT + OCT + clinical/TDS"),
    ("retfound_oct", "RETFound linear probe", "OCT B-scan"),
    (
        "gdp_native_rnflt_tds_multitask_efficientnet",
        "GDP-native multitask helper",
        "RNFLT + visual-field TDS",
    ),
    ("llm_gpt51", "GPT-5.1", "RNFLT + visual-field TDS"),
    ("llm_gpt54", "GPT-5.4", "RNFLT + visual-field TDS"),
    ("llm_gpt56_luna", "GPT-5.6-luna", "RNFLT + visual-field TDS"),
    ("llm_claude_haiku45", "Claude Haiku 4.5", "RNFLT + visual-field TDS"),
    (
        "equi_agent_multitarget",
        "Ours",
        "Native/foundation outputs + reliability arbitration",
    ),
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect the complete six-endpoint GDP result matrix.")
    parser.add_argument(
        "--predictions-root",
        type=Path,
        default=repo_root() / "equi-agent" / "outputs" / "predictions",
    )
    parser.add_argument(
        "--metrics-root",
        type=Path,
        default=repo_root() / "equi-agent" / "outputs" / "metrics",
    )
    parser.add_argument(
        "--llm-root",
        type=Path,
        default=repo_root() / "equi-agent" / "outputs" / "baselines" / "gdp_progression_llm_v1",
    )
    parser.add_argument(
        "--agent-root",
        type=Path,
        default=repo_root() / "equi-agent" / "outputs" / "equi_agent_gdp_progression_multitarget_live",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def prediction_path(args: argparse.Namespace, target: str, method: str) -> Path:
    if method.startswith("llm_"):
        slug = method.removeprefix("llm_")
        return args.llm_root / slug / f"predictions_{target}.csv"
    if method == "equi_agent_multitarget":
        return args.agent_root / f"predictions_{target}.csv"
    return args.predictions_root / f"gdp_progression_forecasting_{target}_{method}.csv"


def metric_paths(args: argparse.Namespace, target: str, method: str) -> tuple[Path, Path]:
    if method.startswith("llm_"):
        slug = method.removeprefix("llm_")
        directory = args.metrics_root / f"exp8_gdp_progression_forecasting_{target}_llm_{slug}"
        stem = f"predictions_{target}"
    elif method == "equi_agent_multitarget":
        directory = args.metrics_root / f"exp8_gdp_progression_forecasting_{target}_equi_agent_multitarget"
        stem = f"predictions_{target}"
    else:
        directory = args.metrics_root / (
            f"exp8_gdp_progression_forecasting_{target}_{method.replace('_logreg', '')}"
        )
        stem = f"gdp_progression_forecasting_{target}_{method}"
    return directory / f"{stem}_aggregate.csv", directory / f"{stem}_disparities.csv"


def run_summary_path(args: argparse.Namespace, method: str) -> Path | None:
    if method.startswith("llm_"):
        return args.llm_root / method.removeprefix("llm_") / "summary.json"
    if method == "equi_agent_multitarget":
        return args.agent_root / "summary.json"
    return None


def audit_method(args: argparse.Namespace, target: str, method: str) -> tuple[dict[str, Any], list[str]]:
    errors = []
    predictions = prediction_path(args, target, method)
    aggregate, disparities = metric_paths(args, target, method)
    summary = run_summary_path(args, method)
    for path in [predictions, aggregate, disparities]:
        if not path.is_file():
            errors.append(f"missing {path}")
    if summary is not None:
        if not summary.is_file():
            errors.append(f"missing {summary}")
        else:
            payload = json.loads(summary.read_text(encoding="utf-8"))
            if not payload.get("complete_locked_cohort"):
                errors.append(f"incomplete run summary {summary}")

    row: dict[str, Any] = {
        "target": target,
        "method_key": method,
        "status": "missing" if errors else "complete",
        "prediction_path": str(predictions),
        "aggregate_path": str(aggregate),
        "disparities_path": str(disparities),
    }
    if errors:
        return row, errors

    prediction_rows = read_csv(predictions)
    positives = sum(str(item.get("y_true", "")).strip() in {"1", "1.0"} for item in prediction_rows)
    if len(prediction_rows) != 200 or positives != EXPECTED_POSITIVES[target]:
        errors.append(
            f"cohort mismatch {predictions}: rows={len(prediction_rows)} positives={positives}"
        )

    aggregate_rows = read_csv(aggregate)
    if len(aggregate_rows) != 1:
        errors.append(f"expected one aggregate row in {aggregate}, found {len(aggregate_rows)}")
    else:
        metric = aggregate_rows[0]
        row.update(
            {
                "f1": finite_float(metric.get("f1")),
                "sensitivity": 1.0 - float(metric["fnr"]),
                "specificity": 1.0 - float(metric["fpr"]),
                "balanced_accuracy": finite_float(metric.get("balanced_accuracy")),
                "auroc": finite_float(metric.get("auroc")),
            }
        )
    disparity_rows = read_csv(disparities)
    worst_group_values = [
        value
        for value in (finite_float(item.get("worst_group_f1")) for item in disparity_rows)
        if value is not None
    ]
    row["worst_group_f1"] = min(worst_group_values) if worst_group_values else None
    row["status"] = "invalid" if errors else "complete"
    return row, errors


def metric_text(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def markdown_table(target: str, rows: list[dict[str, Any]]) -> str:
    lines = [
        f"## `{target}`",
        "",
        f"Locked test cohort: 200 cases; {EXPECTED_POSITIVES[target]} positives.",
        "",
        "| Method | Input | F1 | Worst-group F1 | Sensitivity | Specificity | Balanced accuracy |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    method_details = {key: (name, inputs) for key, name, inputs in METHODS}
    for row in rows:
        name, inputs = method_details[row["method_key"]]
        if row["status"] != "complete":
            lines.append(f"| {name} | {inputs} | MISSING | MISSING | MISSING | MISSING | MISSING |")
            continue
        lines.append(
            f"| {name} | {inputs} | {metric_text(row.get('f1'))} | "
            f"{metric_text(row.get('worst_group_f1'))} | {metric_text(row.get('sensitivity'))} | "
            f"{metric_text(row.get('specificity'))} | {metric_text(row.get('balanced_accuracy'))} |"
        )
    return "\n".join(lines)


def latex_table(target: str, rows: list[dict[str, Any]]) -> str:
    escaped_target = target.replace("_", r"\_")
    label_target = target.replace("_", "-")
    method_details = {key: (name, inputs) for key, name, inputs in METHODS}
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        (
            r"\caption{Harvard GDP glaucoma-progression performance for the "
            f"\\texttt{{{escaped_target}}} endpoint on the locked 200-case test cohort.}}"
        ),
        f"\\label{{tab:gdp-progression-{label_target}}}",
        r"\begin{tabular}{@{}llccccc@{}}",
        r"\toprule",
        "\\textbf{Method} & \\textbf{Input} & \\textbf{F1} & \\textbf{Worst-group F1} & "
        "\\textbf{Sensitivity} & \\textbf{Specificity} & \\textbf{Balanced Acc.} \\\\",
        r"\midrule",
    ]
    for row in rows:
        name, inputs = method_details[row["method_key"]]
        values = [
            metric_text(row.get(column)) if row["status"] == "complete" else "--"
            for column in ["f1", "worst_group_f1", "sensitivity", "specificity", "balanced_accuracy"]
        ]
        escaped_input = inputs.replace("+", r"\(+\)")
        prefix = r"\textbf{" + name + "}" if row["method_key"] == "equi_agent_multitarget" else name
        lines.append(f"{prefix} & {escaped_input} & " + " & ".join(values) + " \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    errors = []
    markdown = ["# Complete Harvard-GDP Progression Results", ""]
    latex = []
    for target in TARGETS:
        target_rows = []
        for method, _, _ in METHODS:
            row, method_errors = audit_method(args, target, method)
            rows.append(row)
            target_rows.append(row)
            errors.extend(method_errors)
        markdown.extend([markdown_table(target, target_rows), ""])
        latex.extend([latex_table(target, target_rows), ""])

    write_csv(args.out_dir / "gdp_progression_complete_results.csv", rows)
    (args.out_dir / "gdp_progression_complete_results.md").write_text(
        "\n".join(markdown).rstrip() + "\n", encoding="utf-8"
    )
    (args.out_dir / "gdp_progression_complete_tables.tex").write_text(
        "\n".join(latex).rstrip() + "\n", encoding="utf-8"
    )
    status = {
        "complete": not errors,
        "required_cells": len(TARGETS) * len(METHODS),
        "complete_cells": sum(row["status"] == "complete" for row in rows),
        "errors": errors,
        "outputs": {
            "csv": str(args.out_dir / "gdp_progression_complete_results.csv"),
            "markdown": str(args.out_dir / "gdp_progression_complete_results.md"),
            "latex": str(args.out_dir / "gdp_progression_complete_tables.tex"),
        },
    }
    (args.out_dir / "completion_status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(status, indent=2, sort_keys=True))
    if errors and not args.allow_incomplete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
