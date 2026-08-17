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

MODELS = {
    "gpt51": "GPT-5.1",
    "gpt56_luna": "GPT-5.6-luna",
    "claude_haiku45": "Claude Haiku 4.5",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def metric_text(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def latex_metric(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def metric_paths(metrics_root: Path, target: str, slug: str) -> tuple[Path, Path]:
    directory = metrics_root / f"exp8_gdp_progression_forecasting_{target}_llm_{slug}"
    stem = f"predictions_{target}"
    return directory / f"{stem}_aggregate.csv", directory / f"{stem}_disparities.csv"


def collect_row(
    run_root: Path,
    metrics_root: Path,
    target: str,
    slug: str,
) -> dict[str, Any]:
    predictions_path = run_root / slug / f"predictions_{target}.csv"
    summary_path = run_root / slug / "summary.json"
    aggregate_path, disparities_path = metric_paths(metrics_root, target, slug)
    required = [predictions_path, summary_path, aggregate_path, disparities_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing completed {target}/{slug} artifacts: {missing}")

    predictions = read_csv(predictions_path)
    positives = sum(str(row.get("y_true", "")).strip() in {"1", "1.0"} for row in predictions)
    if len(predictions) != 200 or positives != EXPECTED_POSITIVES[target]:
        raise ValueError(
            f"Locked cohort mismatch for {target}/{slug}: "
            f"rows={len(predictions)}, positives={positives}"
        )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not summary.get("complete_locked_cohort"):
        raise ValueError(f"Incomplete LLM run: {summary_path}")

    aggregate_rows = read_csv(aggregate_path)
    if len(aggregate_rows) != 1:
        raise ValueError(f"Expected one aggregate row in {aggregate_path}, found {len(aggregate_rows)}")
    aggregate = aggregate_rows[0]
    if int(float(aggregate["n"])) != 200:
        raise ValueError(f"Aggregate n is not 200: {aggregate_path}")

    disparities = read_csv(disparities_path)
    worst_values = [
        value
        for value in (finite_float(row.get("worst_group_f1")) for row in disparities)
        if value is not None
    ]
    return {
        "target": target,
        "positive_test_cases": positives,
        "model": MODELS[slug],
        "model_slug": slug,
        "f1": finite_float(aggregate.get("f1")),
        "worst_group_f1": min(worst_values) if worst_values else None,
        "sensitivity": 1.0 - float(aggregate["fnr"]),
        "specificity": 1.0 - float(aggregate["fpr"]),
        "balanced_accuracy": finite_float(aggregate.get("balanced_accuracy")),
        "auroc": finite_float(aggregate.get("auroc")),
        "predictions_path": str(predictions_path),
        "aggregate_path": str(aggregate_path),
        "disparities_path": str(disparities_path),
    }


def markdown_table(target: str, rows: list[dict[str, Any]]) -> str:
    output = [
        f"## `{target}`",
        "",
        f"Locked test cohort: 200 cases, {EXPECTED_POSITIVES[target]} progression-positive cases.",
        "",
        "| Model | F1 | Worst-group F1 | Sensitivity | Specificity | Balanced accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        output.append(
            f"| {row['model']} | {metric_text(row['f1'])} | "
            f"{metric_text(row['worst_group_f1'])} | {metric_text(row['sensitivity'])} | "
            f"{metric_text(row['specificity'])} | {metric_text(row['balanced_accuracy'])} |"
        )
    return "\n".join(output)


def latex_table(target: str, rows: list[dict[str, Any]]) -> str:
    escaped = target.replace("_", r"\_")
    label = target.replace("_", "-")
    output = [
        r"\begin{table*}[t]",
        r"\centering",
        (
            r"\caption{Standalone LLM glaucoma-progression baselines on Harvard GDP for the "
            rf"\texttt{{{escaped}}} endpoint. Worst-group F1 is computed over available prespecified "
            r"demographic subgroups.}"
        ),
        f"\\label{{tab:gdp-progression-llm-{label}}}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        "\\textbf{Method} & \\textbf{F1} & \\textbf{Worst-group F1} & "
        "\\textbf{Sensitivity} & \\textbf{Specificity} & \\textbf{Balanced Acc.} \\\\",
        r"\midrule",
    ]
    for row in rows:
        output.append(
            f"{row['model']} & {latex_metric(row['f1'])} & "
            f"{latex_metric(row['worst_group_f1'])} & {latex_metric(row['sensitivity'])} & "
            f"{latex_metric(row['specificity'])} & {latex_metric(row['balanced_accuracy'])} \\\\"
        )
    output.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(output)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect six-endpoint GDP progression LLM results.")
    parser.add_argument(
        "--run-root",
        type=Path,
        default=repo_root() / "equi-agent" / "outputs" / "baselines" / "gdp_progression_llm_v1",
    )
    parser.add_argument(
        "--metrics-root",
        type=Path,
        default=repo_root() / "equi-agent" / "outputs" / "metrics",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--targets", nargs="+", choices=TARGETS, default=TARGETS)
    parser.add_argument(
        "--model-slugs",
        nargs="+",
        choices=list(MODELS),
        default=list(MODELS),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    markdown_sections = ["# Harvard-GDP Progression LLM Baselines", ""]
    latex_sections = []
    for target in args.targets:
        target_rows = [
            collect_row(args.run_root, args.metrics_root, target, slug)
            for slug in args.model_slugs
        ]
        all_rows.extend(target_rows)
        markdown_sections.extend([markdown_table(target, target_rows), ""])
        latex_sections.extend([latex_table(target, target_rows), ""])

    write_csv(args.out_dir / "gdp_progression_llm_results.csv", all_rows)
    (args.out_dir / "gdp_progression_llm_results.md").write_text(
        "\n".join(markdown_sections).rstrip() + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "gdp_progression_llm_tables.tex").write_text(
        "\n".join(latex_sections).rstrip() + "\n",
        encoding="utf-8",
    )
    print("\n".join(markdown_sections).rstrip())
    print(f"\nwrote={args.out_dir}")


if __name__ == "__main__":
    main()
