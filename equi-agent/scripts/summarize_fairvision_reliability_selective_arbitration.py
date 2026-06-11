from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_OUT_DIR = Path("equi-agent/outputs/fairvision_reliability_selective_arbitration")
METHOD_ORDER = [
    "best_single_global_reliability",
    "static_mean_probability",
    "global_reliability_weighted",
    "shrunk_subgroup_reliability_weighted",
    "shrunk_plus_disagreement_escalation",
    "full_reliability_conformal_escalation",
]
ABLATION_COLUMNS = [
    "method",
    "forced_n",
    "forced_f1",
    "forced_balanced_accuracy",
    "forced_sensitivity",
    "forced_specificity",
    "forced_ece",
    "coverage",
    "escalation_rate",
    "accepted_f1",
    "accepted_sensitivity",
    "accepted_specificity",
    "accepted_error_rate",
    "worst_group_f1",
    "worst_group_accepted_f1",
]
RISK_COLUMNS = [
    "task",
    "coverage",
    "escalation_rate",
    "f1",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "ece",
    "worst_group_accepted_f1",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print slide-ready summaries from FairVision reliability-selective arbitration outputs."
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--task", default="overall", help="Use overall or a disease task: amd, dr, glaucoma.")
    parser.add_argument("--top-risk-rows", type=int, default=9)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def fnum(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if not math.isnan(number) else math.nan


def fmt(value: Any) -> str:
    number = fnum(value)
    if math.isnan(number):
        return "" if value in {None, ""} else str(value)
    return f"{number:.3f}"


def method_rank(method: str) -> int:
    try:
        return METHOD_ORDER.index(method)
    except ValueError:
        return len(METHOD_ORDER)


def print_table(title: str, rows: list[dict[str, Any]], columns: list[str]) -> None:
    if not rows:
        print(f"\n{title}: no rows")
        return
    widths = {
        col: max(len(col), *(len(fmt(row.get(col, ""))) for row in rows))
        for col in columns
    }
    print(f"\n{title}")
    print(" ".join(col.rjust(widths[col]) for col in columns))
    for row in rows:
        print(" ".join(fmt(row.get(col, "")).rjust(widths[col]) for col in columns))


def main() -> None:
    args = parse_args()
    summary_path = args.out_dir / "selective_arbitration_summary.json"
    ablation_path = args.out_dir / "selective_arbitration_ablation_metrics.csv"
    risk_path = args.out_dir / "risk_coverage_curve.csv"

    if not ablation_path.exists():
        raise SystemExit(f"Missing ablation metrics: {ablation_path}")

    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    print("FairVision Reliability-Selective Arbitration")
    if summary:
        print(f"test_common_cases={summary.get('test_common_cases')}")
        print(f"validation_common_cases={summary.get('validation_common_cases')}")
        counterfactuals = summary.get("counterfactuals", {})
        if counterfactuals.get("run"):
            print(
                "counterfactuals="
                f"rows {counterfactuals.get('rows')}, "
                f"label_flip_rate {fmt(counterfactuals.get('label_flip_rate'))}, "
                f"escalation_flip_rate {fmt(counterfactuals.get('escalation_flip_rate'))}"
            )

    ablation_rows = [
        row
        for row in read_csv(ablation_path)
        if str(row.get("task", "")).lower() == args.task.lower()
    ]
    ablation_rows = sorted(ablation_rows, key=lambda row: method_rank(row.get("method", "")))
    print_table(f"Ablation Metrics ({args.task})", ablation_rows, ABLATION_COLUMNS)

    if risk_path.exists():
        all_risk_rows = read_csv(risk_path)
        if all_risk_rows and "task" in all_risk_rows[0]:
            risk_rows = [
                row
                for row in all_risk_rows
                if str(row.get("task", "")).lower() == args.task.lower()
            ]
        else:
            risk_rows = all_risk_rows
        risk_rows = risk_rows[: args.top_risk_rows]
        print_table("Risk-Coverage Curve", risk_rows, RISK_COLUMNS)


if __name__ == "__main__":
    main()
