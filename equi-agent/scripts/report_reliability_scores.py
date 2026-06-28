from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_PRIORS_JSON = Path("equi-agent/outputs/metrics/validation_subgroup_priors.json")
DEFAULT_OUT = Path("equi-agent/outputs/metrics/reliability_scores_by_subgroup.csv")
RELIABILITY_WEIGHTS = {
    "auroc": 0.45,
    "f1": 0.35,
    "ece": -0.10,
    "fnr": -0.07,
    "fpr": -0.03,
}
DEFAULTS = {
    "auroc": 0.5,
    "f1": 0.0,
    "ece": 0.5,
    "fnr": 0.5,
    "fpr": 0.5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Report R_local, R_global, and shrunk reliability scores for each "
            "model/task/subgroup combo from validation subgroup priors JSON."
        )
    )
    parser.add_argument("--priors-json", type=Path, default=DEFAULT_PRIORS_JSON)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--task", help="Optional task filter, e.g. amd, dr, glaucoma.")
    parser.add_argument("--model", help="Optional model filter, e.g. retfound_oct.")
    parser.add_argument("--attribute", help="Optional attribute filter, e.g. race_x_sex_gender.")
    parser.add_argument("--subgroup", help="Optional subgroup filter, e.g. 'asian x male'.")
    parser.add_argument("--k", type=float, default=50.0, help="Empirical-Bayes shrinkage constant.")
    parser.add_argument("--low-reliability-threshold", type=float, default=0.35)
    parser.add_argument("--include-global", action="store_true", help="Also include GLOBAL/GLOBAL rows.")
    parser.add_argument("--top", type=int, default=20, help="Number of lowest-reliability rows to print.")
    return parser.parse_args()


def norm(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return text.lower()


def finite(value: Any, default: float = math.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(number) else number


def load_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict) and "models" in data:
        raise SystemExit(
            f"{path} is an EquityAgent calibration JSON. It only stores FPR/FNR and cannot "
            "compute R_local/R_global with AUROC, F1, and ECE. Use "
            "equi-agent/outputs/metrics/validation_subgroup_priors.json."
        )
    raise SystemExit(f"Unsupported priors JSON format: {path}")


def reliability_score(row: dict[str, Any]) -> float:
    return sum(RELIABILITY_WEIGHTS[field] * finite(row.get(field), DEFAULTS[field]) for field in RELIABILITY_WEIGHTS)


def row_matches(row: dict[str, Any], args: argparse.Namespace) -> bool:
    filters = {
        "task": args.task,
        "model_name": args.model,
        "attribute": args.attribute,
        "subgroup": args.subgroup,
    }
    for column, wanted in filters.items():
        if wanted is not None and norm(row.get(column)) != norm(wanted):
            return False
    return True


def fmt(value: Any) -> str:
    number = finite(value)
    if math.isnan(number):
        return ""
    return f"{number:.6f}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "dataset",
        "task",
        "model_name",
        "attribute",
        "subgroup",
        "n",
        "n_positive",
        "n_negative",
        "unstable",
        "lambda_g",
        "R_local",
        "R_global",
        "R_shrunk",
        "R_local_minus_global",
        "low_reliability",
        "auroc_local",
        "f1_local",
        "ece_local",
        "fnr_local",
        "fpr_local",
        "auroc_global",
        "f1_global",
        "ece_global",
        "fnr_global",
        "fpr_global",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows: list[dict[str, Any]], top: int) -> None:
    columns = ["task", "model_name", "attribute", "subgroup", "n", "lambda_g", "R_local", "R_global", "R_shrunk", "low_reliability"]
    shown = rows[:top]
    if not shown:
        print("No rows matched.")
        return
    widths = {
        col: max(len(col), *(len(fmt(row[col]) if col.startswith("R_") or col == "lambda_g" else str(row.get(col, ""))) for row in shown))
        for col in columns
    }
    print(" ".join(col.rjust(widths[col]) for col in columns))
    for row in shown:
        values = []
        for col in columns:
            value = fmt(row[col]) if col.startswith("R_") or col == "lambda_g" else str(row.get(col, ""))
            values.append(value.rjust(widths[col]))
        print(" ".join(values))


def main() -> None:
    args = parse_args()
    rows = load_rows(args.priors_json)
    global_lookup = {
        (norm(row.get("dataset")), norm(row.get("task")), norm(row.get("model_name"))): row
        for row in rows
        if norm(row.get("attribute")) == "global" and norm(row.get("subgroup")) == "global"
    }

    output_rows: list[dict[str, Any]] = []
    for row in rows:
        is_global = norm(row.get("attribute")) == "global" and norm(row.get("subgroup")) == "global"
        if is_global and not args.include_global:
            continue
        if not row_matches(row, args):
            continue

        global_row = global_lookup.get((norm(row.get("dataset")), norm(row.get("task")), norm(row.get("model_name"))))
        if global_row is None:
            continue

        n = finite(row.get("n"), 0.0)
        lambda_g = 0.0 if is_global else (n / (n + args.k) if n > 0 and args.k >= 0 else 0.0)
        r_local = reliability_score(row)
        r_global = reliability_score(global_row)
        r_shrunk = lambda_g * r_local + (1.0 - lambda_g) * r_global

        output_rows.append(
            {
                "dataset": row.get("dataset", ""),
                "task": row.get("task", ""),
                "model_name": row.get("model_name", ""),
                "attribute": row.get("attribute", ""),
                "subgroup": row.get("subgroup", ""),
                "n": row.get("n", ""),
                "n_positive": row.get("n_positive", ""),
                "n_negative": row.get("n_negative", ""),
                "unstable": row.get("unstable", ""),
                "lambda_g": lambda_g,
                "R_local": r_local,
                "R_global": r_global,
                "R_shrunk": r_shrunk,
                "R_local_minus_global": r_local - r_global,
                "low_reliability": r_shrunk < args.low_reliability_threshold,
                "auroc_local": finite(row.get("auroc"), DEFAULTS["auroc"]),
                "f1_local": finite(row.get("f1"), DEFAULTS["f1"]),
                "ece_local": finite(row.get("ece"), DEFAULTS["ece"]),
                "fnr_local": finite(row.get("fnr"), DEFAULTS["fnr"]),
                "fpr_local": finite(row.get("fpr"), DEFAULTS["fpr"]),
                "auroc_global": finite(global_row.get("auroc"), DEFAULTS["auroc"]),
                "f1_global": finite(global_row.get("f1"), DEFAULTS["f1"]),
                "ece_global": finite(global_row.get("ece"), DEFAULTS["ece"]),
                "fnr_global": finite(global_row.get("fnr"), DEFAULTS["fnr"]),
                "fpr_global": finite(global_row.get("fpr"), DEFAULTS["fpr"]),
            }
        )

    output_rows.sort(key=lambda item: (finite(item["R_shrunk"]), str(item["task"]), str(item["model_name"])))
    write_csv(args.out, output_rows)

    print(f"wrote={args.out}")
    print(f"rows={len(output_rows)}")
    print(f"formula=0.45*AUROC + 0.35*F1 - 0.10*ECE - 0.07*FNR - 0.03*FPR")
    print(f"shrinkage=lambda_g*R_local + (1-lambda_g)*R_global, lambda_g=n/(n+{args.k:g})")
    print(f"low_reliability_threshold={args.low_reliability_threshold:g}")
    print()
    print_table(output_rows, args.top)


if __name__ == "__main__":
    main()
