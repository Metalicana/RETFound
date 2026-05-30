from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


SINGLE_ATTRIBUTES = ["race", "ethnicity", "sex_gender", "age_group"]
INTERSECTIONAL_ATTRIBUTES = ["race_x_age_group", "race_x_sex_gender", "sex_gender_x_age_group"]
METRIC_FIELDS = [
    "n",
    "support_pos",
    "support_neg",
    "accuracy",
    "f1",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "fpr",
    "fnr",
    "ece",
    "escalation_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build demographic subgroup performance tables from Equi-Agent prediction CSVs."
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out-prefix", type=Path, required=True)
    parser.add_argument("--tex", type=Path)
    parser.add_argument("--model-label", default="Equi-Agent")
    parser.add_argument("--caption", default="")
    parser.add_argument("--min-positive", type=int, default=5)
    parser.add_argument("--min-negative", type=int, default=5)
    parser.add_argument("--ece-bins", type=int, default=10)
    parser.add_argument(
        "--include-intersections",
        action="store_true",
        help="Also include race/age, race/sex, and sex/age subgroup rows in the LaTeX table.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(number) else number


def int_label(value: Any) -> int:
    return int(round(fnum(value)))


def bool_value(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def safe_div(num: float, den: float) -> float | None:
    return None if den == 0 else num / den


def value_or_missing(row: dict[str, str], key: str) -> str:
    value = str(row.get(key, "")).strip()
    return value if value else "missing"


def add_intersections(rows: list[dict[str, str]]) -> None:
    for row in rows:
        row["race_x_age_group"] = f"{value_or_missing(row, 'race')} x {value_or_missing(row, 'age_group')}"
        row["race_x_sex_gender"] = f"{value_or_missing(row, 'race')} x {value_or_missing(row, 'sex_gender')}"
        row["sex_gender_x_age_group"] = f"{value_or_missing(row, 'sex_gender')} x {value_or_missing(row, 'age_group')}"


def ece(rows: list[dict[str, str]], bins: int) -> float | None:
    if not rows:
        return None
    total = len(rows)
    error = 0.0
    for idx in range(bins):
        lo = idx / bins
        hi = (idx + 1) / bins
        if idx == bins - 1:
            bucket = [row for row in rows if lo <= fnum(row["y_prob"]) <= hi]
        else:
            bucket = [row for row in rows if lo <= fnum(row["y_prob"]) < hi]
        if not bucket:
            continue
        conf = sum(fnum(row["y_prob"]) for row in bucket) / len(bucket)
        acc = sum(int_label(row["y_true"]) == int_label(row["y_pred"]) for row in bucket) / len(bucket)
        error += (len(bucket) / total) * abs(acc - conf)
    return error


def metrics(rows: list[dict[str, str]], ece_bins: int, min_positive: int, min_negative: int) -> dict[str, Any]:
    tn = fp = fn = tp = 0
    for row in rows:
        y_true = int_label(row["y_true"])
        y_pred = int_label(row["y_pred"])
        if y_true == 0 and y_pred == 0:
            tn += 1
        elif y_true == 0 and y_pred == 1:
            fp += 1
        elif y_true == 1 and y_pred == 0:
            fn += 1
        elif y_true == 1 and y_pred == 1:
            tp += 1
    n = len(rows)
    support_pos = tp + fn
    support_neg = tn + fp
    sensitivity = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    precision = safe_div(tp, tp + fp)
    f1 = None
    if precision is not None and sensitivity is not None and precision + sensitivity > 0:
        f1 = 2 * precision * sensitivity / (precision + sensitivity)
    return {
        "n": n,
        "support_pos": support_pos,
        "support_neg": support_neg,
        "accuracy": safe_div(tp + tn, n),
        "f1": f1,
        "balanced_accuracy": None if sensitivity is None or specificity is None else (sensitivity + specificity) / 2,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "fpr": safe_div(fp, fp + tn),
        "fnr": safe_div(fn, fn + tp),
        "ece": ece(rows, ece_bins),
        "escalation_rate": safe_div(sum(bool_value(row.get("escalate_to_human")) for row in rows), n),
        "unstable": support_pos < min_positive or support_neg < min_negative,
    }


def fmt_csv(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def fmt_tex(value: Any) -> str:
    if value is None:
        return "--"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def esc(value: Any) -> str:
    text = str(value)
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def build_task_rows(
    rows: list[dict[str, str]],
    attributes: list[str],
    ece_bins: int,
    min_positive: int,
    min_negative: int,
) -> list[dict[str, Any]]:
    output = []
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        task = row.get("task", "all")
        for attr in attributes:
            grouped[(task, attr, value_or_missing(row, attr))].append(row)
    for (task, attr, subgroup), group_rows in sorted(grouped.items()):
        output.append(
            {
                "task": task,
                "attribute": attr,
                "subgroup": subgroup,
                **metrics(group_rows, ece_bins, min_positive, min_negative),
            }
        )
    return output


def mean(values: list[Any]) -> float | None:
    numbers = [value for value in values if isinstance(value, (float, int)) and not math.isnan(float(value))]
    return None if not numbers else sum(float(value) for value in numbers) / len(numbers)


def build_macro_rows(task_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in task_rows:
        grouped[(row["attribute"], row["subgroup"])].append(row)
    output = []
    for (attr, subgroup), rows in sorted(grouped.items()):
        macro = {
            "attribute": attr,
            "subgroup": subgroup,
            "tasks": ";".join(sorted(str(row["task"]) for row in rows)),
            "n": sum(int(row["n"]) for row in rows),
            "support_pos": sum(int(row["support_pos"]) for row in rows),
            "support_neg": sum(int(row["support_neg"]) for row in rows),
            "unstable": any(bool(row["unstable"]) for row in rows),
        }
        for field in [
            "accuracy",
            "f1",
            "balanced_accuracy",
            "sensitivity",
            "specificity",
            "fpr",
            "fnr",
            "ece",
            "escalation_rate",
        ]:
            macro[field] = mean([row[field] for row in rows])
        output.append(macro)
    return output


def write_tex(path: Path, rows: list[dict[str, Any]], caption: str, model_label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    caption = caption or (
        f"Demographic-stratified {model_label} performance. Metrics are macro-averaged across available tasks. "
        "Accuracy is reported with F1, balanced accuracy, error rates, calibration, and escalation rate."
    )
    lines = [
        r"\begin{longtable}{@{}llrrrrrrrrr@{}}",
        rf"\caption{{{esc(caption)}}}\label{{tab:equi_agent_demographic_detail}}\\",
        r"\toprule",
        r"Attribute & Group & $n$ & Acc. & F1 & Bal. acc. & FPR & FNR & ECE & Esc. & Unstable \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Attribute & Group & $n$ & Acc. & F1 & Bal. acc. & FPR & FNR & ECE & Esc. & Unstable \\",
        r"\midrule",
        r"\endhead",
    ]
    for row in rows:
        unstable = "Yes" if row["unstable"] else "No"
        lines.append(
            f"{esc(row['attribute'])} & {esc(row['subgroup'])} & {row['n']} & "
            f"{fmt_tex(row['accuracy'])} & {fmt_tex(row['f1'])} & {fmt_tex(row['balanced_accuracy'])} & "
            f"{fmt_tex(row['fpr'])} & {fmt_tex(row['fnr'])} & {fmt_tex(row['ece'])} & "
            f"{fmt_tex(row['escalation_rate'])} & {unstable} \\\\"
        )
    lines.extend([r"\botrule", r"\end{longtable}"])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    rows = read_csv(args.predictions)
    if not rows:
        raise SystemExit("No prediction rows found.")
    required = {"y_true", "y_pred", "y_prob"}
    missing = sorted(required - set(rows[0]))
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    add_intersections(rows)
    attributes = SINGLE_ATTRIBUTES + INTERSECTIONAL_ATTRIBUTES
    task_rows = build_task_rows(rows, attributes, args.ece_bins, args.min_positive, args.min_negative)
    macro_rows = build_macro_rows(task_rows)

    macro_fields = [
        "attribute",
        "subgroup",
        "tasks",
        *METRIC_FIELDS,
        "unstable",
    ]
    task_fields = [
        "task",
        "attribute",
        "subgroup",
        *METRIC_FIELDS,
        "unstable",
    ]
    macro_path = args.out_prefix.with_name(args.out_prefix.name + "_macro.csv")
    task_path = args.out_prefix.with_name(args.out_prefix.name + "_by_task.csv")
    write_csv(macro_path, [{key: fmt_csv(row.get(key)) for key in macro_fields} for row in macro_rows], macro_fields)
    write_csv(task_path, [{key: fmt_csv(row.get(key)) for key in task_fields} for row in task_rows], task_fields)

    if args.tex:
        tex_rows = macro_rows if args.include_intersections else [row for row in macro_rows if row["attribute"] in SINGLE_ATTRIBUTES]
        write_tex(args.tex, tex_rows, args.caption, args.model_label)

    print(f"wrote={macro_path}")
    print(f"wrote={task_path}")
    if args.tex:
        print(f"wrote={args.tex}")


if __name__ == "__main__":
    main()
