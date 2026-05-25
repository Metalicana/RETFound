from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


DISPLAY_COLS = [
    "task",
    "n",
    "support_neg",
    "support_pos",
    "pred_neg",
    "pred_pos",
    "tn",
    "fp",
    "fn",
    "tp",
    "accuracy",
    "precision",
    "recall",
    "specificity",
    "f1",
    "auroc",
    "escalation_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Equi-Agent live prediction CSVs.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-csv", type=Path)
    parser.add_argument("--threshold-tol", type=float, default=1e-12)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fnum(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number):
        return default
    return number


def int_label(value: object) -> int:
    return int(round(fnum(value)))


def bool_value(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def safe_div(num: float, den: float) -> float | None:
    return None if den == 0 else num / den


def auroc(rows: list[dict[str, str]]) -> float | None:
    pairs = [(fnum(row["y_prob"]), int_label(row["y_true"])) for row in rows]
    pos = sum(label == 1 for _, label in pairs)
    neg = sum(label == 0 for _, label in pairs)
    if pos == 0 or neg == 0:
        return None
    pairs.sort(key=lambda item: item[0])
    rank_sum_pos = 0.0
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        rank_sum_pos += sum(avg_rank for _, label in pairs[i:j] if label == 1)
        i = j
    return (rank_sum_pos - pos * (pos + 1) / 2.0) / (pos * neg)


def confusion(rows: list[dict[str, str]]) -> tuple[int, int, int, int]:
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
    return tn, fp, fn, tp


def summarize_task(task: str, rows: list[dict[str, str]]) -> dict[str, object]:
    tn, fp, fn, tp = confusion(rows)
    n = len(rows)
    support_pos = sum(int_label(row["y_true"]) for row in rows)
    support_neg = n - support_pos
    pred_pos = sum(int_label(row["y_pred"]) for row in rows)
    pred_neg = n - pred_pos
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    precision = safe_div(tp, tp + fp)
    f1 = None
    if precision is not None and recall is not None and precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)

    result: dict[str, object] = {
        "task": task,
        "n": n,
        "support_neg": support_neg,
        "support_pos": support_pos,
        "pred_neg": pred_neg,
        "pred_pos": pred_pos,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": safe_div(tp + tn, n),
        "balanced_accuracy": None if recall is None or specificity is None else (recall + specificity) / 2,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "fpr": safe_div(fp, fp + tn),
        "fnr": safe_div(fn, fn + tp),
        "auroc": auroc(rows),
        "escalations": sum(bool_value(row.get("escalate_to_human")) for row in rows),
        "escalation_rate": safe_div(sum(bool_value(row.get("escalate_to_human")) for row in rows), n),
        "mean_y_prob": safe_div(sum(fnum(row["y_prob"]) for row in rows), n),
        "mean_positive_votes": safe_div(sum(fnum(row.get("positive_votes")) for row in rows), n),
    }

    threshold_counts: dict[str, int] = defaultdict(int)
    action_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        threshold_counts[f"{fnum(row.get('applied_threshold')):g}"] += 1
        action_counts[str(row.get("calibration_action", ""))] += 1
    for threshold, count in sorted(threshold_counts.items()):
        result[f"threshold_{threshold}_n"] = count
    for action, count in sorted(action_counts.items()):
        result[f"calibration_{action}_n"] = count
    return result


def fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def print_table(rows: list[dict[str, object]], cols: list[str]) -> None:
    cols = [col for col in cols if any(col in row for row in rows)]
    widths = {
        col: max(len(col), *(len(fmt(row.get(col))) for row in rows))
        for col in cols
    }
    print(" ".join(col.rjust(widths[col]) for col in cols))
    for row in rows:
        print(" ".join(fmt(row.get(col)).rjust(widths[col]) for col in cols))


def main() -> None:
    args = parse_args()
    predictions = read_csv(args.predictions)
    required = {"task", "y_true", "y_prob", "y_pred", "applied_threshold", "escalate_to_human"}
    missing = sorted(required - set(predictions[0] if predictions else {}))
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    inconsistent = [
        row
        for row in predictions
        if int(fnum(row["y_prob"]) + args.threshold_tol >= fnum(row["applied_threshold"])) != int_label(row["y_pred"])
    ]
    accept_escalate = [
        row
        for row in predictions
        if row.get("safety_decision") == "ACCEPT" and bool_value(row.get("escalate_to_human"))
    ]

    by_task: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in predictions:
        by_task[row["task"]].append(row)
    rows = [summarize_task(task, by_task[task]) for task in sorted(by_task)]
    summary = {
        "predictions": str(args.predictions),
        "n": len(predictions),
        "tasks": sorted(by_task),
        "inconsistent_applied": len(inconsistent),
        "accept_escalate": len(accept_escalate),
        "overall_escalations": sum(bool_value(row.get("escalate_to_human")) for row in predictions),
        "overall_escalation_rate": safe_div(
            sum(bool_value(row.get("escalate_to_human")) for row in predictions),
            len(predictions),
        ),
        "task_metrics": rows,
    }

    print(json.dumps({k: v for k, v in summary.items() if k != "task_metrics"}, indent=2))
    print("\nTask metrics:")
    print_table(rows, DISPLAY_COLS)

    threshold_cols = ["task"] + sorted({key for row in rows for key in row if key.startswith("threshold_")})
    calibration_cols = ["task"] + sorted({key for row in rows for key in row if key.startswith("calibration_")})
    if len(threshold_cols) > 1:
        print("\nThreshold counts:")
        print_table(rows, threshold_cols)
    if len(calibration_cols) > 1:
        print("\nCalibration action counts:")
        print_table(rows, calibration_cols)

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summary, indent=2) + "\n")
    if args.out_csv:
        write_csv(args.out_csv, rows)


if __name__ == "__main__":
    main()
