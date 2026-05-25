from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Equi-Agent live prediction CSVs.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-csv", type=Path)
    parser.add_argument("--threshold-tol", type=float, default=1e-12)
    return parser.parse_args()


def require_runtime_libs():
    import pandas as pd
    from sklearn.metrics import confusion_matrix, roc_auc_score

    return pd, confusion_matrix, roc_auc_score


def safe_div(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def maybe_auc(roc_auc_score, y_true, y_prob) -> float | None:
    if len(set(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_prob))


def bool_series(series):
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def summarize_task(group, confusion_matrix, roc_auc_score) -> dict:
    tn, fp, fn, tp = confusion_matrix(group["y_true"], group["y_pred"], labels=[0, 1]).ravel()
    n = int(len(group))
    support_pos = int(group["y_true"].sum())
    support_neg = int(n - support_pos)
    pred_pos = int(group["y_pred"].sum())
    pred_neg = int(n - pred_pos)

    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    precision = safe_div(tp, tp + fp)
    f1 = None
    if precision is not None and recall is not None and precision + recall > 0:
        f1 = float(2 * precision * recall / (precision + recall))

    result = {
        "task": group["task"].iloc[0],
        "n": n,
        "support_neg": support_neg,
        "support_pos": support_pos,
        "pred_neg": pred_neg,
        "pred_pos": pred_pos,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": safe_div(tp + tn, n),
        "balanced_accuracy": None if recall is None or specificity is None else float((recall + specificity) / 2),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "fpr": safe_div(fp, fp + tn),
        "fnr": safe_div(fn, fn + tp),
        "auroc": maybe_auc(roc_auc_score, group["y_true"].tolist(), group["y_prob"].tolist()),
        "escalations": int(bool_series(group["escalate_to_human"]).sum()),
        "escalation_rate": float(bool_series(group["escalate_to_human"]).mean()),
        "mean_y_prob": float(group["y_prob"].mean()),
        "mean_positive_votes": float(group["positive_votes"].mean()) if "positive_votes" in group else None,
    }

    for threshold, count in group["applied_threshold"].value_counts(dropna=False).sort_index().items():
        result[f"threshold_{threshold:g}_n"] = int(count)

    if "calibration_action" in group:
        for action, count in group["calibration_action"].value_counts(dropna=False).sort_index().items():
            result[f"calibration_{action}_n"] = int(count)

    return result


def main() -> None:
    args = parse_args()
    pd, confusion_matrix, roc_auc_score = require_runtime_libs()

    predictions = pd.read_csv(args.predictions)
    required = {"task", "y_true", "y_prob", "y_pred", "applied_threshold", "escalate_to_human"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    predictions["threshold_y_pred"] = (
        predictions["y_prob"] + args.threshold_tol >= predictions["applied_threshold"]
    ).astype(int)
    inconsistent = predictions[predictions["threshold_y_pred"] != predictions["y_pred"]]
    accept_escalate = predictions[
        predictions["safety_decision"].eq("ACCEPT") & bool_series(predictions["escalate_to_human"])
    ] if "safety_decision" in predictions else predictions.iloc[0:0]

    rows = [
        summarize_task(group, confusion_matrix, roc_auc_score)
        for _, group in predictions.groupby("task", sort=True)
    ]
    summary = {
        "predictions": str(args.predictions),
        "n": int(len(predictions)),
        "tasks": sorted(predictions["task"].dropna().unique().tolist()),
        "inconsistent_applied": int(len(inconsistent)),
        "accept_escalate": int(len(accept_escalate)),
        "overall_escalations": int(bool_series(predictions["escalate_to_human"]).sum()),
        "overall_escalation_rate": float(bool_series(predictions["escalate_to_human"]).mean()),
        "task_metrics": rows,
    }

    table = pd.DataFrame(rows)
    display_cols = [
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
    display_cols = [col for col in display_cols if col in table.columns]

    print(json.dumps({k: v for k, v in summary.items() if k != "task_metrics"}, indent=2))
    print("\nTask metrics:")
    print(table[display_cols].to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    threshold_cols = ["task"] + [col for col in table.columns if col.startswith("threshold_")]
    calibration_cols = ["task"] + [col for col in table.columns if col.startswith("calibration_")]
    if len(threshold_cols) > 1:
        print("\nThreshold counts:")
        print(table[threshold_cols].fillna(0).to_string(index=False))
    if len(calibration_cols) > 1:
        print("\nCalibration action counts:")
        print(table[calibration_cols].fillna(0).to_string(index=False))

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summary, indent=2) + "\n")
    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out_csv, index=False)


if __name__ == "__main__":
    main()
