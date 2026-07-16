"""Calculate binary glaucoma metrics from a predictions CSV.

Example:
    python calculate_csv_metrics.py refuge_test_agentic_cfp_predictions.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


DEFAULT_PROBABILITY_COLUMNS = (
    "Probability_GL",
    "RETFound_Probability_GL",
    "probability",
    "Probability",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Print binary glaucoma metrics from a predictions CSV."
    )
    parser.add_argument("csv_path", type=Path, help="Predictions CSV to evaluate")
    parser.add_argument(
        "--truth-column", default="Ground_Truth", help="Ground-truth column name"
    )
    parser.add_argument(
        "--prediction-column", default="Pred_GL", help="Binary prediction column name"
    )
    parser.add_argument(
        "--probability-column",
        default=None,
        help="Positive-class probability column; auto-detected when omitted",
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=None,
        help="Optional path for a one-row summary metrics CSV",
    )
    return parser.parse_args()


def find_probability_column(frame, requested):
    if requested:
        if requested not in frame.columns:
            raise KeyError(f"Probability column not found: {requested}")
        return requested
    return next(
        (column for column in DEFAULT_PROBABILITY_COLUMNS if column in frame.columns),
        None,
    )


def calculate_metrics(frame, truth_column, prediction_column, probability_column=None):
    required = {truth_column, prediction_column}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(
            f"Missing required columns: {sorted(missing)}. "
            f"Available columns: {list(frame.columns)}"
        )

    working = frame.copy()
    working[truth_column] = pd.to_numeric(working[truth_column], errors="coerce")
    working[prediction_column] = pd.to_numeric(
        working[prediction_column], errors="coerce"
    )
    valid_mask = working[truth_column].isin([0, 1]) & working[prediction_column].isin(
        [0, 1]
    )
    valid = working.loc[valid_mask].copy()
    invalid_count = int((~valid_mask).sum())
    if valid.empty:
        raise ValueError("No rows contain valid binary ground truth and predictions")

    y_true = valid[truth_column].astype(int).to_numpy()
    y_pred = valid[prediction_column].astype(int).to_numpy()
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan

    report_text = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Normal", "Glaucoma"],
        digits=4,
        zero_division=0,
    )
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Normal", "Glaucoma"],
        output_dict=True,
        zero_division=0,
    )

    summary = {
        "total_rows": len(frame),
        "valid_rows": len(valid),
        "invalid_or_failed_rows": invalid_count,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_glaucoma": report["Glaucoma"]["precision"],
        "recall_sensitivity": report["Glaucoma"]["recall"],
        "specificity": specificity,
        "f1_glaucoma": report["Glaucoma"]["f1-score"],
        "support_normal": int(report["Normal"]["support"]),
        "support_glaucoma": int(report["Glaucoma"]["support"]),
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    if probability_column:
        probabilities = pd.to_numeric(
            valid[probability_column], errors="coerce"
        )
        probability_mask = probabilities.notna()
        probability_targets = y_true[probability_mask.to_numpy()]
        if probability_mask.sum() > 0 and len(np.unique(probability_targets)) == 2:
            summary["auroc"] = roc_auc_score(
                probability_targets,
                probabilities.loc[probability_mask].to_numpy(),
            )
        else:
            summary["auroc"] = np.nan

    return report_text, summary, np.array([[tn, fp], [fn, tp]])


def main():
    args = parse_args()
    frame = pd.read_csv(args.csv_path, nrows = 170)
    probability_column = find_probability_column(frame, args.probability_column)
    report, summary, matrix = calculate_metrics(
        frame,
        args.truth_column,
        args.prediction_column,
        probability_column,
    )

    print(f"Metrics for: {args.csv_path}")
    print(f"Ground truth: {args.truth_column} | Prediction: {args.prediction_column}")
    print(f"Probability: {probability_column or 'Not available'}")
    print("\nPer-class classification report:")
    print(report)
    print("Summary:")
    print(pd.Series(summary).to_string())
    print("\nConfusion matrix [[TN, FP], [FN, TP]]:")
    print(matrix)

    if args.metrics_csv:
        args.metrics_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([summary]).to_csv(args.metrics_csv, index=False)
        print(f"\nSummary metrics saved to: {args.metrics_csv}")


if __name__ == "__main__":
    main()
