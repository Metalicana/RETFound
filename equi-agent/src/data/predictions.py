from __future__ import annotations

from pathlib import Path

import pandas as pd


STANDARD_PREDICTION_COLUMNS = [
    "patient_id",
    "eye_id",
    "visit_id",
    "image_id",
    "dataset",
    "task",
    "model_name",
    "y_true",
    "y_prob",
    "y_pred",
    "split",
    "race",
    "ethnicity",
    "sex_gender",
    "age",
    "age_group",
    "metadata_missing_flag",
]

LONGITUDINAL_COLUMNS = [
    "sequence_id",
    "visit_index",
    "visit_date",
    "time_since_baseline",
    "num_visits",
    "progression_label",
]


def read_prediction_file(path: str | Path, longitudinal: bool = False) -> pd.DataFrame:
    predictions = pd.read_csv(path)
    validate_prediction_schema(predictions, longitudinal=longitudinal)
    return predictions


def validate_prediction_schema(df: pd.DataFrame, longitudinal: bool = False) -> None:
    required = set(STANDARD_PREDICTION_COLUMNS)
    if longitudinal:
        required.update(LONGITUDINAL_COLUMNS)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Prediction file is missing required columns: {missing}")

    if not df["y_prob"].between(0, 1).all():
        bad = int((~df["y_prob"].between(0, 1)).sum())
        raise ValueError(f"y_prob must be in [0, 1]. Bad rows: {bad}")

    y_pred_values = set(df["y_pred"].dropna().unique())
    if not y_pred_values <= {0, 1, 0.0, 1.0, False, True}:
        raise ValueError(f"y_pred must be binary. Found values: {sorted(y_pred_values)}")

    y_true_values = set(df["y_true"].dropna().unique())
    if not y_true_values <= {0, 1, 0.0, 1.0, False, True}:
        raise ValueError(f"y_true must be binary for these metric utilities. Found: {sorted(y_true_values)}")

    duplicate_keys = ["patient_id", "eye_id", "visit_id", "image_id", "dataset", "task", "model_name", "split"]
    duplicate_count = int(df.duplicated(duplicate_keys).sum())
    if duplicate_count:
        raise ValueError(f"Prediction file has duplicate prediction keys: {duplicate_count}")

