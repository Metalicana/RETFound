from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


SAMPLE_KEY_COLUMNS = ["patient_id", "eye_id", "visit_id", "image_id", "dataset", "task", "split"]


def mean_probability_ensemble(predictions: pd.DataFrame, method_name: str = "mean_probability") -> pd.DataFrame:
    base = _base_output_frame(predictions)
    probs = predictions.groupby(SAMPLE_KEY_COLUMNS, dropna=False)["y_prob"].mean().reset_index(name="y_prob")
    output = base.merge(probs, on=SAMPLE_KEY_COLUMNS, how="inner")
    output["model_name"] = method_name
    output["y_pred"] = (output["y_prob"] >= 0.5).astype(int)
    return output


def majority_vote_ensemble(predictions: pd.DataFrame, method_name: str = "majority_vote") -> pd.DataFrame:
    base = _base_output_frame(predictions)
    votes = predictions.groupby(SAMPLE_KEY_COLUMNS, dropna=False)["y_pred"].mean().reset_index(name="vote_rate")
    output = base.merge(votes, on=SAMPLE_KEY_COLUMNS, how="inner")
    output["y_prob"] = output["vote_rate"]
    output["y_pred"] = (output["vote_rate"] >= 0.5).astype(int)
    output["model_name"] = method_name
    return output.drop(columns=["vote_rate"])


def confidence_weighted_ensemble(
    predictions: pd.DataFrame,
    confidence_col: str | None = None,
    method_name: str = "confidence_weighted",
) -> pd.DataFrame:
    work = predictions.copy()
    if confidence_col and confidence_col in work.columns:
        work["_confidence"] = work[confidence_col].clip(lower=0)
    else:
        work["_confidence"] = (work["y_prob"] - 0.5).abs() * 2
    work["_weighted_prob"] = work["y_prob"] * work["_confidence"]
    grouped = work.groupby(SAMPLE_KEY_COLUMNS, dropna=False)
    probs = (grouped["_weighted_prob"].sum() / grouped["_confidence"].sum().replace(0, np.nan)).reset_index(name="y_prob")
    fallback = mean_probability_ensemble(predictions)[SAMPLE_KEY_COLUMNS + ["y_prob"]]
    probs = probs.merge(fallback, on=SAMPLE_KEY_COLUMNS, how="left", suffixes=("", "_fallback"))
    probs["y_prob"] = probs["y_prob"].fillna(probs["y_prob_fallback"])
    probs = probs.drop(columns=["y_prob_fallback"])
    output = _base_output_frame(predictions).merge(probs, on=SAMPLE_KEY_COLUMNS, how="inner")
    output["model_name"] = method_name
    output["y_pred"] = (output["y_prob"] >= 0.5).astype(int)
    return output


def fit_logistic_stacking(validation_predictions: pd.DataFrame) -> LogisticRegression:
    matrix, y = _prediction_matrix(validation_predictions)
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(matrix, y)
    return model


def predict_logistic_stacking(
    predictions: pd.DataFrame,
    stacker: LogisticRegression,
    method_name: str = "logistic_stacking",
) -> pd.DataFrame:
    matrix, _ = _prediction_matrix(predictions)
    probs = stacker.predict_proba(matrix)[:, 1]
    base = _base_output_frame(predictions).sort_values(SAMPLE_KEY_COLUMNS).reset_index(drop=True)
    base["model_name"] = method_name
    base["y_prob"] = probs
    base["y_pred"] = (base["y_prob"] >= 0.5).astype(int)
    return base


def _base_output_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    metadata_cols = [
        "y_true",
        "race",
        "ethnicity",
        "sex_gender",
        "age",
        "age_group",
        "metadata_missing_flag",
    ]
    return predictions[SAMPLE_KEY_COLUMNS + metadata_cols].drop_duplicates(SAMPLE_KEY_COLUMNS)


def _prediction_matrix(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    pivot = predictions.pivot_table(
        index=SAMPLE_KEY_COLUMNS,
        columns="model_name",
        values="y_prob",
        aggfunc="first",
    ).sort_index()
    if pivot.isna().any().any():
        raise ValueError("Stacking requires every base model to have predictions for every sample.")
    y = predictions.drop_duplicates(SAMPLE_KEY_COLUMNS).set_index(SAMPLE_KEY_COLUMNS).loc[pivot.index, "y_true"]
    return pivot, y.astype(int)

