from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_true) == 0:
        return np.nan
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = (y_prob > lower) & (y_prob <= upper)
        if lower == 0.0:
            mask = (y_prob >= lower) & (y_prob <= upper)
        if not mask.any():
            continue
        confidence = y_prob[mask].mean()
        accuracy = y_true[mask].mean()
        ece += mask.mean() * abs(accuracy - confidence)
    return float(ece)


def confusion_rates(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "ppv": tp / (tp + fp) if tp + fp else np.nan,
        "npv": tn / (tn + fn) if tn + fn else np.nan,
        "fpr": fp / (fp + tn) if fp + tn else np.nan,
        "fnr": fn / (fn + tp) if fn + tp else np.nan,
    }


def binary_classification_metrics(
    y_true,
    y_prob,
    y_pred=None,
    threshold: float = 0.5,
    ece_bins: int = 10,
) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_pred is None:
        y_pred = (y_prob >= threshold).astype(int)
    else:
        y_pred = np.asarray(y_pred).astype(int)

    metrics = {
        "n": int(len(y_true)),
        "n_positive": int((y_true == 1).sum()),
        "n_negative": int((y_true == 0).sum()),
        "auroc": _safe_score(roc_auc_score, y_true, y_prob),
        "auprc": _safe_score(average_precision_score, y_true, y_prob),
        "f1": _safe_score(f1_score, y_true, y_pred, zero_division=0),
        "accuracy": _safe_score(accuracy_score, y_true, y_pred),
        "balanced_accuracy": _safe_score(balanced_accuracy_score, y_true, y_pred),
        "brier": _safe_score(brier_score_loss, y_true, y_prob),
        "nll": _safe_log_loss(y_true, y_prob),
        "ece": expected_calibration_error(y_true, y_prob, n_bins=ece_bins),
    }
    metrics.update(confusion_rates(y_true, y_pred))
    return {key: float(value) if isinstance(value, (np.floating, float)) else value for key, value in metrics.items()}


def reliability_diagram_data(y_true, y_prob, n_bins: int = 10) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    rows = []
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    for idx, (lower, upper) in enumerate(zip(bins[:-1], bins[1:])):
        mask = (y_prob > lower) & (y_prob <= upper)
        if lower == 0.0:
            mask = (y_prob >= lower) & (y_prob <= upper)
        rows.append(
            {
                "bin": idx,
                "lower": lower,
                "upper": upper,
                "n": int(mask.sum()),
                "mean_probability": float(y_prob[mask].mean()) if mask.any() else np.nan,
                "empirical_rate": float(y_true[mask].mean()) if mask.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _safe_score(func, *args, **kwargs) -> float:
    try:
        return float(func(*args, **kwargs))
    except ValueError:
        return np.nan


def _safe_log_loss(y_true, y_prob) -> float:
    try:
        clipped = np.clip(y_prob, 1e-7, 1 - 1e-7)
        return float(log_loss(y_true, clipped, labels=[0, 1]))
    except ValueError:
        return np.nan
