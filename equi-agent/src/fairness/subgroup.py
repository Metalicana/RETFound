from __future__ import annotations

import numpy as np
import pandas as pd

from metrics.classification import binary_classification_metrics


def add_intersectional_attributes(
    df: pd.DataFrame,
    pairs: list[tuple[str, str]],
    separator: str = " x ",
) -> pd.DataFrame:
    """Add pairwise intersectional columns such as race_x_sex_gender."""
    output = df.copy()
    for left, right in pairs:
        if left not in output.columns or right not in output.columns:
            raise ValueError(f"Missing columns for intersectional attribute: {left}, {right}")
        name = f"{left}_x_{right}"
        output[name] = (
            output[left].fillna("missing").astype(str)
            + separator
            + output[right].fillna("missing").astype(str)
        )
    return output


def subgroup_metrics(
    df: pd.DataFrame,
    attribute: str,
    min_positive: int = 20,
    min_negative: int = 20,
    y_true_col: str = "y_true",
    y_prob_col: str = "y_prob",
    y_pred_col: str = "y_pred",
) -> pd.DataFrame:
    if attribute not in df.columns:
        raise ValueError(f"Missing subgroup attribute: {attribute}")
    rows = []
    for subgroup, group in df.groupby(attribute, dropna=False):
        metrics = binary_classification_metrics(group[y_true_col], group[y_prob_col], group[y_pred_col])
        metrics.update({"attribute": attribute, "subgroup": subgroup})
        metrics["unstable"] = bool(metrics["n_positive"] < min_positive or metrics["n_negative"] < min_negative)
        rows.append(metrics)
    return pd.DataFrame(rows)


def subgroup_disparities(subgroup_df: pd.DataFrame) -> dict[str, float]:
    stable = subgroup_df[~subgroup_df["unstable"]].copy()
    if stable.empty:
        stable = subgroup_df.copy()

    def gap(metric: str) -> float:
        values = stable[metric].dropna()
        return float(values.max() - values.min()) if len(values) else np.nan

    return {
        "worst_group_f1": float(stable["f1"].min()) if stable["f1"].notna().any() else np.nan,
        "worst_group_auroc": float(stable["auroc"].min()) if stable["auroc"].notna().any() else np.nan,
        "max_min_fpr_gap": gap("fpr"),
        "max_min_fnr_gap": gap("fnr"),
        "equal_opportunity_difference": gap("sensitivity"),
        "equalized_odds_difference": max(gap("fpr"), gap("sensitivity")),
        "subgroup_ece_mean": float(stable["ece"].mean()) if stable["ece"].notna().any() else np.nan,
        "subgroup_ece_max": float(stable["ece"].max()) if stable["ece"].notna().any() else np.nan,
    }


def all_subgroup_metrics(
    df: pd.DataFrame,
    attributes: list[str],
    min_positive: int = 20,
    min_negative: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_frames = [
        subgroup_metrics(df, attr, min_positive=min_positive, min_negative=min_negative)
        for attr in attributes
    ]
    metrics_df = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    disparity_rows = []
    for attr in attributes:
        attr_df = metrics_df[metrics_df["attribute"] == attr]
        row = {"attribute": attr}
        row.update(subgroup_disparities(attr_df))
        disparity_rows.append(row)
    return metrics_df, pd.DataFrame(disparity_rows)
