from __future__ import annotations

from pathlib import Path

import pandas as pd

from fairness.subgroup import subgroup_metrics


def build_validation_subgroup_priors(
    validation_predictions: pd.DataFrame,
    attributes: list[str],
    min_positive: int = 20,
    min_negative: int = 20,
) -> pd.DataFrame:
    """Compute validation-only subgroup reliability priors for agent inputs."""
    if set(validation_predictions["split"].unique()) - {"val", "validation"}:
        raise ValueError("Subgroup priors must be built from validation predictions only.")

    rows = []
    group_cols = ["model_name", "task"]
    if "dataset" in validation_predictions.columns:
        group_cols.insert(1, "dataset")

    for keys, group in validation_predictions.groupby(group_cols, dropna=False):
        key_values = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        for attr in attributes:
            attr_metrics = subgroup_metrics(
                group,
                attr,
                min_positive=min_positive,
                min_negative=min_negative,
            )
            for row in attr_metrics.to_dict("records"):
                row.update(key_values)
                rows.append(row)
    return pd.DataFrame(rows)


def save_subgroup_priors(priors: pd.DataFrame, csv_path: str, json_path: str | None = None) -> None:
    csv_file = Path(csv_path)
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    priors.to_csv(csv_file, index=False)
    if json_path:
        json_file = Path(json_path)
        json_file.parent.mkdir(parents=True, exist_ok=True)
        priors.to_json(json_file, orient="records", indent=2)
