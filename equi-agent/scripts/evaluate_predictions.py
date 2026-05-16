from __future__ import annotations

import argparse
import sys
from pathlib import Path


DEFAULT_ATTRIBUTES = ["race", "ethnicity", "sex_gender", "age_group"]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def equi_agent_root() -> Path:
    return Path(__file__).resolve().parents[1]


def require_runtime_libs():
    import pandas as pd

    sys.path.insert(0, str(equi_agent_root() / "src"))
    from data.predictions import validate_prediction_schema
    from fairness.subgroup import add_intersectional_attributes, all_subgroup_metrics
    from metrics.classification import binary_classification_metrics, reliability_diagram_data

    return pd, validate_prediction_schema, add_intersectional_attributes, all_subgroup_metrics, binary_classification_metrics, reliability_diagram_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate standard-schema prediction CSVs.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=equi_agent_root() / "outputs" / "metrics" / "exp1_standalone",
    )
    parser.add_argument("--min-positive", type=int, default=20)
    parser.add_argument("--min-negative", type=int, default=20)
    parser.add_argument("--ece-bins", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    (
        pd,
        validate_prediction_schema,
        add_intersectional_attributes,
        all_subgroup_metrics,
        binary_classification_metrics,
        reliability_diagram_data,
    ) = require_runtime_libs()

    predictions = pd.read_csv(args.predictions)
    validate_prediction_schema(predictions)
    predictions = add_intersectional_attributes(
        predictions,
        [("race", "sex_gender"), ("race", "age_group"), ("sex_gender", "age_group")],
    )
    attributes = DEFAULT_ATTRIBUTES + ["race_x_sex_gender", "race_x_age_group", "sex_gender_x_age_group"]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    aggregate_rows = []
    reliability_frames = []
    subgroup_frames = []
    disparity_frames = []

    group_cols = ["dataset", "task", "model_name", "split"]
    for keys, group in predictions.groupby(group_cols, dropna=False):
        key = dict(zip(group_cols, keys))
        metrics = binary_classification_metrics(
            group["y_true"],
            group["y_prob"],
            group["y_pred"],
            ece_bins=args.ece_bins,
        )
        aggregate_rows.append({**key, **metrics})

        reliability = reliability_diagram_data(group["y_true"], group["y_prob"], n_bins=args.ece_bins)
        for col, value in key.items():
            reliability[col] = value
        reliability_frames.append(reliability)

        subgroup_metrics, disparities = all_subgroup_metrics(
            group,
            attributes,
            min_positive=args.min_positive,
            min_negative=args.min_negative,
        )
        for frame in [subgroup_metrics, disparities]:
            for col, value in key.items():
                frame[col] = value
        subgroup_frames.append(subgroup_metrics)
        disparity_frames.append(disparities)

    aggregate = pd.DataFrame(aggregate_rows)
    subgroup = pd.concat(subgroup_frames, ignore_index=True) if subgroup_frames else pd.DataFrame()
    disparities = pd.concat(disparity_frames, ignore_index=True) if disparity_frames else pd.DataFrame()
    reliability = pd.concat(reliability_frames, ignore_index=True) if reliability_frames else pd.DataFrame()

    stem = args.predictions.stem
    aggregate.to_csv(args.out_dir / f"{stem}_aggregate.csv", index=False)
    subgroup.to_csv(args.out_dir / f"{stem}_subgroups.csv", index=False)
    disparities.to_csv(args.out_dir / f"{stem}_disparities.csv", index=False)
    reliability.to_csv(args.out_dir / f"{stem}_reliability.csv", index=False)

    print(f"predictions={args.predictions}")
    print(f"rows={len(predictions)}")
    print(f"wrote={args.out_dir}")
    print("\nAggregate:")
    display_cols = ["dataset", "task", "model_name", "split", "n", "auroc", "f1", "balanced_accuracy", "fpr", "fnr", "ece"]
    print(aggregate[display_cols].to_string(index=False))
    print("\nLargest disparity rows:")
    if not disparities.empty:
        sort_col = "equalized_odds_difference"
        print(disparities.sort_values(sort_col, ascending=False).head(12).to_string(index=False))


if __name__ == "__main__":
    main()
