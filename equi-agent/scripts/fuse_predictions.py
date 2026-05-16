from __future__ import annotations

import argparse
from pathlib import Path


KEY_COLUMNS = ["patient_id", "eye_id", "visit_id", "image_id", "dataset", "task", "split"]
METADATA_COLUMNS = ["race", "ethnicity", "sex_gender", "age", "age_group", "metadata_missing_flag"]
STANDARD_COLUMNS = [
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


def equi_agent_root() -> Path:
    return Path(__file__).resolve().parents[1]


def require_runtime_libs():
    import numpy as np
    import pandas as pd

    return np, pd


def prepare_predictions(df, model_label: str, pd):
    keep = KEY_COLUMNS + ["y_true", "y_prob"] + METADATA_COLUMNS
    out = df[keep].copy()
    rename = {"y_prob": f"y_prob_{model_label}", "y_true": "y_true_base"}
    return out.rename(columns=rename)


def confidence_weighted_probability(np, left_prob, right_prob):
    left_conf = np.abs(left_prob - 0.5)
    right_conf = np.abs(right_prob - 0.5)
    denom = left_conf + right_conf
    return np.where(denom > 0, (left_prob * left_conf + right_prob * right_conf) / denom, (left_prob + right_prob) / 2.0)


def make_standard_output(merged, y_prob, model_name: str, pd):
    output = merged[KEY_COLUMNS + ["y_true"] + METADATA_COLUMNS].copy()
    output["model_name"] = model_name
    output["y_prob"] = y_prob
    output["y_pred"] = (output["y_prob"] >= 0.5).astype(int)
    return output[STANDARD_COLUMNS]


def summarize(output, name: str) -> None:
    print(f"{name}: rows={len(output)}")
    print(f"  by_task={output['task'].value_counts().to_dict()}")
    print(f"  by_split={output['split'].value_counts().to_dict()}")
    print(f"  prob_summary={output.groupby('task')['y_prob'].describe().to_dict()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build static fusion baselines from two standard prediction CSVs.")
    parser.add_argument("--left", type=Path, required=True, help="First standard prediction CSV, e.g. OCT.")
    parser.add_argument("--right", type=Path, required=True, help="Second standard prediction CSV, e.g. SLO.")
    parser.add_argument("--left-label", default="left")
    parser.add_argument("--right-label", default="right")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=equi_agent_root() / "outputs" / "predictions",
    )
    parser.add_argument("--prefix", default="fairvision_static_fusion")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np, pd = require_runtime_libs()

    left = pd.read_csv(args.left)
    right = pd.read_csv(args.right)
    left_prepared = prepare_predictions(left, args.left_label, pd)
    right_prepared = prepare_predictions(right, args.right_label, pd)

    merged = left_prepared.merge(
        right_prepared[KEY_COLUMNS + ["y_true_base", f"y_prob_{args.right_label}"]],
        on=KEY_COLUMNS,
        how="inner",
        suffixes=("", "_right"),
    )
    if len(merged) != min(len(left), len(right)):
        print(f"Warning: merged rows={len(merged)} left={len(left)} right={len(right)}")

    if not (merged["y_true_base"] == merged["y_true_base_right"]).all():
        raise ValueError("Prediction files disagree on y_true for at least one key.")
    merged = merged.rename(columns={"y_true_base": "y_true"}).drop(columns=["y_true_base_right"])

    left_prob = merged[f"y_prob_{args.left_label}"].astype(float)
    right_prob = merged[f"y_prob_{args.right_label}"].astype(float)

    mean_output = make_standard_output(
        merged,
        (left_prob + right_prob) / 2.0,
        model_name=f"mean_prob_{args.left_label}_{args.right_label}",
        pd=pd,
    )
    confidence_output = make_standard_output(
        merged,
        confidence_weighted_probability(np, left_prob.to_numpy(), right_prob.to_numpy()),
        model_name=f"confidence_weighted_{args.left_label}_{args.right_label}",
        pd=pd,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    mean_path = args.out_dir / f"{args.prefix}_mean.csv"
    confidence_path = args.out_dir / f"{args.prefix}_confidence_weighted.csv"
    mean_output.to_csv(mean_path, index=False)
    confidence_output.to_csv(confidence_path, index=False)

    print(f"wrote_mean={mean_path}")
    summarize(mean_output, "mean")
    print(f"wrote_confidence_weighted={confidence_path}")
    summarize(confidence_output, "confidence_weighted")


if __name__ == "__main__":
    main()
