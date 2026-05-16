from __future__ import annotations

import argparse
import sys
from pathlib import Path


DEFAULT_ATTRIBUTES = ["race", "ethnicity", "sex_gender", "age_group"]
INTERSECTIONAL_PAIRS = [("race", "sex_gender"), ("race", "age_group"), ("sex_gender", "age_group")]


def equi_agent_root() -> Path:
    return Path(__file__).resolve().parents[1]


def require_runtime_libs():
    import pandas as pd

    sys.path.insert(0, str(equi_agent_root() / "src"))
    from fairness.subgroup import add_intersectional_attributes, subgroup_metrics
    from metrics.classification import binary_classification_metrics

    return pd, add_intersectional_attributes, subgroup_metrics, binary_classification_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build validation subgroup reliability priors.")
    parser.add_argument("--predictions", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--global-out", type=Path, default=None)
    parser.add_argument("--min-positive", type=int, default=20)
    parser.add_argument("--min-negative", type=int, default=20)
    parser.add_argument("--include-intersections", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pd, add_intersectional_attributes, subgroup_metrics, binary_classification_metrics = require_runtime_libs()

    frames = []
    for path in args.predictions:
        df = pd.read_csv(path)
        df = df[df["split"].isin(["val", "validation"])].copy()
        if df.empty:
            print(f"Warning: no validation rows in {path}")
            continue
        frames.append(df)
    if not frames:
        raise ValueError("No validation predictions were provided.")

    predictions = pd.concat(frames, ignore_index=True)
    predictions = add_intersectional_attributes(predictions, INTERSECTIONAL_PAIRS)
    attributes = DEFAULT_ATTRIBUTES + [f"{left}_x_{right}" for left, right in INTERSECTIONAL_PAIRS]

    global_rows = []
    subgroup_rows = []
    group_cols = ["dataset", "task", "model_name"]
    for keys, group in predictions.groupby(group_cols, dropna=False):
        key = dict(zip(group_cols, keys))
        global_metrics = binary_classification_metrics(group["y_true"], group["y_prob"], group["y_pred"])
        global_rows.append({**key, "attribute": "GLOBAL", "subgroup": "GLOBAL", **global_metrics})

        for attr in attributes:
            attr_df = subgroup_metrics(
                group,
                attr,
                min_positive=args.min_positive,
                min_negative=args.min_negative,
            )
            for row in attr_df.to_dict("records"):
                subgroup_rows.append({**key, **row})

    global_priors = pd.DataFrame(global_rows)
    subgroup_priors = pd.DataFrame(subgroup_rows)
    priors = pd.concat([global_priors, subgroup_priors], ignore_index=True, sort=False)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    priors.to_csv(args.out, index=False)
    global_path = args.global_out or args.out.with_name(args.out.stem + "_global.csv")
    global_priors.to_csv(global_path, index=False)

    print(f"wrote={args.out}")
    print(f"wrote_global={global_path}")
    print(f"rows={len(priors)}")
    print(f"global_rows={len(global_priors)}")
    print(f"subgroup_rows={len(subgroup_priors)}")
    print("models_tasks:")
    print(global_priors[["dataset", "task", "model_name", "n", "auroc", "f1", "fpr", "fnr", "ece"]].to_string(index=False))
    print("unstable_subgroup_count=", int(subgroup_priors.get("unstable", pd.Series(dtype=bool)).sum()))


if __name__ == "__main__":
    main()
