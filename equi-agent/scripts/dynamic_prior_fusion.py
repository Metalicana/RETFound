from __future__ import annotations

import argparse
import math
import sys
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
INTERSECTIONAL_PAIRS = [("race", "sex_gender"), ("race", "age_group"), ("sex_gender", "age_group")]
DEFAULT_ATTRIBUTE_PRIORITY = [
    "race_x_sex_gender",
    "race_x_age_group",
    "sex_gender_x_age_group",
    "race",
    "ethnicity",
    "sex_gender",
    "age_group",
]
LOWER_IS_BETTER = {"ece", "fpr", "fnr", "brier", "nll"}


def equi_agent_root() -> Path:
    return Path(__file__).resolve().parents[1]


def require_runtime_libs():
    import pandas as pd

    sys.path.insert(0, str(equi_agent_root() / "src"))
    from data.predictions import validate_prediction_schema
    from fairness.subgroup import add_intersectional_attributes

    return pd, validate_prediction_schema, add_intersectional_attributes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Choose a per-example FairVision prediction source using validation priors."
    )
    parser.add_argument("--predictions", nargs="+", type=Path, required=True)
    parser.add_argument("--subgroup-priors", type=Path, required=True)
    parser.add_argument("--global-priors", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--selection-log-out",
        type=Path,
        default=None,
        help="Optional row-level decision log path.",
    )
    parser.add_argument(
        "--threshold-files",
        nargs="*",
        type=Path,
        default=[],
        help="Optional tune_thresholds.py threshold CSVs. Used for y_pred after model selection.",
    )
    parser.add_argument(
        "--select-metric",
        default="auroc",
        help="Prior metric used to choose model source, e.g. auroc, f1, ece, fnr.",
    )
    parser.add_argument(
        "--attribute-priority",
        nargs="*",
        default=DEFAULT_ATTRIBUTE_PRIORITY,
        help="Subgroup attributes checked before falling back to global priors.",
    )
    parser.add_argument("--min-positive", type=int, default=20)
    parser.add_argument("--min-negative", type=int, default=20)
    parser.add_argument(
        "--model-name",
        default=None,
        help="Output model_name. Defaults to dynamic_prior_<select_metric>.",
    )
    return parser.parse_args()


def norm_value(value) -> str:
    if value is None:
        return "missing"
    try:
        if math.isnan(value):
            return "missing"
    except TypeError:
        pass
    text = str(value)
    return text if text and text.lower() != "nan" else "missing"


def finite_float(value, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number):
        return default
    return number


def load_predictions(paths, pd, validate_prediction_schema, add_intersectional_attributes):
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        validate_prediction_schema(df)
        frames.append(df)
    predictions = pd.concat(frames, ignore_index=True)
    predictions = add_intersectional_attributes(predictions, INTERSECTIONAL_PAIRS)

    duplicate_keys = KEY_COLUMNS + ["model_name"]
    duplicates = int(predictions.duplicated(duplicate_keys).sum())
    if duplicates:
        raise ValueError(f"Duplicate prediction rows across inputs for {duplicate_keys}: {duplicates}")
    return predictions


def load_thresholds(paths, pd) -> dict[tuple[str, str, str], float]:
    thresholds: dict[tuple[str, str, str], float] = {}
    for path in paths:
        df = pd.read_csv(path)
        required = {"dataset", "task", "model_name", "threshold"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Threshold file {path} missing columns: {sorted(missing)}")
        for row in df.to_dict("records"):
            key = (str(row["dataset"]), str(row["task"]), str(row["model_name"]))
            thresholds[key] = float(row["threshold"])
    return thresholds


def priors_to_lookup(priors, metric: str) -> dict[tuple[str, str, str, str, str], dict]:
    required = {"dataset", "task", "model_name", "attribute", "subgroup", metric}
    missing = required - set(priors.columns)
    if missing:
        raise ValueError(f"Prior table missing columns for --select-metric {metric!r}: {sorted(missing)}")

    lookup = {}
    for row in priors.to_dict("records"):
        key = (
            str(row["dataset"]),
            str(row["task"]),
            str(row["model_name"]),
            str(row["attribute"]),
            norm_value(row["subgroup"]),
        )
        lookup[key] = row
    return lookup


def is_stable(row: dict, min_positive: int, min_negative: int) -> bool:
    if str(row.get("unstable", "False")).lower() in {"true", "1", "yes"}:
        return False
    n_positive = finite_float(row.get("n_positive"), 0.0)
    n_negative = finite_float(row.get("n_negative"), 0.0)
    return bool(n_positive is not None and n_negative is not None and n_positive >= min_positive and n_negative >= min_negative)


def prior_score(row: dict | None, metric: str) -> float | None:
    if row is None:
        return None
    return finite_float(row.get(metric))


def better_score(score: float | None, best: float | None, metric: str) -> bool:
    if score is None:
        return False
    if best is None:
        return True
    if metric in LOWER_IS_BETTER:
        return score < best
    return score > best


def find_model_prior(
    candidate: dict,
    prior_lookup: dict[tuple[str, str, str, str, str], dict],
    global_lookup: dict[tuple[str, str, str, str, str], dict],
    metric: str,
    attribute_priority: list[str],
    min_positive: int,
    min_negative: int,
) -> tuple[dict | None, str, str]:
    dataset = str(candidate["dataset"])
    task = str(candidate["task"])
    model = str(candidate["model_name"])

    for attr in attribute_priority:
        if attr not in candidate:
            continue
        subgroup = norm_value(candidate[attr])
        row = prior_lookup.get((dataset, task, model, attr, subgroup))
        if row is not None and is_stable(row, min_positive, min_negative) and prior_score(row, metric) is not None:
            return row, attr, subgroup

    row = global_lookup.get((dataset, task, model, "GLOBAL", "GLOBAL"))
    if row is not None and prior_score(row, metric) is not None:
        return row, "GLOBAL", "GLOBAL"
    return None, "NONE", "NONE"


def choose_candidate(
    group,
    prior_lookup,
    global_lookup,
    metric: str,
    attribute_priority: list[str],
    min_positive: int,
    min_negative: int,
):
    best = None
    best_score = None
    best_prior = None
    best_attr = "NONE"
    best_subgroup = "NONE"
    best_global_score = None
    best_ece = None

    for candidate in group.to_dict("records"):
        prior, attr, subgroup = find_model_prior(
            candidate,
            prior_lookup,
            global_lookup,
            metric,
            attribute_priority,
            min_positive,
            min_negative,
        )
        score = prior_score(prior, metric)
        if not better_score(score, best_score, metric):
            if score != best_score:
                continue
            global_prior = global_lookup.get(
                (
                    str(candidate["dataset"]),
                    str(candidate["task"]),
                    str(candidate["model_name"]),
                    "GLOBAL",
                    "GLOBAL",
                )
            )
            global_score = prior_score(global_prior, metric)
            ece = prior_score(global_prior, "ece")
            if best is not None:
                if better_score(global_score, best_global_score, metric):
                    pass
                elif global_score == best_global_score and ece is not None and (best_ece is None or ece < best_ece):
                    pass
                else:
                    continue
        global_prior = global_lookup.get(
            (
                str(candidate["dataset"]),
                str(candidate["task"]),
                str(candidate["model_name"]),
                "GLOBAL",
                "GLOBAL",
            )
        )
        best = candidate
        best_score = score
        best_prior = prior
        best_attr = attr
        best_subgroup = subgroup
        best_global_score = prior_score(global_prior, metric)
        best_ece = prior_score(global_prior, "ece")

    if best is None:
        raise ValueError("Could not select any model for a prediction group.")
    return best, best_prior, best_attr, best_subgroup, best_score


def apply_threshold(row: dict, thresholds: dict[tuple[str, str, str], float]) -> tuple[int, float, str]:
    key = (str(row["dataset"]), str(row["task"]), str(row["model_name"]))
    threshold = thresholds.get(key)
    if threshold is None:
        return int(row["y_pred"]), 0.5, "source_y_pred"
    return int(float(row["y_prob"]) >= threshold), threshold, "threshold_file"


def main() -> None:
    args = parse_args()
    pd, validate_prediction_schema, add_intersectional_attributes = require_runtime_libs()

    predictions = load_predictions(args.predictions, pd, validate_prediction_schema, add_intersectional_attributes)
    subgroup_priors = pd.read_csv(args.subgroup_priors)
    global_priors = pd.read_csv(args.global_priors)
    prior_lookup = priors_to_lookup(subgroup_priors, args.select_metric)
    global_lookup = priors_to_lookup(global_priors, args.select_metric)
    thresholds = load_thresholds(args.threshold_files, pd) if args.threshold_files else {}

    output_rows = []
    log_rows = []
    out_model_name = args.model_name or f"dynamic_prior_{args.select_metric}"
    for _, group in predictions.groupby(KEY_COLUMNS, dropna=False, sort=False):
        y_true_values = set(group["y_true"].dropna().astype(int).unique())
        if len(y_true_values) != 1:
            raise ValueError(f"Input predictions disagree on y_true for key: {group[KEY_COLUMNS].iloc[0].to_dict()}")

        chosen, prior, attr, subgroup, score = choose_candidate(
            group,
            prior_lookup,
            global_lookup,
            args.select_metric,
            args.attribute_priority,
            args.min_positive,
            args.min_negative,
        )
        y_pred, threshold, threshold_source = apply_threshold(chosen, thresholds)
        row = {col: chosen[col] for col in STANDARD_COLUMNS if col != "model_name"}
        row["model_name"] = out_model_name
        row["y_pred"] = y_pred
        output_rows.append(row)

        prior_n = finite_float(prior.get("n") if prior else None)
        log_rows.append(
            {
                **{col: chosen[col] for col in KEY_COLUMNS},
                "y_true": chosen["y_true"],
                "chosen_model": chosen["model_name"],
                "chosen_y_prob": chosen["y_prob"],
                "chosen_y_pred": y_pred,
                "threshold": threshold,
                "threshold_source": threshold_source,
                "prior_attribute": attr,
                "prior_subgroup": subgroup,
                "select_metric": args.select_metric,
                "prior_score": score,
                "prior_n": prior_n,
            }
        )

    output = pd.DataFrame(output_rows)[STANDARD_COLUMNS]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.out, index=False)

    log_path = args.selection_log_out or args.out.with_name(args.out.stem + "_selection_log.csv")
    pd.DataFrame(log_rows).to_csv(log_path, index=False)

    print(f"wrote={args.out}")
    print(f"wrote_selection_log={log_path}")
    print(f"rows={len(output)}")
    print(f"model_name={out_model_name}")
    print(f"select_metric={args.select_metric}")
    print(f"threshold_files={len(args.threshold_files)}")
    print("chosen_models:")
    print(pd.DataFrame(log_rows)["chosen_model"].value_counts().to_string())
    print("by_task:")
    print(output["task"].value_counts().to_string())


if __name__ == "__main__":
    main()
