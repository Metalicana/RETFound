#!/usr/bin/env python
"""Build a versioned model reliability registry for Equi-Agent arbitration.

The registry keeps empirical model priors outside prompts. It combines:
- validation-selected thresholds,
- held-out aggregate model metrics,
- subgroup false-positive / false-negative priors.
- validation margin-bin reliability estimates.

The runtime can then look up a compact case-specific reliability packet and pass
that packet to agents without hard-coding metric values in prompt text.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SUBGROUP_MODEL_ALIASES = {
    "retfound": "retfound_oct",
    "mirage": "mirage_slo",
}

METRIC_FIELDS = [
    "n",
    "n_positive",
    "n_negative",
    "auroc",
    "auprc",
    "f1",
    "accuracy",
    "balanced_accuracy",
    "brier",
    "nll",
    "ece",
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
    "fpr",
    "fnr",
]

DEFAULT_MARGIN_BINS = [
    float("-inf"),
    -0.50,
    -0.25,
    -0.10,
    -0.05,
    0.00,
    0.05,
    0.10,
    0.20,
    0.35,
    0.50,
    float("inf"),
]


def parse_float(value: str | None) -> float | int | None:
    if value in {None, ""}:
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    if number.is_integer():
        return int(number)
    return number


def format_bound(value: float) -> str:
    if value == float("-inf"):
        return "-inf"
    if value == float("inf"):
        return "inf"
    return f"{value:.2f}"


def find_margin_bin(margin: float, bins: list[float]) -> tuple[int, float, float, str]:
    for index in range(len(bins) - 1):
        lower = bins[index]
        upper = bins[index + 1]
        if lower <= margin < upper:
            return index, lower, upper, f"[{format_bound(lower)}, {format_bound(upper)})"
    # Only possible for +inf/NaN-adjacent edge cases; keep deterministic.
    index = len(bins) - 2
    lower = bins[index]
    upper = bins[index + 1]
    return index, lower, upper, f"[{format_bound(lower)}, {format_bound(upper)})"


def safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def model_entry(registry: dict[str, Any], model_name: str) -> dict[str, Any]:
    return registry["models"].setdefault(
        model_name,
        {
            "tasks": {},
            "subgroup_priors": {},
        },
    )


def task_entry(registry: dict[str, Any], model_name: str, task: str) -> dict[str, Any]:
    model = model_entry(registry, model_name)
    return model["tasks"].setdefault(task, {})


def load_thresholds(metrics_dir: Path, registry: dict[str, Any]) -> None:
    for path in sorted(metrics_dir.glob("thresholds_fairvision*.csv")):
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                model_name = row.get("model_name")
                task = row.get("task")
                if not model_name or not task:
                    continue
                entry = task_entry(registry, model_name, task)
                entry["threshold"] = {
                    "value": parse_float(row.get("threshold")),
                    "selection_metric": row.get("metric"),
                    "validation_score": parse_float(row.get("validation_score")),
                    "n": parse_float(row.get("n")),
                    "source_file": str(path),
                }


def threshold_lookup(registry: dict[str, Any]) -> dict[tuple[str, str], float]:
    lookup = {}
    for model_name, model in registry.get("models", {}).items():
        for task, task_values in model.get("tasks", {}).items():
            threshold = task_values.get("threshold", {}).get("value")
            if threshold is not None:
                lookup[(model_name, task)] = float(threshold)
    return lookup


def prediction_files_for_split(predictions_dir: Path, split: str) -> list[Path]:
    return sorted(predictions_dir.glob(f"fairvision*_{split}.csv"))


def load_margin_bins(
    root: Path,
    registry: dict[str, Any],
    split: str,
    bins: list[float],
) -> None:
    predictions_dir = root / "outputs" / "predictions"
    thresholds = threshold_lookup(registry)
    grouped: dict[tuple[str, str, int], dict[str, Any]] = {}

    for path in prediction_files_for_split(predictions_dir, split):
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("split") != split:
                    continue
                model_name = row.get("model_name")
                task = row.get("task")
                if not model_name or not task:
                    continue
                threshold = thresholds.get((model_name, task))
                if threshold is None:
                    continue
                y_prob = parse_float(row.get("y_prob"))
                y_true = parse_float(row.get("y_true"))
                if y_prob is None or y_true is None or math.isnan(float(y_prob)):
                    continue
                margin = float(y_prob) - threshold
                bin_index, lower, upper, label = find_margin_bin(margin, bins)
                key = (model_name, task, bin_index)
                bucket = grouped.setdefault(
                    key,
                    {
                        "bin_index": bin_index,
                        "lower": lower,
                        "upper": upper,
                        "label": label,
                        "n": 0,
                        "n_positive": 0,
                        "n_negative": 0,
                        "tp": 0,
                        "tn": 0,
                        "fp": 0,
                        "fn": 0,
                    },
                )
                y_pred = int(margin >= 0.0)
                y_true_int = int(float(y_true) > 0)
                bucket["n"] += 1
                bucket["n_positive"] += y_true_int
                bucket["n_negative"] += int(y_true_int == 0)
                if y_true_int == 1 and y_pred == 1:
                    bucket["tp"] += 1
                elif y_true_int == 0 and y_pred == 0:
                    bucket["tn"] += 1
                elif y_true_int == 0 and y_pred == 1:
                    bucket["fp"] += 1
                elif y_true_int == 1 and y_pred == 0:
                    bucket["fn"] += 1

    for (model_name, task, _), bucket in sorted(grouped.items()):
        n = bucket["n"]
        pred_pos = bucket["tp"] + bucket["fp"]
        pred_neg = bucket["tn"] + bucket["fn"]
        enriched = {
            **bucket,
            "source_split": split,
            "empirical_positive_rate": safe_rate(bucket["n_positive"], n),
            "accuracy": safe_rate(bucket["tp"] + bucket["tn"], n),
            "precision_when_predicted_positive": safe_rate(bucket["tp"], pred_pos),
            "npv_when_predicted_negative": safe_rate(bucket["tn"], pred_neg),
            "sensitivity_within_bin": safe_rate(bucket["tp"], bucket["tp"] + bucket["fn"]),
            "specificity_within_bin": safe_rate(bucket["tn"], bucket["tn"] + bucket["fp"]),
            "predicted_positive_count": pred_pos,
            "predicted_negative_count": pred_neg,
        }
        # JSON does not support infinity, so keep numeric finite fields nullable
        # and retain the text label for the actual interval.
        if enriched["lower"] in {float("-inf"), float("inf")}:
            enriched["lower"] = None
        if enriched["upper"] in {float("-inf"), float("inf")}:
            enriched["upper"] = None
        task_entry(registry, model_name, task).setdefault("margin_bins", []).append(enriched)


def load_aggregate_metrics(metrics_dir: Path, registry: dict[str, Any]) -> None:
    aggregate_paths = sorted(metrics_dir.glob("exp*/fairvision*_thresholded_aggregate.csv"))
    for path in aggregate_paths:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                model_name = row.get("model_name")
                task = row.get("task")
                if not model_name or not task:
                    continue
                entry = task_entry(registry, model_name, task)
                metrics = {
                    field: parse_float(row.get(field))
                    for field in METRIC_FIELDS
                    if field in row
                }
                metrics["split"] = row.get("split")
                metrics["source_file"] = str(path)
                entry["global_metrics"] = metrics


def load_subgroup_priors(equity_json_dir: Path, registry: dict[str, Any]) -> None:
    for path in sorted(equity_json_dir.glob("equity_*_calibration.json")):
        source_model = path.stem.removeprefix("equity_").removesuffix("_calibration")
        model_name = SUBGROUP_MODEL_ALIASES.get(source_model, source_model)
        with path.open(encoding="utf-8") as handle:
            priors = json.load(handle)
        entry = model_entry(registry, model_name)
        entry["subgroup_priors"] = {
            "source_model_key": source_model,
            "source_file": str(path),
            "false_positive": priors.get("false_positive", {}),
            "false_negative": priors.get("false_negative", {}),
        }


def build_registry(root: Path, margin_split: str, margin_bins: list[float]) -> dict[str, Any]:
    metrics_dir = root / "outputs" / "metrics"
    equity_json_dir = root / "EquityAgent" / "JSONs"
    registry: dict[str, Any] = {
        "schema_version": "1.0",
        "dataset": "harvard_fairvision",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "intended_use": "reliability-aware arbitration priors",
            "label_leakage_exclusions": ["visual-field MD for FairVision glaucoma"],
            "notes": [
                "Thresholds are selected outside the prompt and loaded as data.",
                f"Margin-bin reliability is computed from the {margin_split} split only.",
                "Subgroup priors are used for model reliability lookup, not disease likelihood.",
                "Production deployments should regenerate this registry after monitoring updates.",
            ],
        },
        "margin_bin_edges": [
            format_bound(edge)
            for edge in margin_bins
        ],
        "models": {},
    }
    load_thresholds(metrics_dir, registry)
    load_margin_bins(root, registry, margin_split, margin_bins)
    load_aggregate_metrics(metrics_dir, registry)
    load_subgroup_priors(equity_json_dir, registry)
    return registry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="equi-agent root directory")
    parser.add_argument(
        "--output",
        default="outputs/registry/fairvision_model_reliability_registry.json",
        help="registry JSON path relative to --root unless absolute",
    )
    parser.add_argument(
        "--margin-split",
        default="val",
        choices=["val", "test"],
        help="prediction split used to build margin-bin reliability; use val for runtime priors",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output

    registry = build_registry(root, args.margin_split, DEFAULT_MARGIN_BINS)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(registry, handle, indent=2, sort_keys=True)
        handle.write("\n")

    n_tasks = sum(len(model.get("tasks", {})) for model in registry["models"].values())
    print(f"wrote={output}")
    print(f"models={len(registry['models'])} model_tasks={n_tasks}")


if __name__ == "__main__":
    main()
