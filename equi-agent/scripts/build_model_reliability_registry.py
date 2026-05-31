#!/usr/bin/env python
"""Build a versioned model reliability registry for Equi-Agent arbitration.

The registry keeps empirical model priors outside prompts. It combines:
- validation-selected thresholds,
- held-out aggregate model metrics,
- subgroup false-positive / false-negative priors.

The runtime can then look up a compact case-specific reliability packet and pass
that packet to agents without hard-coding metric values in prompt text.
"""

from __future__ import annotations

import argparse
import csv
import json
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


def build_registry(root: Path) -> dict[str, Any]:
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
                "Subgroup priors are used for model reliability lookup, not disease likelihood.",
                "Production deployments should regenerate this registry after monitoring updates.",
            ],
        },
        "models": {},
    }
    load_thresholds(metrics_dir, registry)
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
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output

    registry = build_registry(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(registry, handle, indent=2, sort_keys=True)
        handle.write("\n")

    n_tasks = sum(len(model.get("tasks", {})) for model in registry["models"].values())
    print(f"wrote={output}")
    print(f"models={len(registry['models'])} model_tasks={n_tasks}")


if __name__ == "__main__":
    main()
