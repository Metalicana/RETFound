#!/usr/bin/env python3
"""Collect FairVision foundation-model results on Yusra's locked 250-case slices.

The input model CSVs are the standard, validation-thresholded FairVision test
predictions. This script restricts them to the exact AMD and DR cases listed in
``OphthalmicAgent/data/fairvision_250each.csv`` and applies the reporting
convention used by the existing RETFound and MIRAGE table rows:

* overall F1 is support-weighted F1;
* sensitivity, specificity, and balanced accuracy use the binary predictions;
* worst-group F1 is the minimum support-weighted F1 over age group, gender,
  and race subgroups present in the locked slice.

No threshold is selected here and no test label is used to alter a prediction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


MODEL_SPECS = {
    "visionfm_oct": ("VisionFM", "OCT"),
    "urfound_oct": ("URFound", "OCT"),
    "visionfm_slo": ("VisionFM", "SLO"),
    "urfound_slo": ("URFound", "SLO"),
    "flair_slo": ("FLAIR", "SLO"),
    "ret_clip_slo": ("RET-CLIP", "SLO"),
    "retizero_slo": ("RetiZero", "SLO"),
}
TASK_DISPLAY = {"amd": "AMD", "dr": "DR"}
RESULT_FIELDS = [
    "task",
    "model_key",
    "model",
    "modality",
    "n",
    "negative",
    "positive",
    "weighted_f1",
    "worst_group_f1",
    "worst_group_attribute",
    "worst_group_value",
    "worst_group_n",
    "sensitivity",
    "specificity",
    "balanced_accuracy",
    "tn",
    "fp",
    "fn",
    "tp",
    "threshold",
    "threshold_metric",
    "prediction_csv",
    "prediction_sha256",
    "locked_csv",
    "locked_csv_sha256",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--locked-csv",
        type=Path,
        default=Path("OphthalmicAgent/data/fairvision_250each.csv"),
    )
    parser.add_argument(
        "--predictions-root",
        type=Path,
        default=Path("equi-agent/outputs/predictions"),
    )
    parser.add_argument(
        "--metrics-root",
        type=Path,
        default=Path("equi-agent/outputs/metrics"),
        help="Directory containing thresholds_fairvision_*.csv files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("equi-agent/outputs/benchmarks/fairvision_yusra_foundations"),
    )
    parser.add_argument("--tasks", nargs="+", choices=sorted(TASK_DISPLAY), default=["amd", "dr"])
    parser.add_argument("--models", nargs="+", choices=sorted(MODEL_SPECS), default=list(MODEL_SPECS))
    parser.add_argument(
        "--expected-cases-per-task",
        type=int,
        default=250,
        help="Fail rather than report a partial locked cohort.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_task(value: object) -> str:
    return str(value or "").strip().lower()


def binary_value(value: object, field: str) -> int:
    try:
        number = int(float(str(value).strip()))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field}: {value!r}") from exc
    if number not in (0, 1):
        raise ValueError(f"Invalid {field}: {value!r}")
    return number


def age_group(value: object) -> str:
    try:
        age = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid locked-cohort age: {value!r}") from exc
    if age < 50:
        return "young"
    if age >= 70:
        return "older"
    return "middle"


def locked_index(path: Path, tasks: list[str], expected: int) -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    counts = {task: 0 for task in tasks}
    labels = {task: {0: 0, 1: 0} for task in tasks}
    for row in read_csv(path):
        task = normalized_task(row.get("Task_Folder"))
        if task not in tasks:
            continue
        image_id = Path(str(row.get("filename", "")).replace("\\", "/")).name
        if not image_id:
            raise ValueError(f"Locked row has no filename: {row}")
        key = (task, image_id)
        if key in index:
            raise ValueError(f"Duplicate locked case: {key}")
        truth = binary_value(row.get("Ground_Truth"), "Ground_Truth")
        index[key] = {
            "task": task,
            "image_id": image_id,
            "y_true": truth,
            "age_group": age_group(row.get("Age")),
            "gender": str(row.get("Gender", "")).strip().lower(),
            "race": str(row.get("Race", "")).strip().lower(),
        }
        counts[task] += 1
        labels[task][truth] += 1

    for task in tasks:
        if counts[task] != expected:
            raise ValueError(f"Locked {task} cohort must contain {expected} cases; found {counts[task]}")
        if not all(labels[task].values()):
            raise ValueError(f"Locked {task} cohort must contain both classes: {labels[task]}")
    return index


def prediction_path(root: Path, task: str, model: str) -> Path:
    return root / f"fairvision_{task}_{model}_test_thresholded.csv"


def threshold_metadata(root: Path, task: str, model: str) -> tuple[str, str]:
    path = root / f"thresholds_fairvision_{task}_{model}.csv"
    if not path.is_file():
        return "", ""
    rows = read_csv(path)
    if len(rows) != 1:
        raise ValueError(f"Expected one threshold row in {path}; found {len(rows)}")
    return rows[0].get("threshold", ""), rows[0].get("metric", "")


def require_metrics():
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score

    return balanced_accuracy_score, confusion_matrix, f1_score, recall_score


def align_predictions(
    path: Path,
    task: str,
    locked: dict[tuple[str, str], dict[str, Any]],
    expected: int,
) -> list[dict[str, Any]]:
    matched: dict[str, dict[str, Any]] = {}
    for row in read_csv(path):
        if normalized_task(row.get("task")) != task:
            continue
        image_id = Path(str(row.get("image_id", "")).replace("\\", "/")).name
        key = (task, image_id)
        meta = locked.get(key)
        if meta is None:
            continue
        if image_id in matched:
            raise ValueError(f"Duplicate prediction for locked case {task}/{image_id} in {path}")
        prediction_truth = binary_value(row.get("y_true"), "y_true")
        if prediction_truth != meta["y_true"]:
            raise ValueError(
                f"Label mismatch for {task}/{image_id}: locked={meta['y_true']} prediction={prediction_truth}"
            )
        matched[image_id] = {
            **meta,
            "y_pred": binary_value(row.get("y_pred"), "y_pred"),
        }

    if len(matched) != expected:
        missing = sorted(
            image_id
            for locked_task, image_id in locked
            if locked_task == task and image_id not in matched
        )
        raise ValueError(
            f"{path} matched {len(matched)}/{expected} locked {task} cases; missing={missing[:10]}"
        )
    return [matched[image_id] for image_id in sorted(matched)]


def result_metrics(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    balanced_accuracy_score, confusion_matrix, f1_score, recall_score = require_metrics()
    y_true = [row["y_true"] for row in rows]
    y_pred = [row["y_pred"] for row in rows]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    group_rows: list[dict[str, Any]] = []
    for attribute in ("age_group", "gender", "race"):
        for value in sorted({row[attribute] for row in rows}):
            group = [row for row in rows if row[attribute] == value]
            group_rows.append(
                {
                    "attribute": attribute,
                    "subgroup": value,
                    "n": len(group),
                    "negative": sum(row["y_true"] == 0 for row in group),
                    "positive": sum(row["y_true"] == 1 for row in group),
                    "weighted_f1": float(
                        f1_score(
                            [row["y_true"] for row in group],
                            [row["y_pred"] for row in group],
                            average="weighted",
                            zero_division=0,
                        )
                    ),
                }
            )
    worst = min(group_rows, key=lambda row: (row["weighted_f1"], row["attribute"], row["subgroup"]))
    metrics = {
        "n": len(rows),
        "negative": sum(value == 0 for value in y_true),
        "positive": sum(value == 1 for value in y_true),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "worst_group_f1": worst["weighted_f1"],
        "worst_group_attribute": worst["attribute"],
        "worst_group_value": worst["subgroup"],
        "worst_group_n": worst["n"],
        "sensitivity": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "specificity": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    return metrics, group_rows


def metric_text(value: object) -> str:
    return f"{float(value):.4f}"


def write_report(out_dir: Path, results: list[dict[str, Any]]) -> None:
    lines = [
        "# FairVision Locked-Slice Foundation Results",
        "",
        "Yusra-compatible reporting on the locked 250-case AMD and DR slices. "
        "F1 and subgroup F1 are support-weighted.",
        "",
    ]
    for task in TASK_DISPLAY:
        task_rows = [row for row in results if row["task"] == task]
        if not task_rows:
            continue
        lines.extend(
            [
                f"## {TASK_DISPLAY[task]}",
                "",
                "| Model | Modality | F1 | Worst-group F1 | Sensitivity | Specificity | Balanced accuracy |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in task_rows:
            lines.append(
                f"| {row['model']} | {row['modality']} | {metric_text(row['weighted_f1'])} | "
                f"{metric_text(row['worst_group_f1'])} | {metric_text(row['sensitivity'])} | "
                f"{metric_text(row['specificity'])} | {metric_text(row['balanced_accuracy'])} |"
            )
        lines.extend(["", "LaTeX rows:", "", "```latex"])
        for row in task_rows:
            lines.append(
                f"{row['model']} & {row['modality']} & {metric_text(row['weighted_f1'])} & "
                f"{metric_text(row['worst_group_f1'])} & {metric_text(row['sensitivity'])} & "
                f"{metric_text(row['specificity'])} & {metric_text(row['balanced_accuracy'])} \\\\"
            )
        lines.extend(["```", ""])
    (out_dir / "fairvision_yusra_foundation_results.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    locked = locked_index(args.locked_csv, args.tasks, args.expected_cases_per_task)
    locked_hash = sha256(args.locked_csv)
    results: list[dict[str, Any]] = []
    subgroup_rows: list[dict[str, Any]] = []

    for task in args.tasks:
        for model_key in args.models:
            path = prediction_path(args.predictions_root, task, model_key)
            aligned = align_predictions(path, task, locked, args.expected_cases_per_task)
            metrics, groups = result_metrics(aligned)
            model, modality = MODEL_SPECS[model_key]
            threshold, threshold_metric = threshold_metadata(args.metrics_root, task, model_key)
            result = {
                "task": task,
                "model_key": model_key,
                "model": model,
                "modality": modality,
                **metrics,
                "threshold": threshold,
                "threshold_metric": threshold_metric,
                "prediction_csv": str(path),
                "prediction_sha256": sha256(path),
                "locked_csv": str(args.locked_csv),
                "locked_csv_sha256": locked_hash,
            }
            results.append(result)
            subgroup_rows.extend(
                {"task": task, "model_key": model_key, "model": model, **group}
                for group in groups
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "fairvision_yusra_foundation_results.csv", results, RESULT_FIELDS)
    write_csv(
        args.out_dir / "fairvision_yusra_foundation_subgroups.csv",
        subgroup_rows,
        ["task", "model_key", "model", "attribute", "subgroup", "n", "negative", "positive", "weighted_f1"],
    )
    write_report(args.out_dir, results)
    provenance = {
        "locked_csv": str(args.locked_csv),
        "locked_csv_sha256": locked_hash,
        "expected_cases_per_task": args.expected_cases_per_task,
        "tasks": args.tasks,
        "models": args.models,
        "reporting": {
            "f1": "support-weighted F1",
            "worst_group_f1": "minimum support-weighted F1 over age_group, gender, and race",
            "age_groups": {"young": "age < 50", "middle": "50 <= age < 70", "older": "age >= 70"},
            "threshold": "read from validation-threshold metadata; never selected by this collector",
        },
    }
    (args.out_dir / "protocol.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print((args.out_dir / "fairvision_yusra_foundation_results.md").read_text(encoding="utf-8"))
    print(f"wrote={args.out_dir}")


if __name__ == "__main__":
    main()
