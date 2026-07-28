#!/usr/bin/env python3
"""Collect complete PAPILA and GAMMA glaucoma results into manuscript rows."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


MODEL_NAMES = {
    "retfound": "RETFound",
    "mirage": "MIRAGE",
    "ret_clip": "RET-CLIP",
    "retizero": "RetiZero",
    "urfound": "UrFound",
}
FOUNDATION_MODELS = tuple(MODEL_NAMES)
GAMMA_CFP_MODELS = ("mirage", "ret_clip", "retizero", "urfound")
GENDER_MAP = {"0": "male", "male": "male", "1": "female", "female": "female"}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def arguments() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--papila-manifest",
        type=Path,
        default=root / "OphthalmicAgent" / "data_papila" / "manifest.csv",
    )
    parser.add_argument(
        "--papila-foundations",
        type=Path,
        default=(
            root
            / "equi-agent"
            / "outputs"
            / "benchmarks"
            / "papila_glaucoma_foundations_oof_v2"
        ),
    )
    parser.add_argument(
        "--papila-agent",
        type=Path,
        default=(
            root
            / "OphthalmicAgent"
            / "outputs"
            / "papila"
            / "agentic_retfound_oof_v2"
            / "predictions.csv"
        ),
    )
    parser.add_argument(
        "--gamma-manifest",
        type=Path,
        default=root / "OphthalmicAgent" / "data_gamma" / "manifest.csv",
    )
    parser.add_argument(
        "--gamma-foundations",
        type=Path,
        default=(
            root
            / "equi-agent"
            / "outputs"
            / "benchmarks"
            / "gamma_glaucoma_foundations_oof_v1"
        ),
    )
    parser.add_argument(
        "--gamma-retfound",
        type=Path,
        default=(
            root
            / "OphthalmicAgent"
            / "outputs"
            / "gamma"
            / "retfound_oct_fairvision_init_v1"
            / "summary.json"
        ),
    )
    parser.add_argument(
        "--gamma-agent",
        type=Path,
        default=(
            root
            / "OphthalmicAgent"
            / "outputs"
            / "gamma"
            / "agentic_retfound_oct_v1"
            / "predictions.csv"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=(
            root
            / "equi-agent"
            / "outputs"
            / "benchmarks"
            / "papila_gamma_glaucoma_manuscript"
        ),
    )
    parser.add_argument("--min-subgroup-n", type=int, default=10)
    parser.add_argument("--min-subgroup-positive", type=int, default=2)
    parser.add_argument("--min-subgroup-negative", type=int, default=2)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def confusion_metrics(labels: list[int], predictions: list[int]) -> dict[str, Any]:
    if len(labels) != len(predictions) or not labels:
        raise ValueError("Labels and predictions must be non-empty and aligned")
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    return {
        "n": len(labels),
        "positive_n": sum(labels),
        "negative_n": len(labels) - sum(labels),
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": (
            (sensitivity + specificity) / 2
            if sensitivity is not None and specificity is not None
            else None
        ),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def latest_valid_agent_rows(path: Path, split: str = "test") -> dict[str, dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(path):
        if str(row.get("split", "")).strip().lower() != split:
            continue
        case_id = str(row.get("case_id", "")).strip()
        if case_id:
            grouped[case_id].append(row)

    selected = {}
    for case_id, attempts in grouped.items():
        valid = [row for row in attempts if str(row.get("Pred_GL", "")).strip() in {"0", "1"}]
        if valid:
            selected[case_id] = valid[-1]
    return selected


def locked_test_manifest(path: Path, dataset: str) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rows = [
        row
        for row in read_csv(path)
        if str(row.get("dataset", dataset)).strip().lower() == dataset
        and str(row.get("split", "")).strip().lower() == "test"
        and str(row.get("label", "")).strip() in {"0", "1"}
    ]
    train = [
        row
        for row in read_csv(path)
        if str(row.get("dataset", dataset)).strip().lower() == dataset
        and str(row.get("split", "")).strip().lower() == "train"
        and str(row.get("label", "")).strip() in {"0", "1"}
    ]
    if not rows:
        raise ValueError(f"No locked test rows in {path}")
    return train, rows


def complete_agent_metrics(
    manifest: Path,
    predictions: Path,
    dataset: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, str]]]:
    train_rows, test_rows = locked_test_manifest(manifest, dataset)
    predicted = latest_valid_agent_rows(predictions)
    expected_ids = [str(row.get("case_id", "")).strip() for row in test_rows]
    missing = [case_id for case_id in expected_ids if case_id not in predicted]
    if missing:
        raise ValueError(
            f"{dataset} agent is incomplete: {len(missing)}/{len(test_rows)} locked test cases "
            f"lack a valid latest prediction: {missing[:10]}"
        )

    aligned = []
    for row in test_rows:
        case_id = str(row["case_id"]).strip()
        agent_row = predicted[case_id]
        y_true = int(row["label"])
        recorded_truth = int(float(agent_row["Ground_Truth"]))
        if y_true != recorded_truth:
            raise ValueError(f"Ground-truth mismatch for {dataset}:{case_id}")
        aligned.append(
            {
                **row,
                "y_true": y_true,
                "y_pred": int(agent_row["Pred_GL"]),
            }
        )
    metrics = confusion_metrics(
        [row["y_true"] for row in aligned],
        [row["y_pred"] for row in aligned],
    )
    return metrics, aligned, train_rows


def papila_worst_group_f1(
    metrics_rows: list[dict[str, Any]],
    train_rows: list[dict[str, str]],
    args: argparse.Namespace,
) -> tuple[float | None, list[dict[str, Any]]]:
    import numpy as np

    train_ages = [
        value
        for row in train_rows
        if (value := optional_float(row.get("age"))) is not None
    ]
    if len(train_ages) < 3:
        raise ValueError("PAPILA training ages are unavailable")
    lower, upper = [float(value) for value in np.quantile(train_ages, [1 / 3, 2 / 3])]

    for row in metrics_rows:
        gender = str(row.get("gender_code", "")).strip().lower()
        if gender not in GENDER_MAP:
            raise ValueError(f"Unknown PAPILA gender code: {gender!r}")
        row["sex_gender"] = GENDER_MAP[gender]
        age = optional_float(row.get("age"))
        if age is None:
            row["age_group"] = "unknown"
        elif age <= lower:
            row["age_group"] = f"age_le_{lower:g}"
        elif age <= upper:
            row["age_group"] = f"age_{lower:g}_to_{upper:g}"
        else:
            row["age_group"] = f"age_gt_{upper:g}"

    groups = []
    for attribute in ("sex_gender", "age_group"):
        for value in sorted({str(row[attribute]) for row in metrics_rows}):
            selected = [row for row in metrics_rows if row[attribute] == value]
            metrics = confusion_metrics(
                [row["y_true"] for row in selected],
                [row["y_pred"] for row in selected],
            )
            eligible = (
                metrics["n"] >= args.min_subgroup_n
                and metrics["positive_n"] >= args.min_subgroup_positive
                and metrics["negative_n"] >= args.min_subgroup_negative
            )
            groups.append(
                {
                    "attribute": attribute,
                    "group": value,
                    "eligible_for_worst_group": eligible,
                    **metrics,
                }
            )
    eligible = [
        float(row["f1"])
        for row in groups
        if row["eligible_for_worst_group"] and row["f1"] is not None
    ]
    return min(eligible) if eligible else None, groups


def summary_result(
    summary_path: Path,
    dataset: str,
    model_key: str,
    modality: str,
) -> dict[str, Any]:
    summary = read_json(summary_path)
    summary_dataset = str(summary.get("dataset", dataset)).strip().lower()
    if summary_dataset != dataset:
        raise ValueError(f"Dataset mismatch in {summary_path}: {summary_dataset}")
    metrics = summary.get("metrics", {}).get("test", {})
    if not isinstance(metrics, dict):
        raise ValueError(f"Missing test metrics in {summary_path}")
    return {
        "dataset": dataset,
        "model": MODEL_NAMES[model_key],
        "modality": modality,
        "n": metrics.get("n", summary.get("rows", {}).get("test")),
        "f1": metrics.get("f1"),
        "worst_group_f1": metrics.get("worst_group_f1"),
        "sensitivity": metrics.get("sensitivity"),
        "specificity": metrics.get("specificity"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "auroc": metrics.get("auroc"),
        "source": str(summary_path),
    }


def agent_result(
    dataset: str,
    modality: str,
    metrics: dict[str, Any],
    worst_group_f1: float | None,
    source: Path,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "model": "Ours",
        "modality": modality,
        "n": metrics["n"],
        "f1": metrics["f1"],
        "worst_group_f1": worst_group_f1,
        "sensitivity": metrics["sensitivity"],
        "specificity": metrics["specificity"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "auroc": None,
        "source": str(source),
    }


def format_metric(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.4f}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(out_dir: Path, rows: list[dict[str, Any]], subgroup_rows: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "papila_gamma_glaucoma_results.csv", rows)
    write_csv(out_dir / "papila_ours_subgroup_metrics.csv", subgroup_rows)

    lines = ["# PAPILA and GAMMA Glaucoma Results", ""]
    latex = []
    for dataset in ("papila", "gamma"):
        lines.extend(
            [
                f"## {dataset.upper()}",
                "",
                "| Model | Modality | F1 | Worst-group F1 | Sensitivity | Specificity | Balanced accuracy |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in [item for item in rows if item["dataset"] == dataset]:
            values = [
                format_metric(row[key])
                for key in (
                    "f1",
                    "worst_group_f1",
                    "sensitivity",
                    "specificity",
                    "balanced_accuracy",
                )
            ]
            lines.append(
                f"| {row['model']} | {row['modality']} | " + " | ".join(values) + " |"
            )
            latex.append(
                f"{row['model']} & {row['modality']} & "
                + " & ".join("--" if value == "N/A" else value for value in values)
                + r" \\"
            )
        lines.append("")
        latex.append("")

    (out_dir / "papila_gamma_glaucoma_results.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    (out_dir / "papila_gamma_glaucoma_rows.tex").write_text(
        "\n".join(latex) + "\n",
        encoding="utf-8",
    )
    print("\n".join(lines))


def main() -> None:
    args = arguments()
    rows = [
        summary_result(
            args.papila_foundations / model / "summary.json",
            "papila",
            model,
            "CFP",
        )
        for model in FOUNDATION_MODELS
    ]
    papila_metrics, papila_aligned, papila_train = complete_agent_metrics(
        args.papila_manifest,
        args.papila_agent,
        "papila",
    )
    papila_worst, papila_subgroups = papila_worst_group_f1(
        papila_aligned,
        papila_train,
        args,
    )
    rows.append(
        agent_result(
            "papila",
            "CFP + evidence arbitration",
            papila_metrics,
            papila_worst,
            args.papila_agent,
        )
    )

    rows.append(
        summary_result(
            args.gamma_retfound,
            "gamma",
            "retfound",
            "OCT",
        )
    )
    rows.extend(
        summary_result(
            args.gamma_foundations / model / "summary.json",
            "gamma",
            model,
            "CFP",
        )
        for model in GAMMA_CFP_MODELS
    )
    gamma_metrics, _, _ = complete_agent_metrics(
        args.gamma_manifest,
        args.gamma_agent,
        "gamma",
    )
    rows.append(
        agent_result(
            "gamma",
            "OCT + CFP evidence arbitration",
            gamma_metrics,
            None,
            args.gamma_agent,
        )
    )
    write_outputs(args.out_dir, rows, papila_subgroups)


if __name__ == "__main__":
    main()
