#!/usr/bin/env python3
"""Collect complete-cohort REFUGE glaucoma results for the manuscript."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


MODEL_LABELS = {
    "retfound": "RETFound",
    "ret_clip": "RET-CLIP",
    "retizero": "RetiZero",
    "urfound": "URFound",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "OphthalmicAgent" / "data_refuge" / "manifest.csv",
    )
    parser.add_argument(
        "--foundations-dir",
        type=Path,
        default=(
            root
            / "equi-agent"
            / "outputs"
            / "benchmarks"
            / "refuge_glaucoma_foundations_v1"
        ),
    )
    parser.add_argument(
        "--agent-predictions",
        type=Path,
        default=(
            root
            / "OphthalmicAgent"
            / "outputs"
            / "refuge"
            / "agentic_retfound_cfp_v1"
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
            / "refuge_glaucoma_manuscript"
        ),
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def confusion(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("Metric inputs must be nonempty and aligned")
    tn = sum(truth == 0 and pred == 0 for truth, pred in zip(y_true, y_pred))
    fp = sum(truth == 0 and pred == 1 for truth, pred in zip(y_true, y_pred))
    fn = sum(truth == 1 and pred == 0 for truth, pred in zip(y_true, y_pred))
    tp = sum(truth == 1 and pred == 1 for truth, pred in zip(y_true, y_pred))
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    return {
        "n": len(y_true),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": (
            (sensitivity + specificity) / 2
            if sensitivity is not None and specificity is not None
            else None
        ),
    }


def latest_valid_agent_rows(path: Path) -> dict[str, dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(path):
        if str(row.get("split", "")).strip().lower() != "test":
            continue
        grouped[str(row.get("case_id", "")).strip()].append(row)
    selected = {}
    for case_id, attempts in grouped.items():
        valid = [
            row
            for row in attempts
            if str(row.get("Pred_GL", "")).strip() in {"0", "1"}
        ]
        if valid:
            selected[case_id] = valid[-1]
    return selected


def metric_text(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.4f}"


def main() -> None:
    args = parse_args()
    manifest_test = {
        row["case_id"]: row
        for row in read_csv(args.manifest)
        if row.get("dataset", "").lower() == "refuge"
        and row.get("split", "").lower() == "test"
    }
    if len(manifest_test) != 400:
        raise ValueError(
            f"Expected 400 locked REFUGE test cases, found {len(manifest_test)}"
        )

    rows: list[dict[str, Any]] = []
    for model_name, display_name in MODEL_LABELS.items():
        path = args.foundations_dir / model_name / "summary.json"
        if not path.exists():
            raise FileNotFoundError(path)
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("dataset") != "refuge":
            raise ValueError(f"Wrong dataset in {path}: {summary.get('dataset')!r}")
        metrics = summary["metrics"]["test"]
        if int(metrics.get("n", 0)) != 400:
            raise ValueError(f"Incomplete {display_name} test metrics in {path}")
        rows.append(
            {
                "model": display_name,
                "modality": "CFP",
                "worst_group_f1": None,
                **{
                    key: metrics.get(key)
                    for key in (
                        "f1",
                        "sensitivity",
                        "specificity",
                        "balanced_accuracy",
                    )
                },
            }
        )

    agent_by_case = latest_valid_agent_rows(args.agent_predictions)
    missing = sorted(set(manifest_test) - set(agent_by_case))
    extra = sorted(set(agent_by_case) - set(manifest_test))
    if missing or extra:
        raise ValueError(
            "Agent predictions must cover the exact 400-case test cohort; "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )
    ordered_ids = sorted(manifest_test)
    truth = [int(manifest_test[case_id]["label"]) for case_id in ordered_ids]
    agent_truth = [
        int(float(agent_by_case[case_id]["Ground_Truth"]))
        for case_id in ordered_ids
    ]
    if truth != agent_truth:
        raise ValueError("Agent and manifest ground truths do not align")
    predictions = [
        int(agent_by_case[case_id]["Pred_GL"]) for case_id in ordered_ids
    ]
    agent_metrics = confusion(truth, predictions)
    rows.append(
        {
            "model": "Ours",
            "modality": "CFP + evidence arbitration",
            "worst_group_f1": None,
            **{
                key: agent_metrics[key]
                for key in ("f1", "sensitivity", "specificity", "balanced_accuracy")
            },
        }
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "refuge_glaucoma_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# REFUGE Glaucoma Results",
        "",
        "Locked official 400-image test cohort. Official Train fits probes and "
        "official Validation selects the probe and threshold. Worst-group F1 "
        "is unavailable because demographic metadata are not supplied.",
        "",
        "| Model | Modality | F1 | Worst-group F1 | Sensitivity | Specificity | Balanced accuracy |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['modality']} | {metric_text(row['f1'])} | "
            f"{metric_text(row['worst_group_f1'])} | "
            f"{metric_text(row['sensitivity'])} | "
            f"{metric_text(row['specificity'])} | "
            f"{metric_text(row['balanced_accuracy'])} |"
        )
    markdown = "\n".join(lines) + "\n"
    (args.out_dir / "refuge_glaucoma_results.md").write_text(
        markdown,
        encoding="utf-8",
    )
    print(markdown)


if __name__ == "__main__":
    main()
