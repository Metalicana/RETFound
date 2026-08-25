#!/usr/bin/env python3
"""Create a mixed FairVision master manifest and four reviewer manifests.

Inputs are the three actual disease-specific agentic result CSVs. The script:

1. Selects 20 cases per disease whose weighted F1 is as close as possible to
   that disease CSV's weighted F1 while preserving its rounded class ratio.
2. Requires both positive and negative cases in every 20-case disease sample.
3. Randomly partitions those 60 cases, without overlap, into four manifests.
4. Writes 5 glaucoma + 5 AMD + 5 DR cases to every reviewer manifest.

Public manifests are blinded. Ground truth and model predictions are retained
only under ``_private_audit``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

try:
    from .generate_fairvision_human_manifests import (
        CASE_CANDIDATES,
        FILE_CANDIDATES,
        choose_composition,
        confusion_counts,
        find_column,
        prepare_source,
        sample_manifest,
    )
except ImportError:  # Support: python scripts/generate_fairvision_mixed_manifests.py
    from generate_fairvision_human_manifests import (
        CASE_CANDIDATES,
        FILE_CANDIDATES,
        choose_composition,
        confusion_counts,
        find_column,
        prepare_source,
        sample_manifest,
    )


DISEASES = ("glaucoma", "amd", "dr")
DEMOGRAPHIC_CANDIDATES = {
    "age": ("Age", "age"),
    "gender": ("Gender", "gender", "sex", "Sex"),
    "race": ("Race", "race"),
    "ethnicity": ("Ethnicity", "ethnicity"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--glaucoma-csv", type=Path, default=Path("fairvision_glaucoma_agentic.csv")
    )
    parser.add_argument(
        "--amd-csv", type=Path, default=Path("fairvision_amd_agentic.csv")
    )
    parser.add_argument(
        "--dr-csv", type=Path, default=Path("fairvision_dr_agentic.csv")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/fairvision_mixed_human_manifests"),
    )
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def select_demographic_columns(frame: pd.DataFrame) -> dict[str, str]:
    return {
        output_name: find_column(frame.columns, candidates)
        for output_name, candidates in DEMOGRAPHIC_CANDIDATES.items()
    }


def public_manifest(frame: pd.DataFrame, manifest_id: str) -> pd.DataFrame:
    file_column = find_column(frame.columns, FILE_CANDIDATES)
    case_column = find_column(frame.columns, CASE_CANDIDATES, required=False)
    demographics = select_demographic_columns(frame)
    filenames = frame[file_column].astype(str)
    if case_column is None or case_column == file_column:
        case_ids = filenames.map(lambda value: Path(value).stem)
    else:
        case_ids = frame[case_column].astype(str)

    case_ids = frame["disease"].astype(str) + ":" + case_ids
    return pd.DataFrame({
        "manifest_id": manifest_id,
        "dataset": "fairvision",
        "disease": frame["disease"].astype(str),
        "review_order": np.arange(1, len(frame) + 1),
        "case_id": case_ids,
        "filename": filenames,
        "age": frame[demographics["age"]],
        "gender": frame[demographics["gender"]],
        "race": frame[demographics["race"]],
        "ethnicity": frame[demographics["ethnicity"]],
        "human_diagnosis": "",
        "human_confidence": "",
        "reviewer_notes": "",
    })


def metric_row(frame: pd.DataFrame, manifest: str, disease: str) -> dict:
    subset = frame.loc[frame["disease"] == disease]
    counts = confusion_counts(subset["_truth"], subset["_prediction"])
    return {
        "manifest": manifest,
        "disease": disease,
        "cases": int(len(subset)),
        "positive_count": int(subset["_truth"].sum()),
        "negative_count": int((subset["_truth"] == 0).sum()),
        "weighted_f1": float(f1_score(
            subset["_truth"], subset["_prediction"],
            average="weighted", zero_division=0,
        )),
        **counts,
    }


def main() -> None:
    args = parse_args()
    source_paths = {
        "glaucoma": args.glaucoma_csv,
        "amd": args.amd_csv,
        "dr": args.dr_csv,
    }
    rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.output_dir / "_private_audit"
    private_dir.mkdir(parents=True, exist_ok=True)

    selected_by_disease = {}
    master_metrics = []
    source_summary = {}
    for disease, source_path in source_paths.items():
        frame, quality = prepare_source(source_path, disease)
        if len(frame) < 20:
            raise ValueError(f"{disease}: only {len(frame)} eligible cases; 20 are required")

        composition = choose_composition(frame, 20, quality["source_weighted_f1"])
        selected = sample_manifest(frame, composition, args.seed + len(selected_by_disease))
        selected.insert(0, "disease", disease)
        selected_by_disease[disease] = selected

        row = metric_row(selected, "master", disease)
        row.update({
            "source_csv": str(source_path),
            "source_cases": quality["eligible_rows"],
            "source_positive_count": quality["source_positive_count"],
            "source_negative_count": quality["source_negative_count"],
            "source_weighted_f1": quality["source_weighted_f1"],
            "absolute_f1_difference": abs(
                row["weighted_f1"] - quality["source_weighted_f1"]
            ),
        })
        master_metrics.append(row)
        source_summary[disease] = quality

    master = pd.concat(selected_by_disease.values(), ignore_index=True)
    master = master.iloc[rng.permutation(len(master))].reset_index(drop=True)
    public_manifest(master, "fairvision_mixed_master").to_csv(
        args.output_dir / "master_manifest.csv", index=False
    )
    master.to_csv(private_dir / "master_selected_rows.csv", index=False)

    assignments = {index: [] for index in range(4)}
    for selected in selected_by_disease.values():
        shuffled = selected.iloc[rng.permutation(len(selected))].reset_index(drop=True)
        for split_index in range(4):
            assignments[split_index].append(
                shuffled.iloc[split_index * 5:(split_index + 1) * 5]
            )

    split_metrics = []
    partition_keys = []
    for split_index, pieces in assignments.items():
        split = pd.concat(pieces, ignore_index=True)
        split = split.iloc[rng.permutation(len(split))].reset_index(drop=True)
        number = split_index + 1
        manifest_id = f"fairvision_mixed_{number:02d}"
        manifest_dir = args.output_dir / f"manifest_{number:02d}"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        public_manifest(split, manifest_id).to_csv(manifest_dir / "manifest.csv", index=False)
        split.to_csv(
            private_dir / f"manifest_{number:02d}_selected_rows.csv", index=False
        )
        partition_keys.extend(
            (disease, int(row))
            for disease, row in zip(split["disease"], split["_source_row"])
        )
        for disease in DISEASES:
            split_metrics.append(metric_row(split, manifest_id, disease))

    expected_counts = {"amd": 20, "dr": 20, "glaucoma": 20}
    if len(master) != 60 or master.groupby("disease").size().to_dict() != expected_counts:
        raise AssertionError("Master manifest must contain exactly 20 cases per disease")
    if any(
        row["positive_count"] == 0 or row["negative_count"] == 0
        for row in master_metrics
    ):
        raise AssertionError("Every disease in the master manifest must contain both classes")
    if len(partition_keys) != 60 or len(set(partition_keys)) != 60:
        raise AssertionError("The four manifests must partition the 60 cases without overlap")

    expected_split_counts = {"amd": 5, "dr": 5, "glaucoma": 5}
    forbidden = {
        "Ground_Truth", "ground_truth", "Pred_GL", "Pred_AMD", "Pred_DR",
        "_truth", "_prediction",
    }
    for number in range(1, 5):
        path = args.output_dir / f"manifest_{number:02d}" / "manifest.csv"
        manifest = pd.read_csv(path)
        if len(manifest) != 15 or manifest.groupby("disease").size().to_dict() != expected_split_counts:
            raise AssertionError(f"Invalid counts in {path}")
        if forbidden.intersection(manifest.columns):
            raise AssertionError(f"Public manifest is not blinded: {path}")

    pd.DataFrame(master_metrics).to_csv(
        private_dir / "master_selection_metrics.csv", index=False
    )
    pd.DataFrame(split_metrics).to_csv(private_dir / "split_metrics.csv", index=False)
    summary = {
        "seed": args.seed,
        "master_cases": 60,
        "master_cases_per_disease": 20,
        "reviewer_manifests": 4,
        "cases_per_reviewer_manifest": 15,
        "cases_per_disease_per_reviewer_manifest": 5,
        "reviewer_manifests_are_disjoint": True,
        "sources": source_summary,
        "private_audit_not_for_reviewers": True,
    }
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Created manifests under: {args.output_dir}")
    for row in master_metrics:
        print(
            f"{row['disease']}: source F1={row['source_weighted_f1']:.6f}, "
            f"selected F1={row['weighted_f1']:.6f}, "
            f"difference={row['absolute_f1_difference']:.6f}, "
            f"positive={row['positive_count']}, negative={row['negative_count']}"
        )


if __name__ == "__main__":
    main()
