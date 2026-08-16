#!/usr/bin/env python3
"""Create reproducible, performance-matched FairVision doctor manifests.

For each of glaucoma, AMD, and DR, this script creates seven independently
sampled 50-case manifests. Each manifest preserves the source ground-truth
class ratio as closely as integer sample counts allow and minimizes the
absolute difference between its weighted F1 and the filtered source weighted
F1. Cases do not repeat within a manifest; overlap across manifests is allowed.

Doctor-facing ``manifest.csv`` files are blinded. Full selected source rows and
selection metrics are written under ``_audit`` and should not be sent to the
reviewers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


DISEASE_CONFIG = {
    "glaucoma": {
        "prediction_candidates": ("Pred_GL", "pred_gl", "prediction", "predicted_label"),
        "default_source": "fairvision_glaucoma_agentic.csv",
    },
    "amd": {
        "prediction_candidates": ("Pred_AMD", "pred_amd", "prediction", "predicted_label"),
        "default_source": "fairvision_amd_agentic.csv",
    },
    "dr": {
        "prediction_candidates": ("Pred_DR", "pred_dr", "prediction", "predicted_label"),
        "default_source": "fairvision_dr_agentic.csv",
    },
}
TRUTH_CANDIDATES = ("Ground_Truth", "ground_truth", "groundtruth", "label", "y_true")
FILE_CANDIDATES = (
    "Filename", "filename", "Filepath", "filepath", "file_path", "path", "npz_path"
)
CASE_CANDIDATES = ("case_id", "Case_ID", "Patient_ID", "patient_id", "Filename", "filename")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 7 x 50 performance-matched FairVision manifests per disease."
    )
    parser.add_argument(
        "--glaucoma-csv", type=Path,
        default=Path(DISEASE_CONFIG["glaucoma"]["default_source"]),
    )
    parser.add_argument(
        "--amd-csv", type=Path,
        default=Path(DISEASE_CONFIG["amd"]["default_source"]),
    )
    parser.add_argument(
        "--dr-csv", type=Path,
        default=Path(DISEASE_CONFIG["dr"]["default_source"]),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/fairvision_human_evaluation_manifests"),
    )
    parser.add_argument("--manifests-per-disease", type=int, default=7)
    parser.add_argument("--cases-per-manifest", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def find_column(columns, candidates, *, required=True):
    exact = {str(column): column for column in columns}
    folded = {str(column).strip().casefold(): column for column in columns}
    for candidate in candidates:
        if candidate in exact:
            return exact[candidate]
        match = folded.get(candidate.strip().casefold())
        if match is not None:
            return match
    if required:
        raise KeyError(f"Expected one of {list(candidates)}; found {list(columns)}")
    return None


def binary_value(value):
    if pd.isna(value):
        return np.nan
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return np.nan
    return int(number) if number in (0.0, 1.0) else np.nan


def confusion_counts(truth, prediction):
    truth = np.asarray(truth, dtype=int)
    prediction = np.asarray(prediction, dtype=int)
    return {
        "tn": int(np.sum((truth == 0) & (prediction == 0))),
        "fp": int(np.sum((truth == 0) & (prediction == 1))),
        "fn": int(np.sum((truth == 1) & (prediction == 0))),
        "tp": int(np.sum((truth == 1) & (prediction == 1))),
    }


def prepare_source(path: Path, disease: str):
    if not path.is_file():
        raise FileNotFoundError(f"Missing {disease} source CSV: {path}")
    raw = pd.read_csv(path)
    if raw.empty:
        raise ValueError(f"Source CSV is empty: {path}")

    last_column = raw.columns[-1]
    dash_mask = pd.Series(False, index=raw.index)
    if disease == "dr":
        # User-specified exclusion: the final DR column contains '-' for rows
        # that must not enter source metrics or any doctor manifest.
        dash_mask = raw[last_column].astype(str).str.strip().eq("-")

    truth_column = find_column(raw.columns, TRUTH_CANDIDATES)
    prediction_column = find_column(
        raw.columns, DISEASE_CONFIG[disease]["prediction_candidates"]
    )
    frame = raw.loc[~dash_mask].copy()
    frame["_source_row"] = frame.index.astype(int)
    frame["_truth"] = frame[truth_column].map(binary_value)
    frame["_prediction"] = frame[prediction_column].map(binary_value)
    invalid_mask = frame[["_truth", "_prediction"]].isna().any(axis=1)
    invalid_count = int(invalid_mask.sum())
    frame = frame.loc[~invalid_mask].copy()
    frame[["_truth", "_prediction"]] = frame[["_truth", "_prediction"]].astype(int)

    if frame.empty or frame["_truth"].nunique() != 2:
        raise ValueError(
            f"{disease}: filtered source must contain valid rows from both ground-truth classes"
        )
    source_f1 = float(f1_score(
        frame["_truth"], frame["_prediction"], average="weighted", zero_division=0
    ))
    source_counts = confusion_counts(frame["_truth"], frame["_prediction"])
    quality = {
        "source_csv": str(path),
        "raw_rows": int(len(raw)),
        "last_column": str(last_column),
        "excluded_last_column_dash_rows": int(dash_mask.sum()),
        "excluded_invalid_truth_or_prediction_rows": invalid_count,
        "eligible_rows": int(len(frame)),
        "truth_column": str(truth_column),
        "prediction_column": str(prediction_column),
        "source_positive_count": int(frame["_truth"].sum()),
        "source_negative_count": int((frame["_truth"] == 0).sum()),
        "source_positive_ratio": float(frame["_truth"].mean()),
        "source_weighted_f1": source_f1,
        **{f"source_{key}": value for key, value in source_counts.items()},
    }
    return frame, quality


def choose_composition(frame, sample_size, source_f1):
    """Find feasible TN/FP/FN/TP counts with closest weighted F1."""
    positive_target = int(round(sample_size * float(frame["_truth"].mean())))
    positive_target = max(1, min(sample_size - 1, positive_target))
    negative_target = sample_size - positive_target
    available = confusion_counts(frame["_truth"], frame["_prediction"])
    source_rates = {
        "tpr": available["tp"] / (available["tp"] + available["fn"]),
        "tnr": available["tn"] / (available["tn"] + available["fp"]),
    }
    candidates = []
    for tp in range(positive_target + 1):
        fn = positive_target - tp
        if tp > available["tp"] or fn > available["fn"]:
            continue
        for tn in range(negative_target + 1):
            fp = negative_target - tn
            if tn > available["tn"] or fp > available["fp"]:
                continue
            truth = np.asarray([1] * positive_target + [0] * negative_target)
            prediction = np.asarray([1] * tp + [0] * fn + [0] * tn + [1] * fp)
            weighted_f1 = float(f1_score(
                truth, prediction, average="weighted", zero_division=0
            ))
            rate_distance = (
                abs(tp / positive_target - source_rates["tpr"])
                + abs(tn / negative_target - source_rates["tnr"])
            )
            candidates.append({
                "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                "weighted_f1": weighted_f1,
                "absolute_f1_difference": abs(weighted_f1 - source_f1),
                "rate_distance": rate_distance,
            })
    if not candidates:
        raise ValueError(
            f"No feasible {sample_size}-case confusion composition at the target class ratio"
        )
    return min(
        candidates,
        key=lambda item: (
            item["absolute_f1_difference"], item["rate_distance"],
            -item["tp"], -item["tn"],
        ),
    )


def sample_manifest(frame, composition, seed):
    rng = np.random.default_rng(seed)
    pieces = []
    strata = (
        (0, 0, "tn"), (0, 1, "fp"), (1, 0, "fn"), (1, 1, "tp")
    )
    for truth, prediction, key in strata:
        count = composition[key]
        if count == 0:
            continue
        pool = frame[(frame["_truth"] == truth) & (frame["_prediction"] == prediction)]
        selected_positions = rng.choice(len(pool), size=count, replace=False)
        pieces.append(pool.iloc[selected_positions])
    selected = pd.concat(pieces, ignore_index=False)
    selected = selected.iloc[rng.permutation(len(selected))].copy().reset_index(drop=True)
    selected["_review_order"] = np.arange(1, len(selected) + 1)
    if selected["_source_row"].duplicated().any():
        raise AssertionError("A case was repeated within one manifest")
    return selected


def doctor_manifest(selected, disease, manifest_number):
    file_column = find_column(selected.columns, FILE_CANDIDATES)
    case_column = find_column(selected.columns, CASE_CANDIDATES, required=False)
    filenames = selected[file_column].astype(str)
    if case_column is None:
        case_ids = filenames.map(lambda value: Path(value).stem)
    else:
        case_ids = selected[case_column].astype(str)
    return pd.DataFrame({
        "manifest_id": f"fairvision_{disease}_{manifest_number:02d}",
        "dataset": "fairvision",
        "disease": disease,
        "review_order": selected["_review_order"].astype(int),
        "case_id": case_ids,
        "filename": filenames,
        "human_diagnosis": "",
        "human_confidence": "",
        "reviewer_notes": "",
    })


def write_disease_manifests(
    disease, source_path, output_root, manifest_count, sample_size, base_seed
):
    frame, source_quality = prepare_source(source_path, disease)
    if len(frame) < sample_size:
        raise ValueError(
            f"{disease}: only {len(frame)} eligible rows, fewer than requested {sample_size}"
        )
    composition = choose_composition(
        frame, sample_size, source_quality["source_weighted_f1"]
    )
    disease_root = output_root / disease
    audit_root = disease_root / "_audit"
    audit_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    selected_sets = []

    for manifest_number in range(1, manifest_count + 1):
        seed = base_seed + {"glaucoma": 1000, "amd": 2000, "dr": 3000}[disease] + manifest_number
        selected = sample_manifest(frame, composition, seed)
        manifest_dir = disease_root / f"manifest_{manifest_number:02d}"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        doctor = doctor_manifest(selected, disease, manifest_number)
        doctor.to_csv(manifest_dir / "manifest.csv", index=False)

        audit = selected.drop(columns=["_review_order"], errors="ignore").copy()
        audit.insert(0, "manifest_id", f"fairvision_{disease}_{manifest_number:02d}")
        audit.to_csv(audit_root / f"manifest_{manifest_number:02d}_selected_rows.csv", index=False)
        weighted_f1 = float(f1_score(
            selected["_truth"], selected["_prediction"],
            average="weighted", zero_division=0,
        ))
        counts = confusion_counts(selected["_truth"], selected["_prediction"])
        selected_set = set(selected["_source_row"].astype(int))
        selected_sets.append(selected_set)
        summaries.append({
            "disease": disease,
            "manifest_number": manifest_number,
            "manifest_id": f"fairvision_{disease}_{manifest_number:02d}",
            "seed": seed,
            "cases": int(len(selected)),
            "unique_cases": int(selected["_source_row"].nunique()),
            "positive_count": int(selected["_truth"].sum()),
            "negative_count": int((selected["_truth"] == 0).sum()),
            "positive_ratio": float(selected["_truth"].mean()),
            "weighted_f1": weighted_f1,
            "source_weighted_f1": source_quality["source_weighted_f1"],
            "absolute_f1_difference": abs(weighted_f1 - source_quality["source_weighted_f1"]),
            **counts,
        })

    overlap = []
    for left in range(manifest_count):
        for right in range(left + 1, manifest_count):
            overlap.append({
                "disease": disease,
                "manifest_a": left + 1,
                "manifest_b": right + 1,
                "shared_cases": len(selected_sets[left] & selected_sets[right]),
            })
    pd.DataFrame(summaries).to_csv(audit_root / "manifest_metrics.csv", index=False)
    pd.DataFrame(overlap).to_csv(audit_root / "pairwise_overlap.csv", index=False)
    (audit_root / "source_quality.json").write_text(
        json.dumps(source_quality, indent=2) + "\n", encoding="utf-8"
    )
    return source_quality, summaries


def validate_outputs(output_root, all_summaries, manifest_count, sample_size):
    expected = len(DISEASE_CONFIG) * manifest_count
    if len(all_summaries) != expected:
        raise AssertionError(f"Expected {expected} summary rows, got {len(all_summaries)}")
    for summary in all_summaries:
        if summary["cases"] != sample_size or summary["unique_cases"] != sample_size:
            raise AssertionError(f"Invalid manifest size/uniqueness: {summary}")
        if summary["positive_count"] == 0 or summary["negative_count"] == 0:
            raise AssertionError(f"Manifest lacks a class: {summary}")
        path = (
            output_root / summary["disease"]
            / f"manifest_{summary['manifest_number']:02d}" / "manifest.csv"
        )
        manifest = pd.read_csv(path)
        forbidden = {"Ground_Truth", "Pred_GL", "Pred_AMD", "Pred_DR", "_truth", "_prediction"}
        if forbidden.intersection(manifest.columns):
            raise AssertionError(f"Doctor manifest is not blinded: {path}")
        if len(manifest) != sample_size or manifest["case_id"].duplicated().any():
            raise AssertionError(f"Doctor manifest failed row/duplicate QA: {path}")


def main():
    args = parse_args()
    if args.manifests_per_disease <= 0 or args.cases_per_manifest <= 1:
        raise ValueError("Manifest count must be positive and sample size must exceed 1")
    source_paths = {
        "glaucoma": args.glaucoma_csv,
        "amd": args.amd_csv,
        "dr": args.dr_csv,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_summaries = []
    source_summaries = {}
    for disease, source_path in source_paths.items():
        source_quality, summaries = write_disease_manifests(
            disease, source_path, args.output_dir,
            args.manifests_per_disease, args.cases_per_manifest, args.seed,
        )
        source_summaries[disease] = source_quality
        all_summaries.extend(summaries)
        print(
            f"{disease}: source weighted F1={source_quality['source_weighted_f1']:.6f}, "
            f"eligible={source_quality['eligible_rows']}, "
            f"excluded '-'={source_quality['excluded_last_column_dash_rows']}"
        )

    validate_outputs(
        args.output_dir, all_summaries,
        args.manifests_per_disease, args.cases_per_manifest,
    )
    summary_frame = pd.DataFrame(all_summaries)
    summary_frame.to_csv(args.output_dir / "all_manifest_metrics.csv", index=False)
    run_summary = {
        "seed": args.seed,
        "manifests_per_disease": args.manifests_per_disease,
        "cases_per_manifest": args.cases_per_manifest,
        "total_manifests": len(all_summaries),
        "sources": source_summaries,
        "maximum_absolute_weighted_f1_difference": float(
            summary_frame["absolute_f1_difference"].max()
        ),
        "doctor_manifests_blinded": True,
        "audit_files_not_for_reviewers": True,
    }
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Created and validated {len(all_summaries)} manifests under {args.output_dir}")
    print(
        "Maximum absolute weighted-F1 difference: "
        f"{run_summary['maximum_absolute_weighted_f1_difference']:.6f}"
    )


if __name__ == "__main__":
    main()
