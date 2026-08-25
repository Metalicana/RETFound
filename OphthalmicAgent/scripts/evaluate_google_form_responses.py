#!/usr/bin/env python3
"""Evaluate FairVision Google Form doctor responses against private truth.

Export the linked response Sheet as CSV, then run this script with the public
``form_cases.csv`` and private ``evaluation_key.csv`` created by
``prepare_google_form_pilot.py``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


DIAGNOSIS_TITLE = "What is your diagnosis for this case?"
CONFIDENCE_TITLE = "How confident are you in this diagnosis?"
QUALITY_TITLE = "Is the supplied imaging adequate for diagnosis?"
REVIEWER_CANDIDATES = ("Reviewer ID", "reviewer_id", "Reviewer_ID")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare Google Form doctor responses with private FairVision ground truth."
    )
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--form-cases", type=Path, required=True)
    parser.add_argument("--evaluation-key", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def base_header(value):
    # pandas renames duplicate CSV headers as Question, Question.1, Question.2, ...
    return re.sub(r"\.\d+$", "", str(value)).strip()


def find_column(columns, candidates):
    folded = {str(column).strip().casefold(): column for column in columns}
    for candidate in candidates:
        match = folded.get(candidate.casefold())
        if match is not None:
            return match
    raise KeyError(f"Expected one of {list(candidates)}; found {list(columns)}")


def repeated_columns(columns, title):
    return [column for column in columns if base_header(column).casefold() == title.casefold()]


def parse_binary_diagnosis(value):
    if pd.isna(value):
        return np.nan
    match = re.match(r"\s*([01])(?:\D|$)", str(value))
    return int(match.group(1)) if match else np.nan


def binary_metrics(truth, prediction):
    truth = np.asarray(truth, dtype=int)
    prediction = np.asarray(prediction, dtype=int)
    tn, fp, fn, tp = confusion_matrix(truth, prediction, labels=[0, 1]).ravel()
    return {
        "cases": int(len(truth)),
        "accuracy": float(accuracy_score(truth, prediction)),
        "precision_positive": float(precision_score(truth, prediction, zero_division=0)),
        "recall_sensitivity": float(recall_score(truth, prediction, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "f1_positive": float(f1_score(truth, prediction, zero_division=0)),
        "f1_weighted": float(f1_score(
            truth, prediction, average="weighted", zero_division=0
        )),
        "support_negative": int(np.sum(truth == 0)),
        "support_positive": int(np.sum(truth == 1)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def fleiss_kappa(long_frame):
    """Fleiss kappa for binary ratings, allowing varying raters per case."""
    ratings = []
    for _, group in long_frame.dropna(subset=["human_prediction"]).groupby("case_id"):
        counts = group["human_prediction"].astype(int).value_counts()
        n0, n1 = int(counts.get(0, 0)), int(counts.get(1, 0))
        if n0 + n1 >= 2:
            ratings.append((n0, n1))
    if not ratings:
        return None
    total_ratings = sum(n0 + n1 for n0, n1 in ratings)
    p0 = sum(n0 for n0, _ in ratings) / total_ratings
    p1 = sum(n1 for _, n1 in ratings) / total_ratings
    expected = p0 ** 2 + p1 ** 2
    observed_items = []
    for n0, n1 in ratings:
        n = n0 + n1
        observed_items.append((n0 * (n0 - 1) + n1 * (n1 - 1)) / (n * (n - 1)))
    observed = float(np.mean(observed_items))
    if expected >= 1:
        return None
    return float((observed - expected) / (1 - expected))


def validate_inputs(responses, form_cases, key):
    required_cases = {"review_order", "case_id"}
    required_key = {"review_order", "case_id", "ground_truth", "model_prediction"}
    if not required_cases.issubset(form_cases.columns):
        raise KeyError(f"form_cases.csv missing {sorted(required_cases - set(form_cases.columns))}")
    if not required_key.issubset(key.columns):
        raise KeyError(f"evaluation_key.csv missing {sorted(required_key - set(key.columns))}")
    if form_cases["case_id"].duplicated().any() or key["case_id"].duplicated().any():
        raise ValueError("case_id must be unique in form cases and evaluation key")
    if len(form_cases) != len(key):
        raise ValueError(f"Case/key row mismatch: {len(form_cases)} versus {len(key)}")
    merged = form_cases[["review_order", "case_id"]].merge(
        key[["review_order", "case_id", "ground_truth", "model_prediction"]],
        on=["review_order", "case_id"], how="left", validate="one_to_one",
    )
    if merged[["ground_truth", "model_prediction"]].isna().any().any():
        raise ValueError("Private evaluation key does not match form_cases.csv")
    for column in ("ground_truth", "model_prediction"):
        values = set(pd.to_numeric(merged[column], errors="coerce").dropna().astype(int))
        if not values.issubset({0, 1}):
            raise ValueError(f"{column} contains non-binary values: {values}")
    if responses.empty:
        raise ValueError("Response CSV contains no submissions")
    return merged.sort_values("review_order").reset_index(drop=True)


def make_long_responses(responses, cases):
    diagnosis_columns = repeated_columns(responses.columns, DIAGNOSIS_TITLE)
    confidence_columns = repeated_columns(responses.columns, CONFIDENCE_TITLE)
    quality_columns = repeated_columns(responses.columns, QUALITY_TITLE)
    expected = len(cases)
    counts = {
        "diagnosis": len(diagnosis_columns),
        "confidence": len(confidence_columns),
        "quality": len(quality_columns),
    }
    if any(value != expected for value in counts.values()):
        raise ValueError(
            f"Expected {expected} repeated columns for each case question; found {counts}. "
            "Do not rename case questions after creating the Form."
        )
    reviewer_column = find_column(responses.columns, REVIEWER_CANDIDATES)
    reviewer_ids = responses[reviewer_column].astype(str).str.strip()
    if reviewer_ids.eq("").any() or reviewer_ids.duplicated().any():
        raise ValueError("Reviewer ID must be non-empty and unique per submitted response")

    rows = []
    for response_index, response in responses.iterrows():
        reviewer_id = str(response[reviewer_column]).strip()
        for case_index, case in cases.iterrows():
            rows.append({
                "reviewer_id": reviewer_id,
                "response_row": int(response_index),
                "review_order": int(case["review_order"]),
                "case_id": str(case["case_id"]),
                "ground_truth": int(case["ground_truth"]),
                "model_prediction": int(case["model_prediction"]),
                "human_prediction": parse_binary_diagnosis(response[diagnosis_columns[case_index]]),
                "confidence": pd.to_numeric(response[confidence_columns[case_index]], errors="coerce"),
                "image_quality": str(response[quality_columns[case_index]]).strip(),
            })
    long_frame = pd.DataFrame(rows)
    if long_frame["human_prediction"].isna().any():
        bad = long_frame[long_frame["human_prediction"].isna()][
            ["reviewer_id", "case_id"]
        ].head(10).to_dict("records")
        raise ValueError(f"Missing or unparseable required diagnoses; examples: {bad}")
    long_frame["human_prediction"] = long_frame["human_prediction"].astype(int)
    return long_frame


def reviewer_metrics(long_frame, adequate_only=False):
    frame = long_frame
    if adequate_only:
        frame = frame[frame["image_quality"].str.casefold() != "inadequate"]
    records = []
    for reviewer_id, group in frame.groupby("reviewer_id", sort=True):
        if group.empty:
            continue
        metrics = binary_metrics(group["ground_truth"], group["human_prediction"])
        metrics.update({
            "reviewer_id": reviewer_id,
            "analysis": "exclude_inadequate" if adequate_only else "all_required_cases",
            "mean_confidence": float(group["confidence"].mean()),
            "inadequate_excluded": int(
                (long_frame[long_frame["reviewer_id"] == reviewer_id]["image_quality"].str.casefold()
                 == "inadequate").sum()
            ) if adequate_only else 0,
        })
        records.append(metrics)
    return pd.DataFrame(records)


def consensus_results(long_frame):
    records = []
    for case_id, group in long_frame.groupby("case_id", sort=False):
        positive_votes = int(group["human_prediction"].sum())
        negative_votes = int(len(group) - positive_votes)
        consensus = 1 if positive_votes > negative_votes else 0 if negative_votes > positive_votes else np.nan
        records.append({
            "review_order": int(group["review_order"].iloc[0]),
            "case_id": case_id,
            "ground_truth": int(group["ground_truth"].iloc[0]),
            "model_prediction": int(group["model_prediction"].iloc[0]),
            "reviewers": int(len(group)),
            "negative_votes": negative_votes,
            "positive_votes": positive_votes,
            "consensus_prediction": consensus,
            "mean_confidence": float(group["confidence"].mean()),
            "unanimous": int(positive_votes == 0 or negative_votes == 0),
        })
    return pd.DataFrame(records).sort_values("review_order").reset_index(drop=True)


def main():
    args = parse_args()
    responses = pd.read_csv(args.responses)
    form_cases = pd.read_csv(args.form_cases).sort_values("review_order").reset_index(drop=True)
    key = pd.read_csv(args.evaluation_key)
    cases = validate_inputs(responses, form_cases, key)
    long_frame = make_long_responses(responses, cases)
    reviewer_all = reviewer_metrics(long_frame, adequate_only=False)
    reviewer_adequate = reviewer_metrics(long_frame, adequate_only=True)
    consensus = consensus_results(long_frame)
    consensus_valid = consensus.dropna(subset=["consensus_prediction"]).copy()
    consensus_valid["consensus_prediction"] = consensus_valid["consensus_prediction"].astype(int)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    long_frame.to_csv(args.output_dir / "responses_long.csv", index=False)
    reviewer_all.to_csv(args.output_dir / "reviewer_metrics_all_cases.csv", index=False)
    reviewer_adequate.to_csv(
        args.output_dir / "reviewer_metrics_excluding_inadequate.csv", index=False
    )
    consensus.to_csv(args.output_dir / "consensus_by_case.csv", index=False)

    model_metrics = binary_metrics(cases["ground_truth"], cases["model_prediction"])
    consensus_metrics = (
        binary_metrics(consensus_valid["ground_truth"], consensus_valid["consensus_prediction"])
        if not consensus_valid.empty else None
    )
    rater_counts = long_frame.groupby("case_id")["reviewer_id"].nunique()
    summary = {
        "responses_csv": str(args.responses),
        "reviewers": int(long_frame["reviewer_id"].nunique()),
        "cases": int(len(cases)),
        "ratings": int(len(long_frame)),
        "ratings_per_case_min": int(rater_counts.min()),
        "ratings_per_case_max": int(rater_counts.max()),
        "model_metrics_on_same_cases": model_metrics,
        "majority_consensus_metrics": consensus_metrics,
        "consensus_tied_cases": int(consensus["consensus_prediction"].isna().sum()),
        "unanimous_case_fraction": float(consensus["unanimous"].mean()),
        "fleiss_kappa": fleiss_kappa(long_frame),
        "image_quality_counts": {
            str(key): int(value) for key, value in long_frame["image_quality"].value_counts().items()
        },
        "primary_analysis": "All required case diagnoses; weighted F1 is f1_weighted.",
        "sensitivity_analysis": "Per-reviewer metrics excluding ratings marked Inadequate.",
        "tie_policy": "Consensus ties are reported and excluded from consensus metrics.",
    }
    (args.output_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"Evaluation outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
