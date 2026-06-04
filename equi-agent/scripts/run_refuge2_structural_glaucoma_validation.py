from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any


FEATURES = ["vertical_cup_to_disc_ratio", "cup_to_disc_area_ratio"]
POLICIES = {
    "structural_sensitivity": "balanced_accuracy",
    "structural_precision": "f1",
}


def fnum(value: object) -> float | None:
    try:
        text = str(value).strip()
        if text == "":
            return None
        return float(text)
    except Exception:
        return None


def int_label(value: object) -> int | None:
    value_num = fnum(value)
    if value_num is None:
        return None
    label = int(value_num)
    return label if label in {0, 1} else None


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def load_labeled_rows(manifest: Path, split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_csv(manifest):
        if row.get("split") != split:
            continue
        label = int_label(row.get("label_glaucoma"))
        if label is None:
            continue
        values = {feature: fnum(row.get(feature)) for feature in FEATURES}
        if any(value is None for value in values.values()):
            continue
        rows.append(
            {
                "dataset": row.get("dataset", "REFUGE2"),
                "source_split": row.get("split", ""),
                "image_id": row.get("image_id", ""),
                "y_true": label,
                "image_path": row.get("image_path", ""),
                "mask_path": row.get("mask_path", ""),
                **values,
            }
        )
    return rows


def stratified_split(rows: list[dict[str, Any]], test_frac: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    by_label: dict[int, list[dict[str, Any]]] = {0: [], 1: []}
    for row in rows:
        by_label[int(row["y_true"])].append(row)

    calibration_rows: list[dict[str, Any]] = []
    evaluation_rows: list[dict[str, Any]] = []
    for label_rows in by_label.values():
        copied = list(label_rows)
        rng.shuffle(copied)
        n_eval = max(1, round(len(copied) * test_frac))
        evaluation_rows.extend(copied[:n_eval])
        calibration_rows.extend(copied[n_eval:])
    rng.shuffle(calibration_rows)
    rng.shuffle(evaluation_rows)
    return calibration_rows, evaluation_rows


def confusion(y_true: list[int], y_pred: list[int]) -> dict[str, int]:
    pairs = Counter(zip(y_true, y_pred))
    return {
        "tn": pairs[(0, 0)],
        "fp": pairs[(0, 1)],
        "fn": pairs[(1, 0)],
        "tp": pairs[(1, 1)],
    }


def auroc(y_true: list[int], y_score: list[float]) -> float | None:
    pos = sum(y_true)
    neg = len(y_true) - pos
    if pos == 0 or neg == 0:
        return None
    ordered = sorted(zip(y_score, y_true), key=lambda item: item[0])
    rank_sum_pos = 0.0
    rank = 1
    idx = 0
    while idx < len(ordered):
        end = idx
        while end + 1 < len(ordered) and ordered[end + 1][0] == ordered[idx][0]:
            end += 1
        avg_rank = (rank + rank + (end - idx)) / 2
        rank_sum_pos += sum(avg_rank for _, label in ordered[idx : end + 1] if label == 1)
        rank += end - idx + 1
        idx = end + 1
    return (rank_sum_pos - pos * (pos + 1) / 2) / (pos * neg)


def metrics(y_true: list[int], y_pred: list[int], y_score: list[float]) -> dict[str, Any]:
    c = confusion(y_true, y_pred)
    tn, fp, fn, tp = c["tn"], c["fp"], c["fn"], c["tp"]
    n = tn + fp + fn + tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": n,
        **c,
        "accuracy": (tp + tn) / n if n else 0.0,
        "precision": precision,
        "recall": recall,
        "sensitivity": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": (recall + specificity) / 2,
        "auroc": auroc(y_true, y_score),
    }


def candidate_thresholds(values: list[float]) -> list[float]:
    unique = sorted(set(values))
    if not unique:
        return []
    thresholds = [unique[0] - 1e-9]
    thresholds.extend((a + b) / 2 for a, b in zip(unique, unique[1:]))
    thresholds.append(unique[-1] + 1e-9)
    return thresholds


def tune_threshold(rows: list[dict[str, Any]], feature: str, objective: str) -> tuple[float, dict[str, Any]]:
    best_threshold = 0.5
    best_metrics: dict[str, Any] = {}
    best_score = -1.0
    for threshold in candidate_thresholds([float(row[feature]) for row in rows]):
        y_true = [int(row["y_true"]) for row in rows]
        y_score = [float(row[feature]) for row in rows]
        y_pred = [int(score >= threshold) for score in y_score]
        item = metrics(y_true, y_pred, y_score)
        score = float(item[objective])
        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_metrics = item
    return best_threshold, best_metrics


def confidence_from_score(score: float, sensitivity_threshold: float, precision_threshold: float, pred: int) -> str:
    if pred == 1 and score >= precision_threshold:
        return "high"
    if pred == 0 and score < sensitivity_threshold:
        return "medium"
    return "low"


def structural_summary(score: float, sensitivity_threshold: float, precision_threshold: float) -> str:
    if score >= precision_threshold:
        return "strong structural glaucoma evidence from vertical cup-to-disc ratio"
    if score >= sensitivity_threshold:
        return "borderline structural glaucoma evidence from vertical cup-to-disc ratio"
    return "weak structural glaucoma evidence from vertical cup-to-disc ratio"


def prediction_rows_for_policy(
    rows: list[dict[str, Any]],
    policy: str,
    objective: str,
    threshold: float,
    sensitivity_threshold: float,
    precision_threshold: float,
) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        score = float(row["vertical_cup_to_disc_ratio"])
        pred = int(score >= threshold)
        out.append(
            {
                "dataset": row["dataset"],
                "source_split": row["source_split"],
                "evaluation_split": "stratified_holdout",
                "image_id": row["image_id"],
                "image_path": row["image_path"],
                "mask_path": row["mask_path"],
                "policy": policy,
                "objective": objective,
                "feature": "vertical_cup_to_disc_ratio",
                "y_true": row["y_true"],
                "y_score": f"{score:.6f}",
                "threshold": f"{threshold:.6f}",
                "y_pred": pred,
                "is_correct": int(pred == int(row["y_true"])),
                "vertical_cup_to_disc_ratio": f"{score:.6f}",
                "cup_to_disc_area_ratio": f"{float(row['cup_to_disc_area_ratio']):.6f}",
                "sensitivity_threshold": f"{sensitivity_threshold:.6f}",
                "precision_threshold": f"{precision_threshold:.6f}",
                "confidence": confidence_from_score(score, sensitivity_threshold, precision_threshold, pred),
                "escalate_to_human": int(sensitivity_threshold <= score < precision_threshold),
                "structural_agent_summary": structural_summary(score, sensitivity_threshold, precision_threshold),
            }
        )
    return out


def rounded_metric_dict(item: dict[str, Any]) -> dict[str, Any]:
    return {key: (round(value, 6) if isinstance(value, float) else value) for key, value in item.items()}


def repeated_holdout(
    rows: list[dict[str, Any]],
    test_frac: float,
    seed: int,
    repeats: int,
) -> list[dict[str, Any]]:
    out = []
    for offset in range(repeats):
        split_seed = seed + offset
        calibration_rows, evaluation_rows = stratified_split(rows, test_frac, split_seed)
        thresholds = {}
        for policy, objective in POLICIES.items():
            threshold, _ = tune_threshold(calibration_rows, "vertical_cup_to_disc_ratio", objective)
            thresholds[policy] = threshold
            y_true = [int(row["y_true"]) for row in evaluation_rows]
            y_score = [float(row["vertical_cup_to_disc_ratio"]) for row in evaluation_rows]
            y_pred = [int(score >= threshold) for score in y_score]
            item = metrics(y_true, y_pred, y_score)
            out.append(
                {
                    "seed": split_seed,
                    "policy": policy,
                    "objective": objective,
                    "threshold": threshold,
                    **item,
                }
            )
    return out


def summarize_repeats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    metric_keys = ["accuracy", "precision", "recall", "specificity", "f1", "balanced_accuracy", "auroc", "threshold"]
    policies = sorted(set(str(row["policy"]) for row in rows))
    for policy in policies:
        policy_rows = [row for row in rows if row["policy"] == policy]
        policy_summary = {}
        for key in metric_keys:
            values = [float(row[key]) for row in policy_rows if row.get(key) is not None]
            if not values:
                continue
            policy_summary[key] = {
                "mean": round(mean(values), 6),
                "sd": round(stdev(values), 6) if len(values) > 1 else 0.0,
                "min": round(min(values), 6),
                "median": round(median(values), 6),
                "max": round(max(values), 6),
            }
        summary[policy] = policy_summary
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run REFUGE2 external structural glaucoma validation using expert mask-derived "
            "vertical cup-to-disc ratio. This evaluates labeled REFUGE2 train images with "
            "a stratified calibration/holdout split because this Kaggle mirror hides val/test labels."
        )
    )
    parser.add_argument("--manifest", type=Path, default=Path("equi-agent/outputs/manifests/refuge2_manifest_v2.csv"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--test-frac", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--out-dir", type=Path, default=Path("equi-agent/outputs/refuge2_structural_glaucoma_validation"))
    args = parser.parse_args()

    rows = load_labeled_rows(args.manifest, args.split)
    if not rows:
        raise RuntimeError(f"No labeled rows found in {args.manifest} split={args.split}")

    calibration_rows, evaluation_rows = stratified_split(rows, args.test_frac, args.seed)
    thresholds: dict[str, float] = {}
    calibration_metrics: dict[str, dict[str, Any]] = {}
    evaluation_metrics: dict[str, dict[str, Any]] = {}
    all_prediction_rows: list[dict[str, Any]] = []

    for policy, objective in POLICIES.items():
        threshold, train_metrics = tune_threshold(calibration_rows, "vertical_cup_to_disc_ratio", objective)
        thresholds[policy] = threshold
        calibration_metrics[policy] = rounded_metric_dict(train_metrics)
        y_true = [int(row["y_true"]) for row in evaluation_rows]
        y_score = [float(row["vertical_cup_to_disc_ratio"]) for row in evaluation_rows]
        y_pred = [int(score >= threshold) for score in y_score]
        evaluation_metrics[policy] = rounded_metric_dict(metrics(y_true, y_pred, y_score))

    sensitivity_threshold = thresholds["structural_sensitivity"]
    precision_threshold = thresholds["structural_precision"]
    for policy, objective in POLICIES.items():
        all_prediction_rows.extend(
            prediction_rows_for_policy(
                evaluation_rows,
                policy,
                objective,
                thresholds[policy],
                sensitivity_threshold,
                precision_threshold,
            )
        )

    repeated_rows = repeated_holdout(rows, args.test_frac, args.seed, args.repeats)
    repeated_summary = summarize_repeats(repeated_rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.out_dir / "refuge2_structural_glaucoma_predictions.csv"
    repeated_path = args.out_dir / "refuge2_structural_glaucoma_repeated_metrics.csv"
    summary_path = args.out_dir / "refuge2_structural_glaucoma_summary.json"
    write_csv(predictions_path, all_prediction_rows)
    write_csv(repeated_path, [rounded_metric_dict(row) for row in repeated_rows])
    summary = {
        "manifest": str(args.manifest),
        "dataset": "REFUGE2",
        "label_note": "Only labeled train images are available in this Kaggle mirror; val/test classification labels are hidden.",
        "split": args.split,
        "labeled_rows": len(rows),
        "label_counts": dict(Counter(row["y_true"] for row in rows)),
        "calibration_rows": len(calibration_rows),
        "evaluation_rows": len(evaluation_rows),
        "test_frac": args.test_frac,
        "seed": args.seed,
        "thresholds": {key: round(value, 6) for key, value in thresholds.items()},
        "calibration_metrics": calibration_metrics,
        "evaluation_metrics": evaluation_metrics,
        "repeats": args.repeats,
        "repeated_summary": repeated_summary,
        "outputs": {
            "predictions": str(predictions_path),
            "repeated_metrics": str(repeated_path),
            "summary": str(summary_path),
        },
    }
    write_json(summary_path, summary)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
