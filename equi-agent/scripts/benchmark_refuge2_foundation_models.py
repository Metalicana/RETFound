#!/usr/bin/env python3
"""Nested-CV foundation-model benchmark on labeled REFUGE2 training images.

The public mirror used by this project contains 400 labeled training images,
while its validation and test classification labels are unavailable. This
runner therefore produces nested, stratified out-of-fold predictions for the
400 labeled images. It must not be described as official REFUGE2 test-set
performance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import benchmark_papila_foundation_models as common


DEFAULT_MODELS = ("retfound", "ret_clip", "retizero", "urfound")
OVERLAP_NOTES = {
    "urfound": (
        "UrFound states that FLAIR supplied its pretraining data; the FLAIR "
        "assembly includes REFUGE. Treat this result as having pretraining-overlap risk."
    ),
    "retizero": (
        "The bundled RetiZero code lists REFUGE among its retinal datasets. "
        "The exact released-checkpoint composition requires verification."
    ),
}


def parse_args() -> argparse.Namespace:
    root = common.repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=root / "equi-agent" / "outputs" / "benchmarks" / "refuge2_glaucoma_foundations_nested_cv",
    )
    parser.add_argument("--models", nargs="+", choices=DEFAULT_MODELS, default=list(DEFAULT_MODELS))
    parser.add_argument("--device", default=None, help="Example: cuda:0 or cpu.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--logreg-c", nargs="+", type=float, default=[0.01, 0.1, 1.0, 10.0])
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument(
        "--reuse-feature-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--summarize-only", action="store_true")

    parser.add_argument(
        "--retfound-weights",
        type=Path,
        default=root / "OphthalmicAgent" / "weights" / "cfp_model.pth",
    )
    parser.add_argument(
        "--ret-clip-root",
        type=Path,
        default=root / "Foundation_Models" / "RET-CLIP-main",
    )
    parser.add_argument(
        "--ret-clip-weights",
        type=Path,
        default=root / "Foundation_Models" / "RET-CLIP-main" / "pretrained" / "ret_clip_vit_b_16.pt",
    )
    parser.add_argument(
        "--retizero-root",
        type=Path,
        default=root / "Foundation_Models" / "RetiZero-main",
    )
    parser.add_argument(
        "--retizero-weights",
        type=Path,
        default=root / "Foundation_Models" / "RetiZero-main" / "pretrained" / "RetiZero.pth",
    )
    parser.add_argument("--retizero-lora-rank", type=int, default=8)
    parser.add_argument(
        "--urfound-root",
        type=Path,
        default=root / "Foundation_Models" / "UrFound-main",
    )
    parser.add_argument(
        "--urfound-weights",
        type=Path,
        default=root / "Foundation_Models" / "UrFound-main" / "pretrained" / "urfound_mm.pth",
    )
    parser.add_argument(
        "--urfound-model",
        choices=("vit_base_patch16", "vit_large_patch16"),
        default="vit_base_patch16",
    )
    parser.add_argument("--urfound-global-pool", action="store_true")

    args = parser.parse_args()
    # Attributes consumed by the shared feature-extraction implementation.
    args.dataset = "refuge2_labeled_train_nested_cv"
    args.limit_per_split = None
    args.mirage_dir = root / "equi-agent" / "VisionAgent" / "MIRAGE"
    args.mirage_feature_module = None
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_image_path(manifest: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return manifest.parent / path


def load_labeled_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"REFUGE2 manifest not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        source_rows = list(csv.DictReader(handle))

    rows: list[dict[str, Any]] = []
    skipped_unlabeled = 0
    for source in source_rows:
        label = str(source.get("label_glaucoma", source.get("label", ""))).strip()
        if label not in {"0", "1"}:
            skipped_unlabeled += 1
            continue
        image_id = str(source.get("image_id", source.get("case_id", ""))).strip()
        image_value = str(source.get("image_path", source.get("cfp_path", ""))).strip()
        image_path = resolve_image_path(path, image_value)
        if not image_id:
            raise ValueError("Every labeled REFUGE2 row must contain image_id or case_id")
        if not image_path.is_file():
            raise FileNotFoundError(f"REFUGE2 CFP image not found for {image_id}: {image_path}")
        rows.append(
            {
                **source,
                "case_id": image_id,
                "patient_id": image_id,
                "source_split": str(source.get("split", "")),
                "y_true": int(label),
                "image_path": str(image_path),
                "sex_gender": "unknown",
                "age_group": "unknown",
            }
        )

    counts = Counter(row["y_true"] for row in rows)
    duplicate_ids = [case_id for case_id, count in Counter(row["case_id"] for row in rows).items() if count > 1]
    if duplicate_ids:
        raise ValueError(f"Duplicate labeled REFUGE2 image IDs: {duplicate_ids[:10]}")
    if len(rows) != 400 or counts != Counter({0: 360, 1: 40}):
        raise ValueError(
            "Expected the public mirror's 400 labeled training images "
            f"(360 negative, 40 positive); found rows={len(rows)} labels={dict(counts)}"
        )
    if skipped_unlabeled != 800:
        raise ValueError(f"Expected 800 unlabeled val/test rows; found {skipped_unlabeled}")
    return rows


def metric_dict(np, y_true, y_pred, y_prob=None) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())

    def divide(numerator, denominator):
        return float(numerator / denominator) if denominator else None

    sensitivity = divide(tp, tp + fn)
    specificity = divide(tn, tn + fp)
    balanced_accuracy = (
        (sensitivity + specificity) / 2
        if sensitivity is not None and specificity is not None
        else None
    )
    metrics = {
        "n": int(len(y_true)),
        "positive_n": int((y_true == 1).sum()),
        "negative_n": int((y_true == 0).sum()),
        "accuracy": divide(tp + tn, len(y_true)),
        "f1": divide(2 * tp, 2 * tp + fp + fn),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }
    if y_prob is not None:
        from sklearn.metrics import roc_auc_score

        metrics["auroc"] = float(roc_auc_score(y_true, y_prob))
    return metrics


def fit_classifier(features, labels, c_value, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=float(c_value),
            class_weight="balanced",
            max_iter=5000,
            random_state=seed,
        ),
    )
    classifier.fit(features, labels)
    return classifier


def select_inner_configuration(np, features, labels, args, outer_fold):
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    inner = StratifiedKFold(
        n_splits=args.inner_folds,
        shuffle=True,
        random_state=args.seed + 1000 + outer_fold,
    )
    best = None
    search_rows = []
    for c_value in args.logreg_c:
        probabilities = np.full(len(labels), np.nan, dtype=float)
        for inner_fold, (train_index, valid_index) in enumerate(inner.split(features, labels)):
            classifier = fit_classifier(
                features[train_index],
                labels[train_index],
                c_value,
                args.seed + outer_fold * 100 + inner_fold,
            )
            probabilities[valid_index] = classifier.predict_proba(features[valid_index])[:, 1]
        if not np.isfinite(probabilities).all():
            raise RuntimeError("Inner-CV probabilities were not fully populated")
        threshold, threshold_metrics = common.select_f1_threshold(np, labels, probabilities)
        auc = float(roc_auc_score(labels, probabilities))
        row = {
            "C": float(c_value),
            "threshold": float(threshold),
            "inner_oof_f1": threshold_metrics["f1"],
            "inner_oof_balanced_accuracy": threshold_metrics["balanced_accuracy"],
            "inner_oof_auroc": auc,
        }
        search_rows.append(row)
        key = (
            row["inner_oof_f1"],
            row["inner_oof_balanced_accuracy"],
            row["inner_oof_auroc"],
            -abs(math.log10(float(c_value))),
        )
        if best is None or key > best[0]:
            best = (key, row)
    assert best is not None
    return best[1], search_rows


def nested_oof_predictions(np, features, labels, args):
    from sklearn.model_selection import StratifiedKFold

    outer = StratifiedKFold(
        n_splits=args.outer_folds,
        shuffle=True,
        random_state=args.seed,
    )
    probabilities = np.full(len(labels), np.nan, dtype=float)
    predictions = np.full(len(labels), -1, dtype=int)
    fold_numbers = np.full(len(labels), -1, dtype=int)
    thresholds = np.full(len(labels), np.nan, dtype=float)
    selected_c = np.full(len(labels), np.nan, dtype=float)
    fold_rows = []
    search_rows = []

    for outer_fold, (train_index, test_index) in enumerate(outer.split(features, labels), start=1):
        configuration, inner_search = select_inner_configuration(
            np,
            features[train_index],
            labels[train_index],
            args,
            outer_fold,
        )
        for row in inner_search:
            search_rows.append({"outer_fold": outer_fold, **row})
        classifier = fit_classifier(
            features[train_index],
            labels[train_index],
            configuration["C"],
            args.seed + outer_fold,
        )
        fold_probabilities = classifier.predict_proba(features[test_index])[:, 1]
        fold_predictions = (fold_probabilities >= configuration["threshold"]).astype(int)
        probabilities[test_index] = fold_probabilities
        predictions[test_index] = fold_predictions
        fold_numbers[test_index] = outer_fold
        thresholds[test_index] = configuration["threshold"]
        selected_c[test_index] = configuration["C"]
        fold_rows.append(
            {
                "outer_fold": outer_fold,
                "train_n": int(len(train_index)),
                "test_n": int(len(test_index)),
                "selected_C": configuration["C"],
                "selected_threshold": configuration["threshold"],
                **metric_dict(np, labels[test_index], fold_predictions, fold_probabilities),
            }
        )

    if not np.isfinite(probabilities).all() or (predictions < 0).any():
        raise RuntimeError("Outer-CV predictions were not fully populated")
    return probabilities, predictions, fold_numbers, thresholds, selected_c, fold_rows, search_rows


def bootstrap_intervals(np, labels, predictions, probabilities, replicates, seed):
    if replicates <= 0:
        return {}
    rng = np.random.default_rng(seed)
    names = ("f1", "sensitivity", "specificity", "balanced_accuracy", "auroc")
    values = {name: [] for name in names}
    for _ in range(replicates):
        sample = rng.integers(0, len(labels), size=len(labels))
        sampled_labels = labels[sample]
        if len(np.unique(sampled_labels)) < 2:
            continue
        metrics = metric_dict(
            np,
            sampled_labels,
            predictions[sample],
            probabilities[sample],
        )
        for name in names:
            if metrics.get(name) is not None:
                values[name].append(float(metrics[name]))
    return {
        name: {
            "low": float(np.quantile(items, 0.025)),
            "high": float(np.quantile(items, 0.975)),
        }
        for name, items in values.items()
        if items
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def model_weight(args, model_name: str) -> Path:
    return common.model_weight_path(args, model_name)


def run_model(np, torch, Image, DataLoader, transforms, tqdm, args, model_name, rows, device):
    model_dir = args.out_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    feature_map, provenance = common.extract_model_features(
        np,
        torch,
        Image,
        DataLoader,
        transforms,
        tqdm,
        args,
        model_name,
        {"all_labeled": rows},
        device,
    )
    features = feature_map["all_labeled"]
    labels = np.asarray([row["y_true"] for row in rows], dtype=int)
    (
        probabilities,
        predictions,
        fold_numbers,
        thresholds,
        selected_c,
        fold_rows,
        search_rows,
    ) = nested_oof_predictions(np, features, labels, args)
    metrics = metric_dict(np, labels, predictions, probabilities)
    intervals = bootstrap_intervals(
        np,
        labels,
        predictions,
        probabilities,
        args.bootstrap_replicates,
        args.seed + sum(ord(character) for character in model_name),
    )

    prediction_rows = []
    for index, row in enumerate(rows):
        prediction_rows.append(
            {
                "dataset": "refuge2",
                "evaluation_cohort": "official_labeled_train400_nested_cv",
                "model_name": model_name,
                "case_id": row["case_id"],
                "source_split": row["source_split"],
                "y_true": int(labels[index]),
                "y_prob": float(probabilities[index]),
                "y_pred": int(predictions[index]),
                "outer_fold": int(fold_numbers[index]),
                "fold_threshold": float(thresholds[index]),
                "selected_C": float(selected_c[index]),
                "image_path": row["image_path"],
            }
        )
    write_csv(model_dir / "predictions_oof.csv", prediction_rows)
    write_csv(model_dir / "outer_fold_metrics.csv", fold_rows)
    write_csv(model_dir / "inner_search.csv", search_rows)

    weights = model_weight(args, model_name)
    summary = {
        "dataset": "REFUGE2",
        "evaluation_cohort": "400 officially labeled training images",
        "official_test_performance": False,
        "protocol": (
            f"{args.outer_folds}-fold stratified outer CV with {args.inner_folds}-fold "
            "inner CV for logistic-regression C and F1-threshold selection"
        ),
        "test_used_for_selection": False,
        "model_name": model_name,
        "rows": len(rows),
        "label_counts": dict(Counter(str(value) for value in labels)),
        "metrics": metrics,
        "bootstrap_95_ci": intervals,
        "feature_provenance": provenance,
        "weights": str(weights),
        "weights_sha256": sha256(weights),
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "pretraining_overlap_note": OVERLAP_NOTES.get(model_name, ""),
        "seed": args.seed,
    }
    (model_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"model_name": model_name, "metrics": metrics}, indent=2))


def summarize(out_dir: Path, requested_models: list[str] | None = None) -> None:
    model_names = requested_models or list(DEFAULT_MODELS)
    rows = []
    for model_name in model_names:
        path = out_dir / model_name / "summary.json"
        if not path.is_file():
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        metrics = summary["metrics"]
        rows.append(
            {
                "model_name": model_name,
                "f1": metrics["f1"],
                "worst_group_f1": "",
                "sensitivity": metrics["sensitivity"],
                "specificity": metrics["specificity"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "auroc": metrics["auroc"],
                "tn": metrics["tn"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "tp": metrics["tp"],
                "official_test_performance": False,
                "pretraining_overlap_note": summary.get("pretraining_overlap_note", ""),
            }
        )
    if not rows:
        raise FileNotFoundError(f"No model summaries found under {out_dir}")
    write_csv(out_dir / "refuge2_glaucoma_foundation_nested_cv.csv", rows)
    lines = [
        "# REFUGE2 Glaucoma Foundation Benchmark",
        "",
        "Nested cross-validation on the 400 officially labeled training images. "
        "These are not official REFUGE2 test-set results.",
        "",
        "| Model | F1 | Worst-group F1 | Sensitivity | Specificity | Balanced accuracy | AUROC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model_name']} | {float(row['f1']):.4f} | N/A | "
            f"{float(row['sensitivity']):.4f} | {float(row['specificity']):.4f} | "
            f"{float(row['balanced_accuracy']):.4f} | {float(row['auroc']):.4f} |"
        )
    notes = [row for row in rows if row["pretraining_overlap_note"]]
    if notes:
        lines.extend(["", "## Pretraining-overlap notes", ""])
        for row in notes:
            lines.append(f"- **{row['model_name']}**: {row['pretraining_overlap_note']}")
    report = "\n".join(lines) + "\n"
    (out_dir / "refuge2_glaucoma_foundation_nested_cv.md").write_text(report, encoding="utf-8")
    print(report)


def main() -> None:
    args = parse_args()
    if args.summarize_only:
        summarize(args.out_dir)
        return

    import numpy as np
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from tqdm import tqdm

    if args.outer_folds < 2 or args.inner_folds < 2:
        raise ValueError("outer-folds and inner-folds must both be at least 2")
    if not args.logreg_c or any(value <= 0 for value in args.logreg_c):
        raise ValueError("All --logreg-c values must be positive")
    rows = load_labeled_rows(args.manifest)
    common.set_seed(np, torch, args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "dataset": "REFUGE2",
        "evaluation_cohort": "400 officially labeled training images",
        "official_test_performance": False,
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "models": args.models,
        "outer_folds": args.outer_folds,
        "inner_folds": args.inner_folds,
        "selection_objective": "inner out-of-fold positive-class F1",
        "probe": "standardized class-balanced logistic regression",
        "logreg_c": args.logreg_c,
        "label_counts": dict(Counter(str(row["y_true"]) for row in rows)),
        "seed": args.seed,
        "device": str(device),
    }
    (args.out_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(protocol, indent=2, sort_keys=True))

    for model_name in args.models:
        run_model(
            np,
            torch,
            Image,
            DataLoader,
            transforms,
            tqdm,
            args,
            model_name,
            rows,
            device,
        )
    summarize(args.out_dir, args.models)


if __name__ == "__main__":
    main()
