#!/usr/bin/env python3
"""Refit PAPILA probes using patient-grouped development-set OOF selection."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
from collections import Counter, defaultdict
from pathlib import Path

import benchmark_papila_foundation_models as benchmark


def parse_args() -> argparse.Namespace:
    root = benchmark.repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "OphthalmicAgent" / "data_papila" / "manifest.csv",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=root / "equi-agent" / "outputs" / "benchmarks" / "papila_glaucoma_foundations_v1",
        help="Directory containing feature_cache/<model>/{train,val,test}.npz.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=root / "equi-agent" / "outputs" / "benchmarks" / "papila_glaucoma_foundations_oof_v2",
    )
    parser.add_argument("--models", nargs="+", choices=benchmark.MODELS, default=list(benchmark.MODELS))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--logreg-c",
        nargs="+",
        type=float,
        default=[0.001, 0.01, 0.1, 1.0, 10.0],
    )
    parser.add_argument(
        "--pca-dims",
        nargs="+",
        type=int,
        default=[0, 16, 32, 64],
        help="Zero means no PCA.",
    )
    parser.add_argument(
        "--class-weights",
        nargs="+",
        choices=("none", "balanced"),
        default=["none", "balanced"],
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--min-subgroup-n", type=int, default=10)
    parser.add_argument("--min-subgroup-positive", type=int, default=2)
    parser.add_argument("--min-subgroup-negative", type=int, default=2)
    parser.add_argument("--summarize-only", action="store_true")
    return parser.parse_args()


def load_feature_cache(np, source_dir: Path, model_name: str, split: str, rows):
    path = source_dir / "feature_cache" / model_name / f"{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Feature cache missing: {path}")
    with np.load(path, allow_pickle=False) as payload:
        case_ids = payload["case_ids"].astype(str).tolist()
        expected = [row["case_id"] for row in rows]
        if case_ids != expected:
            raise ValueError(f"Feature cache case order mismatch: {path}")
        matrix = payload["features"].astype("float32")
    if matrix.shape[0] != len(rows) or not np.isfinite(matrix).all():
        raise ValueError(f"Invalid feature matrix at {path}: {matrix.shape}")
    return matrix


def grouped_fold_ids(np, rows, folds: int, seed: int):
    if folds < 2:
        raise ValueError("--folds must be at least 2")
    patient_rows: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        patient_rows[row["patient_id"]].append(index)

    by_label: dict[int, list[str]] = defaultdict(list)
    for patient, indices in patient_rows.items():
        labels = {int(rows[index]["y_true"]) for index in indices}
        if len(labels) != 1:
            raise ValueError(f"Patient {patient} has conflicting binary labels: {labels}")
        by_label[next(iter(labels))].append(patient)
    if set(by_label) != {0, 1}:
        raise ValueError(f"Development patients must contain both classes: {set(by_label)}")
    if min(len(values) for values in by_label.values()) < folds:
        raise ValueError(f"Not enough patients per class for {folds} folds")

    rng = random.Random(seed)
    patient_fold = {}
    for label in (0, 1):
        patients = sorted(by_label[label])
        rng.shuffle(patients)
        for index, patient in enumerate(patients):
            patient_fold[patient] = index % folds
    fold_ids = np.asarray([patient_fold[row["patient_id"]] for row in rows], dtype=int)

    for fold in range(folds):
        labels = {int(rows[index]["y_true"]) for index in np.flatnonzero(fold_ids == fold)}
        if labels != {0, 1}:
            raise ValueError(f"Fold {fold} does not contain both classes: {labels}")
    return fold_ids


def make_candidate(Pipeline, StandardScaler, PCA, LogisticRegression, pca_dim, c_value, class_weight, seed):
    steps = [("scale", StandardScaler())]
    if pca_dim > 0:
        steps.append(
            (
                "pca",
                PCA(
                    n_components=pca_dim,
                    svd_solver="randomized",
                    random_state=seed,
                ),
            )
        )
    steps.append(
        (
            "classifier",
            LogisticRegression(
                C=c_value,
                class_weight=None if class_weight == "none" else "balanced",
                max_iter=5000,
                random_state=seed,
            ),
        )
    )
    return Pipeline(steps)


def oof_probabilities(np, x, y, fold_ids, folds, factory):
    probabilities = np.full(len(y), np.nan, dtype=float)
    for fold in range(folds):
        train_mask = fold_ids != fold
        valid_mask = fold_ids == fold
        model = factory()
        model.fit(x[train_mask], y[train_mask])
        probabilities[valid_mask] = model.predict_proba(x[valid_mask])[:, 1]
    if not np.isfinite(probabilities).all():
        raise RuntimeError("OOF predictions contain non-finite or missing values")
    return probabilities


def select_candidate(np, args, x_dev, y_dev, fold_ids, model_dir: Path):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    rows = []
    best = None
    max_fold_train = min(int((fold_ids != fold).sum()) for fold in range(args.folds))
    for pca_dim in args.pca_dims:
        if pca_dim < 0:
            raise ValueError("PCA dimensions cannot be negative")
        if pca_dim and pca_dim > min(x_dev.shape[1], max_fold_train):
            print(f"skip_pca_dim={pca_dim} reason=exceeds_fold_matrix")
            continue
        for c_value in args.logreg_c:
            for class_weight in args.class_weights:
                config = {
                    "pca_dim": pca_dim,
                    "C": c_value,
                    "class_weight": class_weight,
                }

                def factory(config=config):
                    return make_candidate(
                        Pipeline,
                        StandardScaler,
                        PCA,
                        LogisticRegression,
                        config["pca_dim"],
                        config["C"],
                        config["class_weight"],
                        args.seed,
                    )

                probabilities = oof_probabilities(
                    np,
                    x_dev,
                    y_dev,
                    fold_ids,
                    args.folds,
                    factory,
                )
                threshold, metrics = benchmark.select_f1_threshold(np, y_dev, probabilities)
                auc = benchmark.roc_auc(np, y_dev, probabilities)
                name = (
                    f"pca_{pca_dim if pca_dim else 'none'}"
                    f"_c_{c_value:g}_weight_{class_weight}"
                )
                row = {
                    "candidate": name,
                    "pca_dim": pca_dim,
                    "C": c_value,
                    "class_weight": class_weight,
                    "oof_threshold": threshold,
                    "oof_f1": metrics["f1"],
                    "oof_balanced_accuracy": metrics["balanced_accuracy"],
                    "oof_sensitivity": metrics["sensitivity"],
                    "oof_specificity": metrics["specificity"],
                    "oof_auroc": auc,
                }
                rows.append(row)
                key = (
                    metrics["f1"] if metrics["f1"] is not None else -1.0,
                    metrics["balanced_accuracy"] if metrics["balanced_accuracy"] is not None else -1.0,
                    auc if auc is not None else -1.0,
                    -abs(threshold - 0.5),
                )
                if best is None or key > best[0]:
                    best = (key, name, config, threshold, metrics, auc, probabilities, factory)
                print(
                    f"candidate={name} oof_f1={metrics['f1']:.4f} "
                    f"oof_ba={metrics['balanced_accuracy']:.4f} threshold={threshold:.4f}"
                )

    if best is None:
        raise RuntimeError("No valid OOF probe candidates")
    model_dir.mkdir(parents=True, exist_ok=True)
    benchmark.write_csv_rows(model_dir / "oof_probe_search.csv", rows)
    return best


def run_model(np, args, model_name, by_split, age_bounds):
    print(f"\n=== OOF refit model={model_name} ===", flush=True)
    model_dir = args.out_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    features = {
        split: load_feature_cache(np, args.source_dir, model_name, split, by_split[split])
        for split in benchmark.SPLITS
    }
    development_rows = by_split["train"] + by_split["val"]
    x_dev = np.concatenate([features["train"], features["val"]], axis=0)
    y_dev = np.asarray([row["y_true"] for row in development_rows], dtype=int)
    fold_ids = grouped_fold_ids(np, development_rows, args.folds, args.seed)

    best = select_candidate(np, args, x_dev, y_dev, fold_ids, model_dir)
    _, candidate, config, threshold, oof_metrics, oof_auc, oof_probs, factory = best
    final_probe = factory()
    final_probe.fit(x_dev, y_dev)
    test_probs = final_probe.predict_proba(features["test"])[:, 1]
    with (model_dir / "selected_probe.pkl").open("wb") as handle:
        pickle.dump(final_probe, handle)

    benchmark.write_predictions(
        model_dir / "predictions_development_oof.csv",
        model_name,
        development_rows,
        oof_probs,
        threshold,
    )
    benchmark.write_predictions(
        model_dir / "predictions_test.csv",
        model_name,
        by_split["test"],
        test_probs,
        threshold,
    )

    test_labels = np.asarray([row["y_true"] for row in by_split["test"]], dtype=int)
    test_metrics = benchmark.confusion_metrics(np, test_labels, test_probs, threshold)
    test_metrics["auroc"] = benchmark.roc_auc(np, test_labels, test_probs)
    test_groups, worst_f1 = benchmark.subgroup_metrics(
        np, by_split["test"], test_probs, threshold, args
    )
    test_metrics["worst_group_f1"] = worst_f1
    benchmark.write_csv_rows(model_dir / "subgroup_metrics_test.csv", test_groups)
    intervals = benchmark.bootstrap_intervals(
        np, by_split["test"], test_probs, threshold, args
    )

    oof_metrics = dict(oof_metrics)
    oof_metrics["auroc"] = oof_auc
    summary = {
        "dataset": "papila",
        "task": "glaucoma",
        "model_name": model_name,
        "manifest": str(args.manifest),
        "feature_source": str(args.source_dir / "feature_cache" / model_name),
        "rows": {
            "development": len(development_rows),
            "test": len(by_split["test"]),
        },
        "age_group_train_tertiles": age_bounds,
        "gender_normalization": benchmark.GENDER_MAP,
        "selection_protocol": {
            "name": "patient_grouped_development_oof",
            "folds": args.folds,
            "development_splits": ["train", "val"],
            "test_used_for_selection": False,
        },
        "selected_probe": {
            "candidate": candidate,
            "kind": "regularized_logistic",
            "threshold_selected_on": "development_oof_f1",
            "threshold": float(threshold),
            "config": config,
        },
        "metrics": {
            "development_oof": oof_metrics,
            "test": test_metrics,
        },
        "test_patient_bootstrap_95_ci": intervals,
        "worst_group_policy": {
            "attributes": ["sex_gender", "age_group"],
            "age_groups": "tertiles derived from original training ages only",
            "minimum_n": args.min_subgroup_n,
            "minimum_positive": args.min_subgroup_positive,
            "minimum_negative": args.min_subgroup_negative,
            "intersections": False,
        },
        "seed": args.seed,
    }
    (model_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "model": model_name,
                "selected_probe": summary["selected_probe"],
                "development_oof": oof_metrics,
                "test": test_metrics,
            },
            indent=2,
        )
    )


def main() -> None:
    args = parse_args()
    if args.summarize_only:
        benchmark.summarize_results(args.out_dir)
        return

    import numpy as np

    by_split, age_bounds = benchmark.load_manifest(np, args.manifest, None)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "manifest": str(args.manifest),
        "feature_source": str(args.source_dir),
        "models": args.models,
        "development_rows": len(by_split["train"]) + len(by_split["val"]),
        "development_labels": dict(
            Counter(
                str(row["y_true"])
                for split in ("train", "val")
                for row in by_split[split]
            )
        ),
        "test_rows": len(by_split["test"]),
        "folds": args.folds,
        "selection_objective": "patient_grouped_development_oof_f1",
        "test_used_for_selection": False,
        "seed": args.seed,
    }
    (args.out_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(protocol, indent=2, sort_keys=True))

    for model_name in args.models:
        run_model(np, args, model_name, by_split, age_bounds)
    if set(args.models) == set(benchmark.MODELS):
        benchmark.summarize_results(args.out_dir)


if __name__ == "__main__":
    main()
