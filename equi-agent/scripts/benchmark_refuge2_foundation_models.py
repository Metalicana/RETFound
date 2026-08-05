#!/usr/bin/env python3
"""Benchmark frozen retinal foundation models on the original REFUGE cohort.

The Kaggle download used by this project is stored under a ``REFUGE2`` name,
but it contains the original REFUGE 1,200-image cohort: 400 Train, 400
Validation, and 400 Test images. Classification labels are merged from
``OphthalmicAgent/data_refuge/data.csv``, the same file used by the tracked
RETFound reference implementation.

The default ``robust_cv`` protocol extracts deterministic frozen features from
the exact same images for every model. It combines Train and Validation into an
800-image development cohort, selects a class-balanced logistic probe by
repeated stratified cross-validation, selects its decision threshold from
development out-of-fold predictions, and evaluates Test exactly once using an
ensemble of the fitted cross-validation probes. Test labels never influence
probe selection, threshold selection, or fitting.

The previous single-seed MLP implementation remains available as
``--protocol reference_mlp`` for sensitivity analysis. It is not the default because
the fixed high-capacity head substantially overfit several encoders.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import benchmark_papila_foundation_models as common


MODELS = ("retfound", "ret_clip", "retizero", "urfound")
SPLITS = ("train", "val", "test")
EXPECTED_SPLIT_COUNTS = Counter({0: 360, 1: 40})
MODEL_DISPLAY_NAMES = {
    "retfound": "RETFound",
    "ret_clip": "RET-CLIP",
    "retizero": "RetiZero",
    "urfound": "UrFound",
}
OVERLAP_NOTES = {
    "retizero": (
        "The bundled RetiZero code lists REFUGE among its retinal datasets; "
        "the released checkpoint's exact pretraining composition must be disclosed."
    ),
    "urfound": (
        "UrFound states that FLAIR supplied pretraining data, and FLAIR includes "
        "REFUGE; interpret this row with pretraining-overlap risk."
    ),
}


class RefugeImageDataset:
    def __init__(self, Image, torch, rows, transform, image_mode: str):
        self.Image = Image
        self.torch = torch
        self.rows = rows
        self.transform = transform
        self.image_mode = image_mode

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image_path = Path(row["image_path"])
        with self.Image.open(image_path) as source:
            image = source.convert(self.image_mode)
        return (
            self.transform(image),
            self.torch.tensor(float(row["y_true"]), dtype=self.torch.float32),
            row["case_id"],
        )


def parse_args() -> argparse.Namespace:
    root = common.repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=(
            root
            / "equi-agent"
            / "outputs"
            / "manifests"
            / "refuge_kaggle_archive_manifest.csv"
        ),
        help="Manifest from build_refuge2_manifest.py containing real image paths.",
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=root / "OphthalmicAgent" / "data_refuge" / "data.csv",
        help="Labeled original-REFUGE CSV used by the tracked reference run.",
    )
    parser.add_argument(
        "--retfound-metadata",
        type=Path,
        default=root / "OphthalmicAgent" / "cfp_glaucoma_training_metadata.json",
        help="Imported RETFound reference result; used only with --protocol reference_mlp.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=(
            root
            / "equi-agent"
            / "outputs"
            / "benchmarks"
            / "refuge_glaucoma_foundations_robust_cv_v1"
        ),
    )
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument(
        "--protocol",
        choices=("robust_cv", "reference_mlp"),
        default="robust_cv",
        help=(
            "robust_cv uses repeated development OOF selection and a linear-probe "
            "ensemble; reference_mlp preserves the older Train/Validation MLP run."
        ),
    )
    parser.add_argument("--device", default=None, help="Example: cuda:0 or cpu.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=5)
    parser.add_argument(
        "--logreg-c",
        nargs="+",
        type=float,
        default=[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument(
        "--reuse-feature-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--audit-mask-labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Audit label association against CDR values derived from supplied masks.",
    )
    parser.add_argument(
        "--hash-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record SHA-256 for every evaluated image.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the 1,200-image label/path join and write protocol.json only.",
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
        default=(
            root
            / "Foundation_Models"
            / "RET-CLIP-main"
            / "pretrained"
            / "ret_clip_vit_b_16.pt"
        ),
    )
    parser.add_argument(
        "--retizero-root",
        type=Path,
        default=root / "Foundation_Models" / "RetiZero-main",
    )
    parser.add_argument(
        "--retizero-weights",
        type=Path,
        default=(
            root
            / "Foundation_Models"
            / "RetiZero-main"
            / "pretrained"
            / "RetiZero.pth"
        ),
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
        default=(
            root
            / "Foundation_Models"
            / "UrFound-main"
            / "pretrained"
            / "urfound_mm.pth"
        ),
    )
    parser.add_argument(
        "--urfound-model",
        choices=("vit_base_patch16", "vit_large_patch16"),
        default="vit_base_patch16",
    )
    parser.add_argument("--urfound-global-pool", action="store_true")

    args = parser.parse_args()
    # Attributes required by the shared, checkpoint-audited model builders.
    args.dataset = "refuge"
    args.manifest = args.source_manifest
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


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def source_split_from_label_path(filename: str) -> str:
    parts = [part.lower() for part in Path(filename.replace("\\", "/")).parts]
    matches = [
        split
        for source_name, split in (
            ("train", "train"),
            ("validation", "val"),
            ("test", "test"),
        )
        if source_name in parts
    ]
    if len(matches) != 1:
        raise ValueError(f"Cannot infer one REFUGE split from label path: {filename}")
    return matches[0]


def resolve_manifest_image_path(manifest: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest.parent / path
    return path.resolve()


def load_official_splits(
    source_manifest: Path,
    labels_csv: Path,
) -> dict[str, list[dict[str, Any]]]:
    source_rows = read_csv(source_manifest)
    source_lookup: dict[tuple[str, str], dict[str, str]] = {}
    for row in source_rows:
        split = str(row.get("split", "")).strip().lower()
        image_id = str(row.get("image_id", "")).strip().lower()
        if split not in SPLITS or not image_id:
            continue
        key = (split, image_id)
        if key in source_lookup:
            raise ValueError(f"Duplicate source-manifest image: {key}")
        source_lookup[key] = row

    rows_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    unmatched: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for label_row in read_csv(labels_csv):
        filename = str(label_row.get("filename", "")).strip()
        label_text = str(label_row.get("Ground_Truth", "")).strip()
        if label_text not in {"0", "1"}:
            raise ValueError(f"Invalid Ground_Truth={label_text!r} for {filename}")
        split = source_split_from_label_path(filename)
        image_id = Path(filename.replace("\\", "/")).stem
        key = (split, image_id.lower())
        if key in seen:
            raise ValueError(f"Duplicate labeled REFUGE image: {key}")
        seen.add(key)
        source = source_lookup.get(key)
        if source is None:
            unmatched.append(key)
            continue
        image_path = resolve_manifest_image_path(
            source_manifest,
            str(source.get("image_path", "")),
        )
        if not image_path.is_file():
            raise FileNotFoundError(f"REFUGE image not found for {key}: {image_path}")
        rows_by_split[split].append(
            {
                "dataset": "refuge",
                "split": split,
                "case_id": image_id,
                "patient_id": image_id,
                "y_true": int(label_text),
                "image_path": str(image_path),
                "mask_path": str(source.get("mask_path", "")),
                "cup_to_disc_area_ratio": str(
                    source.get("cup_to_disc_area_ratio", "")
                ),
                "vertical_cup_to_disc_ratio": str(
                    source.get("vertical_cup_to_disc_ratio", "")
                ),
                "reference_filename": filename,
            }
        )

    if unmatched:
        raise ValueError(
            "Reference labels did not match downloaded images: "
            f"count={len(unmatched)} examples={unmatched[:10]}"
        )
    extra = sorted(set(source_lookup) - seen)
    if extra:
        raise ValueError(
            f"Downloaded manifest contains {len(extra)} images absent from the reference CSV: {extra[:10]}"
        )

    for split, rows in rows_by_split.items():
        rows.sort(key=lambda row: row["case_id"].lower())
        counts = Counter(row["y_true"] for row in rows)
        if len(rows) != 400 or counts != EXPECTED_SPLIT_COUNTS:
            raise ValueError(
                f"Official REFUGE {split} must contain 400 images with labels "
                f"{dict(EXPECTED_SPLIT_COUNTS)}; found rows={len(rows)} labels={dict(counts)}"
            )
    return rows_by_split


def set_seed(np, torch, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    # torch.initial_seed is set by the DataLoader generator for each worker.
    import numpy as np
    import torch

    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def training_transform(transforms, evaluation_transform):
    if not hasattr(evaluation_transform, "transforms"):
        raise TypeError(
            "Foundation evaluation transform is not a torchvision Compose; "
            "cannot insert the reference augmentation reproducibly."
        )
    steps = [copy.deepcopy(step) for step in evaluation_transform.transforms]
    tensor_indices = [
        index for index, step in enumerate(steps) if isinstance(step, transforms.ToTensor)
    ]
    if len(tensor_indices) != 1:
        raise ValueError(
            f"Expected one ToTensor in foundation transform, found {len(tensor_indices)}"
        )
    index = tensor_indices[0]
    steps[index:index] = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
    ]
    return transforms.Compose(steps)


def make_loader(
    np,
    torch,
    Image,
    DataLoader,
    WeightedRandomSampler,
    rows,
    transform,
    image_mode,
    batch_size,
    num_workers,
    device,
    seed,
    balanced=False,
):
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = None
    if balanced:
        labels = np.asarray([row["y_true"] for row in rows], dtype=int)
        counts = np.bincount(labels, minlength=2)
        if np.any(counts == 0):
            raise ValueError("Balanced sampling requires both REFUGE classes")
        weights = torch.as_tensor(
            [1.0 / counts[label] for label in labels],
            dtype=torch.double,
        )
        sampler = WeightedRandomSampler(
            weights,
            num_samples=len(rows),
            replacement=True,
            generator=generator,
        )
    dataset = RefugeImageDataset(Image, torch, rows, transform, image_mode)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=generator,
    )


def foundation_features(torch, extractor, images):
    values = common.feature_tensor(torch, extractor.forward(images))
    return values.to(dtype=torch.float32)


def build_head(torch, feature_dim: int):
    head = torch.nn.Sequential(
        torch.nn.Linear(feature_dim, 256),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.25),
        torch.nn.Linear(256, 1),
    )
    for module in head.modules():
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.01)
            if module.bias is not None:
                module.bias.data.zero_()
    return head


def infer_feature_dim(torch, extractor, loader, device) -> int:
    images, _labels, _case_ids = next(iter(loader))
    extractor.model.eval()
    with torch.inference_mode():
        features = foundation_features(
            torch,
            extractor,
            images.to(device, non_blocking=True),
        )
    if features.ndim != 2 or features.shape[1] <= 0:
        raise RuntimeError(f"Invalid foundation feature shape: {tuple(features.shape)}")
    return int(features.shape[1])


def predict(np, torch, extractor, head, loader, device):
    extractor.model.eval()
    head.eval()
    targets: list[int] = []
    probabilities: list[float] = []
    case_ids: list[str] = []
    with torch.inference_mode():
        for images, labels, batch_case_ids in loader:
            features = foundation_features(
                torch,
                extractor,
                images.to(device, non_blocking=True),
            )
            logits = head(features).reshape(-1)
            probabilities.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            targets.extend(labels.numpy().astype(int).tolist())
            case_ids.extend(list(batch_case_ids))
    return (
        np.asarray(targets, dtype=int),
        np.asarray(probabilities, dtype=float),
        case_ids,
    )


def metric_dict(np, y_true, probabilities, threshold: float) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = np.asarray(y_true, dtype=int)
    predictions = (np.asarray(probabilities, dtype=float) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    sensitivity = recall_score(y_true, predictions, zero_division=0)
    specificity = tn / (tn + fp) if tn + fp else None
    positive_f1 = float(f1_score(y_true, predictions, zero_division=0))
    weighted_f1 = float(
        f1_score(y_true, predictions, average="weighted", zero_division=0)
    )
    return {
        "n": int(len(y_true)),
        "auroc": float(roc_auc_score(y_true, probabilities)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity) if specificity is not None else None,
        "balanced_accuracy": (
            float((sensitivity + specificity) / 2)
            if specificity is not None
            else None
        ),
        "f1": positive_f1,
        "positive_class_f1": positive_f1,
        "weighted_f1": weighted_f1,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def select_f1_threshold(np, y_true, probabilities) -> tuple[float, float]:
    from sklearn.metrics import f1_score

    candidates = np.unique(np.concatenate(([0.0], probabilities, [1.0])))
    scores = np.asarray(
        [
            f1_score(y_true, probabilities >= threshold, zero_division=0)
            for threshold in candidates
        ]
    )
    best_score = float(scores.max())
    tied = candidates[scores == best_score]
    threshold = float(tied[np.argmin(np.abs(tied - 0.5))])
    return threshold, best_score


def weighted_f1_from_counts(tn: int, fp: int, fn: int, tp: int) -> float:
    positive_f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    negative_f1 = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
    total = tn + fp + fn + tp
    return ((tp + fn) * positive_f1 + (tn + fp) * negative_f1) / total


def write_predictions(
    path: Path,
    model_name: str,
    split: str,
    rows,
    case_ids,
    targets,
    probabilities,
    threshold: float,
) -> None:
    expected_ids = [row["case_id"] for row in rows]
    if case_ids != expected_ids:
        raise ValueError(f"Prediction order mismatch for {model_name}/{split}")
    fields = [
        "dataset",
        "task",
        "model_name",
        "case_id",
        "patient_id",
        "split",
        "y_true",
        "y_prob",
        "threshold",
        "y_pred",
        "is_correct",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row, target, probability in zip(rows, targets, probabilities):
            prediction = int(probability >= threshold)
            writer.writerow(
                {
                    "dataset": "refuge",
                    "task": "glaucoma",
                    "model_name": model_name,
                    "case_id": row["case_id"],
                    "patient_id": row["patient_id"],
                    "split": split,
                    "y_true": int(target),
                    "y_prob": float(probability),
                    "threshold": threshold,
                    "y_pred": prediction,
                    "is_correct": int(prediction == target),
                }
            )


def optional_float(value: object) -> float | None:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def write_image_provenance(args, rows_by_split) -> None:
    if not args.hash_images:
        return
    rows = []
    for split in SPLITS:
        for row in rows_by_split[split]:
            image_path = Path(row["image_path"])
            rows.append(
                {
                    "split": split,
                    "case_id": row["case_id"],
                    "image_path": str(image_path),
                    "image_bytes": image_path.stat().st_size,
                    "image_sha256": sha256(image_path),
                }
            )
    common.write_csv_rows(args.out_dir / "image_provenance.csv", rows)


def audit_mask_label_alignment(np, args, rows_by_split) -> dict[str, Any]:
    if not args.audit_mask_labels:
        return {"enabled": False}

    from sklearn.metrics import roc_auc_score
    from build_refuge2_manifest import mask_features

    case_rows = []
    split_summaries = {}
    for split in SPLITS:
        labels = []
        area_values = []
        vertical_values = []
        for row in rows_by_split[split]:
            area = optional_float(row.get("cup_to_disc_area_ratio"))
            vertical = optional_float(row.get("vertical_cup_to_disc_ratio"))
            mask_text = str(row.get("mask_path", "")).strip()
            mask_path = Path(mask_text).expanduser() if mask_text else None
            if mask_path is not None and not mask_path.is_absolute():
                mask_path = (args.source_manifest.parent / mask_path).resolve()
            if (
                (area is None or vertical is None)
                and mask_path is not None
                and mask_path.is_file()
            ):
                measured = mask_features(mask_path)
                area = optional_float(measured.get("cup_to_disc_area_ratio"))
                vertical = optional_float(measured.get("vertical_cup_to_disc_ratio"))
            case_rows.append(
                {
                    "split": split,
                    "case_id": row["case_id"],
                    "y_true": row["y_true"],
                    "cup_to_disc_area_ratio": "" if area is None else area,
                    "vertical_cup_to_disc_ratio": "" if vertical is None else vertical,
                    "mask_path": str(mask_path) if mask_path is not None else "",
                }
            )
            if area is not None and vertical is not None:
                labels.append(int(row["y_true"]))
                area_values.append(area)
                vertical_values.append(vertical)

        labels_array = np.asarray(labels, dtype=int)
        area_array = np.asarray(area_values, dtype=float)
        vertical_array = np.asarray(vertical_values, dtype=float)

        def audit_feature(values):
            if len(values) == 0 or len(np.unique(labels_array)) < 2:
                return {"n": int(len(values)), "auroc": None}
            positive = values[labels_array == 1]
            negative = values[labels_array == 0]
            return {
                "n": int(len(values)),
                "auroc": float(roc_auc_score(labels_array, values)),
                "positive_mean": float(positive.mean()),
                "negative_mean": float(negative.mean()),
                "positive_median": float(np.median(positive)),
                "negative_median": float(np.median(negative)),
            }

        split_summaries[split] = {
            "expected_rows": len(rows_by_split[split]),
            "measured_rows": len(labels),
            "area_cdr": audit_feature(area_array),
            "vertical_cdr": audit_feature(vertical_array),
        }

    common.write_csv_rows(args.out_dir / "mask_label_audit_cases.csv", case_rows)
    audit = {
        "enabled": True,
        "purpose": (
            "Dataset-integrity diagnostic only; mask features and Test labels are "
            "never used to fit probes or select thresholds."
        ),
        "splits": split_summaries,
    }
    (args.out_dir / "mask_label_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return audit


def make_logistic_probe(C: float, seed: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=float(C),
            class_weight="balanced",
            max_iter=5000,
            random_state=seed,
            solver="lbfgs",
        ),
    )


def evaluate_cv_candidate(np, x_dev, y_dev, x_test, C: float, args):
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    oof_sum = np.zeros(len(y_dev), dtype=float)
    oof_count = np.zeros(len(y_dev), dtype=int)
    test_sum = np.zeros(len(x_test), dtype=float)
    fold_aurocs = []
    fitted_models = 0
    for repeat in range(args.cv_repeats):
        splitter = StratifiedKFold(
            n_splits=args.cv_folds,
            shuffle=True,
            random_state=args.seed + repeat * 1009,
        )
        for fold, (train_index, valid_index) in enumerate(
            splitter.split(x_dev, y_dev),
            start=1,
        ):
            seed = args.seed + repeat * 1009 + fold
            probe = make_logistic_probe(C, seed)
            probe.fit(x_dev[train_index], y_dev[train_index])
            valid_probabilities = probe.predict_proba(x_dev[valid_index])[:, 1]
            oof_sum[valid_index] += valid_probabilities
            oof_count[valid_index] += 1
            test_sum += probe.predict_proba(x_test)[:, 1]
            fitted_models += 1
            fold_aurocs.append(
                float(roc_auc_score(y_dev[valid_index], valid_probabilities))
            )

    if np.any(oof_count != args.cv_repeats):
        raise RuntimeError(
            f"OOF coverage failure for C={C}: counts={np.unique(oof_count).tolist()}"
        )
    oof_probabilities = oof_sum / oof_count
    test_probabilities = test_sum / fitted_models
    threshold, oof_metrics = common.select_f1_threshold(
        np,
        y_dev,
        oof_probabilities,
    )
    oof_auc = float(roc_auc_score(y_dev, oof_probabilities))
    return {
        "C": float(C),
        "mean_fold_auroc": float(np.mean(fold_aurocs)),
        "std_fold_auroc": float(np.std(fold_aurocs, ddof=1)),
        "oof_auroc": oof_auc,
        "oof_threshold": float(threshold),
        "oof_metrics": oof_metrics,
        "oof_probabilities": oof_probabilities,
        "test_probabilities": test_probabilities,
        "ensemble_models": fitted_models,
    }


def select_cv_candidate(candidates):
    """Select representation regularization without consulting Test labels."""
    return max(
        candidates,
        key=lambda item: (
            item["mean_fold_auroc"],
            item["oof_auroc"],
            item["oof_metrics"]["f1"],
            item["oof_metrics"]["balanced_accuracy"],
            -item["C"],
        ),
    )


def bootstrap_intervals(np, y_true, probabilities, threshold, replicates, seed):
    from sklearn.metrics import roc_auc_score

    if replicates <= 0:
        return {}
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    negative = np.flatnonzero(y_true == 0)
    positive = np.flatnonzero(y_true == 1)
    rng = np.random.default_rng(seed)
    values = {
        key: []
        for key in (
            "positive_class_f1",
            "weighted_f1",
            "sensitivity",
            "specificity",
            "balanced_accuracy",
            "auroc",
        )
    }
    for _ in range(replicates):
        indices = np.concatenate(
            (
                rng.choice(negative, size=len(negative), replace=True),
                rng.choice(positive, size=len(positive), replace=True),
            )
        )
        sampled_true = y_true[indices]
        sampled_probabilities = probabilities[indices]
        metrics = common.confusion_metrics(
            np,
            sampled_true,
            sampled_probabilities,
            threshold,
        )
        values["positive_class_f1"].append(metrics["f1"])
        values["weighted_f1"].append(
            weighted_f1_from_counts(
                metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]
            )
        )
        for key in ("sensitivity", "specificity", "balanced_accuracy"):
            values[key].append(metrics[key])
        values["auroc"].append(
            float(roc_auc_score(sampled_true, sampled_probabilities))
        )
    return {
        key: {
            "lower_95": float(np.percentile(metric_values, 2.5)),
            "upper_95": float(np.percentile(metric_values, 97.5)),
        }
        for key, metric_values in values.items()
    }


def write_robust_predictions(
    path: Path,
    model_name: str,
    rows,
    probabilities,
    threshold: float,
    evaluation_role: str,
) -> None:
    output = []
    for row, probability in zip(rows, probabilities):
        prediction = int(float(probability) >= threshold)
        output.append(
            {
                "dataset": "refuge",
                "task": "glaucoma",
                "model_name": model_name,
                "case_id": row["case_id"],
                "patient_id": row["patient_id"],
                "source_split": row["split"],
                "evaluation_role": evaluation_role,
                "y_true": int(row["y_true"]),
                "y_prob": float(probability),
                "threshold": threshold,
                "y_pred": prediction,
                "is_correct": int(prediction == int(row["y_true"])),
            }
        )
    common.write_csv_rows(path, output)


def run_robust_model(
    np,
    torch,
    Image,
    DataLoader,
    transforms,
    tqdm,
    args,
    model_name,
    rows_by_split,
    device,
) -> None:
    print(f"\n=== REFUGE model={model_name} protocol=robust_cv ===", flush=True)
    model_dir = args.out_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    features, feature_provenance = common.extract_model_features(
        np,
        torch,
        Image,
        DataLoader,
        transforms,
        tqdm,
        args,
        model_name,
        rows_by_split,
        device,
    )
    development_rows = rows_by_split["train"] + rows_by_split["val"]
    x_dev = np.concatenate((features["train"], features["val"]), axis=0)
    y_dev = np.asarray([row["y_true"] for row in development_rows], dtype=int)
    x_test = features["test"]
    y_test = np.asarray([row["y_true"] for row in rows_by_split["test"]], dtype=int)
    if not np.isfinite(x_dev).all() or not np.isfinite(x_test).all():
        raise RuntimeError(f"Non-finite frozen features for {model_name}")

    candidates = [
        evaluate_cv_candidate(np, x_dev, y_dev, x_test, C, args)
        for C in args.logreg_c
    ]
    selected = select_cv_candidate(candidates)
    threshold = selected["oof_threshold"]
    test_probabilities = selected["test_probabilities"]
    test_metrics = common.confusion_metrics(
        np,
        y_test,
        test_probabilities,
        threshold,
    )
    test_metrics["positive_class_f1"] = test_metrics["f1"]
    test_metrics["weighted_f1"] = weighted_f1_from_counts(
        test_metrics["tn"],
        test_metrics["fp"],
        test_metrics["fn"],
        test_metrics["tp"],
    )
    test_metrics["auroc"] = common.roc_auc(np, y_test, test_probabilities)
    intervals = bootstrap_intervals(
        np,
        y_test,
        test_probabilities,
        threshold,
        args.bootstrap_replicates,
        args.seed + MODELS.index(model_name) * 10000,
    )

    search_rows = []
    for candidate in candidates:
        search_rows.append(
            {
                "C": candidate["C"],
                "mean_fold_auroc": candidate["mean_fold_auroc"],
                "std_fold_auroc": candidate["std_fold_auroc"],
                "oof_auroc": candidate["oof_auroc"],
                "oof_threshold": candidate["oof_threshold"],
                "oof_f1": candidate["oof_metrics"]["f1"],
                "oof_sensitivity": candidate["oof_metrics"]["sensitivity"],
                "oof_specificity": candidate["oof_metrics"]["specificity"],
                "oof_balanced_accuracy": candidate["oof_metrics"]["balanced_accuracy"],
                "selected": candidate is selected,
            }
        )
    common.write_csv_rows(model_dir / "probe_search.csv", search_rows)
    write_robust_predictions(
        model_dir / "predictions_development_oof.csv",
        model_name,
        development_rows,
        selected["oof_probabilities"],
        threshold,
        "development_repeated_oof",
    )
    write_robust_predictions(
        model_dir / "predictions_test.csv",
        model_name,
        rows_by_split["test"],
        test_probabilities,
        threshold,
        "untouched_test",
    )

    weight_path = common.model_weight_path(args, model_name)
    summary = {
        "dataset_for_reporting": "REFUGE",
        "archive_directory_name": "REFUGE2",
        "cohort_identity": "original REFUGE 1,200-image cohort",
        "model_name": model_name,
        "protocol": "robust_cv",
        "feature_provenance": feature_provenance,
        "checkpoint": str(weight_path),
        "checkpoint_sha256": sha256(weight_path),
        "source_manifest": str(args.source_manifest),
        "source_manifest_sha256": sha256(args.source_manifest),
        "labels_csv": str(args.labels_csv),
        "labels_csv_sha256": sha256(args.labels_csv),
        "feature_shapes": {
            split: list(features[split].shape) for split in SPLITS
        },
        "development": {
            "n": len(development_rows),
            "positive_n": int(y_dev.sum()),
            "negative_n": int(len(y_dev) - y_dev.sum()),
            "source_splits": ["train", "val"],
            "cv_folds": args.cv_folds,
            "cv_repeats": args.cv_repeats,
            "selected_C": selected["C"],
            "selection_objective": "highest mean repeated-fold AUROC",
            "oof_threshold_objective": "highest positive-class F1",
            "oof_threshold": threshold,
            "oof_auroc": selected["oof_auroc"],
            "oof_metrics": selected["oof_metrics"],
            "ensemble_models": selected["ensemble_models"],
        },
        "test": {
            "used_for_selection": False,
            "metrics": test_metrics,
            "bootstrap_95_ci": intervals,
        },
        "worst_group_f1": None,
        "worst_group_note": "Unavailable because REFUGE has no demographic metadata.",
        "pretraining_overlap_note": OVERLAP_NOTES.get(
            model_name,
            "Checkpoint training-data overlap must be documented from its model card.",
        ),
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
                "selected_C": selected["C"],
                "development_oof_auroc": selected["oof_auroc"],
                "development_oof_f1": selected["oof_metrics"]["f1"],
                "selected_threshold": threshold,
                "test": test_metrics,
            },
            indent=2,
        )
    )


def train_model(
    np,
    torch,
    Image,
    DataLoader,
    WeightedRandomSampler,
    transforms,
    args,
    model_name: str,
    rows_by_split,
    device,
) -> None:
    from sklearn.metrics import roc_auc_score

    print(f"\n=== REFUGE model={model_name} protocol=reference_mlp ===", flush=True)
    model_dir = args.out_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    extractor = common.BUILDERS[model_name](args, torch, transforms, device)
    for parameter in extractor.model.parameters():
        parameter.requires_grad = False
    extractor.model.eval()

    train_loader = make_loader(
        np,
        torch,
        Image,
        DataLoader,
        WeightedRandomSampler,
        rows_by_split["train"],
        training_transform(transforms, extractor.transform),
        extractor.image_mode,
        args.batch_size,
        args.num_workers,
        device,
        args.seed,
        balanced=True,
    )
    evaluation_loaders = {
        split: make_loader(
            np,
            torch,
            Image,
            DataLoader,
            WeightedRandomSampler,
            rows_by_split[split],
            extractor.transform,
            extractor.image_mode,
            args.batch_size,
            args.num_workers,
            device,
            args.seed + index + 1,
            balanced=False,
        )
        for index, split in enumerate(("val", "test"))
    }

    feature_dim = infer_feature_dim(
        torch,
        extractor,
        evaluation_loaders["val"],
        device,
    )
    head = build_head(torch, feature_dim).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=args.head_lr,
        weight_decay=args.weight_decay,
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    checkpoint_path = model_dir / "best_head.pth"
    best_auc = -float("inf")
    best_epoch = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        extractor.model.eval()
        head.train()
        running_loss = 0.0
        batches = 0
        for images, labels, _case_ids in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.no_grad():
                features = foundation_features(torch, extractor, images)
            optimizer.zero_grad(set_to_none=True)
            logits = head(features).reshape(-1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            batches += 1

        val_targets, val_probabilities, _ = predict(
            np,
            torch,
            extractor,
            head,
            evaluation_loaders["val"],
            device,
        )
        val_auc = float(roc_auc_score(val_targets, val_probabilities))
        mean_loss = running_loss / max(1, batches)
        history.append(
            {"epoch": epoch, "train_loss": mean_loss, "validation_auroc": val_auc}
        )
        print(
            f"model={model_name} epoch={epoch}/{args.epochs} "
            f"loss={mean_loss:.6f} val_auc={val_auc:.6f}",
            flush=True,
        )
        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            torch.save(
                {
                    "model_name": model_name,
                    "feature_dim": feature_dim,
                    "head_state_dict": head.state_dict(),
                    "epoch": epoch,
                    "validation_auroc": val_auc,
                    "protocol": "reference_refuge_frozen_encoder_mlp",
                },
                checkpoint_path,
            )

    checkpoint = common.safe_torch_load(torch, checkpoint_path)
    head.load_state_dict(checkpoint["head_state_dict"])
    head.eval()
    val_targets, val_probabilities, val_case_ids = predict(
        np,
        torch,
        extractor,
        head,
        evaluation_loaders["val"],
        device,
    )
    threshold, validation_f1 = select_f1_threshold(
        np,
        val_targets,
        val_probabilities,
    )
    test_targets, test_probabilities, test_case_ids = predict(
        np,
        torch,
        extractor,
        head,
        evaluation_loaders["test"],
        device,
    )
    metrics = {
        "val": metric_dict(np, val_targets, val_probabilities, threshold),
        "test": metric_dict(np, test_targets, test_probabilities, threshold),
    }
    write_predictions(
        model_dir / "predictions_val.csv",
        model_name,
        "val",
        rows_by_split["val"],
        val_case_ids,
        val_targets,
        val_probabilities,
        threshold,
    )
    write_predictions(
        model_dir / "predictions_test.csv",
        model_name,
        "test",
        rows_by_split["test"],
        test_case_ids,
        test_targets,
        test_probabilities,
        threshold,
    )
    common.write_csv_rows(model_dir / "epoch_history.csv", history)

    weight_path = common.model_weight_path(args, model_name)
    summary = {
        "dataset": "refuge",
        "archive_directory_name": "REFUGE2",
        "cohort_identity": "original REFUGE 1,200-image official Train/Validation/Test cohort",
        "model_name": model_name,
        "feature_provenance": extractor.provenance,
        "checkpoint_sha256": sha256(weight_path),
        "source_manifest": str(args.source_manifest),
        "source_manifest_sha256": sha256(args.source_manifest),
        "labels_csv": str(args.labels_csv),
        "labels_csv_sha256": sha256(args.labels_csv),
        "rows": {split: len(rows) for split, rows in rows_by_split.items()},
        "split_labels": {
            split: dict(Counter(str(row["y_true"]) for row in rows))
            for split, rows in rows_by_split.items()
        },
        "protocol": {
            "encoder": "frozen; evaluation mode during head training",
            "head": f"{feature_dim} -> 256 -> ReLU -> Dropout(0.25) -> 1",
            "balanced_training_sampler": True,
            "loss": "BCEWithLogitsLoss without pos_weight",
            "optimizer": "AdamW",
            "head_learning_rate": args.head_lr,
            "weight_decay": args.weight_decay,
            "maximum_training_epochs": args.epochs,
            "checkpoint_selection": "highest Validation AUROC",
            "threshold_selection": "highest positive-class Validation F1; tie nearest 0.5",
            "test_used_for_selection": False,
            "train_augmentation": "model-native preprocessing plus horizontal flip and +/-15 degree rotation",
            "comparison_to_reference_retfound": (
                "Same split, sampler, MLP hidden width, augmentation, optimizer, "
                "checkpoint criterion, and threshold criterion. Model-native input "
                "normalization is retained; only RETFound has its architecture-specific fc_norm."
            ),
        },
        "best_epoch": best_epoch,
        "best_validation_auroc": best_auc,
        "selected_validation_threshold": threshold,
        "validation_f1_at_selected_threshold": validation_f1,
        "metrics": metrics,
        "worst_group_f1": None,
        "worst_group_note": "Not available: REFUGE classification CSV has no demographic metadata.",
        "pretraining_overlap_note": OVERLAP_NOTES.get(model_name, "No overlap claim made; verify checkpoint documentation."),
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
                "best_epoch": best_epoch,
                "best_validation_auroc": best_auc,
                "selected_validation_threshold": threshold,
                "test": metrics["test"],
            },
            indent=2,
        )
    )


def retfound_report_row(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metrics = metadata.get("final_test_metrics", {})
    if not metrics:
        return None
    sensitivity = metrics.get("sensitivity")
    specificity = metrics.get("specificity")
    balanced_accuracy = (
        (float(sensitivity) + float(specificity)) / 2
        if sensitivity is not None and specificity is not None
        else None
    )
    tn = int(metrics.get("tn", 0))
    fp = int(metrics.get("fp", 0))
    fn = int(metrics.get("fn", 0))
    tp = int(metrics.get("tp", 0))
    weighted_f1 = weighted_f1_from_counts(tn, fp, fn, tp)
    return {
        "model_name": "retfound",
        "f1": weighted_f1,
        "positive_class_f1": metrics.get("f1"),
        "worst_group_f1": None,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "auroc": metrics.get("auroc"),
        "threshold": metadata.get("selected_validation_threshold"),
        "best_epoch": metadata.get("best_epoch"),
        "source": str(path),
        "pretraining_overlap_note": "Tracked RETFound REFUGE reference run.",
    }


def summarize_results(args) -> None:
    rows: list[dict[str, Any]] = []
    retfound = retfound_report_row(args.retfound_metadata)
    if retfound:
        rows.append(retfound)
    for model_name in ("ret_clip", "retizero", "urfound"):
        path = args.out_dir / model_name / "summary.json"
        if not path.is_file():
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        metrics = summary["metrics"]["test"]
        weighted_f1 = metrics.get("weighted_f1")
        if weighted_f1 is None:
            weighted_f1 = weighted_f1_from_counts(
                int(metrics["tn"]),
                int(metrics["fp"]),
                int(metrics["fn"]),
                int(metrics["tp"]),
            )
        rows.append(
            {
                "model_name": model_name,
                "f1": weighted_f1,
                "positive_class_f1": metrics.get("positive_class_f1", metrics.get("f1")),
                "worst_group_f1": None,
                "sensitivity": metrics.get("sensitivity"),
                "specificity": metrics.get("specificity"),
                "balanced_accuracy": metrics.get("balanced_accuracy"),
                "auroc": metrics.get("auroc"),
                "threshold": summary.get("selected_validation_threshold"),
                "best_epoch": summary.get("best_epoch"),
                "source": str(path),
                "pretraining_overlap_note": summary.get("pretraining_overlap_note", ""),
            }
        )
    if not rows:
        raise FileNotFoundError(f"No REFUGE result summaries found under {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    common.write_csv_rows(args.out_dir / "refuge_glaucoma_reference_protocol.csv", rows)

    def number(row, key):
        value = row.get(key)
        return "N/A" if value is None else f"{float(value):.4f}"

    lines = [
        "# REFUGE Glaucoma Foundation Benchmark",
        "",
        (
            "The downloaded archive directory is named REFUGE2, but this experiment "
            "uses the original REFUGE 1,200-image official 400/400/400 cohort and "
            "the tracked reference classification labels."
        ),
        "",
        (
            "All comparison heads follow the reference frozen-encoder MLP protocol. "
            "F1 is support-weighted, matching the manuscript table convention. "
            "Worst-group F1 is unavailable because demographic metadata are absent."
        ),
        "",
        "| Model | F1 | Worst-group F1 | Sensitivity | Specificity | Balanced accuracy | AUROC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {MODEL_DISPLAY_NAMES[row['model_name']]} | {number(row, 'f1')} | "
            f"N/A | {number(row, 'sensitivity')} | {number(row, 'specificity')} | "
            f"{number(row, 'balanced_accuracy')} | {number(row, 'auroc')} |"
        )
    lines.extend(["", "## Provenance Notes", ""])
    for row in rows:
        lines.append(
            f"- **{MODEL_DISPLAY_NAMES[row['model_name']]}:** "
            f"{row.get('pretraining_overlap_note', '')}"
        )
    report = "\n".join(lines) + "\n"
    (args.out_dir / "refuge_glaucoma_reference_protocol.md").write_text(
        report,
        encoding="utf-8",
    )
    print(report)


def summarize_robust_results(args) -> None:
    rows = []
    for model_name in MODELS:
        path = args.out_dir / model_name / "summary.json"
        if not path.is_file():
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("protocol") != "robust_cv":
            continue
        metrics = summary["test"]["metrics"]
        intervals = summary["test"].get("bootstrap_95_ci", {})
        weighted_f1 = metrics.get("weighted_f1")
        if weighted_f1 is None:
            weighted_f1 = weighted_f1_from_counts(
                int(metrics["tn"]),
                int(metrics["fp"]),
                int(metrics["fn"]),
                int(metrics["tp"]),
            )
        row = {
            "model_name": model_name,
            "f1": weighted_f1,
            "positive_class_f1": metrics.get("positive_class_f1", metrics.get("f1")),
            "f1_ci_lower": intervals.get("weighted_f1", {}).get("lower_95"),
            "f1_ci_upper": intervals.get("weighted_f1", {}).get("upper_95"),
            "worst_group_f1": None,
            "sensitivity": metrics.get("sensitivity"),
            "specificity": metrics.get("specificity"),
            "balanced_accuracy": metrics.get("balanced_accuracy"),
            "auroc": metrics.get("auroc"),
            "threshold": summary["development"].get("oof_threshold"),
            "selected_C": summary["development"].get("selected_C"),
            "development_oof_f1": summary["development"]
            .get("oof_metrics", {})
            .get("f1"),
            "development_oof_auroc": summary["development"].get("oof_auroc"),
            "test_n": metrics.get("n"),
            "tn": metrics.get("tn"),
            "fp": metrics.get("fp"),
            "fn": metrics.get("fn"),
            "tp": metrics.get("tp"),
            "source": str(path),
            "pretraining_overlap_note": summary.get("pretraining_overlap_note", ""),
        }
        rows.append(row)
    if not rows:
        raise FileNotFoundError(
            f"No robust REFUGE result summaries found under {args.out_dir}"
        )
    common.write_csv_rows(args.out_dir / "refuge_glaucoma_robust_cv.csv", rows)

    def number(row, key):
        value = row.get(key)
        return "N/A" if value is None else f"{float(value):.4f}"

    lines = [
        "# REFUGE Glaucoma Frozen-Feature Benchmark",
        "",
        (
            "The downloaded directory is named REFUGE2, but this is the original "
            "REFUGE 1,200-image official cohort. It is not a full REFUGE2 result."
        ),
        "",
        (
            "Every model uses deterministic frozen features from the same files. "
            "Train and Validation form the 800-image development set. A standardized "
            "class-balanced logistic probe is selected by repeated stratified "
            "cross-validation, the threshold is selected from development OOF "
            "predictions, and the untouched 400-image Test split is evaluated once. "
            "F1 is support-weighted to match the manuscript table convention."
        ),
        "",
        "| Model | F1 | 95% CI | Worst-group F1 | Sensitivity | Specificity | Balanced accuracy | AUROC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        ci = (
            "N/A"
            if row["f1_ci_lower"] is None
            else f"{row['f1_ci_lower']:.4f}-{row['f1_ci_upper']:.4f}"
        )
        lines.append(
            f"| {MODEL_DISPLAY_NAMES[row['model_name']]} | {number(row, 'f1')} | "
            f"{ci} | N/A | {number(row, 'sensitivity')} | "
            f"{number(row, 'specificity')} | {number(row, 'balanced_accuracy')} | "
            f"{number(row, 'auroc')} |"
        )
    lines.extend(["", "## Provenance Notes", ""])
    for row in rows:
        lines.append(
            f"- **{MODEL_DISPLAY_NAMES[row['model_name']]}:** "
            f"{row.get('pretraining_overlap_note', '')}"
        )
    report = "\n".join(lines) + "\n"
    (args.out_dir / "refuge_glaucoma_robust_cv.md").write_text(
        report,
        encoding="utf-8",
    )
    print(report)


def write_robust_protocol(args, rows_by_split, device, mask_audit) -> None:
    script_path = Path(__file__).resolve()
    protocol = {
        "dataset_for_reporting": "REFUGE",
        "archive_directory_name": "REFUGE2",
        "cohort_identity": "original REFUGE 1,200-image cohort",
        "full_refuge2_result": False,
        "source_manifest": str(args.source_manifest),
        "source_manifest_sha256": sha256(args.source_manifest),
        "labels_csv": str(args.labels_csv),
        "labels_csv_sha256": sha256(args.labels_csv),
        "script_path": str(script_path),
        "script_sha256": sha256(script_path),
        "command_argv": sys.argv,
        "models": args.models,
        "official_split_rows": {
            split: len(rows) for split, rows in rows_by_split.items()
        },
        "official_split_labels": {
            split: dict(Counter(str(row["y_true"]) for row in rows))
            for split, rows in rows_by_split.items()
        },
        "development_rows": len(rows_by_split["train"])
        + len(rows_by_split["val"]),
        "development_source_splits": ["train", "val"],
        "probe": "StandardScaler + class-balanced logistic regression",
        "regularization_grid_C": args.logreg_c,
        "regularization_selection": "highest mean repeated-fold development AUROC",
        "threshold_selection": "highest development repeated-OOF positive-class F1",
        "test_prediction": "mean probability from all selected-C CV probes",
        "cv_folds": args.cv_folds,
        "cv_repeats": args.cv_repeats,
        "bootstrap_replicates": args.bootstrap_replicates,
        "test_used_for_selection": False,
        "mask_label_audit": mask_audit,
        "image_provenance_path": (
            str(args.out_dir / "image_provenance.csv") if args.hash_images else ""
        ),
        "device": str(device),
        "seed": args.seed,
    }
    (args.out_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(protocol, indent=2, sort_keys=True))


def write_protocol(args, rows_by_split, device) -> None:
    protocol = {
        "dataset_for_reporting": "REFUGE",
        "archive_directory_name": "REFUGE2",
        "cohort_identity": "original REFUGE 1,200-image cohort",
        "source_manifest": str(args.source_manifest),
        "source_manifest_sha256": sha256(args.source_manifest),
        "labels_csv": str(args.labels_csv),
        "labels_csv_sha256": sha256(args.labels_csv),
        "models": args.models,
        "official_split_rows": {
            split: len(rows) for split, rows in rows_by_split.items()
        },
        "official_split_labels": {
            split: dict(Counter(str(row["y_true"]) for row in rows))
            for split, rows in rows_by_split.items()
        },
        "training_strategy": (
            "balanced sampling on Train; one fixed 256-unit MLP; best checkpoint "
            "by Validation AUROC; threshold by Validation F1; one Test evaluation"
        ),
        "test_used_for_selection": False,
        "device": str(device),
        "seed": args.seed,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(protocol, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    if args.summarize_only:
        if args.protocol == "robust_cv":
            summarize_robust_results(args)
        else:
            summarize_results(args)
        return

    import numpy as np
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from torchvision import transforms
    from tqdm import tqdm

    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.cv_folds < 2 or args.cv_repeats < 1:
        raise ValueError("--cv-folds must be >=2 and --cv-repeats must be >=1")
    if not args.logreg_c or any(value <= 0 for value in args.logreg_c):
        raise ValueError("--logreg-c values must all be positive")
    if args.bootstrap_replicates < 0:
        raise ValueError("--bootstrap-replicates must be >=0")
    set_seed(np, torch, args.seed)
    device = torch.device(
        args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    rows_by_split = load_official_splits(args.source_manifest, args.labels_csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_image_provenance(args, rows_by_split)
    mask_audit = audit_mask_label_alignment(np, args, rows_by_split)

    if args.protocol == "robust_cv":
        write_robust_protocol(args, rows_by_split, device, mask_audit)
    else:
        write_protocol(args, rows_by_split, device)
    if args.validate_only:
        print("REFUGE path/label validation passed; no models were loaded.")
        return

    if args.protocol == "robust_cv":
        for model_name in args.models:
            set_seed(np, torch, args.seed)
            run_robust_model(
                np,
                torch,
                Image,
                DataLoader,
                transforms,
                tqdm,
                args,
                model_name,
                rows_by_split,
                device,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
        summarize_robust_results(args)
        return

    legacy_models = [model for model in args.models if model != "retfound"]
    if "retfound" in args.models:
        print(
            "reference_mlp_retfound=imported_from_metadata_not_rerun",
            flush=True,
        )
    for model_name in legacy_models:
        set_seed(np, torch, args.seed)
        train_model(
            np,
            torch,
            Image,
            DataLoader,
            WeightedRandomSampler,
            transforms,
            args,
            model_name,
            rows_by_split,
            device,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
    summarize_results(args)


if __name__ == "__main__":
    main()
