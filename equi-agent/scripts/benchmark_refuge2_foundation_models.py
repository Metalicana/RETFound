#!/usr/bin/env python3
"""Benchmark REFUGE CFP foundation models using Yusra's RETFound protocol.

The Kaggle download used by this project is stored under a ``REFUGE2`` name,
but it contains the original REFUGE 1,200-image cohort: 400 Train, 400
Validation, and 400 Test images. Classification labels are merged from
``OphthalmicAgent/data_refuge/data.csv``, the same file used by Yusra's
RETFound implementation.

For each comparison model this runner keeps the foundation encoder frozen,
trains one 256-unit MLP head for 30 epochs with balanced sampling, selects the
checkpoint by Validation AUROC, selects the threshold by Validation F1, and
evaluates Test exactly once. Test labels never influence training, checkpoint
selection, or threshold selection.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import benchmark_papila_foundation_models as common


MODELS = ("ret_clip", "retizero", "urfound")
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
        help="Yusra's labeled original-REFUGE CSV.",
    )
    parser.add_argument(
        "--retfound-metadata",
        type=Path,
        default=root / "OphthalmicAgent" / "cfp_glaucoma_training_metadata.json",
        help="Yusra's RETFound result, included unchanged in the combined report.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=(
            root
            / "equi-agent"
            / "outputs"
            / "benchmarks"
            / "refuge_glaucoma_foundations_yusra_protocol_v1"
        ),
    )
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--device", default=None, help="Example: cuda:0 or cpu.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the 1,200-image label/path join and write protocol.json only.",
    )
    parser.add_argument("--summarize-only", action="store_true")

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
    args.retfound_weights = root / "OphthalmicAgent" / "weights" / "cfp_model.pth"
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
                "yusra_filename": filename,
            }
        )

    if unmatched:
        raise ValueError(
            "Yusra labels did not match downloaded images: "
            f"count={len(unmatched)} examples={unmatched[:10]}"
        )
    extra = sorted(set(source_lookup) - seen)
    if extra:
        raise ValueError(
            f"Downloaded manifest contains {len(extra)} images absent from Yusra's CSV: {extra[:10]}"
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
            "cannot insert Yusra's augmentation reproducibly."
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
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
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

    print(f"\n=== REFUGE model={model_name} protocol=yusra ===", flush=True)
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
                    "protocol": "yusra_refuge_frozen_encoder_mlp",
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
            "comparison_to_yusra_retfound": (
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
    return {
        "model_name": "retfound",
        "f1": metrics.get("f1"),
        "worst_group_f1": None,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "auroc": metrics.get("auroc"),
        "threshold": metadata.get("selected_validation_threshold"),
        "best_epoch": metadata.get("best_epoch"),
        "source": str(path),
        "pretraining_overlap_note": "Yusra's tracked RETFound REFUGE run.",
    }


def summarize_results(args) -> None:
    rows: list[dict[str, Any]] = []
    retfound = retfound_report_row(args.retfound_metadata)
    if retfound:
        rows.append(retfound)
    for model_name in MODELS:
        path = args.out_dir / model_name / "summary.json"
        if not path.is_file():
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        metrics = summary["metrics"]["test"]
        rows.append(
            {
                "model_name": model_name,
                "f1": metrics.get("f1"),
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
    common.write_csv_rows(args.out_dir / "refuge_glaucoma_yusra_protocol.csv", rows)

    def number(row, key):
        value = row.get(key)
        return "N/A" if value is None else f"{float(value):.4f}"

    lines = [
        "# REFUGE Glaucoma Foundation Benchmark",
        "",
        (
            "The downloaded archive directory is named REFUGE2, but this experiment "
            "uses the original REFUGE 1,200-image official 400/400/400 cohort and "
            "Yusra's tracked classification labels."
        ),
        "",
        (
            "All comparison heads follow Yusra's frozen-encoder MLP protocol. "
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
    (args.out_dir / "refuge_glaucoma_yusra_protocol.md").write_text(
        report,
        encoding="utf-8",
    )
    print(report)


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
        summarize_results(args)
        return
    if args.validate_only:
        rows_by_split = load_official_splits(args.source_manifest, args.labels_csv)
        write_protocol(args, rows_by_split, args.device or "not_loaded")
        print("REFUGE path/label validation passed; no models were loaded.")
        return

    import numpy as np
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from torchvision import transforms

    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    set_seed(np, torch, args.seed)
    device = torch.device(
        args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    rows_by_split = load_official_splits(args.source_manifest, args.labels_csv)
    write_protocol(args, rows_by_split, device)
    for model_name in args.models:
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
