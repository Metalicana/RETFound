from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path


STANDARD_COLUMNS = [
    "patient_id",
    "eye_id",
    "visit_id",
    "image_id",
    "dataset",
    "task",
    "model_name",
    "y_true",
    "y_prob",
    "y_pred",
    "split",
    "race",
    "ethnicity",
    "sex_gender",
    "age",
    "age_group",
    "metadata_missing_flag",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def equi_agent_root() -> Path:
    return Path(__file__).resolve().parents[1]


def require_runtime_libs():
    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models, transforms
    from tqdm import tqdm

    return np, pd, torch, Image, DataLoader, Dataset, models, transforms, tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train simple supervised ResNet/ViT FairVision baselines and emit standard predictions."
    )
    parser.add_argument("--manifest-dir", type=Path, default=equi_agent_root() / "outputs" / "manifests")
    parser.add_argument("--task", choices=("amd", "dr", "glaucoma"), required=True)
    parser.add_argument("--modality", choices=("oct", "slo"), required=True)
    parser.add_argument("--arch", choices=("resnet50", "vit_b_16"), required=True)
    parser.add_argument("--out-val", type=Path, required=True)
    parser.add_argument("--out-test", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None, help="Example: cuda, cuda:0, or cpu.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--oct-representation",
        choices=("center", "mean", "max", "mean_max_center", "three_slices"),
        default="center",
        help="How to convert an OCT volume to a 2D/RGB image for 2D backbones.",
    )
    parser.add_argument("--freeze-backbone", action="store_true", help="Train only the final classifier head.")
    parser.add_argument("--balanced-sampler", action="store_true", help="Sample positives and negatives with equal weight.")
    parser.add_argument(
        "--imagenet-weights",
        action="store_true",
        help="Use torchvision ImageNet weights if available locally. May try to download if not cached.",
    )
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    parser.add_argument(
        "--path-prefix-from",
        default=None,
        help="Optional stale prefix in manifest image_path values to replace before loading NPZ files.",
    )
    parser.add_argument(
        "--path-prefix-to",
        default=None,
        help="Replacement prefix for --path-prefix-from. Useful when manifests were built on another machine.",
    )
    return parser.parse_args()


def set_seed(torch, seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_name(arch: str, modality: str) -> str:
    return f"{arch}_{modality}_supervised"


def build_model(torch, models, arch: str, imagenet_weights: bool):
    if arch == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if imagenet_weights else None
        model = models.resnet50(weights=weights)
        model.fc = torch.nn.Linear(model.fc.in_features, 1)
        return model
    if arch == "vit_b_16":
        weights = models.ViT_B_16_Weights.DEFAULT if imagenet_weights else None
        model = models.vit_b_16(weights=weights)
        model.heads.head = torch.nn.Linear(model.heads.head.in_features, 1)
        return model
    raise ValueError(f"Unsupported arch: {arch}")


def freeze_backbone(torch, model, arch: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    if arch == "resnet50":
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
        return
    if arch == "vit_b_16":
        for parameter in model.heads.head.parameters():
            parameter.requires_grad = True
        return
    raise ValueError(f"Unsupported arch: {arch}")


def normalize_to_uint8(np, image):
    image = np.asarray(image, dtype="float32")
    finite = np.isfinite(image)
    if not finite.any():
        return np.zeros(image.shape, dtype="uint8")
    lo, hi = np.percentile(image[finite], [1.0, 99.0])
    if hi <= lo:
        lo = float(np.nanmin(image[finite]))
        hi = float(np.nanmax(image[finite]))
    if hi <= lo:
        return np.zeros(image.shape, dtype="uint8")
    image = (image - lo) / (hi - lo)
    image = np.clip(image, 0.0, 1.0)
    return (image * 255.0).astype("uint8")


def slice_axis(np, volume) -> int:
    volume = np.asarray(volume)
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D OCT volume, got shape={volume.shape}")
    # FairVision and GDP commonly store OCT as slices x H x W. If this differs,
    # the slice axis is usually the smallest dimension.
    return min(range(3), key=lambda axis: volume.shape[axis])


def take_slice(np, volume, axis: int, fraction: float):
    index = int(round((volume.shape[axis] - 1) * fraction))
    return np.take(volume, index, axis=axis)


def oct_to_image(np, volume, representation: str):
    volume = np.asarray(volume, dtype="float32")
    axis = slice_axis(np, volume)
    if representation == "center":
        return take_slice(np, volume, axis, 0.5)
    if representation == "mean":
        return np.mean(volume, axis=axis)
    if representation == "max":
        return np.max(volume, axis=axis)
    if representation == "mean_max_center":
        center = take_slice(np, volume, axis, 0.5)
        mean = np.mean(volume, axis=axis)
        max_proj = np.max(volume, axis=axis)
        return np.stack([center, mean, max_proj], axis=-1)
    if representation == "three_slices":
        return np.stack(
            [
                take_slice(np, volume, axis, 0.25),
                take_slice(np, volume, axis, 0.5),
                take_slice(np, volume, axis, 0.75),
            ],
            axis=-1,
        )
    raise ValueError(f"Unknown OCT representation: {representation}")


class FairVisionImageDataset:
    def __init__(self, np, Image, frame, modality: str, transform, oct_representation: str):
        self.np = np
        self.Image = Image
        self.frame = frame.reset_index(drop=True)
        self.modality = modality
        self.transform = transform
        self.oct_representation = oct_representation

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        with self.np.load(row["image_path"]) as data:
            if self.modality == "oct":
                volume = data["oct_bscans"]
                image = oct_to_image(self.np, volume, self.oct_representation)
            else:
                image = data["slo_fundus"]
        image = normalize_to_uint8(self.np, image)
        image = self.Image.fromarray(image).convert("RGB")
        return self.transform(image), float(row["y_true"]), index


def split_frame(pd, manifest, split: str, limit: int | None):
    frame = manifest[manifest["split"] == split].copy()
    if limit:
        frame = frame.head(limit).copy()
    frame["y_true"] = pd.to_numeric(frame["y_true"], errors="coerce")
    frame = frame[frame["y_true"].isin([0, 1])].copy()
    return frame


def rewrite_image_paths(manifest, path_prefix_from: str | None, path_prefix_to: str | None):
    if not path_prefix_from:
        return manifest
    if path_prefix_to is None:
        raise ValueError("--path-prefix-to is required when --path-prefix-from is set")
    manifest = manifest.copy()
    manifest["image_path"] = manifest["image_path"].astype(str).str.replace(
        path_prefix_from,
        path_prefix_to,
        n=1,
        regex=False,
    )
    return manifest


def train_one_epoch(torch, tqdm, model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    n = 0
    for images, labels, _ in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        labels = labels.float().to(device).view(-1, 1)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += float(loss.detach().cpu().item()) * len(labels)
        n += len(labels)
    return running_loss / max(n, 1)


def build_balanced_sampler(torch, train_df):
    labels = train_df["y_true"].astype(int).to_numpy()
    positives = max(int(labels.sum()), 1)
    negatives = max(int(len(labels) - labels.sum()), 1)
    weights = [0.5 / positives if label == 1 else 0.5 / negatives for label in labels]
    return torch.utils.data.WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )


def predict(torch, tqdm, model, loader, device):
    model.eval()
    probs_by_index = {}
    with torch.no_grad():
        for images, _, indices in tqdm(loader, desc="predict", leave=False):
            images = images.to(device)
            logits = model(images).view(-1)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            for index, prob in zip(indices.numpy().tolist(), probs.tolist()):
                probs_by_index[index] = float(prob)
    return probs_by_index


def classification_summary(np, sklearn_metrics, frame, probs_by_index: dict[int, float]) -> dict[str, float]:
    labels = frame.reset_index(drop=True)["y_true"].astype(int).to_numpy()
    probs = np.asarray([probs_by_index[i] for i in range(len(labels))], dtype="float32")
    preds = (probs >= 0.5).astype(int)
    try:
        auroc = float(sklearn_metrics.roc_auc_score(labels, probs))
    except Exception:
        auroc = float("nan")
    return {
        "auroc": auroc,
        "f1": float(sklearn_metrics.f1_score(labels, preds, zero_division=0)),
        "balanced_accuracy": float(sklearn_metrics.balanced_accuracy_score(labels, preds)),
    }


def to_standard_predictions(pd, frame, probs_by_index: dict[int, float], arch: str, modality: str):
    rows = []
    for index, row in frame.reset_index(drop=True).iterrows():
        y_prob = probs_by_index[index]
        rows.append(
            {
                "patient_id": row["patient_id"],
                "eye_id": row.get("eye_id", ""),
                "visit_id": row.get("visit_id", ""),
                "image_id": row["image_id"],
                "dataset": row["dataset"],
                "task": row["task"],
                "model_name": model_name(arch, modality),
                "y_true": int(row["y_true"]),
                "y_prob": y_prob,
                "y_pred": int(y_prob >= 0.5),
                "split": row["split"],
                "race": row["race"],
                "ethnicity": row["ethnicity"],
                "sex_gender": row["sex_gender"],
                "age": row["age"],
                "age_group": row["age_group"],
                "metadata_missing_flag": bool(row["metadata_missing_flag"]),
            }
        )
    return pd.DataFrame(rows, columns=STANDARD_COLUMNS)


def main() -> None:
    args = parse_args()
    np, pd, torch, Image, DataLoader, Dataset, models, transforms, tqdm = require_runtime_libs()
    from sklearn import metrics as sklearn_metrics

    set_seed(torch, args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    manifest = pd.read_csv(args.manifest_dir / f"fairvision_{args.task}.csv")
    manifest = rewrite_image_paths(manifest, args.path_prefix_from, args.path_prefix_to)

    train_df = split_frame(pd, manifest, "train", args.limit_train)
    val_df = split_frame(pd, manifest, "val", args.limit_val)
    test_df = split_frame(pd, manifest, "test", args.limit_test)
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError(
            f"Missing split rows: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )

    transform_train = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    transform_eval = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_set = FairVisionImageDataset(np, Image, train_df, args.modality, transform_train, args.oct_representation)
    val_set = FairVisionImageDataset(np, Image, val_df, args.modality, transform_eval, args.oct_representation)
    test_set = FairVisionImageDataset(np, Image, test_df, args.modality, transform_eval, args.oct_representation)

    sampler = build_balanced_sampler(torch, train_df) if args.balanced_sampler else None

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(torch, models, args.arch, args.imagenet_weights).to(device)
    if args.freeze_backbone:
        freeze_backbone(torch, model, args.arch)
    positives = float(train_df["y_true"].sum())
    negatives = float(len(train_df) - positives)
    pos_weight = torch.tensor([negatives / max(positives, 1.0)], device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    print(
        f"train_rows={len(train_df)} val_rows={len(val_df)} test_rows={len(test_df)} "
        f"train_pos={int(positives)} train_neg={int(negatives)} "
        f"oct_representation={args.oct_representation} freeze_backbone={args.freeze_backbone} "
        f"balanced_sampler={args.balanced_sampler}"
    )

    best_state = None
    best_val_auroc = float("-inf")
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(torch, tqdm, model, train_loader, optimizer, criterion, device)
        val_probs = predict(torch, tqdm, model, val_loader, device)
        val_summary = classification_summary(np, sklearn_metrics, val_df, val_probs)
        val_auroc = val_summary["auroc"]
        if np.isfinite(val_auroc) and val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(
            "epoch={epoch} train_loss={loss:.6f} val_auroc={auroc:.6f} "
            "val_f1@0.5={f1:.6f} val_bal_acc@0.5={bal:.6f} best_epoch={best_epoch}".format(
                epoch=epoch,
                loss=loss,
                auroc=val_summary["auroc"],
                f1=val_summary["f1"],
                bal=val_summary["balanced_accuracy"],
                best_epoch=best_epoch,
            )
        )

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"loaded_best_epoch={best_epoch} best_val_auroc={best_val_auroc:.6f}")

    if args.checkpoint:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "arch": args.arch,
                "modality": args.modality,
                "task": args.task,
                "imagenet_weights": args.imagenet_weights,
                "oct_representation": args.oct_representation,
                "freeze_backbone": args.freeze_backbone,
                "balanced_sampler": args.balanced_sampler,
                "best_epoch": best_epoch,
                "best_val_auroc": best_val_auroc,
            },
            args.checkpoint,
        )
        print(f"wrote_checkpoint={args.checkpoint}")

    val_predictions = to_standard_predictions(
        pd, val_df, predict(torch, tqdm, model, val_loader, device), args.arch, args.modality
    )
    test_predictions = to_standard_predictions(
        pd, test_df, predict(torch, tqdm, model, test_loader, device), args.arch, args.modality
    )

    args.out_val.parent.mkdir(parents=True, exist_ok=True)
    args.out_test.parent.mkdir(parents=True, exist_ok=True)
    val_predictions.to_csv(args.out_val, index=False)
    test_predictions.to_csv(args.out_test, index=False)
    print(f"wrote_val={args.out_val} rows={len(val_predictions)}")
    print(f"wrote_test={args.out_test} rows={len(test_predictions)}")


if __name__ == "__main__":
    main()
