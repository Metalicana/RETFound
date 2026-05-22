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


class FairVisionImageDataset:
    def __init__(self, np, Image, frame, modality: str, transform):
        self.np = np
        self.Image = Image
        self.frame = frame.reset_index(drop=True)
        self.modality = modality
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        with self.np.load(row["image_path"]) as data:
            if self.modality == "oct":
                volume = data["oct_bscans"]
                image = volume[volume.shape[0] // 2]
            else:
                image = data["slo_fundus"]
        if image.max() <= 1.0:
            image = (image * 255).astype("uint8")
        else:
            image = image.astype("uint8")
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

    train_set = FairVisionImageDataset(np, Image, train_df, args.modality, transform_train)
    val_set = FairVisionImageDataset(np, Image, val_df, args.modality, transform_eval)
    test_set = FairVisionImageDataset(np, Image, test_df, args.modality, transform_eval)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(torch, models, args.arch, args.imagenet_weights).to(device)
    positives = float(train_df["y_true"].sum())
    negatives = float(len(train_df) - positives)
    pos_weight = torch.tensor([negatives / max(positives, 1.0)], device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(torch, tqdm, model, train_loader, optimizer, criterion, device)
        print(f"epoch={epoch} train_loss={loss:.6f}")

    if args.checkpoint:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "arch": args.arch,
                "modality": args.modality,
                "task": args.task,
                "imagenet_weights": args.imagenet_weights,
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
