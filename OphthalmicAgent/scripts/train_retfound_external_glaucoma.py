#!/usr/bin/env python3
"""Train reproducible RETFound CFP/OCT glaucoma heads from a canonical manifest."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--modality", choices=("cfp", "oct"), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--backbone-weights", type=Path, default=None)
    parser.add_argument("--init-model-weights", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--head-type", choices=("linear", "mlp"), default="mlp")
    parser.add_argument("--unfreeze-blocks", type=int, default=0)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--oct-slices", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    return parser.parse_args()


def resolve_path(manifest: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    repo_candidate = manifest.parent.parent / path
    return repo_candidate if repo_candidate.exists() else manifest.parent / path


def load_rows(manifest: Path, dataset: str, modality: str) -> list[dict[str, str]]:
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    path_column = "cfp_path" if modality == "cfp" else "oct_path"
    selected = []
    for row in rows:
        if row.get("dataset", "").strip().lower() != dataset.lower():
            continue
        value = row.get(path_column, "").strip()
        if not value:
            continue
        copied = dict(row)
        copied["image_path"] = str(resolve_path(manifest, value))
        selected.append(copied)
    if not selected:
        raise SystemExit(f"No {dataset}/{modality} rows found in {manifest}")
    return selected


def load_oct_slices(path: Path, count: int) -> list[np.ndarray]:
    from PIL import Image

    if path.is_dir():
        files = sorted(
            p for p in path.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        )
        if not files:
            raise ValueError(f"No B-scan images found in {path}")
        arrays = [np.asarray(Image.open(item).convert("L")) for item in files]
        volume = np.stack(arrays)
    elif path.suffix.lower() == ".npz":
        with np.load(path) as data:
            key = next((candidate for candidate in ("oct_bscans", "volume", "oct", "bscans") if candidate in data), None)
            if key is None:
                raise ValueError(f"No recognized OCT key in {path}; found {list(data.files)}")
            volume = np.asarray(data[key])
    elif path.suffix.lower() == ".npy":
        volume = np.asarray(np.load(path))
    else:
        raise ValueError(f"Unsupported OCT input: {path}")
    volume = np.squeeze(volume)
    if volume.ndim != 3:
        raise ValueError(f"Expected OCT volume [slices,height,width], got {volume.shape} at {path}")
    indices = np.linspace(0, volume.shape[0] - 1, count, dtype=int)
    return [volume[index] for index in indices]


def main() -> None:
    args = parse_args()
    import torch
    import torch.nn as nn
    from PIL import Image
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
    from torchvision import transforms
    from VisionAgent.models_vit import RETFound_mae

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    weights = args.backbone_weights or ROOT / "VisionAgent" / "weights" / (
        "RETFound_mae_natureCFP.pth" if args.modality == "cfp" else "RETFound_mae_natureOCT.pth"
    )
    if not weights.exists():
        raise SystemExit(f"Backbone weights not found: {weights}")

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    class ManifestDataset(Dataset):
        def __init__(self, rows, transform):
            self.rows = rows
            self.transform = transform
        def __len__(self):
            return len(self.rows)
        def __getitem__(self, index):
            row = self.rows[index]
            path = Path(row["image_path"])
            if args.modality == "cfp":
                image = self.transform(Image.open(path).convert("RGB"))
            else:
                images = []
                for array in load_oct_slices(path, args.oct_slices):
                    array = np.asarray(array)
                    if array.dtype != np.uint8:
                        low, high = float(np.nanmin(array)), float(np.nanmax(array))
                        array = np.zeros_like(array, dtype=np.uint8) if high <= low else ((array-low)/(high-low)*255).astype(np.uint8)
                    images.append(self.transform(Image.fromarray(array).convert("RGB")))
                image = torch.stack(images)
            return image, torch.tensor(float(row["label"]), dtype=torch.float32), row["case_id"]

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = RETFound_mae(img_size=224, num_classes=0, drop_path_rate=0.0, global_pool="")
            checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
            state = checkpoint.get("model", checkpoint)
            state = {k: v for k, v in state.items() if "head" not in k and "decoder" not in k and "fc_norm" not in k}
            self.backbone.load_state_dict(state, strict=False)
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
            if args.unfreeze_blocks:
                if args.unfreeze_blocks > len(self.backbone.blocks):
                    raise ValueError(f"unfreeze-blocks exceeds {len(self.backbone.blocks)}")
                for block in self.backbone.blocks[-args.unfreeze_blocks:]:
                    for parameter in block.parameters():
                        parameter.requires_grad = True
                # The final normalization participates in the adapted feature.
                for name in ("norm", "fc_norm"):
                    module = getattr(self.backbone, name, None)
                    if module is not None:
                        for parameter in module.parameters():
                            parameter.requires_grad = True
            self.head = (nn.Linear(1024, 1) if args.head_type == "linear" else
                nn.Sequential(nn.Linear(1024, args.hidden_dim), nn.ReLU(), nn.Dropout(args.dropout), nn.Linear(args.hidden_dim, 1)))
        def forward(self, images):
            if args.modality == "oct":
                batch, slices, channels, height, width = images.shape
                features = self.backbone(images.reshape(batch*slices, channels, height, width))
                features = features.reshape(batch, slices, -1).mean(dim=1)
            else:
                features = self.backbone(images)
            return self.head(features).squeeze(1)

    rows = load_rows(args.manifest, args.dataset, args.modality)
    limits = {"train": args.limit_train, "val": args.limit_val, "test": args.limit_test}
    split_rows = {}
    for split in ("train", "val", "test"):
        values = [row for row in rows if row["split"].strip().lower() == split]
        split_rows[split] = values[: limits[split]] if limits[split] else values
    if not split_rows["train"] or not split_rows["val"]:
        raise SystemExit("Both train and val splits are required")

    labels = np.array([int(float(row["label"])) for row in split_rows["train"]])
    counts = np.bincount(labels, minlength=2)
    if np.any(counts == 0):
        raise SystemExit(f"Training split must contain both labels; counts={counts.tolist()}")
    sample_weights = np.array([1.0 / counts[label] for label in labels], dtype=np.float64)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(ManifestDataset(split_rows["train"], train_transform), batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=device.type == "cuda", generator=generator)
    val_loader = DataLoader(ManifestDataset(split_rows["val"], eval_transform), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = Model()
    if args.init_model_weights:
        if not args.init_model_weights.exists():
            raise SystemExit(f"Initial model weights not found: {args.init_model_weights}")
        initial = torch.load(args.init_model_weights, map_location="cpu", weights_only=False)
        initial = initial.get("model", initial)
        if args.modality == "cfp" and any(name.startswith("glaucoma_head.") for name in initial):
            initial = {
                ("head." + name.removeprefix("glaucoma_head.")) if name.startswith("glaucoma_head.") else name: value
                for name, value in initial.items()
                if not name.startswith(("amd_head.", "dr_head."))
            }
        missing, unexpected = model.load_state_dict(initial, strict=False)
        if missing or unexpected:
            raise SystemExit(f"Initial checkpoint mismatch: missing={missing}, unexpected={unexpected}")
        print(f"initialized_model={args.init_model_weights}", flush=True)
    model = model.to(device)
    backbone_parameters = [parameter for parameter in model.backbone.parameters() if parameter.requires_grad]
    parameter_groups = [{"params": model.head.parameters(), "lr": args.lr}]
    if backbone_parameters:
        parameter_groups.append({"params": backbone_parameters, "lr": args.backbone_lr})
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)
    # The sampler already balances classes; adding pos_weight here would count
    # class imbalance twice and distort probability calibration.
    loss_fn = nn.BCEWithLogitsLoss()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.out_dir / "best_head.pth"
    best_auc = -1.0
    epochs_without_improvement = 0

    def predict(loader):
        model.eval(); probs=[]; targets=[]; ids=[]
        with torch.no_grad():
            for images, labels_batch, case_ids in loader:
                values = torch.sigmoid(model(images.to(device))).cpu().numpy()
                probs.extend(values.tolist()); targets.extend(labels_batch.numpy().tolist()); ids.extend(case_ids)
        return ids, np.asarray(targets, dtype=int), np.asarray(probs, dtype=float)

    for epoch in range(1, args.epochs + 1):
        model.train(); losses=[]
        for images, labels_batch, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(images.to(device))
            loss = loss_fn(logits, labels_batch.to(device))
            loss.backward(); optimizer.step(); losses.append(float(loss.item()))
        _, targets, probs = predict(val_loader)
        auc = roc_auc_score(targets, probs) if len(np.unique(targets)) == 2 else float("nan")
        print(f"epoch={epoch} loss={np.mean(losses):.6f} val_auc={auc:.6f}", flush=True)
        if np.isfinite(auc) and auc > best_auc:
            best_auc = float(auc)
            epochs_without_improvement = 0
            trainable_state = {
                name: value.detach().cpu()
                for name, value in model.state_dict().items()
                if name.startswith("head.") or name in {
                    parameter_name for parameter_name, parameter in model.named_parameters()
                    if parameter.requires_grad
                }
            }
            torch.save({"trainable_state": trainable_state, "epoch": epoch, "val_auc": best_auc, "args": vars(args)}, checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"early_stop epoch={epoch} best_val_auc={best_auc:.6f}", flush=True)
                break

    saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(saved["trainable_state"], strict=False)
    _, val_targets, val_probs = predict(val_loader)
    candidates = np.unique(np.concatenate(([0.0], val_probs, [1.0])))
    threshold = max(candidates, key=lambda value: balanced_accuracy_score(val_targets, val_probs >= value))

    metrics = {}
    for split in ("val", "test"):
        if not split_rows[split]:
            continue
        loader = DataLoader(ManifestDataset(split_rows[split], eval_transform), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        ids, targets, probs = predict(loader)
        with (args.out_dir / f"predictions_{split}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["dataset", "task", "model_name", "case_id", "split", "y_true", "y_prob", "threshold", "y_pred"])
            writer.writeheader()
            for case_id, target, prob in zip(ids, targets, probs):
                writer.writerow({"dataset": args.dataset, "task": "glaucoma", "model_name": f"retfound_{args.modality}", "case_id": case_id, "split": split, "y_true": int(target), "y_prob": float(prob), "threshold": float(threshold), "y_pred": int(prob >= threshold)})

        predictions = (probs >= threshold).astype(int)
        tn = int(((targets == 0) & (predictions == 0)).sum())
        fp = int(((targets == 0) & (predictions == 1)).sum())
        fn = int(((targets == 1) & (predictions == 0)).sum())
        tp = int(((targets == 1) & (predictions == 1)).sum())
        divide = lambda numerator, denominator: float(numerator / denominator) if denominator else None
        metrics[split] = {
            "auroc": float(roc_auc_score(targets, probs)) if len(np.unique(targets)) == 2 else None,
            "accuracy": divide(tp + tn, len(targets)),
            "f1": divide(2 * tp, 2 * tp + fp + fn),
            "sensitivity": divide(tp, tp + fn),
            "specificity": divide(tn, tn + fp),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        }

    summary = {"dataset": args.dataset, "modality": args.modality, "best_val_auc": best_auc, "threshold": float(threshold), "rows": {k: len(v) for k, v in split_rows.items()}, "seed": args.seed, "checkpoint": str(checkpoint_path), "head_type": args.head_type, "unfreeze_blocks": args.unfreeze_blocks, "metrics": metrics}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
