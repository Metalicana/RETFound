from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def equi_agent_root() -> Path:
    return Path(__file__).resolve().parents[1]


sys.path.insert(0, str(equi_agent_root()))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from VisionAgent.fairvision_npz import FairVisionNPZ


def print_dataset_diagnostics(name, dataset):
    print(f"{name} label summary: {dataset.label_summary()}")
    sample_count = int(os.environ.get("MIRAGE_AUDIT_SAMPLES", 2))
    if sample_count <= 0:
        return
    for source in dataset.sources:
        seen = 0
        for item in dataset.files:
            if item["source"] != source:
                continue
            with np.load(item["path"]) as data:
                label, metadata = dataset._build_label_and_metadata(item, data)
                raw = {
                    key: dataset._scalar_to_string(data[key])
                    for key in ("amd_condition", "dr_subtype", "glaucoma")
                    if key in data.files
                }
                shape = tuple(data["slo_fundus"].shape) if "slo_fundus" in data.files else None
            print(
                f"{name} audit {source}: file={metadata['filename']} "
                f"slo_shape={shape} raw={raw} label={label.tolist()} groundtruth={metadata['groundtruth']}"
            )
            seen += 1
            if seen >= sample_count:
                break


class MultiAgentLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.bce_amd = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0, 1.5, 2.0], device=device))
        self.bce_dr = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0], device=device))
        self.bce_standard = nn.BCEWithLogitsLoss()

    def forward(self, outputs, targets):
        amd_mask = targets[:, 0] != -1
        dr_mask = targets[:, 3] != -1
        gl_mask = targets[:, 4] != -1

        loss = torch.zeros((), device=targets.device)
        if amd_mask.any():
            loss = loss + self.bce_amd(outputs["amd"][amd_mask], targets[amd_mask, 0:3])
        if dr_mask.any():
            loss = loss + self.bce_dr(outputs["dr"][dr_mask], targets[dr_mask, 3:4])
        if gl_mask.any():
            loss = loss + self.bce_standard(outputs["glaucoma"][gl_mask], targets[gl_mask, 4:5])
        return loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MIRAGE SLO multi-head linear probing on FairVision.")
    parser.add_argument("--data-root", type=Path, default=repo_root() / "Datasets" / "FairVision")
    parser.add_argument("--mirage-dir", type=Path, default=equi_agent_root() / "VisionAgent" / "MIRAGE")
    parser.add_argument("--output-weights", type=Path, default=equi_agent_root() / "weights" / "slo_model_best.pth")
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("MIRAGE_EPOCHS", 60)))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("MIRAGE_BATCH_SIZE", 64)))
    parser.add_argument("--lr", type=float, default=float(os.environ.get("MIRAGE_LR", 1e-3)))
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("MIRAGE_NUM_WORKERS", 16)))
    parser.add_argument("--device", default=None, help="Example: cuda, cuda:0, or cpu. Defaults to CUDA if available.")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def require_mirage(mirage_dir: Path):
    if not mirage_dir.exists():
        raise FileNotFoundError(f"MIRAGE directory not found: {mirage_dir}")
    sys.path.insert(0, str(mirage_dir))
    from linear_probing_slo import get_model_slo

    return get_model_slo


def build_model(get_model_slo, mirage_dir: Path, device):
    original_dir = Path.cwd()
    os.chdir(mirage_dir)
    try:
        model = get_model_slo().to(device)
    finally:
        os.chdir(original_dir)
    return model


def collect_validation_outputs(model, loader, device):
    val_data = {"amd": [], "dr": [], "gl": [], "targets": []}
    model.eval()
    with torch.no_grad():
        for imgs, labels, _ in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            val_data["amd"].append(torch.sigmoid(outputs["amd"]).cpu().numpy())
            val_data["dr"].append(torch.sigmoid(outputs["dr"]).cpu().numpy())
            val_data["gl"].append(torch.sigmoid(outputs["glaucoma"]).cpu().numpy())
            val_data["targets"].append(labels.cpu().numpy())
    return (
        np.vstack(val_data["targets"]),
        np.vstack(val_data["amd"]),
        np.vstack(val_data["dr"]),
        np.vstack(val_data["gl"]),
    )


def validation_metrics(all_targets, all_amd_preds, all_dr_preds, all_gl_preds):
    metrics = {}
    amd_mask = all_targets[:, 0] != -1
    if amd_mask.any():
        amd_aucs = [
            roc_auc_score(all_targets[amd_mask, i], all_amd_preds[amd_mask, i])
            for i in range(3)
            if len(np.unique(all_targets[amd_mask, i])) > 1
        ]
        if amd_aucs:
            metrics["AMD_Avg_AUC"] = float(np.mean(amd_aucs))

    dr_mask = all_targets[:, 3] != -1
    if dr_mask.any() and len(np.unique(all_targets[dr_mask, 3])) > 1:
        metrics["DR_AUC"] = float(roc_auc_score(all_targets[dr_mask, 3], all_dr_preds[dr_mask, 0]))

    gl_mask = all_targets[:, 4] != -1
    if gl_mask.any() and len(np.unique(all_targets[gl_mask, 4])) > 1:
        metrics["Glaucoma_AUC"] = float(roc_auc_score(all_targets[gl_mask, 4], all_gl_preds[gl_mask, 0]))

    auc_values = [value for key, value in metrics.items() if "AUC" in key]
    metrics["Avg_AUC"] = float(np.mean(auc_values)) if auc_values else 0.0
    return metrics


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    mirage_dir = args.mirage_dir.expanduser().resolve()
    output_weights = args.output_weights.expanduser().resolve()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    get_model_slo = require_mirage(mirage_dir)
    print(f"Using FairVision data root: {data_root}")
    print(f"Using MIRAGE directory: {mirage_dir}")
    print(f"Saving MIRAGE SLO weights to: {output_weights}")

    train_transform = transforms.Compose(
        [
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    train_ds = FairVisionNPZ(data_root, split="Training", transform=train_transform, image_kind="slo")
    val_ds = FairVisionNPZ(data_root, split="Validation", transform=val_transform, image_kind="slo")
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(f"FairVision splits are empty: train={len(train_ds)} val={len(val_ds)}")
    print_dataset_diagnostics("train", train_ds)
    print_dataset_diagnostics("val", val_ds)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=device.type == "cuda",
    )

    model = build_model(get_model_slo, mirage_dir, device)
    if args.resume and output_weights.exists():
        model.load_state_dict(torch.load(output_weights, map_location=device, weights_only=True))
        print("Successfully loaded SLO specialist weights. Resuming...")

    optimizer = torch.optim.AdamW(
        filter(lambda param: param.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.05,
    )
    criterion = MultiAgentLoss(device)
    best_avg_auc = 0.0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for imgs, labels, _ in loop:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())
            loop.set_postfix(loss=loss.item())

        all_targets, all_amd_preds, all_dr_preds, all_gl_preds = collect_validation_outputs(model, val_loader, device)
        metrics = validation_metrics(all_targets, all_amd_preds, all_dr_preds, all_gl_preds)
        print(
            f"\n[Epoch {epoch + 1}] Avg Loss: {train_loss / len(train_loader):.4f} "
            f"| AMD Avg AUC: {metrics.get('AMD_Avg_AUC', 0):.4f} "
            f"| DR AUC: {metrics.get('DR_AUC', 0):.4f} "
            f"| GL AUC: {metrics.get('Glaucoma_AUC', 0):.4f}"
        )

        if metrics["Avg_AUC"] > best_avg_auc:
            best_avg_auc = metrics["Avg_AUC"]
            output_weights.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output_weights)
            print(f"New Best MIRAGE SLO Model Saved! (Avg AUC: {best_avg_auc:.4f})")


if __name__ == "__main__":
    main()
