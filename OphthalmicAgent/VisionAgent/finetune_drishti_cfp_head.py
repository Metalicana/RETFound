"""Train a frozen RETFound-CFP glaucoma head on official Drishti splits."""

import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

from VisionAgent.linear_probing_fundus import get_model_cfp


DATA_ROOT = Path(os.getenv("DRISHTI_DATA_ROOT", "./data_drishti"))
MANIFEST = Path(os.getenv("DRISHTI_MANIFEST", DATA_ROOT / "manifest.csv"))
OUTPUT_WEIGHTS = Path(
    os.getenv("DRISHTI_CFP_OUTPUT_WEIGHTS", "./weights/drishti_cfp_glaucoma_best.pth")
)
METADATA_PATH = Path(
    os.getenv("DRISHTI_CFP_TRAINING_METADATA", "drishti_cfp_training_metadata.json")
)
PREDICTIONS_DIR = Path(
    os.getenv("DRISHTI_CFP_PREDICTIONS_DIR", "./outputs/drishti_cfp_head")
)
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "4"))
EPOCHS = int(os.getenv("EPOCHS", "100"))
PATIENCE = int(os.getenv("PATIENCE", "15"))
HEAD_LR = float(os.getenv("HEAD_LR", "1e-3"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "1e-4"))
SEED = int(os.getenv("RANDOM_SEED", "2026"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_image_path(row):
    value = str(row.get("cfp_path", "")).strip()
    if value and value.lower() != "nan":
        path = Path(value)
        if path.is_absolute():
            return path
        # Manifest paths may be relative either to OphthalmicAgent or data_drishti.
        repo_candidate = MANIFEST.parent.parent / path
        return repo_candidate if repo_candidate.exists() else MANIFEST.parent / path
    folder = "Glaucoma" if int(row["label"]) == 1 else "Normal"
    return DATA_ROOT / folder / f"{row['case_id']}.png"


def read_manifest():
    frame = pd.read_csv(MANIFEST)
    required = {"case_id", "split", "label"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Manifest is missing required columns: {sorted(missing)}")
    frame["split"] = frame["split"].astype(str).str.strip().str.lower()
    frame["label"] = frame["label"].astype(int)
    if not set(frame["label"]).issubset({0, 1}):
        raise ValueError("label must contain only 0 (normal) and 1 (glaucoma)")
    frame["image_path"] = frame.apply(resolve_image_path, axis=1).astype(str)
    missing_images = frame.loc[~frame["image_path"].map(lambda value: Path(value).is_file())]
    if not missing_images.empty:
        preview = missing_images[["case_id", "image_path"]].head(10).to_dict("records")
        raise FileNotFoundError(f"Missing {len(missing_images)} CFP images; examples: {preview}")
    splits = {name: frame[frame["split"] == name].reset_index(drop=True)
              for name in ("train", "val", "test")}
    if any(split.empty for split in splits.values()):
        raise ValueError("Manifest must contain non-empty train, val, and test splits")
    return frame, splits


class DrishtiDataset(Dataset):
    def __init__(self, frame, transform):
        self.frame = frame
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        image = self.transform(Image.open(row["image_path"]).convert("RGB"))
        label = torch.tensor([float(row["label"])], dtype=torch.float32)
        return image, label, str(row["case_id"])


def transforms_for_training():
    normalization = transforms.Normalize(
        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    )
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
        transforms.ToTensor(),
        normalization,
    ])
    evaluation_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalization,
    ])
    return train_transform, evaluation_transform


def make_train_loader(frame, transform):
    labels = frame["label"].to_numpy(dtype=int)
    counts = np.bincount(labels, minlength=2)
    if np.any(counts == 0):
        raise ValueError(f"Train must contain both classes; counts={counts.tolist()}")
    weights = torch.as_tensor([1.0 / counts[label] for label in labels], dtype=torch.double)
    sampler = WeightedRandomSampler(weights, num_samples=len(frame), replacement=True)
    return DataLoader(
        DrishtiDataset(frame, transform), batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=NUM_WORKERS, pin_memory=DEVICE.type == "cuda",
    )


def make_eval_loader(frame, transform):
    return DataLoader(
        DrishtiDataset(frame, transform), batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=DEVICE.type == "cuda",
    )


def predict(model, loader):
    model.eval()
    labels, probabilities, case_ids = [], [], []
    with torch.inference_mode():
        for images, targets, ids in loader:
            values = torch.sigmoid(model(images.to(DEVICE))).reshape(-1).cpu().numpy()
            probabilities.extend(values.tolist())
            labels.extend(targets.reshape(-1).numpy().astype(int).tolist())
            case_ids.extend(ids)
    return np.asarray(labels), np.asarray(probabilities), case_ids


def select_threshold(labels, probabilities):
    candidates = np.unique(np.concatenate(([0.0], probabilities, [1.0])))
    scored = [
        (balanced_accuracy_score(labels, probabilities >= value), value)
        for value in candidates
    ]
    best_score = max(score for score, _ in scored)
    tied = [value for score, value in scored if score == best_score]
    threshold = min(tied, key=lambda value: abs(value - 0.5))
    return float(threshold), float(best_score)


def evaluate_split(model, frame, transform, split, threshold):
    labels, probabilities, case_ids = predict(model, make_eval_loader(frame, transform))
    predictions = (probabilities >= threshold).astype(int)
    output = pd.DataFrame({
        "case_id": case_ids, "split": split, "Ground_Truth": labels,
        "Probability_GL": probabilities, "threshold": threshold,
        "Pred_GL": predictions, "Is_Correct": (labels == predictions).astype(int),
    })
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    output.to_csv(PREDICTIONS_DIR / f"predictions_{split}.csv", index=False)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    metrics = {
        "cases": len(labels),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else None,
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    print(f"\n{split.upper()} report at threshold {threshold:.6f}:")
    print(classification_report(
        labels, predictions, labels=[0, 1], target_names=["Normal", "Glaucoma"],
        digits=4, zero_division=0,
    ))
    print(f"AUROC: {metrics['auroc']:.4f}")
    print(f"Confusion matrix [[TN, FP], [FN, TP]]:\n{np.array([[tn, fp], [fn, tp]])}")
    return metrics


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    frame, splits = read_manifest()
    print(f"Drishti cases={len(frame)} | " + " | ".join(
        f"{name}={len(values)} labels={values['label'].value_counts().sort_index().to_dict()}"
        for name, values in splits.items()
    ))
    train_transform, evaluation_transform = transforms_for_training()
    train_loader = make_train_loader(splits["train"], train_transform)
    validation_loader = make_eval_loader(splits["val"], evaluation_transform)

    model = get_model_cfp().to(DEVICE)
    # Strict head probing: retain the original RETFound representation.
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    for parameter in model.glaucoma_head.parameters():
        parameter.requires_grad = True
    optimizer = torch.optim.AdamW(
        model.glaucoma_head.parameters(), lr=HEAD_LR, weight_decay=WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()  # sampler already balances the classes
    OUTPUT_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    best_auc, best_epoch, stale_epochs = -np.inf, 0, 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        model.backbone.eval()
        losses = []
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for images, targets, _ in loop:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
            loop.set_postfix(loss=loss.item())
        labels, probabilities, _ = predict(model, validation_loader)
        validation_auc = roc_auc_score(labels, probabilities)
        print(f"Epoch {epoch}: loss={np.mean(losses):.6f} val_AUROC={validation_auc:.6f}")
        if validation_auc > best_auc:
            best_auc, best_epoch, stale_epochs = float(validation_auc), epoch, 0
            torch.save(model.state_dict(), OUTPUT_WEIGHTS)
            print(f"Saved best checkpoint: {OUTPUT_WEIGHTS}")
        else:
            stale_epochs += 1
            if stale_epochs >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(OUTPUT_WEIGHTS, map_location=DEVICE, weights_only=False))
    val_labels, val_probabilities, _ = predict(model, validation_loader)
    threshold, validation_balanced_accuracy = select_threshold(
        val_labels, val_probabilities
    )
    validation_metrics = evaluate_split(
        model, splits["val"], evaluation_transform, "val", threshold
    )
    test_metrics = evaluate_split(
        model, splits["test"], evaluation_transform, "test", threshold
    )
    metadata = {
        "dataset": "drishti", "source_weights": "RETFound_mae_natureCFP.pth",
        "output_weights": str(OUTPUT_WEIGHTS), "split_sizes": {
            name: len(values) for name, values in splits.items()
        },
        "split_label_counts": {
            name: {str(key): int(value) for key, value in values["label"].value_counts().sort_index().items()}
            for name, values in splits.items()
        },
        "best_epoch": best_epoch, "best_validation_auroc": best_auc,
        "selected_validation_threshold": threshold,
        "validation_balanced_accuracy_at_threshold": validation_balanced_accuracy,
        "balanced_training_sampler": True, "backbone_frozen": True,
        "maximum_epochs": EPOCHS, "patience": PATIENCE, "seed": SEED,
        "validation_metrics": validation_metrics, "test_metrics": test_metrics,
        "test_used_for_selection": False,
        "small_validation_set_warning": "Validation has only 10 cases; threshold and checkpoint estimates are high variance.",
    }
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
