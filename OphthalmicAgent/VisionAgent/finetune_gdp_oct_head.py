"""Fine-tune only the RETFound-OCT glaucoma head on GDP training cases."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from VisionAgent.linear_probing_oct3 import get_model_oct


GDP_CSV = Path(os.getenv("GDP_CSV", "./data_gdp/data_summary.csv"))
GDP_BSCAN_DIR = Path(os.getenv("GDP_BSCAN_DIR", "./data_gdp/BScan"))
RETFOUND_OCT_WEIGHTS = Path(
    os.getenv(
        "RETFOUND_OCT_WEIGHTS",
        "/lustre/fs1/home/yu395012/RETFound/OphthalmicAgent/VisionAgent/weights/"
        "RETFound_mae_natureOCT.pth",
    )
)
OUTPUT_WEIGHTS = Path(
    os.getenv("GDP_OCT_OUTPUT_WEIGHTS", "./weights/gdp_oct_glaucoma_head_best.pth")
)
METADATA_PATH = Path(
    os.getenv("GDP_OCT_TRAINING_METADATA", "gdp_oct_head_training_metadata.json")
)
VALIDATION_PREDICTIONS = Path(
    os.getenv("GDP_OCT_VALIDATION_PREDICTIONS", "gdp_oct_validation_predictions.csv")
)

OCT_SLICES = int(os.getenv("OCT_SLICES", "8"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "8"))
EPOCHS = int(os.getenv("EPOCHS", "40"))
HEAD_LR = float(os.getenv("HEAD_LR", "1e-4"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "0.05"))
VALIDATION_FRACTION = float(os.getenv("VALIDATION_FRACTION", "0.20"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))
TARGET_SENSITIVITY = float(os.getenv("TARGET_SENSITIVITY", "0.90"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def available_training_rows():
    frame = pd.read_csv(GDP_CSV)
    required = {"filename", "glaucoma", "glaucoma_detection_use"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"GDP CSV is missing columns: {sorted(missing)}")
    frame = frame[
        frame["glaucoma_detection_use"].astype(str).str.lower() == "training"
    ].copy()
    paths, keep = [], []
    for index, row in frame.iterrows():
        name = str(row["filename"])
        filename = name if name.lower().endswith(".npz") else f"{name}.npz"
        path = GDP_BSCAN_DIR / filename
        if path.is_file():
            keep.append(index)
            paths.append(str(path))
        else:
            print(f"Skipping missing GDP BScan: {path}")
    frame = frame.loc[keep].reset_index(drop=True)
    frame["bscan_path"] = paths
    if frame.empty:
        raise ValueError("No available GDP training BScans were found")
    return frame


class GDPTrainingDataset(Dataset):
    def __init__(self, frame, augment=False):
        self.frame = frame.reset_index(drop=True)
        self.augment = augment
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with np.load(row["bscan_path"]) as data:
            volume = np.asarray(data["bscans"])
        if volume.ndim != 3:
            raise ValueError(f"Expected 3D bscans, got {volume.shape}: {row['bscan_path']}")
        volume = np.moveaxis(volume, -1, 0)
        indices = np.linspace(0, volume.shape[0] - 1, OCT_SLICES, dtype=int)
        selected = volume[indices]
        images = torch.stack([
            self.transform(Image.fromarray(slice_).convert("RGB")) for slice_ in selected
        ])
        # Apply the same augmentation to every slice in a cube.
        if self.augment and torch.rand(()) < 0.5:
            images = torch.flip(images, dims=[3])
        label = torch.tensor([float(row["glaucoma"])], dtype=torch.float32)
        return images, label, str(row["filename"])


def loader(frame, augment, shuffle):
    return DataLoader(
        GDPTrainingDataset(frame, augment=augment),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE.type == "cuda",
    )


def load_original_retfound_model():
    # get_model_oct loads the original RETFound OCT backbone and creates fresh
    # downstream heads. No FairVision fine-tuned checkpoint is loaded here.
    model = get_model_oct(weight_path=str(RETFOUND_OCT_WEIGHTS))
    # Strict linear probing: freeze RETFound and every unused disease head.
    for parameter in model.parameters():
        parameter.requires_grad = False
    # The glaucoma head was randomly initialized by RETFoundMultiHead.
    for parameter in model.glaucoma_head.parameters():
        parameter.requires_grad = True
    return model.to(DEVICE)


def predict(model, dataloader):
    model.eval()
    labels, probabilities, patient_ids = [], [], []
    with torch.inference_mode():
        for images, targets, ids in dataloader:
            logits = model(images.to(DEVICE))["glaucoma"]
            probabilities.extend(torch.sigmoid(logits).reshape(-1).cpu().numpy().tolist())
            labels.extend(targets.reshape(-1).numpy().astype(int).tolist())
            patient_ids.extend(list(ids))
    return np.asarray(labels), np.asarray(probabilities), patient_ids


def select_threshold(labels, probabilities, target_sensitivity):
    candidates = np.unique(np.concatenate(([0.0], probabilities, [1.0])))
    choices = []
    for threshold in candidates:
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        choices.append((specificity, threshold, sensitivity))
    eligible = [choice for choice in choices if choice[2] >= target_sensitivity]
    selected = max(eligible, key=lambda value: (value[0], value[1]))
    return {
        "threshold": float(selected[1]),
        "sensitivity": float(selected[2]),
        "specificity": float(selected[0]),
    }


def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    frame = available_training_rows()
    train_frame, validation_frame = train_test_split(
        frame,
        test_size=VALIDATION_FRACTION,
        random_state=RANDOM_SEED,
        stratify=frame["glaucoma"],
    )
    train_loader = loader(train_frame, augment=True, shuffle=True)
    validation_loader = loader(validation_frame, augment=False, shuffle=False)
    positives = int((train_frame["glaucoma"] == 1).sum())
    negatives = int((train_frame["glaucoma"] == 0).sum())
    pos_weight = negatives / positives
    print(
        f"GDP head fine-tuning: train={len(train_frame)} ({positives} positive), "
        f"validation={len(validation_frame)}, pos_weight={pos_weight:.4f}"
    )

    model = load_original_retfound_model()
    optimizer = torch.optim.AdamW(
        model.glaucoma_head.parameters(), lr=HEAD_LR, weight_decay=WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=DEVICE)
    )
    best_auc, best_epoch = -np.inf, 0
    OUTPUT_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        # Frozen modules must remain deterministic while the glaucoma head trains.
        model.backbone.eval()
        model.amd_head.eval()
        model.dr_head.eval()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for images, targets, _ in loop:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(images)["glaucoma"], targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        labels, probabilities, _ = predict(model, validation_loader)
        auc = roc_auc_score(labels, probabilities)
        print(
            f"Epoch {epoch}: loss={running_loss / len(train_loader):.4f} "
            f"validation_AUROC={auc:.4f}"
        )
        if auc > best_auc:
            best_auc, best_epoch = auc, epoch
            torch.save(model.state_dict(), OUTPUT_WEIGHTS)
            print(f"Saved best GDP OCT head: {OUTPUT_WEIGHTS}")

    model.load_state_dict(torch.load(OUTPUT_WEIGHTS, map_location=DEVICE, weights_only=False))
    labels, probabilities, patient_ids = predict(model, validation_loader)
    selected = select_threshold(labels, probabilities, TARGET_SENSITIVITY)
    predictions = (probabilities >= THRESHOLD).astype(int)
    print("\nValidation report at fixed threshold:")
    print(classification_report(
        labels, predictions, labels=[0, 1], target_names=["Normal", "Glaucoma"],
        digits=4, zero_division=0,
    ))
    print(f"Validation-selected screening threshold: {selected}")

    VALIDATION_PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "Patient_ID": patient_ids,
        "Ground_Truth": labels,
        "Probability_GL": probabilities,
        "Pred_GL_Fixed_Threshold": predictions,
    }).to_csv(VALIDATION_PREDICTIONS, index=False)
    metadata = {
        "source_weights": str(RETFOUND_OCT_WEIGHTS),
        "initialization": "original frozen RETFound OCT backbone plus new random glaucoma head",
        "output_weights": str(OUTPUT_WEIGHTS),
        "available_gdp_training_cases": len(frame),
        "train_cases": len(train_frame),
        "validation_cases": len(validation_frame),
        "oct_slices": OCT_SLICES,
        "best_epoch": best_epoch,
        "best_validation_auroc": float(best_auc),
        "pos_weight": pos_weight,
        "fixed_threshold": THRESHOLD,
        "validation_selected_threshold": selected,
        "target_sensitivity": TARGET_SENSITIVITY,
        "random_seed": RANDOM_SEED,
    }
    with METADATA_PATH.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
