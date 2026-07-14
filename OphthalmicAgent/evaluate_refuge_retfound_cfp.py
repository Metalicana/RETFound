"""Evaluate the fine-tuned RETFound-CFP glaucoma model on REFUGE/Test."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from VisionAgent.linear_probing_fundus import get_model_cfp


DATA_ROOT = Path(os.getenv("REFUGE_DATA_ROOT", "./"))
CSV_PATH = Path(os.getenv("REFUGE_CSV", "./data_refuge/data.csv"))
CFP_WEIGHTS = os.getenv("CFP_WEIGHTS", "./weights/cfp_glaucoma_best.pth")
OUTPUT_CSV = os.getenv("OUTPUT_CSV", "refuge_test_retfound_cfp_predictions.csv")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "8"))
TRAINING_METADATA = Path(
    os.getenv("CFP_TRAINING_METADATA", "cfp_glaucoma_training_metadata.json")
)


def resolve_threshold():
    if "THRESHOLD" in os.environ:
        return float(os.environ["THRESHOLD"])
    if TRAINING_METADATA.exists():
        with TRAINING_METADATA.open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        threshold = metadata.get(
            "fixed_threshold", metadata.get("selected_validation_threshold")
        )
        if threshold is not None:
            print(f"Using saved training threshold from {TRAINING_METADATA}: {threshold}")
            return float(threshold)
    print("No saved validation threshold found; falling back to 0.5")
    return 0.5


class RefugeTestDataset(Dataset):
    def __init__(self, csv_path, root):
        frame = pd.read_csv(csv_path)
        required = {"filename", "Ground_Truth"}
        missing = required.difference(frame.columns)
        if missing:
            raise KeyError(f"REFUGE CSV is missing columns: {sorted(missing)}")

        self.frame = frame[
            frame["filename"].astype(str).str.contains(r"[\\/]Test[\\/]", case=False, regex=True)
        ].reset_index(drop=True)
        if self.frame.empty:
            raise ValueError(f"No /Test/ rows found in {csv_path}")

        self.root = root
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        relative_path = Path(str(row["filename"]))
        image_path = relative_path if relative_path.is_absolute() else self.root / relative_path
        image = self.transform(Image.open(image_path).convert("RGB"))
        return image, int(row["Ground_Truth"]), str(image_path)


def load_model(device):
    model = get_model_cfp()
    checkpoint = torch.load(CFP_WEIGHTS, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]
    model.load_state_dict(checkpoint)
    return model.to(device).eval()


def main():
    threshold = resolve_threshold()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = RefugeTestDataset(CSV_PATH, DATA_ROOT)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    model = load_model(device)
    rows = []

    print(f"Evaluating RETFound-CFP on {len(dataset)} REFUGE test images using {device}")
    with torch.no_grad():
        for images, labels, paths in loader:
            probabilities = torch.sigmoid(model(images.to(device))).reshape(-1).cpu().numpy()
            predictions = (probabilities >= threshold).astype(np.int64)
            for path, label, probability, prediction in zip(
                paths, labels.numpy(), probabilities, predictions
            ):
                rows.append({
                    "Filename": path,
                    "Ground_Truth": int(label),
                    "Probability_GL": float(probability),
                    "Pred_GL": int(prediction),
                    "Is_Correct": int(prediction == label),
                })

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_CSV, index=False)
    y_true = results["Ground_Truth"].to_numpy()
    y_probability = results["Probability_GL"].to_numpy()
    y_prediction = results["Pred_GL"].to_numpy()
    tn, fp, fn, tp = confusion_matrix(y_true, y_prediction, labels=[0, 1]).ravel()

    metrics = {
        "images": len(results),
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_prediction),
        "precision": precision_score(y_true, y_prediction, zero_division=0),
        "recall_sensitivity": recall_score(y_true, y_prediction, zero_division=0),
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "f1": f1_score(y_true, y_prediction, zero_division=0),
        "auroc": roc_auc_score(y_true, y_probability),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    print("\nPer-class classification report:")
    print(
        classification_report(
            y_true,
            y_prediction,
            labels=[0, 1],
            target_names=["Normal", "Glaucoma"],
            digits=4,
            zero_division=0,
        )
    )
    print(pd.Series(metrics).to_string())
    print(f"\nPredictions saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
