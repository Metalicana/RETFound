"""Evaluate GDP Test using only the fine-tuned RETFound-OCT model."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from data.gdp_loader import GDPTestLoader
from VisionAgent.linear_probing_oct3 import get_model_oct


CSV_PATH = os.getenv("GDP_CSV", "./data_gdp/data_summary.csv")
BSCAN_DIR = os.getenv("GDP_BSCAN_DIR", "./data_gdp/BScan")
RNFLT_DIR = os.getenv("GDP_RNFLT_DIR", "./data_gdp/RNFLT")
OCT_WEIGHTS = os.getenv("OCT_WEIGHTS", "./weights/gdp_oct_glaucoma_head_best.pth")
OUTPUT_CSV = os.getenv("OUTPUT_CSV", "gdp_test_retfound_oct_predictions.csv")
THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "8"))
MAX_CASES = int(os.getenv("MAX_CASES", "0"))
OCT_SLICES = int(os.getenv("OCT_SLICES", "8"))


class GDPDataset(Dataset):
    def __init__(self, loader):
        self.loader = loader
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.loader)

    def __getitem__(self, index):
        case = self.loader.load(index, include_rnflt=False)
        images = torch.stack([
            self.transform(Image.fromarray(slice_).convert("RGB"))
            for slice_ in case["oct_slices"]
        ])
        return images, case["ground_truth"], case["patient_id"], str(case["bscan_path"])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source = GDPTestLoader(
        CSV_PATH,
        BSCAN_DIR,
        RNFLT_DIR,
        OCT_SLICES,
        MAX_CASES,
        require_rnflt=False,
    )
    dataset = GDPDataset(source)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    model = get_model_oct()
    checkpoint = torch.load(OCT_WEIGHTS, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]
    model.load_state_dict(checkpoint)
    model.to(device).eval()

    rows = []
    print(f"Evaluating RETFound-OCT on {len(dataset)} GDP Test cases using {device}")
    with torch.inference_mode():
        for images, labels, patient_ids, paths in dataloader:
            logits = model(images.to(device))["glaucoma"]
            probabilities = torch.sigmoid(logits).reshape(-1).cpu().numpy()
            predictions = (probabilities >= THRESHOLD).astype(int)
            for patient_id, path, label, probability, prediction in zip(
                patient_ids, paths, labels.numpy(), probabilities, predictions
            ):
                rows.append({
                    "Patient_ID": patient_id,
                    "Filename": path,
                    "Ground_Truth": int(label),
                    "Probability_GL": float(probability),
                    "Pred_GL": int(prediction),
                    "Is_Correct": int(prediction == label),
                })

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_CSV, index=False)
    y_true = results["Ground_Truth"].to_numpy()
    y_pred = results["Pred_GL"].to_numpy()
    y_prob = results["Probability_GL"].to_numpy()
    print("\nPer-class classification report:")
    print(classification_report(
        y_true, y_pred, labels=[0, 1], target_names=["Normal", "Glaucoma"],
        digits=4, zero_division=0,
    ))
    print(f"AUROC: {roc_auc_score(y_true, y_prob):.4f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))
    print(f"\nPredictions saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
