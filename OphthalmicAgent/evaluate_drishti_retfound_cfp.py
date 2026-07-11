"""Evaluate only the fine-tuned RETFound-CFP model on Drishti folders."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from VisionAgent.linear_probing_fundus import get_model_cfp


DATA_ROOT = Path(os.getenv("DRISHTI_DATA_ROOT", "./data_drishti"))
CFP_WEIGHTS = os.getenv("CFP_WEIGHTS", "./weights/cfp_glaucoma_best.pth")
OUTPUT_CSV = os.getenv("OUTPUT_CSV", "drishti_retfound_cfp_predictions.csv")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "8"))
THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class DrishtiCFPDataset(Dataset):
    def __init__(self, root):
        self.samples = []
        for folder_name, label in (("Normal", 0), ("Glaucoma", 1)):
            folder = root / folder_name
            if not folder.is_dir():
                raise FileNotFoundError(f"Required Drishti folder not found: {folder}")
            self.samples.extend(
                (path, label) for path in sorted(folder.rglob("*"))
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = self.transform(Image.open(path).convert("RGB"))
        return image, label, str(path)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = DrishtiCFPDataset(DATA_ROOT)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = get_model_cfp()
    checkpoint = torch.load(CFP_WEIGHTS, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]
    model.load_state_dict(checkpoint)
    model.to(device).eval()

    rows = []
    with torch.no_grad():
        for images, labels, paths in loader:
            probabilities = torch.sigmoid(model(images.to(device))).reshape(-1).cpu().numpy()
            predictions = (probabilities >= THRESHOLD).astype(int)
            for path, label, probability, prediction in zip(paths, labels.numpy(), probabilities, predictions):
                rows.append({
                    "Filename": path,
                    "Ground_Truth": int(label),
                    "Probability_GL": float(probability),
                    "Pred_GL": int(prediction),
                    "Is_Correct": int(prediction == label),
                })

    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT_CSV, index=False)
    y_true = frame["Ground_Truth"].to_numpy()
    y_prob = frame["Probability_GL"].to_numpy()
    y_pred = frame["Pred_GL"].to_numpy()
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "images": len(frame),
        "threshold": THRESHOLD,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall_sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auroc": roc_auc_score(y_true, y_prob),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    print(pd.Series(metrics).to_string())
    print(f"\nPredictions saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
