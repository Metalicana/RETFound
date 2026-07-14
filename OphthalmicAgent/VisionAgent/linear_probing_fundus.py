# Binary glaucoma classification for CFP using the official REFUGE splits.

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    from VisionAgent.models_vit import RETFound_mae
except ImportError:
    raise ImportError("Error: 'models_vit.py' not found.")

# --- CONFIGURATION ---
DATA_ROOT = os.getenv("REFUGE_DATA_ROOT", "./")
CSV_PATH = os.getenv("REFUGE_CSV", "data_refuge/data.csv")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "8"))
LR = float(os.getenv("HEAD_LR", "1e-3"))
BACKBONE_LR = float(os.getenv("BACKBONE_LR", "1e-5"))
UNFREEZE_BLOCKS = int(os.getenv("UNFREEZE_BLOCKS", "0"))
EPOCHS = int(os.getenv("EPOCHS", "30"))
THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATH = os.getenv("CFP_OUTPUT_WEIGHTS", "./weights/cfp_glaucoma_best.pth")
METADATA_PATH = os.getenv("CFP_TRAINING_METADATA", "cfp_glaucoma_training_metadata.json")
TEST_PREDICTIONS_PATH = os.getenv("CFP_TEST_PREDICTIONS", "refuge_test_retfound_cfp_predictions.csv")


class RefugeCFPDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Build the exact image path
        img_path = os.path.join(self.root_dir, row['filename'])
        
        try:
            # CFP are standard 3-channel RGB JPEG images
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
                
            # Ground_Truth is already binary (0 or 1)
            label = torch.tensor([float(row['Ground_Truth'])], dtype=torch.float32)
            
            metadata = {
                'filename': row['filename'],
                'groundtruth': int(row['Ground_Truth'])
            }
            return image, label, metadata
            
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Safe fallback loop to avoid crashing mid-training
            return self.__getitem__((idx + 1) % len(self.df))


class RETFoundGlaucoma(nn.Module):
    def __init__(self, backbone):
        super(RETFoundGlaucoma, self).__init__()
        self.backbone = backbone
        
        # Specialist binary linear probing head matching your OCT structure
        self.glaucoma_head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 1)
        )

        # Initialize weights
        for m in self.glaucoma_head.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        features = self.backbone(x)
        return self.glaucoma_head(features)


def get_model_cfp():
    # Path to the pre-trained RETFound Color Fundus Photography weights
    weight_path = "/lustre/fs1/home/yu395012/RETFound/OphthalmicAgent/VisionAgent/weights/RETFound_mae_natureCFP.pth"
        
    backbone = RETFound_mae(
        img_size=224,
        num_classes=0, 
        drop_path_rate=0.0,
        global_pool='',
    )
    
    # Freeze the backbone
    for param in backbone.parameters():
        param.requires_grad = False
    
    # Load pre-trained weights
    checkpoint = torch.load(weight_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint

    keys_to_remove = [k for k in state_dict.keys() if 'head' in k or 'decoder' in k or 'fc_norm' in k]
    for k in keys_to_remove:
        if k in state_dict:
            del state_dict[k]
            
    load_result = backbone.load_state_dict(state_dict, strict=False)
    print(f"RETFound missing checkpoint keys: {load_result.missing_keys}")
    print(f"RETFound unexpected checkpoint keys: {load_result.unexpected_keys}")
    
    # Wrap backbone with the single binary glaucoma head
    model = RETFoundGlaucoma(backbone)
    
    # Unfreeze head
    for param in model.glaucoma_head.parameters():
        param.requires_grad = True
    # fc_norm is created for downstream global pooling and is not learned by
    # the MAE pretraining objective, so train it together with the CFP head.
    for param in model.backbone.fc_norm.parameters():
        param.requires_grad = True
    
    return model


def _contains_split(series, split):
    return series.astype(str).str.contains(
        rf"[\\/]{split}[\\/]", case=False, regex=True
    )


def _transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    evaluation_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_transform, evaluation_transform


def _loader(frame, transform, shuffle):
    return DataLoader(
        RefugeCFPDataset(frame, DATA_ROOT, transform=transform),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE.type == "cuda",
    )


def _positive_weight(frame):
    positives = int((frame["Ground_Truth"] == 1).sum())
    negatives = int((frame["Ground_Truth"] == 0).sum())
    if positives == 0 or negatives == 0:
        raise ValueError("Training data must contain both positive and negative cases")
    return negatives / positives


def _predict(model, loader):
    model.eval()
    targets, probabilities, filenames = [], [], []
    with torch.no_grad():
        for images, labels, metadata in loader:
            logits = model(images.to(DEVICE))
            targets.extend(labels.reshape(-1).numpy().astype(int).tolist())
            probabilities.extend(torch.sigmoid(logits).reshape(-1).cpu().numpy().tolist())
            filenames.extend(list(metadata["filename"]))
    return np.asarray(targets), np.asarray(probabilities), filenames


def _train_epochs(frame, epochs, output_path=None, validation_loader=None):
    train_transform, _ = _transforms()
    train_loader = _loader(frame, train_transform, shuffle=True)
    model = get_model_cfp().to(DEVICE)
    head_parameters = list(model.glaucoma_head.parameters()) + list(
        model.backbone.fc_norm.parameters()
    )
    parameter_groups = [{"params": head_parameters, "lr": LR}]
    if UNFREEZE_BLOCKS > 0:
        blocks = model.backbone.blocks
        if UNFREEZE_BLOCKS > len(blocks):
            raise ValueError(f"UNFREEZE_BLOCKS={UNFREEZE_BLOCKS} exceeds {len(blocks)} ViT blocks")
        backbone_parameters = []
        for block in blocks[-UNFREEZE_BLOCKS:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
                backbone_parameters.append(parameter)
        parameter_groups.append({"params": backbone_parameters, "lr": BACKBONE_LR})
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=0.05)
    pos_weight = _positive_weight(frame)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=DEVICE)
    )
    best_auc, best_epoch = -np.inf, epochs

    for epoch in range(epochs):
        model.train()
        model.backbone.eval()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for images, labels, _ in loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        message = f"loss={running_loss / len(train_loader):.4f}"
        if validation_loader is not None:
            targets, probabilities, _ = _predict(model, validation_loader)
            auc = roc_auc_score(targets, probabilities)
            message += f" validation_AUROC={auc:.4f}"
            if auc > best_auc:
                best_auc, best_epoch = auc, epoch + 1
                os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
                torch.save(model.state_dict(), output_path)
        print(f"Epoch {epoch + 1}: {message}")

    if validation_loader is None and output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        torch.save(model.state_dict(), output_path)
    return model, best_epoch, best_auc, pos_weight


def _metrics(targets, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(targets, predictions, labels=[0, 1]).ravel()
    return predictions, {
        "auroc": roc_auc_score(targets, probabilities),
        "accuracy": accuracy_score(targets, predictions),
        "precision": precision_score(targets, predictions, zero_division=0),
        "sensitivity": recall_score(targets, predictions, zero_division=0),
        "specificity": tn / (tn + fp) if tn + fp else float("nan"),
        "f1": f1_score(targets, predictions, zero_division=0),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def train():
    master = pd.read_csv(CSV_PATH)
    train_df = master[_contains_split(master["filename"], "Train")].copy()
    validation_df = master[_contains_split(master["filename"], "Validation")].copy()
    test_df = master[_contains_split(master["filename"], "Test")].copy()
    if any(frame.empty for frame in (train_df, validation_df, test_df)):
        raise ValueError("REFUGE Train, Validation, and Test splits must all be present")
    print(
        f"Official splits: Train={len(train_df)}, Validation={len(validation_df)}, "
        f"Test={len(test_df)}"
    )
    combined_df = pd.concat([train_df, validation_df], ignore_index=True)
    print(
        f"Training once on Train+Validation: {len(combined_df)} images for "
        f"{EPOCHS} fixed epochs. Test remains untouched."
    )
    _, _, _, combined_pos_weight = _train_epochs(
        combined_df, EPOCHS, PATH, validation_loader=None
    )

    final_model = get_model_cfp().to(DEVICE)
    final_model.load_state_dict(torch.load(PATH, map_location=DEVICE, weights_only=False))
    _, evaluation_transform = _transforms()
    test_loader = _loader(test_df, evaluation_transform, shuffle=False)
    test_targets, test_probabilities, test_filenames = _predict(final_model, test_loader)
    test_predictions, test_metrics = _metrics(
        test_targets, test_probabilities, THRESHOLD
    )
    pd.DataFrame({
        "Filename": test_filenames,
        "Ground_Truth": test_targets,
        "Probability_GL": test_probabilities,
        "Pred_GL": test_predictions,
        "Is_Correct": (test_predictions == test_targets).astype(int),
    }).to_csv(TEST_PREDICTIONS_PATH, index=False)

    metadata = {
        "official_split_sizes": {"train": len(train_df), "validation": len(validation_df), "test": len(test_df)},
        "training_strategy": "single fixed run on Train+Validation; one final evaluation on Test",
        "fixed_training_epochs": EPOCHS,
        "fixed_threshold": THRESHOLD,
        "combined_train_validation_pos_weight": combined_pos_weight,
        "head_learning_rate": LR,
        "unfrozen_backbone_blocks": UNFREEZE_BLOCKS,
        "backbone_learning_rate": BACKBONE_LR if UNFREEZE_BLOCKS else None,
        "global_pool": True,
        "drop_path_rate": 0.0,
        "frozen_backbone_eval_mode": True,
        "final_test_metrics": test_metrics,
        "final_weights": PATH,
        "test_predictions": TEST_PREDICTIONS_PATH,
        "threshold_note": "The threshold was fixed before Test evaluation and was not selected on Test.",
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(json.dumps(metadata, indent=2))

if __name__ == "__main__":
    train()
