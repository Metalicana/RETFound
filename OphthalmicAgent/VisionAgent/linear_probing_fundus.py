# Pure Binary Glaucoma Classification for Color Fundus Photography (CFP)
# Merges Train + Test folders for training, and uses Validation folder for validation

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

try:
    from VisionAgent.models_vit import RETFound_mae
except ImportError:
    raise ImportError("Error: 'models_vit.py' not found.")

# --- CONFIGURATION ---
DATA_ROOT = "./"
CSV_PATH = "data_refuge/data.csv"
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 60
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATH = "cfp_glaucoma_best.pth"


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
        drop_path_rate=0.2,
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
            
    backbone.load_state_dict(state_dict, strict=False)
    
    # Wrap backbone with the single binary glaucoma head
    model = RETFoundGlaucoma(backbone)
    
    # Unfreeze head
    for param in model.glaucoma_head.parameters():
        param.requires_grad = True
    
    return model


def train():
    # 1. Image Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Filter CSV into training and validation splits
    master_df = pd.read_csv(CSV_PATH)
    
    # Combine rows containing 'Train' or 'Test' for training
    train_df = master_df[master_df['filename'].str.contains('/Train/|/Test/', case=False, regex=True)]
    # Use rows containing 'Validation' for validation
    val_df = master_df[master_df['filename'].str.contains('/Validation/', case=False)]
    
    print(f"Dataset summary: Training on {len(train_df)} images | Validating on {len(val_df)} images.")

    # 3. Create DataLoaders
    train_ds = RefugeCFPDataset(train_df, DATA_ROOT, transform=train_transform)
    val_ds = RefugeCFPDataset(val_df, DATA_ROOT, transform=val_transform)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)
    
    # 4. Initialize Model, Loss, and Optimizer
    model = get_model_cfp().to(DEVICE)
    optimizer = torch.optim.AdamW(model.glaucoma_head.parameters(), lr=LR, weight_decay=0.05)
    criterion = nn.BCEWithLogitsLoss()
    
    best_auc = 0.0

    # 5. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for imgs, labels, _ in loop:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            
            outputs = model(imgs) 
            loss = criterion(outputs, labels)
                
            loss.backward()
            optimizer.step()
        
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        # 6. Validation Pass
        model.eval()
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for imgs, labels, _ in val_loader:
                imgs = imgs.to(DEVICE)
                outputs = model(imgs)
                
                # Apply sigmoid to convert logits to probabilities [0.0 - 1.0]
                val_preds.append(torch.sigmoid(outputs).cpu().numpy())
                val_targets.append(labels.cpu().numpy())

        all_targets = np.vstack(val_targets)
        all_preds = np.vstack(val_preds)

        # 7. Evaluate Performance
        if len(np.unique(all_targets)) > 1:
            auc = roc_auc_score(all_targets, all_preds)
            print(f"\n[Epoch {epoch+1}] Avg Loss: {train_loss/len(train_loader):.4f} | Validation AUC: {auc:.4f}")
            
            if auc > best_auc:
                best_auc = auc
                torch.save(model.state_dict(), PATH)
                print(f"--> Saved New Best CFP Model! (AUC: {best_auc:.4f})")
        else:
            print(f"\n[Epoch {epoch+1}] Avg Loss: {train_loss/len(train_loader):.4f} | Cannot calculate AUC (only 1 class present in batch).")

if __name__ == "__main__":
    train()