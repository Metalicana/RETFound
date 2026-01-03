import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from huggingface_hub import hf_hub_download

# --- IMPORT RETFOUND MODEL DEFINITION ---
# This script must be run inside the cloned 'RETFound' repository
try:
    import models_vit
except ImportError:
    raise ImportError("Error: 'models_vit.py' not found. Please run this script inside the 'RETFound' cloned repository.")

# --- CONFIGURATION ---
DATA_ROOT = "/home/ab575577/projects_spring_2026/HarvardFairVision30K/FairVision"  # Updated to your renamed folder
HF_REPO_ID = "YukunZhou/RETFound_mae_natureCFP"
HF_FILENAME = "RETFound_mae_natureCFP.pth"

# H100 Hyperparameters
BATCH_SIZE = 64      # Optimized for 80GB VRAM
LR = 1e-4            # Fine-tuning rate
EPOCHS = 20
NUM_CLASSES = 3      # [AMD, DR, Glaucoma]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- CUSTOM DATASET ---
class FairVisionNPZ(Dataset):
    def __init__(self, root_dir, split='Training', transform=None):
        self.files = []
        self.transform = transform
        self.sources = ['AMD', 'DR', 'Glaucoma']
        
        # Mappings: 0.0 = Healthy, 1.0 = Disease
        self.amd_map = {
            'not.in.icd.table': 0., 'no.amd.diagnosis': 0.,
            'early.dry': 1., 'intermediate.dry': 1., 
            'advanced.atrophic.dry.with.subfoveal.involvement': 1.,
            'advanced.atrophic.dry.without.subfoveal.involvement': 1.,
            'wet.amd.active.choroidal.neovascularization': 1.,
            'wet.amd.inactive.choroidal.neovascularization': 1.,
            'wet.amd.inactive.scar': 1.
        }
        self.dr_map = {
            'not.in.icd.table': 0., 'no.dr.diagnosis': 0.,
            'mild.npdr': 0., 'moderate.npdr': 0.,
            'severe.npdr': 1., 'pdr': 1.
        }

        print(f"Scanning {split} data in {root_dir}...")
        for source in self.sources:
            path = os.path.join(root_dir, source, split)
            if not os.path.exists(path):
                print(f"Warning: Directory not found: {path}")
                continue
            
            # Find all .npz files
            files_found = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.npz')]
            for f_path in files_found:
                self.files.append({'path': f_path, 'source': source})
                
        print(f"Found {len(self.files)} images for {split}.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        item = self.files[idx]
        try:
            # Load NPZ
            data = np.load(item['path'])
            
            # 1. LOAD IMAGE
            img_array = data['slo_fundus']
            # Normalize to 0-255 uint8 if float
            if img_array.max() <= 1.0:
                img_array = (img_array * 255).astype(np.uint8)
            else:
                img_array = img_array.astype(np.uint8)
            
            image = Image.fromarray(img_array).convert('RGB')
            if self.transform:
                image = self.transform(image)
            
            # 2. PARSE LABEL
            label = torch.zeros(NUM_CLASSES)
            source = item['source']
            
            if source == 'AMD':
                cond = str(data['amd_condition'])
                if self.amd_map.get(cond, 0.) >= 1.0: label[0] = 1.0
            elif source == 'DR':
                cond = str(data['dr_subtype'])
                if self.dr_map.get(cond, 0.) >= 1.0: label[1] = 1.0
            elif source == 'Glaucoma':
                if int(data['glaucoma']) == 1: label[2] = 1.0

            race = int(data['race']) if 'race' in data else -1
            return image, label, race

        except Exception as e:
            print(f"Error loading {item['path']}: {e}")
            # Robust fallback
            return self.__getitem__(idx - 1 if idx > 0 else 0)

# --- MODEL SETUP ---
# --- FIXED MODEL SETUP ---
def get_model():
    print("Loading RETFound ViT-Large...")
    
    # 1. Download/Cache Weights
    repo_id = "YukunZhou/RETFound_mae_natureCFP"
    filename = "RETFound_mae_natureCFP.pth"
    print(f"Checking for weights...")
    try:
        weights_path = hf_hub_download(repo_id=repo_id, filename=filename)
    except Exception as e:
        print(f"Auto-download failed ({e}). Checking local './RETFound_cfp_weights.pth'...")
        weights_path = "./RETFound_cfp_weights.pth"
        
    # --- FIX: Use 'RETFound_mae' (Case Sensitive) ---
    try:
        model = models_vit.RETFound_mae(
            img_size=224,
            num_classes=NUM_CLASSES,
            drop_path_rate=0.2,
            global_pool=True,
        )
    except AttributeError as e:
        # Emergency print to see what IS available if this fails again
        print(f"CRITICAL ERROR: Could not find model function. Available functions in models_vit: {dir(models_vit)}")
        raise e
    
    # Load State Dict
    checkpoint = torch.load(weights_path, map_location='cpu')
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint

    # Remove incompatible head weights (Surgery)
    keys_to_remove = [k for k in state_dict.keys() if 'head' in k or 'decoder' in k]
    for k in keys_to_remove:
        del state_dict[k]
            
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Weights loaded. (Missing keys expected): {len(msg.missing_keys)}")
    return model

# --- TRAINING LOOP ---
def train():
    # Transforms (ImageNet Stats)
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

    # Datasets & Loaders
    train_ds = FairVisionNPZ(DATA_ROOT, split='Training', transform=train_transform)
    val_ds = FairVisionNPZ(DATA_ROOT, split='Validation', transform=val_transform)
    
    # H100 Optimization: 16 workers, pin_memory
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=16, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=8, pin_memory=True)
    
    model = get_model().to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    criterion = nn.BCEWithLogitsLoss() 
    scaler = torch.cuda.amp.GradScaler() # Mixed Precision
    
    best_auc = 0.0

    print(f"Starting Training on {DEVICE} for {EPOCHS} epochs...")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for imgs, labels, _ in loop:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for imgs, labels, _ in val_loader:
                imgs = imgs.to(DEVICE)
                with torch.cuda.amp.autocast():
                    outputs = model(imgs)
                    probs = torch.sigmoid(outputs)
                
                val_preds.append(probs.cpu().numpy())
                val_targets.append(labels.cpu().numpy())
                
        val_preds = np.vstack(val_preds)
        val_targets = np.vstack(val_targets)
        
        try:
            auc = roc_auc_score(val_targets, val_preds, average="macro")
            print(f"Epoch {epoch+1} Results - Loss: {train_loss/len(train_loader):.4f} - Val AUC: {auc:.4f}")
            
            if auc > best_auc:
                best_auc = auc
                torch.save(model.state_dict(), "best_fair_eye_model.pth")
                print(f">>> SAVED BEST MODEL (AUC: {best_auc:.4f})")
                
        except Exception as e:
            print(f"Warning: Could not calculate AUC: {e}")

if __name__ == "__main__":
    train()