##Linear Probing Fundus using AMD, DR, Glaucoma as 0,1 
#
#import os
#import torch
#import torch.nn as nn
#import torch.optim as optim
#import numpy as np
#from torch.utils.data import Dataset, DataLoader
#from torchvision import transforms
#from PIL import Image
#from tqdm import tqdm
#from sklearn.metrics import roc_auc_score
#
#try:
#    from VisionAgent.models_vit import RETFound_mae
#except ImportError:
#    raise ImportError("Error: 'models_vit.py' not found.")
#
## --- CONFIGURATION ---
#DATA_ROOT = "/lustre/fs1/home/yu395012/RETFound/OphthalmicAgent/data/"
#
## H100 Hyperparameters
#BATCH_SIZE = 64      # Optimized for 80GB VRAM
#LR = 1e-4            # Fine-tuning rate
#EPOCHS = 20
#NUM_CLASSES = 3      # [AMD, DR, Glaucoma]
#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#
#class FairVisionNPZ(Dataset):
#    def __init__(self, root_dir, split='Training', transform=None):
#        self.files = []
#        self.transform = transform
#        self.sources = ['AMD', 'DR', 'Glaucoma']
#        
#        # Mappings: 0.0 = Healthy, 1.0 = Disease
#        self.amd_map = {
#            'not.in.icd.table': 0., 'no.amd.diagnosis': 0.,
#            'early.dry': 1., 'intermediate.dry': 1., 
#            'advanced.atrophic.dry.with.subfoveal.involvement': 1.,
#            'advanced.atrophic.dry.without.subfoveal.involvement': 1.,
#            'wet.amd.active.choroidal.neovascularization': 1.,
#            'wet.amd.inactive.choroidal.neovascularization': 1.,
#            'wet.amd.inactive.scar': 1.
#        }
#        self.dr_map = {
#            'not.in.icd.table': 0., 'no.dr.diagnosis': 0.,
#            'mild.npdr': 0., 'moderate.npdr': 0.,
#            'severe.npdr': 1., 'pdr': 1.
#        }
#
#        print(f"Scanning {split} data in {root_dir}...")
#        for source in self.sources:
#            path = os.path.join(root_dir, source, split)
#            if not os.path.exists(path):
#                print(f"Warning: Directory not found: {path}")
#                continue
#            
#            # Find all .npz files
#            files_found = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.npz')]
#            for f_path in files_found:
#                self.files.append({'path': f_path, 'source': source})
#                
#        print(f"Found {len(self.files)} images for {split}.")
#
#    def __len__(self):
#        return len(self.files)
#
#    def __getitem__(self, idx):
#        item = self.files[idx]
#        try:
#            # Load NPZ
#            data = np.load(item['path'])
#            
#            fundus_img = data['slo_fundus']
#            if fundus_img.max() <= 1.0:
#                fundus_img = (fundus_img * 255).astype(np.uint8)
#            else:
#                fundus_img = fundus_img.astype(np.uint8)
#                
#                
#            # Convert to RGB (RETFound expects 3 channels)
#            image = Image.fromarray(fundus_img).convert('RGB')   
#            
#            if self.transform:
#                image = self.transform(image) 
#                
#            label = torch.zeros(NUM_CLASSES)
#            source = item['source']
#                   
#            if source == 'AMD':
#                cond = str(data['amd_condition'])
#                if self.amd_map.get(cond, 0.) >= 1.0: label[0] = 1.0
#            elif source == 'DR':
#                cond = str(data['dr_subtype'])
#                if self.dr_map.get(cond, 0.) >= 1.0: label[1] = 1.0
#            elif source == 'Glaucoma':
#                if int(data['glaucoma']) == 1: label[2] = 1.0
#
#            race = int(data['race']) if 'race' in data else -1
#            return image, label, race
#
#        except Exception as e:
#            print(f"Error loading {item['path']}: {e}")
#            # Robust fallback
#            return self.__getitem__(idx - 1 if idx > 0 else 0)
#
#
#def get_model():
#    print("Loading RETFound ViT-Large with Frozen Backbone...")
#    
#    weight_path = "/lustre/fs1/home/yu395012/RETFound/OphthalmicAgent/VisionAgent/weights/RETFound_mae_natureCFP.pth"
#        
#    # 1. Initialize the architecture
#    model = RETFound_mae(
#        img_size=224,
#        num_classes=NUM_CLASSES, # This creates model.head with size [1024, 3]
#        drop_path_rate=0.2,
#        global_pool='',
#    )
#    
#    # 2. FREEZE THE ENTIRE MODEL (Backbone)
#    for param in model.parameters():
#        param.requires_grad = False
#    
#    # 3. LOAD PRE-TRAINED WEIGHTS
#    checkpoint = torch.load(weight_path, map_location='cpu', weights_only=False)
#    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
#
#    keys_to_remove = [k for k in state_dict.keys() if 'head' in k or 'decoder' in k or 'fc_norm' in k]
#    for k in keys_to_remove:
#        if k in state_dict:
#            del state_dict[k]
#            
#    # Load the backbone weights (strict=False is mandatory here)
#    msg = model.load_state_dict(state_dict, strict=False)
#    
#    # 4. RE-INITIALIZE AND UNFREEZE THE HEAD
#    # This ensures the diagnostic head is fresh and trainable
#    model.head = torch.nn.Linear(1024, NUM_CLASSES)
#    model.head.weight.data.normal_(mean=0.0, std=0.01)
#    model.head.bias.data.zero_()
#    
#    for param in model.head.parameters():
#        param.requires_grad = True
#
#    print(f"Backbone frozen. Head initialized for {NUM_CLASSES} classes.")
#    
#    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#    total_params = sum(p.numel() for p in model.parameters())
#    print(f"Total Parameters: {total_params:,}")
#    print(f"Trainable Parameters (Head): {trainable_params:,}") 
#    # This should be around 3,075
#    
#    return model
#
#
#def train():
#    # Transforms (ImageNet Stats)
#    train_transform = transforms.Compose([
#        transforms.Resize((224, 224)),
#        transforms.RandomHorizontalFlip(),
#        transforms.RandomRotation(15),
#        transforms.ToTensor(),
#        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#    ])
#    
#    val_transform = transforms.Compose([
#        transforms.Resize((224, 224)),
#        transforms.ToTensor(),
#        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#    ])
#
#    # Datasets & Loaders
#    train_ds = FairVisionNPZ(DATA_ROOT, split='Training', transform=train_transform)
#    val_ds = FairVisionNPZ(DATA_ROOT, split='Validation', transform=val_transform)
#    
#    # H100 Optimization: 16 workers, pin_memory
#    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
#                              num_workers=16, pin_memory=True)
#    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, 
#                            num_workers=8, pin_memory=True)
#    
#    model = get_model().to(DEVICE)
#    
#    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
#    criterion = nn.BCEWithLogitsLoss() 
#    scaler = torch.cuda.amp.GradScaler() # Mixed Precision
#    
#    best_auc = 0.0
#
#    print(f"Starting Training on {DEVICE} for {EPOCHS} epochs...")
#    
#    for epoch in range(EPOCHS):
#        model.train()
#        train_loss = 0.0
#        
#        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
#        for imgs, labels, _ in loop:
#            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
#            
#            optimizer.zero_grad()
#            
#            with torch.cuda.amp.autocast():
#                outputs = model(imgs)
#                loss = criterion(outputs, labels)
#            
#            scaler.scale(loss).backward()
#            scaler.step(optimizer)
#            scaler.update()
#            
#            train_loss += loss.item()
#            loop.set_postfix(loss=loss.item())
#            
#        # Validation
#        model.eval()
#        val_preds = []
#        val_targets = []
#        
#        with torch.no_grad():
#            for imgs, labels, _ in val_loader:
#                imgs = imgs.to(DEVICE)
#                with torch.cuda.amp.autocast():
#                    outputs = model(imgs)
#                    probs = torch.sigmoid(outputs)
#                
#                val_preds.append(probs.cpu().numpy())
#                val_targets.append(labels.cpu().numpy())
#                
#        val_preds = np.vstack(val_preds)
#        val_targets = np.vstack(val_targets)
#        
#        try:
#            auc = roc_auc_score(val_targets, val_preds, average="macro")
#            print(f"Epoch {epoch+1} Results - Loss: {train_loss/len(train_loader):.4f} - Val AUC: {auc:.4f}")
#            
#            if auc > best_auc:
#                best_auc = auc
#                torch.save(model.state_dict(), "fundus_model.pth")
#                print(f">>> SAVED BEST MODEL (AUC: {best_auc:.4f})")
#                
#        except Exception as e:
#            print(f"Warning: Could not calculate AUC: {e}")
#
#if __name__ == "__main__":
#    train()


# Linear Probing SLO Fundus using Binary AMD, DR, Glaucoma
# Styled completely in parity with the multi-head OCT Probing architecture

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
DATA_ROOT = "/lustre/fs1/home/yu395012/RETFound/OphthalmicAgent/data/"
BATCH_SIZE = 64      # Optimized for 80GB VRAM
LR = 1e-3            # Mirrors OCT learning rate
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATH = "fundus_model_best.pth"


class FairVisionNPZ(Dataset):
    def __init__(self, root_dir, split='Training', transform=None):
        self.files = []
        self.transform = transform
        self.sources = ['AMD', 'DR', 'Glaucoma']
        self.csv_base_path = root_dir
        
        self.metadata_lookup = self._load_all_metadata()

        # Binary Mappings: 0.0 = Healthy, 1.0 = Disease Presence
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
                continue
            
            all_files_found = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.npz')]
            all_files_found.sort()
            files_found = all_files_found
            
            for f_path in files_found:
                fname = os.path.basename(f_path)
                file_meta = self.metadata_lookup.get(fname, {})
                
                self.files.append({
                    'path': f_path, 
                    'source': source,
                    'meta': file_meta
                })

        print(f"Found {len(self.files)} images with metadata for {split}.")

    def _load_all_metadata(self):
        """Pre-loads CSVs and converts them to a fast-access dictionary."""
        combined_meta = {}
        for source in self.sources:
            csv_name = f"data_summary_{source.lower()}.csv"
            csv_path = os.path.join(self.csv_base_path, source, csv_name)
            
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                records = df.to_dict('records')
                for rec in records:
                    combined_meta[rec['filename']] = rec
            else:
                print(f"Warning: Metadata CSV not found at {csv_path}")
        return combined_meta

    def __getitem__(self, idx):
        item = self.files[idx]
        try:
            data = np.load(item['path'])
            
            # --- Image Processing ---
            fundus_img = data['slo_fundus']
            if fundus_img.max() <= 1.0:
                fundus_img = (fundus_img * 255).astype(np.uint8)
            else:
                fundus_img = fundus_img.astype(np.uint8)
                
            image = Image.fromarray(fundus_img).convert('RGB')   
            if self.transform: 
                image = self.transform(image)
            
            # --- Label Processing ---
            # Index 0: AMD Binary, Index 1: DR Binary, Index 2: Glaucoma Binary
            label = torch.full((3,), -1.0) 
            source = item['source']
            csv_meta = item['meta']
            
            if source == 'AMD':
                cond = str(data['amd_condition'])
                severity = int(self.amd_map.get(cond, 0.))
                label[0] = 1.0 if severity == 1 else 0.0
                
            elif source == 'DR':
                cond = str(data['dr_subtype'])
                severity = int(self.dr_map.get(cond, 0.))
                label[1] = 1.0 if severity >= 1.0 else 0.0
                
            elif source == 'Glaucoma':
                severity = int(data['glaucoma'])
                label[2] = 1.0 if severity == 1 else 0.0
                
            # --- Metadata Assembly ---
            metadata = {
                'disease': source,
                'age': csv_meta.get('age', 'unknown'),
                'gender': csv_meta.get('gender', 'unknown'),
                'race': csv_meta.get('race', 'unknown'),
                'ethnicity': csv_meta.get('ethnicity', 'unknown'),
                'language': csv_meta.get('language', 'unknown'),
                'maritalstatus': csv_meta.get('maritalstatus', 'unknown'),
                'filename': os.path.basename(item['path']),
                'groundtruth': severity
            }
            
            return image, label, metadata

        except Exception as e:
            print(f"Error at index {idx}: {e}")
            return self.__getitem__(idx - 1 if idx > 0 else 0)

    def __len__(self):
        return len(self.files)

        
class RETFoundMultiHead(nn.Module):
    def __init__(self, backbone):
        super(RETFoundMultiHead, self).__init__()
        self.backbone = backbone
        
        # Sequentially styled 3-Head Specialist Architecture matching the OCT module
        self.amd_head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 1)
        )
        self.dr_head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 1)
        )
        self.glaucoma_head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 1)
        )

        # Initialize weights for MLP linear layers
        for head in [self.amd_head, self.dr_head, self.glaucoma_head]:
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    m.weight.data.normal_(mean=0.0, std=0.01)
                    if m.bias is not None:
                        m.bias.data.zero_()

    def forward(self, x):
        features = self.backbone(x)
        return {
            'amd': self.amd_head(features),
            'dr': self.dr_head(features),
            'glaucoma': self.glaucoma_head(features)
        }


def get_model_slo():
    weight_path = "/lustre/fs1/home/yu395012/RETFound/OphthalmicAgent/VisionAgent/weights/RETFound_mae_natureCFP.pth"
        
    backbone = RETFound_mae(
        img_size=224,
        num_classes=0, 
        drop_path_rate=0.2,
        global_pool='',
    )
    
    # Freeze the entire backbone parameters
    for param in backbone.parameters():
        param.requires_grad = False
    
    checkpoint = torch.load(weight_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint

    keys_to_remove = [k for k in state_dict.keys() if 'head' in k or 'decoder' in k or 'fc_norm' in k]
    for k in keys_to_remove:
        if k in state_dict:
            del state_dict[k]
            
    backbone.load_state_dict(state_dict, strict=False)
    model = RETFoundMultiHead(backbone)
    
    # Explicitly unfreeze and map gradients for head networks only
    for param in model.amd_head.parameters(): param.requires_grad = True
    for param in model.dr_head.parameters(): param.requires_grad = True
    for param in model.glaucoma_head.parameters(): param.requires_grad = True
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters (All Heads): {trainable_params:,}") 
    
    return model


class MultiAgentLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        # Keeps exact pos_weight scalars matching OCT loss constraints
        self.amd_pos_weight = torch.tensor([1.5]).to(device)
        self.bce_amd = nn.BCEWithLogitsLoss(pos_weight=self.amd_pos_weight)
        
        self.dr_pos_weight = torch.tensor([5.0]).to(device)
        self.bce_dr = nn.BCEWithLogitsLoss(pos_weight=self.dr_pos_weight)
        
        self.bce_standard = nn.BCEWithLogitsLoss()

    def forward(self, outputs, targets):
        # targets positions mapped: [AMD, DR, Glaucoma]
        
        # --- 1. AMD Loss ---
        amd_mask = (targets[:, 0] != -1)
        loss_amd = self.bce_amd(outputs['amd'][amd_mask], targets[amd_mask, 0:1]) if amd_mask.any() else 0.0

        # --- 2. DR Loss ---
        dr_mask = (targets[:, 1] != -1)
        loss_dr = self.bce_dr(outputs['dr'][dr_mask], targets[dr_mask, 1:2]) if dr_mask.any() else 0.0

        # --- 3. Glaucoma Loss ---
        gl_mask = (targets[:, 2] != -1)
        loss_gl = self.bce_standard(outputs['glaucoma'][gl_mask], targets[gl_mask, 2:3]) if gl_mask.any() else 0.0

        return loss_amd + loss_dr + loss_gl


def train(resume = True):
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

    train_ds = FairVisionNPZ(DATA_ROOT, split='Training', transform=train_transform)
    val_ds = FairVisionNPZ(DATA_ROOT, split='Validation', transform=val_transform)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)
    
    model = get_model_slo().to(DEVICE)
    checkpoint_path = PATH
    
    if resume and os.path.exists(checkpoint_path):
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
            print("Successfully loaded specialist weights. Resuming...")
        except Exception as e:
            print(f"Could not restore parameters. Starting fresh. Error: {e}")
        
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=LR, 
        weight_decay=0.05
    )
    criterion = MultiAgentLoss(DEVICE)
    
    best_avg_auc = 0.0

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

        # --- VALIDATION ---
        model.eval()
        val_data = { 'amd': [], 'dr': [], 'gl': [], 'targets': [] }
        
        with torch.no_grad():
            for imgs, labels, _ in val_loader:
                imgs = imgs.to(DEVICE)
                outputs = model(imgs)
                
                val_data['amd'].append(torch.sigmoid(outputs['amd']).cpu().numpy())
                val_data['dr'].append(torch.sigmoid(outputs['dr']).cpu().numpy())
                val_data['gl'].append(torch.sigmoid(outputs['glaucoma']).cpu().numpy())
                val_data['targets'].append(labels.cpu().numpy())

        all_targets = np.vstack(val_data['targets'])
        all_amd_preds = np.vstack(val_data['amd'])
        all_dr_preds = np.vstack(val_data['dr'])
        all_gl_preds = np.vstack(val_data['gl'])

        metrics = {}
        
        # 1. Binary AMD AUC Check
        amd_mask = all_targets[:, 0] != -1
        if amd_mask.any() and len(np.unique(all_targets[amd_mask, 0])) > 1:
            metrics['AMD_AUC'] = roc_auc_score(all_targets[amd_mask, 0], all_amd_preds[amd_mask, 0])

        # 2. DR AUC Check
        dr_mask = all_targets[:, 1] != -1
        if dr_mask.any() and len(np.unique(all_targets[dr_mask, 1])) > 1:
            metrics['DR_AUC'] = roc_auc_score(all_targets[dr_mask, 1], all_dr_preds[dr_mask, 0])

        # 3. Glaucoma AUC Check
        gl_mask = all_targets[:, 2] != -1
        if gl_mask.any() and len(np.unique(all_targets[gl_mask, 2])) > 1:
            metrics['Glaucoma_AUC'] = roc_auc_score(all_targets[gl_mask, 2], all_gl_preds[gl_mask, 0])

        # --- LOGGING & MODEL EXPORT ---
        avg_auc = np.mean([v for k, v in metrics.items() if 'AUC' in k])
        print(f"\n[Epoch {epoch+1}] Avg Loss: {train_loss/len(train_loader):.4f}")
        print(f"AMD AUC: {metrics.get('AMD_AUC', 0):.4f} | DR AUC: {metrics.get('DR_AUC', 0):.4f} | GL AUC: {metrics.get('Glaucoma_AUC', 0):.4f}")

        if avg_auc > best_avg_auc:
            best_avg_auc = avg_auc
            torch.save(model.state_dict(), PATH)
            print(f"New Best Model Saved! (Avg AUC: {best_avg_auc:.4f})")


if __name__ == "__main__":
    train()