
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

DATA_ROOT = "/lustre/fs1/home/yu395012/RETFound/OphthalmicAgent/data/"
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATH = "slo_model_retfound.pth"

class FairVisionNPZ(Dataset):
    def __init__(self, root_dir, split='Training', transform=None):
        self.files = []
        self.transform = transform
        self.sources = ['AMD', 'DR', 'Glaucoma']
        self.csv_base_path = root_dir
        
        self.metadata_lookup = self._load_all_metadata()

        # AMD binarization lookup values
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
            slo = data['slo_fundus']
            
            if slo.max() <= 1.0:
                slo = (slo * 255).astype(np.uint8)
            else:
                slo = slo.astype(np.uint8)
            
            image = Image.fromarray(slo).convert("RGB")
            
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
        
        # Define 3 separate specialist binary heads
        # AMD changed from 3 nodes down to 1 node for simple binary tracking
        self.amd_head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 1)
        )
        # DR Head (1 binary node)
        self.dr_head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 1)
        )
        # Glaucoma Head (1 binary node)
        self.glaucoma_head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 1)
        )

        # Initialize weights for the new MLP heads
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

def get_model_oct():
    weight_path = "/lustre/fs1/home/yu395012/RETFound/OphthalmicAgent/VisionAgent/weights/RETFound_mae_natureCFP.pth"
        
    backbone = RETFound_mae(
        img_size=224,
        num_classes=0, 
        drop_path_rate=0.2,
        global_pool='',
    )
    
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
    
    for param in model.amd_head.parameters(): param.requires_grad = True
    for param in model.dr_head.parameters(): param.requires_grad = True
    for param in model.glaucoma_head.parameters(): param.requires_grad = True
    
    return model

class MultiAgentLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        # Shifted to a 1D single-element positive scale modifier tensor
        self.amd_pos_weight = torch.tensor([1.5]).to(device)
        self.bce_amd = nn.BCEWithLogitsLoss(pos_weight=self.amd_pos_weight)
        
        self.dr_pos_weight = torch.tensor([5.0]).to(device)
        self.bce_dr = nn.BCEWithLogitsLoss(pos_weight=self.dr_pos_weight)
        
        self.bce_standard = nn.BCEWithLogitsLoss()

    def forward(self, outputs, targets):
        # targets positions: [AMD, DR, Glaucoma]
        
        # --- 1. AMD Loss ---
        amd_mask = (targets[:, 0] != -1)
        if amd_mask.any():
            loss_amd = self.bce_amd(outputs['amd'][amd_mask], targets[amd_mask, 0:1])
        else:
            loss_amd = 0.0

        # --- 2. DR Loss ---
        dr_mask = (targets[:, 1] != -1)
        if dr_mask.any():
            loss_dr = self.bce_dr(outputs['dr'][dr_mask], targets[dr_mask, 1:2])
        else:
            loss_dr = 0.0

        # --- 3. Glaucoma Loss ---
        gl_mask = (targets[:, 2] != -1)
        if gl_mask.any():
            loss_gl = self.bce_standard(outputs['glaucoma'][gl_mask], targets[gl_mask, 2:3])
        else:
            loss_gl = 0.0

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
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)
    
    model = get_model_oct().to(DEVICE)
    checkpoint_path = PATH
    
    if resume and os.path.exists(checkpoint_path):
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
            print("Successfully loaded specialist weights. Resuming...")
        except Exception as e:
            print(f"Could not restore previous model parameters (likely due to shape updates). Starting from scratch. Error: {e}")
        
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
        
        # 1. Binary AMD AUC
        amd_mask = all_targets[:, 0] != -1
        if amd_mask.any() and len(np.unique(all_targets[amd_mask, 0])) > 1:
            metrics['AMD_AUC'] = roc_auc_score(all_targets[amd_mask, 0], all_amd_preds[amd_mask, 0])

        # 2. DR AUC
        dr_mask = all_targets[:, 1] != -1
        if dr_mask.any() and len(np.unique(all_targets[dr_mask, 1])) > 1:
            metrics['DR_AUC'] = roc_auc_score(all_targets[dr_mask, 1], all_dr_preds[dr_mask, 0])

        # 3. Glaucoma AUC
        gl_mask = all_targets[:, 2] != -1
        if gl_mask.any() and len(np.unique(all_targets[gl_mask, 2])) > 1:
            metrics['Glaucoma_AUC'] = roc_auc_score(all_targets[gl_mask, 2], all_gl_preds[gl_mask, 0])

        # --- LOGGING & SAVING ---
        avg_auc = np.mean([v for k, v in metrics.items() if 'AUC' in k])
        print(f"\n[Epoch {epoch+1}] Avg Loss: {train_loss/len(train_loader):.4f}")
        print(f"AMD AUC: {metrics.get('AMD_AUC', 0):.4f} | DR AUC: {metrics.get('DR_AUC', 0):.4f} | GL AUC: {metrics.get('Glaucoma_AUC', 0):.4f}")

        if avg_auc > best_avg_auc:
            best_avg_auc = avg_auc
            torch.save(model.state_dict(), PATH)
            print(f"New Best Model Saved! (Avg AUC: {best_avg_auc:.4f})")

if __name__ == "__main__":
    train()