#With AMD 0,1,2,3
#Separate heads for all
#changed normalization 

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, mean_absolute_error

try:
    from VisionAgent.models_vit import RETFound_mae
except ImportError:
    raise ImportError("Error: 'models_vit.py' not found.")

DATA_ROOT = "/lustre/fs1/home/yu395012/RETFound/OphthalmicAgent/data/"
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 60
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATH = "oct_model_check.pth"


class FairVisionNPZ(Dataset):
    def __init__(self, root_dir, split='Training', transform=None):
        self.files = []
        self.transform = transform
        self.sources = ['AMD', 'DR', 'Glaucoma']
                
        self.amd_map = {
            'not.in.icd.table': 0., 'no.amd.diagnosis': 0.,
            'early.dry': 1., 'intermediate.dry': 2., 
            'advanced.atrophic.dry.with.subfoveal.involvement': 3.,
            'advanced.atrophic.dry.without.subfoveal.involvement': 3.,
            'wet.amd.active.choroidal.neovascularization': 3.,
            'wet.amd.inactive.choroidal.neovascularization': 3.,
            'wet.amd.inactive.scar': 3.
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
            all_files_found = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.npz')]
            all_files_found.sort()
            
#            #taking first 100 only for each disease
#            files_found = all_files_found[:100]
            files_found = all_files_found
#            count = 0
            for f_path in files_found:
                self.files.append({'path': f_path, 'source': source})
#                if count <= 2 or count>=97:
#                  print(f"count: {count}, f_path: {f_path}")
#                count+=1
                
        print(f"Found {len(self.files)} images for {split}.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        item = self.files[idx]
        try:
            data = np.load(item['path'])
            oct_volume = data['oct_bscans']
            oct_slice = oct_volume[oct_volume.shape[0] // 2]
            
            if oct_slice.max() <= 1.0: oct_slice = (oct_slice * 255).astype(np.uint8)
            else: oct_slice = oct_slice.astype(np.uint8)

#            min_val = oct_slice.min()
#            max_val = oct_slice.max()           
#            if max_val - min_val > 0:
#              oct_slice = 255 * (oct_slice - min_val) / (max_val - min_val)      
#            oct_slice = oct_slice.astype(np.uint8)    
#            
            image = Image.fromarray(oct_slice).convert('RGB')
            if self.transform: image = self.transform(image)
            
            label = torch.full((5,), -1.0) 
            source = item['source']
            
            if source == 'AMD':
                cond = str(data['amd_condition'])
                severity = int(self.amd_map.get(cond, 0.))
                # AMD uses indices 0, 1, and 2
                label[0] = 1.0 if severity >= 1 else 0.0
                label[1] = 1.0 if severity >= 2 else 0.0
                label[2] = 1.0 if severity >= 3 else 0.0
                
            elif source == 'DR':
                cond = str(data['dr_subtype'])
                label[3] = 1.0 if self.dr_map.get(cond, 0.) >= 1.0 else 0.0
                
            elif source == 'Glaucoma':
                label[4] = 1.0 if int(data['glaucoma']) == 1 else 0.0

            return image, label, -1
        except Exception:
            return self.__getitem__(idx - 1 if idx > 0 else 0)


class RETFoundMultiHead(nn.Module):
    def __init__(self, backbone):
        super(RETFoundMultiHead, self).__init__()
        self.backbone = backbone
        
        # Define 3 separate specialist heads
        # AMD: 3 nodes for cumulative multi-label classification
        self.amd_head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 3)
        )
        # DR: 1 node for binary classification
#        self.dr_head = nn.Linear(1024, 1)
        
        # Instead of: self.dr_head = nn.Linear(1024, 1)
        self.dr_head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 1)
        )
        # Glaucoma: 1 node for binary classification
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
        # Pass through the frozen backbone to get features
        # Note: RETFound (ViT) usually returns a feature vector of 1024 for the [CLS] token
        features = self.backbone(x)
        
        # Return a dictionary of outputs from each specialist
        return {
            'amd': self.amd_head(features),
            'dr': self.dr_head(features),
            'glaucoma': self.glaucoma_head(features)
        }

def get_model_oct():
#    print("Loading RETFound ViT-Large with Separate Specialist Heads...")
    
    weight_path = "/lustre/fs1/home/yu395012/RETFound/OphthalmicAgent/VisionAgent/weights/RETFound_mae_natureOCT.pth"
        
    # 1. Initialize the backbone architecture

    backbone = RETFound_mae(
        img_size=224,
        num_classes=0, # Setting to 0 often removes the default head in timm-based models
        drop_path_rate=0.2,
        global_pool='',
    )
    
    # 2. FREEZE THE ENTIRE BACKBONE
    for param in backbone.parameters():
        param.requires_grad = False
    
    # 3. LOAD PRE-TRAINED WEIGHTS INTO BACKBONE
    checkpoint = torch.load(weight_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint

    # Clean the state_dict
    keys_to_remove = [k for k in state_dict.keys() if 'head' in k or 'decoder' in k or 'fc_norm' in k]
    for k in keys_to_remove:
        if k in state_dict:
            del state_dict[k]
            
    # Load into the backbone
    msg = backbone.load_state_dict(state_dict, strict=False)
#    print(f"Backbone weights loaded. Missing keys (expected): {msg.missing_keys}")
    
    # 4. WRAP IN MULTI-HEAD ARCHITECTURE
    model = RETFoundMultiHead(backbone)
    
    # Ensure only the heads are trainable
    for param in model.amd_head.parameters(): param.requires_grad = True
    for param in model.dr_head.parameters(): param.requires_grad = True
    for param in model.glaucoma_head.parameters(): param.requires_grad = True

#    print(f"Specialists Initialized: AMD (3 nodes), DR (1 node), Glaucoma (1 node)")
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
#    print(f"Total Parameters: {total_params:,}")
#    print(f"Trainable Parameters (All Heads): {trainable_params:,}") 
    # This should be around 1024 * 5 + 5 = 5,125
    
    return model

class MultiAgentLoss(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.amd_pos_weight = torch.tensor([1.0, 1.5, 2.0]).to(device)
        self.bce_amd = nn.BCEWithLogitsLoss(pos_weight=self.amd_pos_weight)
        
        self.dr_pos_weight = torch.tensor([5.0])
        self.bce_dr = nn.BCEWithLogitsLoss(pos_weight=self.dr_pos_weight.to(DEVICE))
        
        self.bce_standard = nn.BCEWithLogitsLoss()

    def forward(self, outputs, targets):
        # targets: [Batch, 5] -> [AMD1, AMD2, AMD3, DR, Glaucoma]
        
        # --- 1. AMD Loss ---
        # Mask only samples belonging to AMD folder
        amd_mask = (targets[:, 0] != -1)
        if amd_mask.any():
            loss_amd = self.bce_amd(outputs['amd'][amd_mask], targets[amd_mask, 0:3])
        else:
            loss_amd = 0.0

        # --- 2. DR Loss ---
        dr_mask = (targets[:, 3] != -1)
        if dr_mask.any():
            # Targets[:, 3] is [Batch], we need [Batch, 1] for BCE
            loss_dr = self.bce_dr(outputs['dr'][dr_mask], targets[dr_mask, 3:4])
        else:
            loss_dr = 0.0

        # --- 3. Glaucoma Loss ---
        gl_mask = (targets[:, 4] != -1)
        if gl_mask.any():
            loss_gl = self.bce_standard(outputs['glaucoma'][gl_mask], targets[gl_mask, 4:5])
        else:
            loss_gl = 0.0

        return loss_amd + loss_dr + loss_gl
        
        
def train(resume = True):
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

    train_ds = FairVisionNPZ(DATA_ROOT, split='Training', transform=train_transform)
    val_ds = FairVisionNPZ(DATA_ROOT, split='Validation', transform=val_transform)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)
    
    model = get_model_oct().to(DEVICE)
    checkpoint_path = PATH
    
    if resume and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        print("Successfully loaded specialist weights. Resuming...")
        
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
            
            # 1. Forward pass returns a DICTIONARY
            outputs = model(imgs) 
            loss = criterion(outputs, labels)
                
            loss.backward()
            optimizer.step()
        
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        # --- VALIDATION ---
        model.eval()
        # Separate lists for each specialist since they have different shapes
        val_data = { 'amd': [], 'dr': [], 'gl': [], 'targets': [] }
        
        with torch.no_grad():
            for imgs, labels, _ in val_loader:
                imgs = imgs.to(DEVICE)
                outputs = model(imgs)
                
                # Apply Sigmoid to convert logits to probabilities [0, 1]
                val_data['amd'].append(torch.sigmoid(outputs['amd']).cpu().numpy())
                val_data['dr'].append(torch.sigmoid(outputs['dr']).cpu().numpy())
                val_data['gl'].append(torch.sigmoid(outputs['glaucoma']).cpu().numpy())
                val_data['targets'].append(labels.cpu().numpy())

        # Concatenate all batches
        all_targets = np.vstack(val_data['targets'])
        all_amd_preds = np.vstack(val_data['amd'])
        all_dr_preds = np.vstack(val_data['dr'])
        all_gl_preds = np.vstack(val_data['gl'])

        metrics = {}
        
        # 1. AMD AUC (Calculate for each cumulative level)
        amd_mask = all_targets[:, 0] != -1
        if amd_mask.any():
            # Calculate AUC for Stage >= 1
            metrics['AMD_AUC_L1'] = roc_auc_score(all_targets[amd_mask, 0], all_amd_preds[amd_mask, 0])
            # Average the three AMD layers for a single AMD score
            metrics['AMD_Avg_AUC'] = np.mean([
                roc_auc_score(all_targets[amd_mask, i], all_amd_preds[amd_mask, i]) for i in range(3)
            ])

        # 2. DR AUC
        dr_mask = all_targets[:, 3] != -1
        if dr_mask.any() and len(np.unique(all_targets[dr_mask, 3])) > 1:
            metrics['DR_AUC'] = roc_auc_score(all_targets[dr_mask, 3], all_dr_preds[dr_mask, 0])

        # 3. Glaucoma AUC
        gl_mask = all_targets[:, 4] != -1
        if gl_mask.any() and len(np.unique(all_targets[gl_mask, 4])) > 1:
            metrics['Glaucoma_AUC'] = roc_auc_score(all_targets[gl_mask, 4], all_gl_preds[gl_mask, 0])

        # --- LOGGING & SAVING ---
        avg_auc = np.mean([v for k, v in metrics.items() if 'AUC' in k])
        print(f"\n[Epoch {epoch+1}] Avg Loss: {train_loss/len(train_loader):.4f}")
        print(f"AMD Avg AUC: {metrics.get('AMD_Avg_AUC', 0):.4f} | DR AUC: {metrics.get('DR_AUC', 0):.4f} | GL AUC: {metrics.get('Glaucoma_AUC', 0):.4f}")

        if avg_auc > best_avg_auc:
            best_avg_auc = avg_auc
            torch.save(model.state_dict(), PATH)
            print(f"New Best Model Saved! (Avg AUC: {best_avg_auc:.4f})")
if __name__ == "__main__":
    train()