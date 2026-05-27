#With AMD 0,1,2,3
#Separate heads for all
#changed normalization 

import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, mean_absolute_error

try:
    from VisionAgent.models_vit import RETFound_mae
    from VisionAgent.fairvision_npz import FairVisionNPZ
except ImportError as exc:
    raise ImportError("Error loading RETFound model or FairVision dataset helpers.") from exc

EQUI_AGENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EQUI_AGENT_ROOT.parent


def _resolve_fairvision_root():
    configured = os.environ.get("FAIRVISION_DATA_ROOT")
    candidates = []
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend(
        [
            REPO_ROOT / "Datasets" / "FairVision",
            EQUI_AGENT_ROOT / "data",
            REPO_ROOT / "Datasets" / "FairVision" / "HarvardFairVision30k",
        ]
    )

    for candidate in candidates:
        candidate = candidate.expanduser()
        has_diseases = all((candidate / disease).exists() for disease in ("AMD", "DR", "Glaucoma"))
        has_flat_splits = all((candidate / split).exists() for split in ("Training", "Validation", "Test"))
        has_hf_metadata = all(
            (candidate / "HarvardFairVision30k" / disease / "ReadMe").exists()
            for disease in ("AMD", "DR", "Glaucoma")
        )
        parent_has_flat_splits = all((candidate.parent / split).exists() for split in ("Training", "Validation", "Test"))
        has_legacy_splits = all(
            any((candidate / disease / split).exists() for split in ("Training", "Validation", "Test"))
            for disease in ("AMD", "DR", "Glaucoma")
        )
        if (has_flat_splits and has_hf_metadata) or (has_diseases and (parent_has_flat_splits or has_legacy_splits)):
            return str(candidate)

    return str(candidates[0])


DATA_ROOT = _resolve_fairvision_root()
BATCH_SIZE = int(os.environ.get("RETFOUND_BATCH_SIZE", 64))
LR = float(os.environ.get("RETFOUND_LR", 1e-3))
EPOCHS = int(os.environ.get("RETFOUND_EPOCHS", 60))
NUM_WORKERS = int(os.environ.get("RETFOUND_NUM_WORKERS", 16))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATH = os.environ.get(
    "RETFOUND_OCT_MODEL_WEIGHTS",
    str(EQUI_AGENT_ROOT / "weights" / "oct_model_best.pth"),
)


def _env_flag(name, default=True):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def print_dataset_diagnostics(name, dataset):
    print(f"{name} label summary: {dataset.label_summary()}")
    sample_count = int(os.environ.get("RETFOUND_AUDIT_SAMPLES", 2))
    if sample_count <= 0:
        return

    for source in dataset.sources:
        seen = 0
        for item in dataset.files:
            if item["source"] != source:
                continue
            with np.load(item["path"]) as data:
                label, metadata = dataset._build_label_and_metadata(item, data)
                raw = {
                    key: dataset._scalar_to_string(data[key])
                    for key in ("amd_condition", "dr_subtype", "glaucoma")
                    if key in data.files
                }
                shape = tuple(data["oct_bscans"].shape) if "oct_bscans" in data.files else None
            print(
                f"{name} audit {source}: file={metadata['filename']} "
                f"oct_shape={shape} raw={raw} label={label.tolist()} groundtruth={metadata['groundtruth']}"
            )
            seen += 1
            if seen >= sample_count:
                break

#class FairVisionNPZ(Dataset):
#    def __init__(self, root_dir, split='Training', transform=None):
#        self.files = []
#        self.transform = transform
#        self.sources = ['AMD', 'DR', 'Glaucoma']
#                
#        self.amd_map = {
#            'not.in.icd.table': 0., 'no.amd.diagnosis': 0.,
#            'early.dry': 1., 'intermediate.dry': 2., 
#            'advanced.atrophic.dry.with.subfoveal.involvement': 3.,
#            'advanced.atrophic.dry.without.subfoveal.involvement': 3.,
#            'wet.amd.active.choroidal.neovascularization': 3.,
#            'wet.amd.inactive.choroidal.neovascularization': 3.,
#            'wet.amd.inactive.scar': 3.
#        }
#        
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
#            all_files_found = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.npz')]
#            all_files_found.sort()
#            
##            taking first 100 only for each disease
#            files_found = all_files_found[:100]
#            
##            files_found = all_files_found
#
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
#            data = np.load(item['path'])
#            oct_volume = data['oct_bscans']
#            oct_slice = oct_volume[oct_volume.shape[0] // 2]
#            
#            if oct_slice.max() <= 1.0: oct_slice = (oct_slice * 255).astype(np.uint8)
#            else: oct_slice = oct_slice.astype(np.uint8)
##            min_val = oct_slice.min()
##            max_val = oct_slice.max()           
##            if max_val - min_val > 0:
##              oct_slice = 255 * (oct_slice - min_val) / (max_val - min_val)      
##            oct_slice = oct_slice.astype(np.uint8)    
##            
#            image = Image.fromarray(oct_slice).convert('RGB')
#            if self.transform: image = self.transform(image)
#            
#            label = torch.full((5,), -1.0) 
#            source = item['source']
#            
#            if source == 'AMD':
#                cond = str(data['amd_condition'])
#                severity = int(self.amd_map.get(cond, 0.))
#                # AMD uses indices 0, 1, and 2
#                label[0] = 1.0 if severity >= 1 else 0.0
#                label[1] = 1.0 if severity >= 2 else 0.0
#                label[2] = 1.0 if severity >= 3 else 0.0
#                
#            elif source == 'DR':
#                cond = str(data['dr_subtype'])
#                severity = int(self.dr_map.get(cond, 0.))
#                label[3] = 1.0 if severity  >= 1.0 else 0.0
#                
#            elif source == 'Glaucoma':
#                severity = int(data['glaucoma'])
#                label[4] = 1.0 if severity == 1 else 0.0
#                
#            return image, label, -1
#        except Exception:
#            return self.__getitem__(idx - 1 if idx > 0 else 0)

class _LegacyFairVisionNPZ(Dataset):
    def __init__(self, root_dir, split='Training', transform=None):
        self.files = []
        self.transform = transform
        self.sources = ['AMD', 'DR', 'Glaucoma']
        self.root_dir = Path(root_dir).expanduser()
        self.split_folder = self._normalize_split(split)
        self.csv_base_path = self._metadata_root()
        self.image_base_path = self._image_root()

        self.amd_map = {
            'not.in.icd.table': 0., 'no.amd.diagnosis': 0.,
            'normal': 0., 'no amd': 0.,
            'early.dry': 1., 'intermediate.dry': 2., 
            'early amd': 1., 'intermediate amd': 2., 'late amd': 3.,
            'advanced.atrophic.dry.with.subfoveal.involvement': 3.,
            'advanced.atrophic.dry.without.subfoveal.involvement': 3.,
            'wet.amd.active.choroidal.neovascularization': 3.,
            'wet.amd.inactive.choroidal.neovascularization': 3.,
            'wet.amd.inactive.scar': 3.
        }
        
        self.dr_map = {
            'not.in.icd.table': 0., 'no.dr.diagnosis': 0.,
            'no dr': 0., 'non-vision threatening dr': 0.,
            'mild.npdr': 0., 'moderate.npdr': 0.,
            'severe.npdr': 1., 'pdr': 1., 'vision threatening dr': 1.
        }

        self.metadata_lookup = self._load_all_metadata()

        print(f"Scanning {self.split_folder} data in {self.root_dir}...")
        for source in self.sources:
            records = self._metadata_records_for_split(source)
            if records:
                for file_meta in records:
                    fname = str(file_meta["filename"])
                    f_path = self._find_npz_path(source, self.split_folder, fname)
                    if f_path is None:
                        print(f"Warning: NPZ not found for {source}/{self.split_folder}/{fname}")
                        continue
                    self.files.append({
                        'path': str(f_path),
                        'source': source,
                        'meta': file_meta,
                    })
                continue

            legacy_dir = self.root_dir / source / self.split_folder
            if not legacy_dir.exists():
                print(f"Warning: No metadata or legacy directory found for {source}: {legacy_dir}")
                continue

            files_found = sorted(path for path in legacy_dir.iterdir() if path.suffix == ".npz")
            for f_path in files_found:
                fname = f_path.name
                file_meta = self.metadata_lookup.get((source, fname), self.metadata_lookup.get(fname, {}))
                self.files.append({
                    'path': str(f_path),
                    'source': source,
                    'meta': file_meta,
                })

        print(f"Found {len(self.files)} images with metadata for {self.split_folder}.")

    @staticmethod
    def _normalize_split(split):
        key = str(split).strip().lower()
        return {
            'train': 'Training',
            'training': 'Training',
            'val': 'Validation',
            'valid': 'Validation',
            'validation': 'Validation',
            'test': 'Test',
        }.get(key, str(split))

    @staticmethod
    def _scalar_to_string(value):
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="ignore")
        return str(value).strip().lower()

    def _metadata_root(self):
        if (self.root_dir / "HarvardFairVision30k").exists():
            return self.root_dir / "HarvardFairVision30k"
        return self.root_dir

    def _image_root(self):
        if all((self.root_dir / split).exists() for split in ("Training", "Validation", "Test")):
            return self.root_dir
        if all((self.root_dir.parent / split).exists() for split in ("Training", "Validation", "Test")):
            return self.root_dir.parent
        return self.root_dir

    def _metadata_csv_path(self, source):
        task = source.lower()
        candidates = [
            self.csv_base_path / source / "ReadMe" / f"data_summary_{task}.csv",
            self.csv_base_path / source / f"data_summary_{task}.csv",
            self.root_dir / source / "ReadMe" / f"data_summary_{task}.csv",
            self.root_dir / source / f"data_summary_{task}.csv",
        ]
        return next((path for path in candidates if path.exists()), None)

    def _metadata_records_for_split(self, source):
        csv_path = self._metadata_csv_path(source)
        if csv_path is None:
            return []
        df = pd.read_csv(csv_path)
        if "use" in df.columns:
            split_mask = df["use"].map(self._normalize_split) == self.split_folder
            df = df[split_mask].copy()
        return df.to_dict('records')

    def _find_npz_path(self, source, split_folder, filename):
        candidates = [
            self.image_base_path / split_folder / filename,
            self.image_base_path / source / split_folder / filename,
            self.root_dir / split_folder / filename,
            self.root_dir / source / split_folder / filename,
            self.csv_base_path / source / split_folder / filename,
        ]
        return next((path for path in candidates if path.exists()), None)

    def _load_all_metadata(self):
        """Pre-loads CSVs and converts them to a fast-access dictionary."""
        combined_meta = {}
        for source in self.sources:
            csv_path = self._metadata_csv_path(source)

            if csv_path is not None:
                df = pd.read_csv(csv_path)
                records = df.to_dict('records')
                for rec in records:
                    combined_meta[(source, rec['filename'])] = rec
                    combined_meta.setdefault(rec['filename'], rec)
            else:
                print(f"Warning: Metadata CSV not found for {source} under {self.root_dir}")
        return combined_meta

    def _build_label_and_metadata(self, item, data):
        label = torch.full((5,), -1.0)
        source = item['source']
        csv_meta = item['meta']

        if source == 'AMD':
            raw_value = data['amd_condition'] if 'amd_condition' in data.files else csv_meta.get('amd', '')
            severity = int(self.amd_map.get(self._scalar_to_string(raw_value), 0.))
            label[0] = 1.0 if severity >= 1 else 0.0
            label[1] = 1.0 if severity >= 2 else 0.0
            label[2] = 1.0 if severity >= 3 else 0.0
        elif source == 'DR':
            raw_value = data['dr_subtype'] if 'dr_subtype' in data.files else csv_meta.get('dr', '')
            severity = int(self.dr_map.get(self._scalar_to_string(raw_value), 0.))
            label[3] = 1.0 if severity >= 1.0 else 0.0
        elif source == 'Glaucoma':
            raw_value = data['glaucoma'] if 'glaucoma' in data.files else csv_meta.get('glaucoma', 0)
            key = self._scalar_to_string(raw_value)
            severity = 1 if key in {'1', '1.0', 'true', 'yes', 'y'} else 0
            label[4] = 1.0 if severity == 1 else 0.0
        else:
            severity = -1

        metadata = {
            'disease': source,
            'age': csv_meta.get('age', 'unknown'),
            'gender': csv_meta.get('gender', 'unknown'),
            'race': csv_meta.get('race', 'unknown'),
            'ethnicity': csv_meta.get('ethnicity', 'unknown'),
            'language': csv_meta.get('language', 'unknown'),
            'maritalstatus': csv_meta.get('maritalstatus', 'unknown'),
            'filename': os.path.basename(item['path']),
            'groundtruth': severity,
        }
        return label, metadata

    def __getitem__(self, idx):
        item = self.files[idx]
        try:
            data = np.load(item['path'])
            
            # --- Image Processing ---
            oct_volume = data['oct_bscans']
            oct_slice = oct_volume[oct_volume.shape[0] // 2]
            if oct_slice.max() <= 1.0: oct_slice = (oct_slice * 255).astype(np.uint8)
            else: oct_slice = oct_slice.astype(np.uint8)
            image = Image.fromarray(oct_slice).convert('RGB')
            if self.transform: image = self.transform(image)
            label, metadata = self._build_label_and_metadata(item, data)
            
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
    
    weight_path = os.environ.get(
        "RETFOUND_OCT_BACKBONE_WEIGHTS",
        str(EQUI_AGENT_ROOT / "weights" / "RETFound_mae_natureOCT.pth"),
    )
        
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
    print(f"Using FairVision data root: {DATA_ROOT}")
    print(f"Saving RETFound OCT weights to: {PATH}")
    print(f"Resume existing RETFound OCT checkpoint: {resume}")
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
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(
            f"FairVision splits are empty: train={len(train_ds)} val={len(val_ds)}. "
            f"Check FAIRVISION_DATA_ROOT={DATA_ROOT!r}."
        )
    print_dataset_diagnostics("train", train_ds)
    print_dataset_diagnostics("val", val_ds)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=max(1, NUM_WORKERS // 2),
        pin_memory=DEVICE.type == "cuda",
    )
    
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
            amd_aucs = [
                roc_auc_score(all_targets[amd_mask, i], all_amd_preds[amd_mask, i])
                for i in range(3)
                if len(np.unique(all_targets[amd_mask, i])) > 1
            ]
            if amd_aucs:
                metrics['AMD_AUC_L1'] = amd_aucs[0]
                metrics['AMD_Avg_AUC'] = np.mean(amd_aucs)

        # 2. DR AUC
        dr_mask = all_targets[:, 3] != -1
        if dr_mask.any() and len(np.unique(all_targets[dr_mask, 3])) > 1:
            metrics['DR_AUC'] = roc_auc_score(all_targets[dr_mask, 3], all_dr_preds[dr_mask, 0])

        # 3. Glaucoma AUC
        gl_mask = all_targets[:, 4] != -1
        if gl_mask.any() and len(np.unique(all_targets[gl_mask, 4])) > 1:
            metrics['Glaucoma_AUC'] = roc_auc_score(all_targets[gl_mask, 4], all_gl_preds[gl_mask, 0])

        # --- LOGGING & SAVING ---
        auc_values = [v for k, v in metrics.items() if 'AUC' in k]
        avg_auc = np.mean(auc_values) if auc_values else 0.0
        print(f"\n[Epoch {epoch+1}] Avg Loss: {train_loss/len(train_loader):.4f}")
        print(f"AMD Avg AUC: {metrics.get('AMD_Avg_AUC', 0):.4f} | DR AUC: {metrics.get('DR_AUC', 0):.4f} | GL AUC: {metrics.get('Glaucoma_AUC', 0):.4f}")

        if avg_auc > best_avg_auc:
            best_avg_auc = avg_auc
            os.makedirs(os.path.dirname(PATH) or ".", exist_ok=True)
            torch.save(model.state_dict(), PATH)
            print(f"New Best Model Saved! (Avg AUC: {best_avg_auc:.4f})")
if __name__ == "__main__":
    train(resume=_env_flag("RETFOUND_RESUME", True))
