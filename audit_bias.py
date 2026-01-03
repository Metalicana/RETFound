import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import models_vit # Ensure this is accessible
from phase1 import FairVisionNPZ, NUM_CLASSES, DEVICE, get_model # Re-use your setup

# CONFIG
DATA_ROOT = "./FairVision" # Update if needed
MODEL_PATH = "best_fair_eye_model.pth"
BATCH_SIZE = 64

def audit_model():
    print(f"Loading best model from {MODEL_PATH}...")
    model = get_model() # This uses your robust loading logic
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # Use 'Test' split for the Audit
    test_ds = FairVisionNPZ(DATA_ROOT, split='Test', transform=None) 
    # Note: Transform needs to be re-defined here if not importing `val_transform`
    # Let's quickly redefine standard transform for safety
    from torchvision import transforms
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_ds.transform = val_transform

    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

    print("Running Inference for Equity Audit...")
    all_preds = []
    all_labels = []
    all_races = []

    with torch.no_grad():
        for imgs, labels, races in tqdm(loader):
            imgs = imgs.to(DEVICE)
            with torch.amp.autocast('cuda'): # Updated for new PyTorch
                 outputs = model(imgs)
                 probs = torch.sigmoid(outputs)
            
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
            all_races.append(races.numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    all_races = np.concatenate(all_races)

    # --- CALCULATE BIAS ---
    # Race Map: 0=Asian, 1=Black, 2=White (Check your README/NPZ key to confirm)
    race_map = {0: 'Asian', 1: 'Black', 2: 'White'} 
    diseases = ['AMD', 'DR', 'Glaucoma']
    
    results = []

    for race_code, race_name in race_map.items():
        indices = (all_races == race_code)
        if np.sum(indices) == 0: continue
        
        race_preds = all_preds[indices]
        race_labels = all_labels[indices]
        
        for i, disease in enumerate(diseases):
            # Calculate AUC for this Race + Disease
            try:
                auc = roc_auc_score(race_labels[:, i], race_preds[:, i])
                
                # Calculate False Negative Rate (at 0.5 threshold)
                binary_preds = (race_preds[:, i] > 0.5).astype(int)
                binary_labels = race_labels[:, i].astype(int)
                
                # FNR = FN / (FN + TP)
                fn = np.sum((binary_preds == 0) & (binary_labels == 1))
                tp = np.sum((binary_preds == 1) & (binary_labels == 1))
                fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
                
                results.append({
                    'Race': race_name,
                    'Disease': disease,
                    'AUC': auc,
                    'FNR': fnr,
                    'Count': np.sum(indices)
                })
            except:
                pass

    df = pd.DataFrame(results)
    print("\n=== EQUITY AUDIT TABLE ===")
    print(df.sort_values(['Disease', 'Race']))
    
    # Save for the Agent to use
    df.to_csv("equity_audit_table.csv", index=False)
    print("Saved bias stats to 'equity_audit_table.csv'")

if __name__ == "__main__":
    audit_model()