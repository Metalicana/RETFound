#With AMD 0,1,2,3

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, mean_absolute_error

# Import your existing classes from your training file
from VisionAgent.linear_probing_oct2 import FairVisionNPZ, get_model 

DATA_ROOT = "/lustre/fs1/home/yu395012/RETFound/OphthalmicAgent/data/"
MODEL_WEIGHTS = "oct_model_best2.pth"
NUM_CLASSES = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_trained_model(weights_path):
    print(f"Rebuilding model architecture and loading weights from {weights_path}...")
    model = get_model() # This should have global_pool=True inside it now
    
    # Load the state_dict saved during training
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE, weights_only=False))
    model.to(DEVICE)
    model.eval() 
    return model

def main():
    # 1. Setup Data
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_ds = FairVisionNPZ(DATA_ROOT, split='Test', transform=test_transform)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=8)

    # 2. Load Model
    model = load_trained_model(MODEL_WEIGHTS)

    # 3. Run Inference
    results = []
    print(f"Running OCT Specialist Inference on {len(test_ds)} cases...")

    with torch.no_grad():
        for i, (imgs, labels, races) in enumerate(tqdm(test_loader)):
            imgs = imgs.to(DEVICE)
            
            # Forward pass
            outputs = model(imgs) # Raw logits/regression values
            
            # SPLIT PROCESSING:
            # Node 0 (AMD): Keep raw for MAE
            amd_preds = outputs[:, 0].cpu().numpy()
            
            # Nodes 1 & 2 (DR/Glaucoma): Sigmoid for AUC
            binary_probs = torch.sigmoid(outputs[:, 1:]).cpu().numpy()
            
            for j in range(len(imgs)):
                file_info = test_ds.files[i * 32 + j] 
                
                results.append({
                    'file_path': file_info['path'],
                    'source_folder': file_info['source'],
                    'pred_AMD': amd_preds[j],           # Numerical severity
                    'prob_DR': binary_probs[j, 0],      # Probability 0-1
                    'prob_Glaucoma': binary_probs[j, 1], # Probability 0-1
                    'target_AMD': labels[j, 0].item(),
                    'target_DR': labels[j, 1].item(),
                    'target_Glaucoma': labels[j, 2].item()
                })

    # 4. Save to CSV
    df = pd.DataFrame(results)
    df.to_csv("oct_specialist_inference.csv", index=False)
    print("\nInference Complete! Results saved to 'oct_specialist_inference2.csv'")

    # 5. METRICS CALCULATION
    print("\n" + "="*30)
    print("TEST SET PERFORMANCE")
    print("="*30)

    # --- AMD: Mean Absolute Error (MAE) ---
    y_true_amd = df['target_AMD'].values
    y_pred_amd = df['pred_AMD'].values
    mask_amd = y_true_amd != -1
    
    if mask_amd.any():
        mae = mean_absolute_error(y_true_amd[mask_amd], y_pred_amd[mask_amd])
        print(f"AMD MAE:      {mae:.4f} (Avg grade error)")
    
    # --- DR & Glaucoma: AUC ---
    for idx, disease in enumerate(['DR', 'Glaucoma']):
        y_true = df[f'target_{disease}'].values
        y_score = df[f'prob_{disease}'].values
        
        mask = y_true != -1
        if len(np.unique(y_true[mask])) > 1:
            auc = roc_auc_score(y_true[mask], y_score[mask])
            print(f"{disease:13} AUC: {auc:.4f}")
    print("="*30)

if __name__ == "__main__":
    main()