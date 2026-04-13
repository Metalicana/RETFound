##With AMD 0,1,2,3
##Separate heads for all

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report

from VisionAgent.linear_probing_oct3 import FairVisionNPZ, get_model_oct

DATA_ROOT = "/lustre/fs1/home/yu395012/RETFound/OphthalmicAgent/data/"
MODEL_WEIGHTS = "weights/oct_model_best.pth" 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

THRESHOLDS = {
    'AMD_L1': 0.5,
    'AMD_L2': 0.5,
    'AMD_L3': 0.5,
    'DR': 0.5,
    'Glaucoma': 0.5
}

def load_trained_model(weights_path):
    print(f"Rebuilding Multi-Head RETFound architecture and loading weights...")
    model = get_model_oct() 
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE, weights_only=False))
    model.to(DEVICE)
    model.eval() 
    return model

def get_amd_stage(row):
    """Compute mutually exclusive AMD stage from L1,L2,L3."""
    if row['target_AMD_L1']==0 and row['target_AMD_L2']==0 and row['target_AMD_L3']==0:
        return 0
    elif row['target_AMD_L1']==1 and row['target_AMD_L2']==0 and row['target_AMD_L3']==0:
        return 1
    elif row['target_AMD_L1']==1 and row['target_AMD_L2']==1 and row['target_AMD_L3']==0:
        return 2
    elif row['target_AMD_L1']==1 and row['target_AMD_L2']==1 and row['target_AMD_L3']==1:
        return 3
    else:
        return -1  # invalid/ambiguous

def get_predicted_amd_stage(row, thresholds):
    """Compute predicted AMD stage from L1,L2,L3 probabilities."""
    L1 = row['amd_p_L1'] >= thresholds['AMD_L1']
    L2 = row['amd_p_L2'] >= thresholds['AMD_L2']
    L3 = row['amd_p_L3'] >= thresholds['AMD_L3']
    
    if not L1 and not L2 and not L3:
        return 0
    elif L1 and not L2 and not L3:
        return 1
    elif L1 and L2 and not L3:
        return 2
    elif L1 and L2 and L3:
        return 3
    else:
        return -1  # ambiguous prediction

def calculate_binary_metrics(y_true, y_pred):
    """Compute precision, recall, F1, accuracy for binary labels."""
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    return p, r, f1, acc

def main():
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_ds = FairVisionNPZ(DATA_ROOT, split='Test', transform=test_transform)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=8)

    model = load_trained_model(MODEL_WEIGHTS)

    results = []
    equity = []
    print(f"Running inference on {len(test_ds)} test cases...")
#    with torch.no_grad():
#        for i, (imgs, labels, _) in enumerate(tqdm(test_loader)):
#            imgs = imgs.to(DEVICE)
#            outputs = model(imgs) 
#            
#            amd_probs = torch.sigmoid(outputs['amd']).cpu().numpy()
#            dr_probs = torch.sigmoid(outputs['dr']).cpu().numpy()
#            gl_probs = torch.sigmoid(outputs['glaucoma']).cpu().numpy()
#            
#            for j in range(len(imgs)):
#                idx = i * imgs.size(0) + j
#                if idx >= len(test_ds.files): break
#                
#                results.append({
#                    'amd_p_L1': amd_probs[j, 0], 'amd_p_L2': amd_probs[j, 1], 'amd_p_L3': amd_probs[j, 2],
#                    'prob_DR': dr_probs[j, 0], 'prob_Glaucoma': gl_probs[j, 0],
#                    'target_AMD_L1': labels[j, 0].item(), 'target_AMD_L2': labels[j, 1].item(),
#                    'target_AMD_L3': labels[j, 2].item(), 'target_DR': labels[j, 3].item(),
#                    'target_Glaucoma': labels[j, 4].item()
#                })
#
#    df = pd.DataFrame(results)
#    
#    # 1. Update the loop to catch 'metas' instead of '_'
    with torch.no_grad():
        for i, (imgs, labels, metas) in enumerate(tqdm(test_loader)):
            imgs = imgs.to(DEVICE)
            outputs = model(imgs) 
            
            # Convert logits to probabilities
            amd_probs = torch.sigmoid(outputs['amd']).cpu().numpy()
            dr_probs = torch.sigmoid(outputs['dr']).cpu().numpy()
            gl_probs = torch.sigmoid(outputs['glaucoma']).cpu().numpy()
            
            for j in range(len(imgs)):
                idx = i * imgs.size(0) + j
                if idx >= len(test_ds.files): break

                if amd_probs[j, 2] >= 0.5:
                  pred_amd = 3
                elif amd_probs[j,1] >= 0.5:
                  pred_amd = 2
                elif amd_probs[j,0] >= 0.5:
                  pred_amd = 1
                else:
                  pred_amd = 0
                                                
                # DR and Glaucoma: Simple binary threshold
                pred_dr = 1 if dr_probs[j, 0] >= 0.5 else 0
                pred_gl = 1 if gl_probs[j, 0] >= 0.5 else 0

                # --- Create the Row Dictionary ---
                row = {
                    # Metadata from our custom dataset/CSV
                    'filename': metas['filename'][j],
                    'disease_folder': metas['disease'][j],
                    'age': metas['age'][j].item() if torch.is_tensor(metas['age']) else metas['age'][j],
                    'gender': metas['gender'][j],
                    'race': metas['race'][j],
                    'ethnicity': metas['ethnicity'][j],
                    'language': metas['language'][j],
                    'maritalstatus': metas['maritalstatus'][j],
                    'groundtruth': metas['groundtruth'][j].item() if torch.is_tensor(metas['groundtruth']) else metas['groundtruth'][j],
                            
                    # Categorical Predictions
                    'Pred_AMD': pred_amd,
                    'Pred_DR': pred_dr,
                    'Pred_GL': pred_gl,
                }
                
                equity.append(row)
                
                results.append({
                    'amd_p_L1': amd_probs[j, 0], 'amd_p_L2': amd_probs[j, 1], 'amd_p_L3': amd_probs[j, 2],
                    'prob_DR': dr_probs[j, 0], 'prob_Glaucoma': gl_probs[j, 0],
                    'target_AMD_L1': labels[j, 0].item(), 'target_AMD_L2': labels[j, 1].item(),
                    'target_AMD_L3': labels[j, 2].item(), 'target_DR': labels[j, 3].item(),
                    'target_Glaucoma': labels[j, 4].item()
                })

    # 2. Convert to DataFrame and Save
    df = pd.DataFrame(results)
    df_equity = pd.DataFrame(equity)
    
    df_equity.to_csv('equity_retfound.csv', index=False)
    
    print("\n" + "="*80)
    print(f"{'Task':<16} | {'Prec':<7} | {'Rec':<7} | {'F1':<6} | {'Count':<6} | {'Correct':<7}")
    print("-" * 80)
    
    for disease in ['Glaucoma', 'AMD', 'DR']:
    
        if disease == 'AMD':
        # Compute AMD stages
          df['AMD_stage'] = df.apply(get_amd_stage, axis=1)
          df['AMD_pred_stage'] = df.apply(get_predicted_amd_stage, axis=1, thresholds=THRESHOLDS)
      
          # --- AMD metrics using classification_report (per stage) ---
          amd_mask = df['AMD_stage'] != -1
          if amd_mask.any():
              y_true = df.loc[amd_mask, 'AMD_stage']
              y_pred = df.loc[amd_mask, 'AMD_pred_stage']
      
              # Per stage metrics
              report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
              for stage_key in sorted(report.keys()):
                  if stage_key.replace('.', '').isdigit():  # stages 0,1,2,3
                      metrics = report[stage_key]
                      stage_label = f"AMD -> Stage {stage_key}"
                      stage_mask = y_true == int(stage_key)
                      stage_count = stage_mask.sum()
                      stage_correct = (y_pred[stage_mask] == y_true[stage_mask]).sum()
                      print(f"{stage_label:<16} | {metrics['precision']:.4f}  | {metrics['recall']:.4f}  | {metrics['f1-score']:.4f} | {stage_count:<6} | {stage_correct:<7}")
        else:  
          target_col, prob_col = f'target_{disease}', f'prob_{disease}'
          mask = df[target_col] != -1
          if mask.any():
              y_true = df.loc[mask, target_col]
              y_pred = (df.loc[mask, prob_col] >= THRESHOLDS[disease]).astype(int)
              
              total_count = len(y_true)
              correct_count = (y_true == y_pred).sum()
              
              p, r, f1, acc = calculate_binary_metrics(y_true, y_pred)
#              print(f"{disease:<16} | {p:.4f}  | {r:.4f}  | {f1:.4f}  | {acc:.4f}")      
              print(f"{disease:<16} | {p:.4f}  | {r:.4f}  | {f1:.4f} | {total_count:<6} | {correct_count:<7}")       
    
    print("="*80)

if __name__ == "__main__":
    main()