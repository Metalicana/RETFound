import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# Ensure matching architecture wrappers are accessible
from VisionAgent.linear_probing_oct3 import get_model_oct
# Adjust the import path below depending on where your MIRAGE linear probing model is saved
from linear_probing_slo import get_model_slo  

# --- CONFIGURATION ---
EXCEL_PATH = "master_ophthalmic_data.xlsx"  # Path to your master Excel file
PATH_OCT_WEIGHTS = "oct_model_best.pth"     # Path to your fine-tuned binary OCT weights
PATH_SLO_WEIGHTS = "fundus_model_best.pth"  # Path to your fine-tuned binary SLO weights
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RawModelEvaluator:
    def __init__(self, path_oct, path_slo, device):
        self.device = device
        
        # 1. Initialize and load the OCT model (RETFound)
        print("Loading fine-tuned RETFound OCT model...")
        self.model_oct = get_model_oct()
        self.model_oct.load_state_dict(torch.load(path_oct, map_location=device, weights_only=True))
        self.model_oct.to(device)
        self.model_oct.eval()
        
        # 2. Initialize and load the SLO model (MIRAGE)
        print("Loading fine-tuned MIRAGE Fundus model...")
        self.model_slo = get_model_slo()
        self.model_slo.load_state_dict(torch.load(path_slo, map_location=device, weights_only=True))
        self.model_slo.to(device)
        self.model_slo.eval()

        # 3. Standardize Image Transforms to match your training code
        self.transform_oct = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.transform_slo = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])

    def preprocess_image(self, img_array, modality):
        if modality == 'oct':
            # Handle center slice extraction logic exactly like the dataloader
            mid_idx = img_array.shape[0] // 2
            oct_slice = img_array[mid_idx]
            if oct_slice.max() <= 1.0:
                oct_slice = (oct_slice * 255).astype(np.uint8)
            else:
                oct_slice = oct_slice.astype(np.uint8)
            image = Image.fromarray(oct_slice).convert('RGB')
            return self.transform_oct(image).unsqueeze(0).to(self.device)
            
        else: # SLO / Fundus
            fundus_img = img_array.astype(np.float32)
            f_min, f_max = fundus_img.min(), fundus_img.max()
            if f_max - f_min > 0:
                fundus_img = 255 * (fundus_img - f_min) / (f_max - f_min)             
            fundus_img = fundus_img.astype(np.uint8)  
            image = Image.fromarray(fundus_img).convert('L')
            return self.transform_slo(image).unsqueeze(0).to(self.device)

    def evaluate_excel(self, df):
        # Master evaluation storage arrays
        eval_data = {
            'oct': {'amd': [], 'dr': [], 'glaucoma': [], 'targets': []},
            'slo': {'amd': [], 'dr': [], 'glaucoma': [], 'targets': []}
        }
        
        print(f"\nStarting raw backbone testing across {len(df)} file records...")
        
        for index, row in df.iterrows():
            # Dynamically extract necessary column lookups
            row_dict = {k.lower(): v for k, v in row.to_dict().items()}
            npz_path = row_dict.get('filepath') or row_dict.get('file_path') or row_dict.get('path')
            disease = str(row_dict.get('task_folder') or row_dict.get('disease')).strip().lower()
            raw_gt = float(row_dict.get('ground_truth') or row_dict.get('groundtruth'))
            
            # Map the clean target profile to binary values [0 or 1]
            if disease == 'amd':
                gt_binary = 1.0 if raw_gt > 0 else 0.0
                task_idx = 0
            elif disease == 'dr':
                gt_binary = float(raw_gt)
                task_idx = 1
            else: # Glaucoma
                gt_binary = float(raw_gt)
                task_idx = 2

            if not os.path.exists(npz_path):
                print(f"Skipping row {index}: file not found at {npz_path}")
                continue

            try:
                # Load images
                container = np.load(npz_path)
                
                # --- OCT Processing (RETFound) ---
                oct_tensor = self.preprocess_image(container['oct_bscans'], 'oct')
                with torch.no_grad():
                    oct_outputs = self.model_oct(oct_tensor)
                    # Pick the appropriate branch logic output depending on current task array context
                    if disease == 'amd': out_logits = oct_outputs['amd']
                    elif disease == 'dr': out_logits = oct_outputs['dr']
                    else: out_logits = oct_outputs['glaucoma']
                    oct_prob = torch.sigmoid(out_logits).item()
                
                # --- SLO Processing (MIRAGE) ---
                slo_tensor = self.preprocess_image(container['slo_fundus'], 'slo')
                with torch.no_grad():
                    slo_outputs = self.model_slo(slo_tensor)
                    if disease == 'amd': out_logits = slo_outputs['amd']
                    elif disease == 'dr': out_logits = slo_outputs['dr']
                    else: out_logits = slo_outputs['glaucoma']
                    slo_prob = torch.sigmoid(out_logits).item()

                # Store evaluations grouped by condition target indices
                eval_data['oct'][disease].append(oct_prob)
                eval_data['oct']['targets'].append((disease, gt_binary))
                
                eval_data['slo'][disease].append(slo_prob)
                eval_data['slo']['targets'].append((disease, gt_binary))

            except Exception as e:
                print(f"Error parsing row {index}: {e}")
                continue

        # Print final metric breakdown tables
        self.print_performance_report(eval_data)

    def print_performance_report(self, eval_data):
        for modality in ['oct', 'slo']:
            model_name = "RETFound (OCT Backbone)" if modality == 'oct' else "MIRAGE (SLO Backbone)"
            print(f"\n\n{'='*20} {model_name} RAW PERFORMANCE {'='*20}")
            print(f"{'Condition':<12} | {'Precision':<9} | {'Recall':<8} | {'F1-Score':<8} | {'ROC-AUC':<8} | {'Support':<8}")
            print("-" * 65)
            
            for disease in ['amd', 'dr', 'glaucoma']:
                # Filter down targets specifically belonging to current disease track
                tgt_list = [tgt[1] for tgt in eval_data[modality]['targets'] if tgt[0] == disease]
                pred_probs = eval_data[modality][disease]
                
                if len(tgt_list) == 0 or len(pred_probs) == 0:
                    print(f"{disease.upper():<12} | No evaluation data collected.")
                    continue
                
                y_true = np.array(tgt_list)
                y_prob = np.array(pred_probs)
                y_pred = (y_prob > 0.5).astype(int)
                
                # Compute classic baseline ML validation values
                prec = precision_score(y_true, y_pred, zero_division=0)
                rec = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                
                # Guard AUC computation from crashes if specific slices only contain 1 class
                auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
                
                print(f"{disease.upper():<12} | {prec:.4f}    | {rec:.4f}   | {f1:.4f}   | {auc:.4f}   | {len(y_true):<8}")
            print("="*65)

if __name__ == "__main__":
    # Load the Master tracking spreadsheet file paths
    if EXCEL_PATH.endswith('.csv'):
        master_df = pd.read_csv(EXCEL_PATH)
    else:
        master_df = pd.read_excel(EXCEL_PATH)
        
    # Group file paths dynamically to automatically load 250 samples per condition
    task_col = [col for col in master_df.columns if col.lower() in ['task_folder', 'disease_folder', 'disease']][0]
    evaluation_subset = master_df.groupby(task_col).head(250).copy()
    
    # Run absolute model testing
    evaluator = RawModelEvaluator(PATH_OCT_WEIGHTS, PATH_SLO_WEIGHTS, DEVICE)
    evaluator.evaluate_excel(evaluation_subset)