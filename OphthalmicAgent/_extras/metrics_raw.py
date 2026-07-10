### OCT MIDDLE SLICE
#
#import os
#import sys
#import torch
#import torch.nn as nn
#import numpy as np
#import pandas as pd
#from PIL import Image
#from torchvision import transforms
#from sklearn.metrics import classification_report
#
## Ensure architecture wrappers are accessible
#from VisionAgent.linear_probing_oct3 import get_model_oct
#
#MIRAGE_DIR = os.path.abspath("./VisionAgent/MIRAGE")
#sys.path.append(MIRAGE_DIR)
#from linear_probing_slo import get_model_slo  
#
## --- CONFIGURATION ---
#EXCEL_PATH = "data/fairvision_250each.csv"  
#PATH_OCT_WEIGHTS = "weights/oct_model_best_all_binary.pth"     
#PATH_SLO_WEIGHTS = "weights/slo_model_best_all_binary.pth"  
#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#class RawModelEvaluator:
#    def __init__(self, path_oct, path_slo, device):
#        self.device = device
#        
#        # 1. Load OCT Model
#        print("Loading fine-tuned RETFound OCT model...")
#        self.model_oct = get_model_oct()
#        self.model_oct.load_state_dict(torch.load(path_oct, map_location=device, weights_only=True))
#        self.model_oct.to(device)
#        self.model_oct.eval()
#        
#        # 2. Load SLO Model
#        print("Loading fine-tuned MIRAGE Fundus model...")
#        original_dir = os.getcwd()
#        os.chdir(MIRAGE_DIR)
#        self.model_slo = get_model_slo()
#        os.chdir(original_dir)
#        self.model_slo.load_state_dict(torch.load(path_slo, map_location=device, weights_only=True))
#        self.model_slo.to(device)
#        self.model_slo.eval()
#
#        # 3. Standardize Image Transforms
#        self.transform_oct = transforms.Compose([
#            transforms.Resize((224, 224)),
#            transforms.ToTensor(),
#            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#        ])
#        
#        self.transform_slo = transforms.Compose([
#            transforms.Resize((512, 512)),
#            transforms.ToTensor()
#        ])
#
#    def preprocess_image(self, img_array, modality):
#        if modality == 'oct':
#        
#            mid_idx = img_array.shape[0] // 2
#            oct_slice = img_array[mid_idx]
#            if oct_slice.max() <= 1.0:
#                oct_slice = (oct_slice * 255).astype(np.uint8)
#            else:
#                oct_slice = oct_slice.astype(np.uint8)
#            image = Image.fromarray(oct_slice).convert('RGB')
#            return self.transform_oct(image).unsqueeze(0).to(self.device)
#            
#        else: 
#            fundus_img = img_array.astype(np.float32)
#            f_min, f_max = fundus_img.min(), fundus_img.max()
#            if f_max - f_min > 0:
#                fundus_img = 255 * (fundus_img - f_min) / (f_max - f_min)             
#            fundus_img = fundus_img.astype(np.uint8)  
#            image = Image.fromarray(fundus_img).convert('L')
#            return self.transform_slo(image).unsqueeze(0).to(self.device)
#
#    def evaluate_excel(self, df):
#        # Master evaluation storage arrays
#        eval_data = {
#            'oct': {'amd': [], 'dr': [], 'glaucoma': [], 'amd_targets': [], 'dr_targets': [], 'glaucoma_targets': []},
#            'slo': {'amd': [], 'dr': [], 'glaucoma': [], 'amd_targets': [], 'dr_targets': [], 'glaucoma_targets': []}
#        }
#        
#        print(f"\nStarting raw backbone testing across {len(df)} file records...")
#        
#        for index, row in df.iterrows():
#            # Direct lookups matching your exact column names
#            npz_path = row['filename']
#            disease = str(row['Task_Folder']).strip().lower()
#            raw_gt = row['Ground_Truth']
#            
#            # Skip genuinely empty or invalid lines
#            if pd.isna(raw_gt) or pd.isna(npz_path) or not os.path.exists(npz_path):
#                continue
#            # 2. Explicitly ignore rows where ground truth is -1
#            if float(raw_gt) == -1.0:
#                continue
#                
#            gt_binary = 1.0 if float(raw_gt) > 0 else 0.0
#
#            try:
#                container = np.load(npz_path)
#                
#                # --- OCT Processing ---
#                oct_tensor = self.preprocess_image(container['oct_bscans'], 'oct')
#
#                with torch.no_grad():
##                    print("mai")
#                    oct_outputs = self.model_oct(oct_tensor)
##                    print("aya")  
#                    if disease == 'amd': out_logits = oct_outputs['amd']
#                    elif disease == 'dr': out_logits = oct_outputs['dr']
#                    else: out_logits = oct_outputs['glaucoma']
#
#                    oct_prob = torch.sigmoid(out_logits).item()
#                
#                # --- SLO Processing ---
#                slo_tensor = self.preprocess_image(container['slo_fundus'], 'slo')
#                with torch.no_grad():
#                    slo_outputs = self.model_slo(slo_tensor)
#                    if disease == 'amd': out_logits = slo_outputs['amd']
#                    elif disease == 'dr': out_logits = slo_outputs['dr']
#                    else: out_logits = slo_outputs['glaucoma']
#                    slo_prob = torch.sigmoid(out_logits).item()
#                     
#                # Store evaluations grouped by condition target keys
#                eval_data['oct'][disease].append(oct_prob)
#                eval_data['oct'][f"{disease}_targets"].append(gt_binary)
#                
#                eval_data['slo'][disease].append(slo_prob)
#                eval_data['slo'][f"{disease}_targets"].append(gt_binary)
#
#            except Exception:
#                continue
#
#        self.print_performance_report(eval_data)
#
#    def print_performance_report(self, eval_data):
#        # Process performance dynamically per disease condition
#        for disease in ['amd', 'dr', 'glaucoma']:
#            print(f"\n\n" + "#"*30 + f" {disease.upper()} CLINICAL EVALUATION COMPARISON " + "#"*30)
#            
#            for modality in ['oct', 'slo']:
#                model_name = "RETFound (OCT Backbone)" if modality == 'oct' else "MIRAGE (SLO Backbone)"
#                
#                y_true = np.array(eval_data[modality][f"{disease}_targets"])
#                y_prob = np.array(eval_data[modality][disease])
#                
#                if len(y_true) == 0 or len(y_prob) == 0:
#                    print(f"\n[{model_name}] No validation rows collected for this target category.")
#                    continue
#                
#                # Default clinical classification rule (50% diagnostic boundary)
#                y_pred = (y_prob > 0.5).astype(int)
#                
#                # Print metrics exactly mapping the structure of the agent reporting systems
#                print(f"\n" + "="*20 + f" {model_name} RAW METRICS " + "="*20)
#                print(classification_report(
#                    y_true, 
#                    y_pred, 
#                    target_names=['Healthy (0)', 'Pathological (1)'], 
#                    digits=4, 
#                    zero_division=0
#                ))
#                print("="*68)
#        print("\n" + "#"*92)
#
#if __name__ == "__main__":
#    master_df = pd.read_csv(EXCEL_PATH)
#    
#    # Run absolute model testing
#    evaluator = RawModelEvaluator(PATH_OCT_WEIGHTS, PATH_SLO_WEIGHTS, DEVICE)
#    evaluator.evaluate_excel(master_df)

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from sklearn.metrics import classification_report

# Ensure architecture wrappers are accessible
from VisionAgent.linear_probing_oct3 import get_model_oct

MIRAGE_DIR = os.path.abspath("./VisionAgent/MIRAGE")
sys.path.append(MIRAGE_DIR)
from linear_probing_slo import get_model_slo  

# --- CONFIGURATION ---
EXCEL_PATH = "data/fairvision_250each.csv"  
PATH_OCT_WEIGHTS = "weights/oct_model_8_slices_not_center.pth"     
PATH_SLO_WEIGHTS = "weights/slo_model_best_all_binary.pth"  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RawModelEvaluator:
    def __init__(self, path_oct, path_slo, device):
        self.device = device

        # ---------------- OCT MODEL ----------------
        print("Loading fine-tuned RETFound OCT model...")
        self.model_oct = get_model_oct()
        self.model_oct.load_state_dict(
            torch.load(path_oct, map_location=device, weights_only=True)
        )
        self.model_oct.to(device)
        self.model_oct.eval()

        # ---------------- SLO MODEL ----------------
        print("Loading fine-tuned MIRAGE Fundus model...")
        original_dir = os.getcwd()
        os.chdir(MIRAGE_DIR)
        self.model_slo = get_model_slo()
        os.chdir(original_dir)

        self.model_slo.load_state_dict(
            torch.load(path_slo, map_location=device, weights_only=True)
        )
        self.model_slo.to(device)
        self.model_slo.eval()

        # ---------------- Transforms ----------------
        self.transform_oct = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.transform_slo = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])

    ####################################################################
    # IMAGE PREPROCESSING
    ####################################################################

    def preprocess_image(self, img_array, modality):

        if modality == "oct":

            num_slices = 8

            indices = np.linspace(
                0,
                img_array.shape[0] - 1,
                num_slices,
                dtype=int
            )

            selected_slices = img_array[indices]

            images = []

            for oct_slice in selected_slices:

                if oct_slice.max() <= 1.0:
                    oct_slice = (oct_slice * 255).astype(np.uint8)
                else:
                    oct_slice = oct_slice.astype(np.uint8)

                img = Image.fromarray(oct_slice).convert("RGB")
                img = self.transform_oct(img)

                images.append(img)

            image = torch.stack(images)

            return image.unsqueeze(0).to(self.device)

        # ---------------- SLO ----------------

        fundus_img = img_array.astype(np.float32)

        f_min = fundus_img.min()
        f_max = fundus_img.max()

        if f_max - f_min > 0:
            fundus_img = 255 * (fundus_img - f_min) / (f_max - f_min)

        fundus_img = fundus_img.astype(np.uint8)

        image = Image.fromarray(fundus_img).convert("L")

        return self.transform_slo(image).unsqueeze(0).to(self.device)

    ####################################################################
    # PRINT CLASSIFICATION REPORT
    ####################################################################

    def print_subset_report(self, records, title):

        if len(records) == 0:
            print(f"\n{title}: No samples.")
            return

        y_true = np.array([r["target"] for r in records])
        y_prob = np.array([r["prob"] for r in records])

        y_pred = (y_prob > 0.5).astype(int)

        print("\n" + "=" * 75)
        print(title)
        print(f"Number of samples: {len(records)}")

        print(
            classification_report(
                y_true,
                y_pred,
                labels=[0,1],
                target_names=[
                    "Healthy (0)",
                    "Pathological (1)"
                ],
                digits=4,
                zero_division=0
            )
        )
    def evaluate_excel(self, df):
        eval_data = {
            "oct": {
                "amd": [],
                "dr": [],
                "glaucoma": []
            },
            "slo": {
                "amd": [],
                "dr": [],
                "glaucoma": []
            }
        }
    
        print(f"\nStarting raw backbone testing across {len(df)} file records...")
    
        for _, row in df.iterrows():
    
            npz_path = row["filename"]
            disease = str(row["Task_Folder"]).strip().lower()
            raw_gt = row["Ground_Truth"]
    
            # Skip invalid rows
            if (
                pd.isna(raw_gt)
                or pd.isna(npz_path)
                or not os.path.exists(npz_path)
            ):
                continue
    
            if float(raw_gt) == -1:
                continue
    
            if pd.isna(row["Age"]):
                continue
    
            gt_binary = 1 if float(raw_gt) > 0 else 0
    
            ############################################################
            # Demographics
            ############################################################
    
            gender = str(row["Gender"]).strip().lower()
            race = str(row["Race"]).strip().lower()
    
            age = float(row["Age"])
    
            if age < 50:
                age_group = "young"
            elif age >= 70:
                age_group = "older"
            else:
                age_group = "middle"
    
            ############################################################
    
            try:
    
                container = np.load(npz_path)
    
                ###########################
                # OCT
                ###########################
    
                oct_tensor = self.preprocess_image(
                    container["oct_bscans"],
                    "oct"
                )
    
                with torch.no_grad():
    
                    outputs = self.model_oct(oct_tensor)
    
                    if disease == "amd":
                        logits = outputs["amd"]
                    elif disease == "dr":
                        logits = outputs["dr"]
                    else:
                        logits = outputs["glaucoma"]
    
                    oct_prob = torch.sigmoid(logits).item()
    
                ###########################
                # SLO
                ###########################
    
                slo_tensor = self.preprocess_image(
                    container["slo_fundus"],
                    "slo"
                )
    
                with torch.no_grad():
    
                    outputs = self.model_slo(slo_tensor)
    
                    if disease == "amd":
                        logits = outputs["amd"]
                    elif disease == "dr":
                        logits = outputs["dr"]
                    else:
                        logits = outputs["glaucoma"]
    
                    slo_prob = torch.sigmoid(logits).item()
    
                ##########################################################
                # Store ONE record
                ##########################################################
    
                eval_data["oct"][disease].append({
                    "prob": oct_prob,
                    "target": gt_binary,
                    "gender": gender,
                    "race": race,
                    "age_group": age_group
                })
    
                eval_data["slo"][disease].append({
                    "prob": slo_prob,
                    "target": gt_binary,
                    "gender": gender,
                    "race": race,
                    "age_group": age_group
                })
    
            except Exception as e:
                print(e)
                continue
    
        self.print_performance_report(eval_data)

    def print_performance_report(self, eval_data):

        demographic_groups = {
            "Gender": ["male", "female"],
            "Race": ["asian", "white", "black"],
            "Age Group": ["young", "middle", "older"]
        }
    
        attribute_map = {
            "Gender": "gender",
            "Race": "race",
            "Age Group": "age_group"
        }
    
        for disease in ["amd", "dr", "glaucoma"]: 
            print("\n\n" + "#" * 30 +
                  f" {disease.upper()} CLINICAL EVALUATION COMPARISON " +
                  "#" * 30)
    
            for modality in ["oct", "slo"]:
    
                model_name = (
                    "RETFound (OCT Backbone)"
                    if modality == "oct"
                    else "MIRAGE (SLO Backbone)"
                )
    
                records = eval_data[modality][disease]
    
                if len(records) == 0:
                    print(f"\n[{model_name}] No samples found.")
                    continue
    
                ####################################################
                # Overall Report
                ####################################################
    
                print("\n" + "=" * 20 +
                      f" {model_name} RAW METRICS " +
                      "=" * 20)
    
                self.print_subset_report(
                    records,
                    f"{model_name} | Overall (N={len(records)})"
                )
    
                ####################################################
                # Demographic Reports
                ####################################################
    
                for attribute_name, key in attribute_map.items():
    
                    print("\n" + "#" * 25)
                    print(f"{attribute_name} Breakdown")
                    print("#" * 25)
    
                    for subgroup in demographic_groups[attribute_name]:
    
                        subset = [
                            record
                            for record in records
                            if record[key] == subgroup
                        ]
    
                        self.print_subset_report(
                            subset,
                            f"{model_name} | "
                            f"{disease.upper()} | "
                            f"{attribute_name}: {subgroup} "
                            f"(N={len(subset)})"
                        )
    
            print("\n" + "#" * 92)


if __name__ == "__main__":
    master_df = pd.read_csv(EXCEL_PATH)
    
    # Run absolute model testing
    evaluator = RawModelEvaluator(PATH_OCT_WEIGHTS, PATH_SLO_WEIGHTS, DEVICE)
    evaluator.evaluate_excel(master_df)