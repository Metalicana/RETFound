import os
import torch
import numpy as np
import pandas as pd
import json
from PIL import Image
from torchvision import transforms
import models_vit 

# --- CONFIG ---
BASE_DIR = "/home/ab575577/projects_spring_2026/HarvardFairVision30K/FairVision/AMD/"
CSV_PATH = os.path.join(BASE_DIR, "ReadMe/data_summary_amd.csv")
IMAGE_DIR = os.path.join(BASE_DIR, "Validation/")
MODEL_PATH = "best_fair_eye_model.pth"
SAVE_DIR = "amd_failure_analysis"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(SAVE_DIR, exist_ok=True)

def hunt_failures():
    print(f"🚀 Loading Real GT from {CSV_PATH} and scanning {IMAGE_DIR}...")
    
    # 1. Load CSV - Map anything not 'healthy' or 'control' to label 1
    df = pd.read_csv(CSV_PATH)
    gt_lookup = {}
    for _, row in df.iterrows():
        diag = str(row['amd']).lower()
        is_positive = 0 if "healthy" in diag or "control" in diag or "no amd" in diag else 1
        gt_lookup[row['filename']] = {
            "label": is_positive,
            "race": str(row['race']).lower(),
            "age": row['age'],
            "diagnosis": diag
        }

    # 2. Model Loading
    model = models_vit.RETFound_mae(num_classes=3, drop_path_rate=0.2, global_pool=True).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint, strict=False)
    model.eval()

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    files = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.npz')]
    failure_registry = []

    for f in files:
        if f not in gt_lookup: continue
        
        info = gt_lookup[f]
        
        try:
            data = np.load(os.path.join(IMAGE_DIR, f))
            img_arr = data['slo_fundus']
            if img_arr.max() <= 1.0: img_arr = (img_arr * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_arr.astype(np.uint8)).convert('RGB')
            
            img_t = tfm(img_pil).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                out = model(img_t)
                prob = torch.sigmoid(out).cpu().numpy()[0][0] # AMD index

            failure_type = None
            
            # CASE 1: False Positive (Model thinks sick (>70%), but CSV says Healthy)
            # Best for showing Agent cross-referencing metadata to prevent misdiagnosis.
            if info['amd'] != 'normal' and prob > 0.70:
                failure_type = "FALSE_POSITIVE"
                
            # CASE 2: False Negative (Model misses it (<30%), but CSV says Late/Early AMD)
            # Best for showing Agent saving the day by flagging high-risk patient age/smoking.
            elif info['amd'] == 'normal' and prob < 0.30:
                failure_type = "FALSE_NEGATIVE"

            if failure_type:
                save_path = f"{failure_type}_{f.replace('.npz', '')}.jpg"
                img_pil.save(os.path.join(SAVE_DIR, save_path))
                
                failure_registry.append({
                    "file": f,
                    "type": failure_type,
                    "vision_prob": float(prob),
                    "actual_diagnosis": info['diagnosis'],
                    "race": info['race'],
                    "age": info['age']
                })
                print(f"📍 FOUND {failure_type}: {f} | Prob: {prob:.2%} | Real: {info['diagnosis']}")

        except Exception:
            continue

    with open(os.path.join(SAVE_DIR, "amd_failures.json"), "w") as jf:
        json.dump(failure_registry, jf, indent=4)

    print(f"\n✅ Scan complete. Found {len(failure_registry)} high-confidence failures.")

if __name__ == "__main__":
    hunt_failures()