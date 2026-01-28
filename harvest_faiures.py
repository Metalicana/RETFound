import os
import torch
import numpy as np
import pandas as pd
import json
from PIL import Image
from torchvision import transforms
import models_vit 

# --- CONFIG ---
BASE_DIR = "/home/ab575577/projects_spring_2026/HarvardFairVision30K/FairVision/Glaucoma/"
CSV_PATH = os.path.join(BASE_DIR, "ReadMe/data_summary_glaucoma.csv")
IMAGE_DIR = os.path.join(BASE_DIR, "Validation/")
MODEL_PATH = "best_fair_eye_model.pth"
SAVE_DIR = "glaucoma_failure_analysis"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(SAVE_DIR, exist_ok=True)

def hunt_glaucoma_failures():
    print(f"🚀 Loading Glaucoma GT from {CSV_PATH}...")
    
    # 1. Load CSV - 'no' is label 0, 'yes' is label 1
    df = pd.read_csv(CSV_PATH)
    gt_lookup = {}
    for _, row in df.iterrows():
        diag = str(row['glaucoma']).lower().strip()
        is_positive = 1 if diag == "yes" else 0
        
        gt_lookup[row['filename']] = {
            "label": is_positive,
            "race": str(row['race']).lower(),
            "age": row['age'],
            "md": row['md'], # Mean Deviation - very useful for Agent reasoning
            "diagnosis": diag
        }

    # 2. Model Loading
    print(f"Loading model from {MODEL_PATH}...")
    model = models_vit.RETFound_mae(num_classes=3, drop_path_rate=0.2, global_pool=True).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    files = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.npz')]
    print(f"Scanning {len(files)} glaucoma validation images...")
    
    failure_registry = []

    for f in files:
        if f not in gt_lookup:
            continue
        
        info = gt_lookup[f]
        
        try:
            data = np.load(os.path.join(IMAGE_DIR, f))
            img_arr = data['slo_fundus']
            
            # Standardization
            if img_arr.max() <= 1.0: 
                img_arr = (img_arr * 255).astype(np.uint8)
            else:
                img_arr = img_arr.astype(np.uint8)
                
            img_pil = Image.fromarray(img_arr).convert('RGB')
            img_t = tfm(img_pil).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                out = model(img_t)
                # GLAUCOMA INDEX IS 2
                prob = torch.sigmoid(out).cpu().numpy()[0][2]

            failure_type = None
            
            # --- FAILURE CRITERIA ---
            # False Positive: CSV says 'no' (0), but Model is > 65% sure
            if info['label'] == 0 and prob > 0.65:
                failure_type = "FALSE_POSITIVE"
                
            # False Negative: CSV says 'yes' (1), but Model is < 35% sure
            elif info['label'] == 1 and prob < 0.35:
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
                    "age": info['age'],
                    "md_value": info['md']
                })
                print(f"📍 FOUND {failure_type}: {f} | Prob: {prob:.2%} | MD: {info['md']}")

        except Exception:
            continue

    # 3. Save results
    with open(os.path.join(SAVE_DIR, "glaucoma_failures_metadata.json"), "w") as jf:
        json.dump(failure_registry, jf, indent=4)

    print(f"\n✅ Done. Found {len(failure_registry)} glaucoma cases.")

if __name__ == "__main__":
    hunt_glaucoma_failures()