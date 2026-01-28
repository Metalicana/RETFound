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
IMAGE_DIR = os.path.join(BASE_DIR, "Validation/") # or "Test/"
MODEL_PATH = "best_fair_eye_model.pth"
SAVE_DIR = "hero_cases_output"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(SAVE_DIR, exist_ok=True)

def hunt_with_real_gt():
    print(f"🚀 Loading Ground Truth from {CSV_PATH}...")
    
    # 1. Load CSV and create a lookup for labels
    # We map any form of AMD (early, intermediate, late) to 1
    df = pd.read_csv(CSV_PATH)
    gt_lookup = {}
    for _, row in df.iterrows():
        # 'amd' column contains the diagnosis string
        is_positive = 0 if "healthy" in row['amd'].lower() or "control" in row['amd'].lower() else 1
        gt_lookup[row['filename']] = {
            "label": is_positive,
            "race": row['race'].lower(),
            "age": row['age'],
            "diagnosis": row['amd']
        }

    # 2. Load Model
    model = models_vit.RETFound_mae(num_classes=3, drop_path_rate=0.2, global_pool=True).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint, strict=False)
    model.eval()

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. Scan Files
    files = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.npz')]
    print(f"Scanning {len(files)} images in {IMAGE_DIR}...")
    
    hero_registry = []

    for f in files:
        if f not in gt_lookup: continue
        
        case_info = gt_lookup[f]
        
        # HERO FILTER: Black patient (matching your successful 'race=1' logic)
        if "black" not in case_info['race']:
            continue

        try:
            data = np.load(os.path.join(IMAGE_DIR, f))
            img_arr = data['slo_fundus']
            if img_arr.max() <= 1.0: img_arr = (img_arr * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_arr.astype(np.uint8)).convert('RGB')
            
            img_t = tfm(img_pil).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                out = model(img_t)
                prob = torch.sigmoid(out).cpu().numpy()[0][0] # AMD is index 0

            # HERO LOGIC: Model is uncertain (20-48%) but GT is POSITIVE
            if 0.20 < prob < 0.48 and case_info['label'] == 1:
                save_name = f"HERO_{f.replace('.npz', '')}.jpg"
                img_pil.save(os.path.join(SAVE_DIR, save_name))
                
                entry = {
                    "file": f,
                    "vision_prob": float(prob),
                    "gt_label": case_info['label'],
                    "diagnosis": case_info['diagnosis'],
                    "race": case_info['race'],
                    "age": case_info['age']
                }
                hero_registry.append(entry)
                print(f"  ✅ [FOUND] {f} | Prob: {prob:.2%} | GT: {case_info['diagnosis']}")

            if len(hero_registry) >= 5: break

        except Exception as e:
            continue

    with open(os.path.join(SAVE_DIR, "hero_metadata.json"), "w") as jf:
        json.dump(hero_registry, jf, indent=4)

    print(f"\n🚀 Done. Found {len(hero_registry)} true hero cases using CSV ground truth.")

if __name__ == "__main__":
    hunt_with_real_gt()