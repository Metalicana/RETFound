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
# Adjust to Training or Validation as needed
IMAGE_DIR = os.path.join(BASE_DIR, "Validation/") 
MODEL_PATH = "best_fair_eye_model.pth"
SAVE_DIR = "glaucoma_multimodal_failures"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(SAVE_DIR, exist_ok=True)

def hunt_multimodal_glaucoma():
    print(f"🚀 Hunting Glaucoma failures with separate Fundus JPEGs...")
    
    # 1. Load CSV
    df = pd.read_csv(CSV_PATH)
    gt_lookup = {row['filename']: row for _, row in df.iterrows()}

    # 2. Model Loading
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

    # Find all NPZ files (which contain the OCT)
    npz_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.npz')]
    failure_registry = []

    for f in npz_files:
        if f not in gt_lookup: continue
        row = gt_lookup[f]
        label = 1 if str(row['glaucoma']).lower().strip() == "yes" else 0
        
        # Construct the matching Fundus JPG path
        # Example: data_00001.npz -> slo_fundus_00001.jpg
        num_part = f.split('_')[1].replace('.npz', '')
        fundus_path = os.path.join(IMAGE_DIR, f"slo_fundus_{num_part}.jpg")
        
        if not os.path.exists(fundus_path):
            continue

        try:
            # --- VISION MODEL INFERENCE ON JPG ---
            fundus_img = Image.fromarray(np.array(Image.open(fundus_path))).convert('RGB')
            img_t = tfm(fundus_img).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                out = model(img_t)
                prob = torch.sigmoid(out).cpu().numpy()[0][2] # Glaucoma Index 2

            failure_type = None
            if label == 0 and prob > 0.65: failure_type = "FALSE_POSITIVE"
            elif label == 1 and prob < 0.35: failure_type = "FALSE_NEGATIVE"

            if failure_type:
                # 3. Save the "Hero Case" Pair
                # Copy/Save Fundus
                save_fundus_name = f"{failure_type}_{num_part}_fundus.jpg"
                fundus_img.save(os.path.join(SAVE_DIR, save_fundus_name))
                
                # Extract and Save OCT Center Slice from NPZ
                data = np.load(os.path.join(IMAGE_DIR, f))
                oct_save_path = "N/A"
                if 'oct' in data:
                    oct_vol = data['oct']
                    # Get middle slice for visual reasoning
                    mid_idx = oct_vol.shape[0] // 2
                    mid_slice = oct_vol[mid_idx]
                    
                    if mid_slice.max() <= 1.0: mid_slice = (mid_slice * 255).astype(np.uint8)
                    oct_img = Image.fromarray(mid_slice.astype(np.uint8)).convert('L')
                    
                    oct_save_path = f"{failure_type}_{num_part}_oct_slice.jpg"
                    oct_img.save(os.path.join(SAVE_DIR, oct_save_path))

                failure_registry.append({
                    "id": num_part,
                    "type": failure_type,
                    "vision_prob": float(prob),
                    "actual_label": label,
                    "race": str(row['race']).lower(),
                    "age": row['age'],
                    "md": row['md'],
                    "fundus_file": save_fundus_name,
                    "oct_file": oct_save_path
                })
                print(f"✅ Saved Pair for {num_part} | Prob: {prob:.2%} | Type: {failure_type}")

        except Exception as e:
            print(f"Error processing {f}: {e}")
            continue

    with open(os.path.join(SAVE_DIR, "glaucoma_hero_metadata.json"), "w") as jf:
        json.dump(failure_registry, jf, indent=4)

    print(f"\nDone. Found {len(failure_registry)} multi-modal cases for your grant.")

if __name__ == "__main__":
    hunt_multimodal_glaucoma()