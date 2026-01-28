import os
import torch
import numpy as np
import json
from PIL import Image
from torchvision import transforms
import models_vit 

# --- CONFIG ---
SEARCH_DIR = "/home/ab575577/projects_spring_2026/HarvardFairVision30K/FairVision/AMD/Test/"
MODEL_PATH = "best_fair_eye_model.pth"
SAVE_DIR = "failure_cases"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 3

os.makedirs(SAVE_DIR, exist_ok=True)

def hunt_all_failures():
    print(f"🚀 Initializing Full-Set Failure Hunter on {DEVICE}...")
    
    # 1. Model Loading
    model = models_vit.RETFound_mae(
        img_size=224, num_classes=NUM_CLASSES, drop_path_rate=0.2, global_pool=True
    ).to(DEVICE)
    
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # 2. Preprocessing (Matching your training pipeline)
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    files = [f for f in os.listdir(SEARCH_DIR) if f.endswith('.npz')]
    total_files = len(files)
    print(f"Total test samples to scan: {total_files}")
    
    failure_registry = []
    
    # Counter for progress
    processed = 0

    for f in files:
        try:
            path = os.path.join(SEARCH_DIR, f)
            data = np.load(path)
            label = int(data['label'])
            race = int(data.get('race', -1))
            
            # Prepare Image
            img_arr = data['slo_fundus']
            # Ensure it is in 0-255 uint8 format before PIL
            if img_arr.max() <= 1.0:
                img_arr = (img_arr * 255).astype(np.uint8)
            else:
                img_arr = img_arr.astype(np.uint8)
            
            img_pil = Image.fromarray(img_arr).convert('RGB')
            img_t = tfm(img_pil).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = model(img_t)
                probs = torch.sigmoid(logits).cpu().numpy()[0]
                # Assuming AMD is Index 0 (based on your project structure)
                prob = probs[0] 

            # --- DYNAMIC HERO LOGIC ---
            # We look for ANY case where the vision model is wrong (on the wrong side of 0.5)
            failure_type = None
            if label == 0 and prob > 0.50:
                failure_type = "FALSE_POSITIVE"
            elif label == 1 and prob < 0.50:
                failure_type = "FALSE_NEGATIVE"

            if failure_type:
                # Save the image so you can inspect it for the grant
                img_filename = f"{failure_type}_{f.replace('.npz', '')}.jpg"
                img_pil.save(os.path.join(SAVE_DIR, img_filename))
                
                case_data = {
                    "file": f,
                    "type": failure_type,
                    "vision_prob": float(prob),
                    "all_probs": probs.tolist(),
                    "actual_label": label,
                    "race": race,
                    "age": int(data.get('age', -1))
                }
                failure_registry.append(case_data)

            processed += 1
            if processed % 100 == 0:
                print(f"Processed {processed}/{total_files}... Found {len(failure_registry)} failures.")

        except Exception as e:
            continue

    # 3. Final Export
    with open(os.path.join(SAVE_DIR, "failures_metadata.json"), "w") as jf:
        json.dump(failure_registry, jf, indent=4)
    
    print("\n" + "="*40)
    print(f"SCAN COMPLETE: Found {len(failure_registry)} total failures.")
    
    # Filter for the 'Perfect' Hero cases (Black patients where the model failed)
    hero_candidates = [c for c in failure_registry if c['race'] == 1]
    print(f"Specifically found {len(hero_candidates)} failures involving Black patients (Race=1).")
    print(f"All data saved in: {SAVE_DIR}/")
    print("="*40)

if __name__ == "__main__":
    hunt_all_failures()