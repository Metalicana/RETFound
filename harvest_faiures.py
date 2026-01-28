import os
import torch
import numpy as np
import json
from PIL import Image
from torchvision import transforms
import models_vit 

# --- CONFIG ---
# We use the Test set to find the most robust unseen examples
BASE_DIR = "/home/ab575577/projects_spring_2026/HarvardFairVision30K/FairVision/"
DISEASES = ["AMD", "Glaucoma", "DR"] 
MODEL_PATH = "best_fair_eye_model.pth"
SAVE_DIR = "hero_cases_output"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(SAVE_DIR, exist_ok=True)

def hunt_all_heroes():
    print(f"🚀 Initializing Multi-Disease Hero Hunter on {DEVICE}...")
    
    # 1. Load Model
    model = models_vit.RETFound_mae(
        img_size=224, num_classes=3, drop_path_rate=0.2, global_pool=True
    ).to(DEVICE)
    
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    hero_registry = []

    # 2. Iterate through each disease folder
    for disease_idx, disease in enumerate(DISEASES):
        search_path = os.path.join(BASE_DIR, disease, "Test")
        if not os.path.exists(search_path):
            print(f"Skipping {disease}, path not found: {search_path}")
            continue

        print(f"\n--- Scanning {disease} (Index {disease_idx}) ---")
        files = [f for f in os.listdir(search_path) if f.endswith('.npz')]
        
        found_for_this_disease = 0

        for f in files:
            try:
                data = np.load(os.path.join(search_path, f))
                
                # Metadata for the Agent
                race = int(data.get('race', -1))
                label = int(data.get('label', -1))
                
                # Image Prep
                img_arr = data['slo_fundus']
                if img_arr.max() <= 1.0: img_arr = (img_arr * 255).astype(np.uint8)
                else: img_arr = img_arr.astype(np.uint8)
                img_pil = Image.fromarray(img_arr).convert('RGB')
                
                img_t = tfm(img_pil).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    out = model(img_t)
                    probs = torch.sigmoid(out).cpu().numpy()[0]
                    prob = probs[disease_idx] # Match prob to the disease folder

                # HERO LOGIC: The "Sweet Spot" 
                # Model is uncertain (20-48%), but ground truth is Positive (1)
                # This is where the Agent's reasoning "saves" the diagnosis.
                if 0.20 < prob < 0.48 and label == 1:
                    save_name = f"HERO_{disease}_{f.replace('.npz', '')}.jpg"
                    img_pil.save(os.path.join(SAVE_DIR, save_name))
                    
                    case_data = {
                        "disease": disease,
                        "file": f,
                        "vision_prob": float(prob),
                        "label": label,
                        "race": race,
                        "image_path": save_name
                    }
                    hero_registry.append(case_data)
                    found_for_this_disease += 1
                    print(f"  [+] Found {disease} Hero: {f} (Prob: {prob:.2%})")

                # Limit to 5 high-quality cases per disease for the grant
                if found_for_this_disease >= 5:
                    break
            
            except Exception:
                continue

    # 3. Save Registry for the Agent script
    with open(os.path.join(SAVE_DIR, "hero_metadata.json"), "w") as jf:
        json.dump(hero_registry, jf, indent=4)

    print(f"\n✅ Done! Found {len(hero_registry)} hero cases total.")
    print(f"Metadata and images saved in: {SAVE_DIR}/")

if __name__ == "__main__":
    hunt_all_heroes()