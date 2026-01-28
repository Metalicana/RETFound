import os
import torch
import numpy as np
import json
from PIL import Image
from torchvision import transforms
import models_vit 

# --- CONFIG ---
SEARCH_DIR = "/home/ab575577/projects_spring_2026/HarvardFairVision30K/FairVision/AMD/Validation/"
MODEL_PATH = "best_fair_eye_model.pth"
SAVE_DIR = "failure_cases"
os.makedirs(SAVE_DIR, exist_ok=True)

def hunt_failures():
    # ... [Model Loading Code from previous step] ...
    model.eval()

    files = [f for f in os.listdir(SEARCH_DIR) if f.endswith('.npz')]
    failure_registry = []

    for f in files:
        data = np.load(os.path.join(SEARCH_DIR, f))
        label = int(data['label']) 
        race = int(data.get('race', -1))
        
        # Process image and get prediction
        img_t = tfm(Image.fromarray(data['slo_fundus']).convert('RGB')).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            prob = torch.sigmoid(model(img_t)).cpu().numpy()[0][0]

        # --- CLASSIFY FAILURES ---
        failure_type = None
        
        # 1. False Positive (Model thinks sick, but is healthy)
        # Excellent for showing the Agent preventing unnecessary treatment/referral.
        if label == 0 and prob > 0.70:
            failure_type = "FALSE_POSITIVE"
        
        # 2. False Negative (Model misses the disease)
        # Excellent for showing the Agent "saving" a patient by requesting more tests.
        elif label == 1 and prob < 0.30:
            failure_type = "FALSE_NEGATIVE"

        if failure_type:
            # Save data for the Agent script
            case_id = f.replace('.npz', '')
            img_path = os.path.join(SAVE_DIR, f"{failure_type}_{case_id}.jpg")
            Image.fromarray(data['slo_fundus']).save(img_path)
            
            failure_registry.append({
                "file": f,
                "type": failure_type,
                "vision_prob": float(prob),
                "actual_label": label,
                "race": race,
                "age": int(data.get('age', 60)) # Example metadata
            })

    with open(os.path.join(SAVE_DIR, "failures.json"), "w") as jf:
        json.dump(failure_registry, jf, indent=4)