import os
import torch
import numpy as np
import json
from PIL import Image
from torchvision import transforms
import models_vit # Assumes this is in your python path

# --- CONFIG ---
SEARCH_DIR = "/home/ab575577/projects_spring_2026/HarvardFairVision30K/FairVision/AMD/Validation/"
MODEL_PATH = "best_fair_eye_model.pth"
SAVE_DIR = "failure_cases"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure save directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

def get_model():
    """Initializes RETFound and loads weights correctly."""
    model = models_vit.RETFound_mae(
        img_size=224, 
        num_classes=3, 
        drop_path_rate=0.2, 
        global_pool=True
    ).to(DEVICE)
    
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    # Handle both wrapped and unwrapped state dicts
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

def hunt_failures():
    print(f"🚀 Initializing Failure Hunter on {DEVICE}...")
    model = get_model()

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    files = [f for f in os.listdir(SEARCH_DIR) if f.endswith('.npz')]
    print(f"Scanning {len(files)} samples for failures...")
    
    failure_registry = []

    for f in files:
        try:
            data = np.load(os.path.join(SEARCH_DIR, f))
            label = int(data['label']) 
            race = int(data.get('race', -1))
            
            # Prepare image
            img_arr = data['slo_fundus']
            if img_arr.max() <= 1.0: 
                img_arr = (img_arr * 255).astype(np.uint8)
            img = Image.fromarray(img_arr.astype(np.uint8)).convert('RGB')
            
            img_t = tfm(img).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                out = model(img_t)
                # Sigmoid for binary probability of the disease class
                prob = torch.sigmoid(out).cpu().numpy()[0][0]

            failure_type = None
            
            # CASE A: False Positive (Model thinks sick, actually healthy)
            # Great for showing Agent preventing unnecessary procedures
            if label == 0 and prob > 0.75:
                failure_type = "FALSE_POSITIVE"
            
            # CASE B: False Negative (Model misses disease)
            # Great for showing Agent 'saving' a patient using metadata
            elif label == 1 and prob < 0.25:
                failure_type = "FALSE_NEGATIVE"

            if failure_type:
                case_id = f.replace('.npz', '')
                img_filename = f"{failure_type}_{case_id}.jpg"
                img.save(os.path.join(SAVE_DIR, img_filename))
                
                case_data = {
                    "filename": f,
                    "image_saved_as": img_filename,
                    "type": failure_type,
                    "vision_prob": float(prob),
                    "actual_label": label,
                    "race": race,
                    "age": int(data.get('age', 0))
                }
                failure_registry.append(case_data)
                print(f"Found {failure_type}: {f} | Prob: {prob:.2%}")

        except Exception as e:
            continue

    # Save metadata for the Agent script to read
    with open(os.path.join(SAVE_DIR, "failures_metadata.json"), "w") as jf:
        json.dump(failure_registry, jf, indent=4)
    
    print(f"\n✅ Hunt complete. Found {len(failure_registry)} failure cases.")
    print(f"Metadata saved to {SAVE_DIR}/failures_metadata.json")

if __name__ == "__main__":
    hunt_failures()