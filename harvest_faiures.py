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
os.makedirs(SAVE_DIR, exist_ok=True)

def hunt_failures():
    print(f"🚀 Loosening thresholds to find 'Hero' opportunities...")
    
    # 1. Model Setup
    model = models_vit.RETFound_mae(
        img_size=224, num_classes=3, drop_path_rate=0.2, global_pool=True
    ).to(DEVICE)
    
    # Using weights_only=True to silence that future warning
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    files = [f for f in os.listdir(SEARCH_DIR) if f.endswith('.npz')]
    failure_registry = []

    for f in files[:500]: # Check first 500 to see if we get hits
        try:
            data = np.load(os.path.join(SEARCH_DIR, f))
            label = int(data['label']) 
            
            img_arr = data['slo_fundus']
            if img_arr.max() <= 1.0: img_arr = (img_arr * 255).astype(np.uint8)
            img = Image.fromarray(img_arr.astype(np.uint8)).convert('RGB')
            img_t = tfm(img).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                out = model(img_t)
                # sigmoid converts logits to 0-1 range
                probs = torch.sigmoid(out).cpu().numpy()[0] 
                # Check the probability of the disease we are looking for (index 0 for AMD)
                prob = probs[0] 

            failure_type = None
            
            # Relaxed Thresholds: Just needs to be on the wrong side of 0.5
            if label == 0 and prob > 0.55:
                failure_type = "FALSE_POSITIVE"
            elif label == 1 and prob < 0.45:
                failure_type = "FALSE_NEGATIVE"

            if failure_type:
                img_filename = f"{failure_type}_{f.replace('.npz', '')}.jpg"
                img.save(os.path.join(SAVE_DIR, img_filename))
                
                failure_registry.append({
                    "file": f,
                    "type": failure_type,
                    "vision_prob": float(prob),
                    "actual_label": label,
                    "race": int(data.get('race', -1)),
                    "all_probs": probs.tolist()
                })
                print(f"FOUND {failure_type}: {f} | Prob: {prob:.4f} | Label: {label}")

        except Exception as e:
            continue

    with open(os.path.join(SAVE_DIR, "failures_metadata.json"), "w") as jf:
        json.dump(failure_registry, jf, indent=4)
    
    print(f"\n✅ Found {len(failure_registry)} potential hero cases.")

if __name__ == "__main__":
    hunt_failures()