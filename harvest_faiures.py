import os
import torch
import numpy as np
import json
from PIL import Image
from torchvision import transforms
import models_vit 

# --- CONFIG ---
# Path updated to Test set as requested
SEARCH_DIR = "/home/ab575577/projects_spring_2026/HarvardFairVision30K/FairVision/AMD/Test/"
MODEL_PATH = "best_fair_eye_model.pth"
SAVE_DIR = "failure_cases"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(SAVE_DIR, exist_ok=True)

def hunt_failures():
    print(f"🚀 Initializing Full-Set Deep Scan on {DEVICE}...")
    
    # 1. Model Setup
    model = models_vit.RETFound_mae(
        img_size=224, num_classes=3, drop_path_rate=0.2, global_pool=True
    ).to(DEVICE)
    
    # weights_only=False ensures compatibility with standard saved checkpoints
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
    total_files = len(files)
    print(f"Total samples found in Test set: {total_files}")
    
    failure_registry = []
    stats = {"FALSE_POSITIVE": 0, "FALSE_NEGATIVE": 0}

    # Loop through the ENTIRE set (removed the [:500] limit)
    for idx, f in enumerate(files):
        try:
            data = np.load(os.path.join(SEARCH_DIR, f))
            label = int(data['label']) 
            
            # Extract Image
            img_arr = data['slo_fundus']
            if img_arr.max() <= 1.0: 
                img_arr = (img_arr * 255).astype(np.uint8)
            img = Image.fromarray(img_arr.astype(np.uint8)).convert('RGB')
            img_t = tfm(img).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                out = model(img_t)
                probs = torch.sigmoid(out).cpu().numpy()[0] 
                prob = probs[0] # AMD index

            failure_type = None
            
            # Hero Hunter Logic: Cases where the model is on the wrong side of the coin
            if label == 0 and prob > 0.50:
                failure_type = "FALSE_POSITIVE"
            elif label == 1 and prob < 0.50:
                failure_type = "FALSE_NEGATIVE"

            if failure_type:
                stats[failure_type] += 1
                img_filename = f"{failure_type}_{f.replace('.npz', '')}.jpg"
                img.save(os.path.join(SAVE_DIR, img_filename))
                
                failure_registry.append({
                    "file": f,
                    "type": failure_type,
                    "vision_prob": float(prob),
                    "actual_label": label,
                    "race": int(data.get('race', -1)),
                    "age": int(data.get('age', -1)),
                    "all_probs": probs.tolist()
                })
                # Print every 10th failure to avoid flooding the terminal
                if stats[failure_type] % 10 == 0:
                    print(f"[{idx}/{total_files}] Found {failure_type}: {f} | Prob: {prob:.4f}")

        except Exception as e:
            # Skip corrupt files or missing keys silently
            continue

    # 2. Save Final Registry
    with open(os.path.join(SAVE_DIR, "failures_metadata.json"), "w") as jf:
        json.dump(failure_registry, jf, indent=4)
    
    print("\n" + "="*30)
    print("✅ FULL TEST SCAN COMPLETE")
    print(f"Total Samples Processed: {total_files}")
    print(f"False Positives Found: {stats['FALSE_POSITIVE']}")
    print(f"False Negatives Found: {stats['FALSE_NEGATIVE']}")
    print(f"Metadata saved to: {SAVE_DIR}/failures_metadata.json")
    print("="*30)

if __name__ == "__main__":
    hunt_failures()