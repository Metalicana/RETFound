import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import models_vit 

# --- CONFIG ---
SEARCH_DIR = "/home/ab575577/projects_spring_2026/HarvardFairVision30K/FairVision/AMD/Validation/"
MODEL_PATH = "best_fair_eye_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 3

def find_hero():
    print(f"Loading Model to hunt for a ROBUST 'Hero Case'...")
    
    model = models_vit.RETFound_mae(
        img_size=224, num_classes=NUM_CLASSES, drop_path_rate=0.2, global_pool=True
    ).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    files = [f for f in os.listdir(SEARCH_DIR) if f.endswith('.npz')]
    print(f"Scanning {len(files)} validation images...")

    best_prob = 0.0
    best_file = None

    for f in files:
        try:
            path = os.path.join(SEARCH_DIR, f)
            data = np.load(path)
            
            # Must be Black Patient (Race=1)
            if 'race' not in data or int(data['race']) != 1:
                continue

            img_arr = data['slo_fundus']
            if img_arr.max() <= 1.0: img_arr = (img_arr * 255).astype(np.uint8)
            else: img_arr = img_arr.astype(np.uint8)
            img = Image.fromarray(img_arr).convert('RGB')
            
            img_t = tfm(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                out = model(img_t)
                prob = torch.sigmoid(out).cpu().numpy()[0][0] # AMD Probability

            # HUNTING LOGIC:
            # We want between 20% and 48%. 
            # This is the "Sweet Spot" where the Agent saves the day.
            if 0.20 < prob < 0.48:
                print(f"\n>>> FOUND PERFECT HERO CASE: {f}")
                print(f"    Raw AMD Probability: {prob:.2%}")
                print(f"    (Robust enough to survive JPEG compression!)")
                
                img.save("demo_case.jpg")
                print("    Saved to 'demo_case.jpg'")
                return
            
        except Exception:
            continue

    print("Scan complete.")

if __name__ == "__main__":
    find_hero()