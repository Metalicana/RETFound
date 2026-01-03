import os
import numpy as np
from PIL import Image

# Config - We want a BLACK patient with AMD (The "Villain" Scenario)
# Based on your README/Code: Race 1 = Black, Disease = AMD
SOURCE_DIR = "/home/ab575577/projects_spring_2026/HarvardFairVision30K/FairVision/AMD/Validation/"
OUTPUT_FILE = "demo_case.jpg"

def extract_demo_image():
    print(f"Searching for a high-risk demo case in {SOURCE_DIR}...")
    
    files = [f for f in os.listdir(SOURCE_DIR) if f.endswith('.npz')]
    
    found = False
    for f in files:
        path = os.path.join(SOURCE_DIR, f)
        try:
            data = np.load(path)
            # Check if patient is Black (Race=1) and actually has AMD
            # (Your dataset sorts by folder, so we know it's AMD, just need Race)
            if 'race' in data and int(data['race']) == 1:
                print(f"FOUND MATCH: {f} (Black Patient with AMD)")
                
                # Extract Image
                img_array = data['slo_fundus']
                if img_array.max() <= 1.0: img_array = (img_array * 255).astype(np.uint8)
                else: img_array = img_array.astype(np.uint8)
                
                img = Image.fromarray(img_array).convert('RGB')
                img.save(OUTPUT_FILE)
                print(f"Saved demo image to: {OUTPUT_FILE}")
                found = True
                break
        except:
            continue
            
    if not found:
        print("Could not find a perfect match. Grabbing the first available image instead...")
        # Fallback
        data = np.load(os.path.join(SOURCE_DIR, files[0]))
        img_array = (data['slo_fundus'] * 255).astype(np.uint8)
        Image.fromarray(img_array).convert('RGB').save(OUTPUT_FILE)

if __name__ == "__main__":
    extract_demo_image()