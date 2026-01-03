import cv2
import numpy as np
from PIL import Image

INPUT_FILE = "demo_case.jpg"
OUTPUT_FILE = "demo_case_enhanced.jpg"

def enhance_image():
    # 1. Load the "Black" image
    print(f"Loading {INPUT_FILE}...")
    img = cv2.imread(INPUT_FILE, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print("Error: Could not read image. Did you run get_demo_image.py?")
        return

    # 2. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # This is what radiologists/ophthalmologists use to see details in dark scans.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_img = clahe.apply(img)
    
    # 3. Save
    cv2.imwrite(OUTPUT_FILE, enhanced_img)
    print(f"Saved enhanced version to: {OUTPUT_FILE}")
    print("Open this file. You should see the veins and the 'white stuff' (disease) clearly now.")

if __name__ == "__main__":
    enhance_image()