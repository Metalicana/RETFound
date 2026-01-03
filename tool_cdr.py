import os
import cv2
import torch
import numpy as np
from PIL import Image, ImageOps
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
MODEL_ID = "pamixsun/segformer_for_optic_disc_cup_segmentation"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class GlaucomaTool:
    def __init__(self):
        print(f"Loading Glaucoma Tool (SegFormer) on {DEVICE}...")
        self.processor = SegformerImageProcessor.from_pretrained(MODEL_ID)
        self.model = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID).to(DEVICE)
        self.model.eval()

    def preprocess_slo(self, image_path):
        """
        Enhances dark SLO images so the model can see the Optic Disc boundary.
        """
        # 1. Load as grayscale first to analyze structure
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # 2. Check if image is too dark (SLO issue)
        if np.mean(img) < 40: # If average brightness is low
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            img = clahe.apply(img)
            
            # Optional: Gamma Correction to brighten midtones
            gamma = 1.5
            invGamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            img = cv2.LUT(img, table)

        # 3. Convert back to RGB (Model expects 3 channels)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(img_rgb)

    def calculate_cdr(self, mask_array):
        # 0=Background, 1=Disc, 2=Cup
        disc_mask = (mask_array == 1) | (mask_array == 2)
        cup_mask = (mask_array == 2)

        if not np.any(disc_mask): return 0.0, "No Disc Detected"
        if not np.any(cup_mask): return 0.0, "No Cup Detected"

        # Get vertical heights
        rows_d, _ = np.where(disc_mask)
        disc_height = rows_d.max() - rows_d.min()
        
        rows_c, _ = np.where(cup_mask)
        cup_height = rows_c.max() - rows_c.min()

        if disc_height == 0: return 0.0, "Disc Error"
        
        cdr = cup_height / disc_height
        return cdr, "Success"

    def analyze_image(self, image_path, save_path=None):
        # --- NEW PREPROCESSING STEP ---
        image = self.preprocess_slo(image_path)
        original_size = image.size[::-1] 
        
        inputs = self.processor(images=image, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = torch.nn.functional.interpolate(
                outputs.logits, size=original_size, mode="bilinear", align_corners=False
            )
            pred_mask = logits.argmax(dim=1)[0].cpu().numpy()

        cdr, status = self.calculate_cdr(pred_mask)

        if save_path:
            plt.figure(figsize=(10, 5))
            
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.title(f"Enhanced Input\nCDR: {cdr:.3f}")
            plt.axis('off')

            overlay = np.zeros((*original_size, 3), dtype=np.uint8)
            overlay[pred_mask == 1] = [0, 255, 0] # Green Disc
            overlay[pred_mask == 2] = [255, 0, 0] # Red Cup

            plt.subplot(1, 2, 2)
            plt.imshow(image)
            plt.imshow(overlay, alpha=0.4)
            plt.title(f"Segmentation ({status})")
            plt.axis('off')
            
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            print(f"Saved visualization to {save_path}")

        return cdr, status

if __name__ == "__main__":
    # Point this to your image again
    TEST_IMG = "/home/ab575577/projects_spring_2026/HarvardFairVision30K/FairVision/Glaucoma/Training/slo_fundus_00001.jpg" 
    
    if os.path.exists(TEST_IMG):
        tool = GlaucomaTool()
        cdr, status = tool.analyze_image(TEST_IMG, save_path="demo_cdr_fixed.png")
        print(f"\n>>> RESULT: CDR = {cdr:.3f} ({status})")