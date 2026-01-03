import os
import cv2
import torch
import numpy as np
from PIL import Image
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

    def calculate_cdr(self, mask_array):
        """
        Calculates Vertical Cup-to-Disc Ratio (vCDR) from segmentation mask.
        Assumes: 0=Background, 1=Disc, 2=Cup (Standard REFUGE mapping)
        """
        # Extract binary masks
        # Note: Often SegFormer outputs 0:Back, 1:Disc, 2:Cup
        # But physically, the Cup is INSIDE the Disc. 
        # So 'Disc Area' usually includes the Cup pixels too.
        
        disc_mask = (mask_array == 1) | (mask_array == 2)
        cup_mask = (mask_array == 2)

        if not np.any(disc_mask) or not np.any(cup_mask):
            return 0.0, "No Disc/Cup Detected"

        # Get Bounding Box for Disc
        rows_d, _ = np.where(disc_mask)
        disc_height = rows_d.max() - rows_d.min()

        # Get Bounding Box for Cup
        rows_c, _ = np.where(cup_mask)
        cup_height = rows_c.max() - rows_c.min()

        if disc_height == 0: return 0.0, "Disc Error"
        
        cdr = cup_height / disc_height
        return cdr, "Success"

    def analyze_image(self, image_path, save_path=None):
        # 1. Load and Preprocess
        image = Image.open(image_path).convert("RGB")
        # Keep original size for overlay later
        original_size = image.size[::-1] # (H, W)
        
        inputs = self.processor(images=image, return_tensors="pt").to(DEVICE)

        # 2. Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Resize logits back to original image size
            logits = torch.nn.functional.interpolate(
                outputs.logits,
                size=original_size,
                mode="bilinear",
                align_corners=False,
            )
            pred_mask = logits.argmax(dim=1)[0].cpu().numpy()

        # 3. Calculate CDR
        cdr, status = self.calculate_cdr(pred_mask)

        # 4. Visualization (The "Doctor Candy")
        if save_path:
            plt.figure(figsize=(10, 5))
            
            # Original Image
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.title(f"Input Fundus\nCDR: {cdr:.3f}")
            plt.axis('off')

            # Segmentation Overlay
            # Create colored mask: Disc=Blue, Cup=Red
            overlay = np.zeros((*original_size, 3), dtype=np.uint8)
            # Disc (Class 1) -> Green/Blue
            overlay[pred_mask == 1] = [0, 255, 0] 
            # Cup (Class 2) -> Red
            overlay[pred_mask == 2] = [255, 0, 0]

            plt.subplot(1, 2, 2)
            plt.imshow(image)
            plt.imshow(overlay, alpha=0.4) # Transparent overlay
            plt.title("SegFormer Segmentation")
            plt.axis('off')
            
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            print(f"Saved visualization to {save_path}")

        return cdr, status

# --- RUN IT ON A SAMPLE ---
if __name__ == "__main__":
    # Test on one of your Glaucoma images
    # Update this path to a real image in your dataset
    TEST_IMG = "./FairVision/Glaucoma/Training/slo_fundus_00001.jpg" 
    
    if os.path.exists(TEST_IMG):
        tool = GlaucomaTool()
        cdr_value, status = tool.analyze_image(TEST_IMG, save_path="demo_cdr_overlay.png")
        print(f"\n>>> ANALYSIS RESULT: CDR = {cdr_value:.3f} ({status})")
    else:
        print(f"Please update TEST_IMG path. Could not find {TEST_IMG}")