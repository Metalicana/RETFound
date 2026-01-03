import os
import cv2
import torch
import numpy as np
import json
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import matplotlib.pyplot as plt

# --- 1. THE CORE LOGIC (The "Brain") ---
# This is the fixed version with the SLO Enhancer (CLAHE)
class GlaucomaTool:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[System] Loading Glaucoma Tool on {self.device}...")
        self.model_id = "pamixsun/segformer_for_optic_disc_cup_segmentation"
        self.processor = SegformerImageProcessor.from_pretrained(self.model_id)
        self.model = SegformerForSemanticSegmentation.from_pretrained(self.model_id).to(self.device)
        self.model.eval()

    def preprocess_slo(self, image_path):
        """Enhances dark SLO images using CLAHE."""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # Safety check: if image load fails
        if img is None: raise ValueError(f"Could not load image at {image_path}")
        
        # Apply CLAHE if image is dark (SLO characteristic)
        if np.mean(img) < 60: 
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            img = clahe.apply(img)
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(img_rgb)

    def calculate_cdr(self, mask_array):
        # 0=Background, 1=Disc, 2=Cup
        disc_mask = (mask_array == 1) | (mask_array == 2)
        cup_mask = (mask_array == 2)

        if not np.any(disc_mask): return 0.0, "No Disc Detected"
        if not np.any(cup_mask): return 0.0, "No Cup Detected"

        rows_d, _ = np.where(disc_mask)
        disc_height = rows_d.max() - rows_d.min()
        
        rows_c, _ = np.where(cup_mask)
        cup_height = rows_c.max() - rows_c.min()

        if disc_height == 0: return 0.0, "Disc Error"
        return cup_height / disc_height, "Success"

    def run_analysis(self, image_path):
        try:
            image = self.preprocess_slo(image_path)
            original_size = image.size[::-1]
            
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = torch.nn.functional.interpolate(
                    outputs.logits, size=original_size, mode="bilinear", align_corners=False
                )
                pred_mask = logits.argmax(dim=1)[0].cpu().numpy()
            
            cdr, status = self.calculate_cdr(pred_mask)
            
            # Save debug visualization for the user/demo
            debug_path = "latest_agent_debug.png"
            self.save_visualization(image, pred_mask, cdr, debug_path)
            
            return cdr, status, debug_path
        except Exception as e:
            return 0.0, f"Error: {str(e)}", None

    def save_visualization(self, image, mask, cdr, path):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title(f"Enhanced Input\nCDR: {cdr:.3f}")
        plt.axis('off')
        
        overlay = np.zeros((image.size[1], image.size[0], 3), dtype=np.uint8)
        overlay[mask == 1] = [0, 255, 0] # Green Disc
        overlay[mask == 2] = [255, 0, 0] # Red Cup
        
        plt.subplot(1, 2, 2)
        plt.imshow(image)
        plt.imshow(overlay, alpha=0.4)
        plt.title("Segmentation")
        plt.axis('off')
        plt.savefig(path, bbox_inches='tight')
        plt.close()

# Initialize Global Instance (So we don't reload model on every call)
_GLAUCOMA_TOOL_INSTANCE = None

def get_tool_instance():
    global _GLAUCOMA_TOOL_INSTANCE
    if _GLAUCOMA_TOOL_INSTANCE is None:
        _GLAUCOMA_TOOL_INSTANCE = GlaucomaTool()
    return _GLAUCOMA_TOOL_INSTANCE

# --- 2. THE WRAPPER (The Function the Agent Calls) ---
def tool_glaucoma_check(image_path: str):
    """
    Analyzes the optic nerve structure to assess Glaucoma risk.
    Args:
        image_path: The full file path to the fundus image.
    Returns:
        JSON string containing CDR value, Risk Level, and visual proof path.
    """
    tool = get_tool_instance()
    cdr, status, debug_path = tool.run_analysis(image_path)
    
    if "Error" in status:
        return json.dumps({"status": "failed", "reason": status})
    
    # Medical Risk Logic
    risk = "Low"
    if cdr > 0.55: risk = "Moderate"
    if cdr > 0.70: risk = "High (Likely Glaucoma)"
    
    return json.dumps({
        "status": "success",
        "vertical_cdr": round(cdr, 3),
        "risk_assessment": risk,
        "visual_proof": debug_path,
        "note": "A CDR > 0.7 strongly indicates structural damage."
    })

# --- 3. THE REGISTRY (The Menu for GPT-4) ---
# This dictionary maps the function name to the actual Python function
AVAILABLE_TOOLS = {
    "tool_glaucoma_check": tool_glaucoma_check
}

# This List Definition informs OpenAI/Azure what tools are available
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "tool_glaucoma_check",
            "description": "Use this tool when you need to physically verify Glaucoma. It segments the optic disc and cup to calculate the Cup-to-Disc Ratio (CDR). Use it when the diagnosis is uncertain or requires structural proof.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "The file path of the fundus image to analyze."
                    }
                },
                "required": ["image_path"]
            }
        }
    }
]