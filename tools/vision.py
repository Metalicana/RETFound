import torch
from torchvision import transforms
from PIL import Image
import sys
import os

# --- PATH HACK: Allow importing models_vit from the parent directory ---
# This tells Python to look one folder up (RETFound/) for modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import models_vit
except ImportError:
    print("\n[CRITICAL WARNING] 'models_vit.py' not found in the project root.")
    print("Please ensure this script is inside 'RETFound/tools/' and models_vit.py is in 'RETFound/'\n")

class VisionTool:
    def __init__(self, model_path, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[VisionTool] Loading weights from {model_path}...")

        # 1. DEFINE ARCHITECTURE
        # Must match your training script exactly (3 classes, global_pool=True)
        self.model = models_vit.RETFound_mae(
            img_size=224,
            num_classes=3,  # [AMD, DR, Glaucoma]
            drop_path_rate=0,
            global_pool=True,
        )

        # 2. LOAD WEIGHTS
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            # Handle standard vs DDP state_dicts
            state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
            
            # Clean keys if they have "module." prefix
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            
            # Load
            msg = self.model.load_state_dict(state_dict, strict=False)
            # print(f"[VisionTool] Missing keys (expected for fine-tuning): {len(msg.missing_keys)}")
            
            self.model.to(self.device)
            self.model.eval()
            print("[VisionTool] Model loaded successfully.")
            
        except FileNotFoundError:
            print(f"[VisionTool] ERROR: Model file not found at {model_path}")
            # We don't crash here so you can debug imports, but inference will fail later
        except Exception as e:
            print(f"[VisionTool] ERROR loading state_dict: {e}")

        # 3. DEFINE TRANSFORMS (Standard ImageNet)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image_path):
        """
        Input: Path to fundus image
        Output: Dictionary of probabilities
        """
        try:
            # Load and Preprocess
            img = Image.open(image_path).convert('RGB')
            img_t = self.transform(img).unsqueeze(0).to(self.device)

            # Inference
            with torch.no_grad():
                logits = self.model(img_t)
                probs = torch.sigmoid(logits).cpu().numpy()[0]

            # Map outputs based on your training order: [AMD, DR, Glaucoma]
            return {
                "AMD": float(probs[0]),
                "DR": float(probs[1]),       # Expect this to be low (the bug)
                "Glaucoma": float(probs[2]),
                "status": "success"
            }

        except Exception as e:
            return {
                "status": "error", 
                "message": str(e),
                "AMD": 0.0, "DR": 0.0, "Glaucoma": 0.0
            }

# Quick Test (Run this file directly to check imports)
if __name__ == "__main__":
    # Create a dummy model path to test import logic
    print("Testing VisionTool imports...")
    tool = VisionTool("checkpoints/best_fair_eye_model.pth")