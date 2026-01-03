import torch
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
import models_vit 

# --- CONFIG ---
MODEL_PATH = "best_fair_eye_model.pth"
AUDIT_CSV = "equity_audit_table.csv"
DEMO_IMAGE = "demo_case.jpg"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 3

class EquiAgent:
    def __init__(self):
        print("Initializing Equi-Agent (Nature-Tier Logic)...")
        self.device = DEVICE
        self.audit_df = pd.read_csv(AUDIT_CSV)
        
        # 1. LOAD MODEL (The 'Biased' Vision Tool)
        print("Loading Vision Backbone...")
        try:
            # Using the specific function name we found earlier
            self.model = models_vit.RETFound_mae(
                img_size=224, 
                num_classes=NUM_CLASSES, 
                drop_path_rate=0.2, 
                global_pool=True
            ).to(self.device)
            
            checkpoint = torch.load(MODEL_PATH, map_location=self.device)
            state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
            self.model.load_state_dict(state_dict, strict=False)
            self.model.eval()
        except Exception as e:
            print(f"Model Load Error: {e}")
            print("Make sure you run this in the RETFound folder!")
            exit()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def run_inference(self, img_path):
        img = Image.open(img_path).convert('RGB')
        img_t = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(img_t)
            probs = torch.sigmoid(outputs).cpu().numpy()[0]
        return {'AMD': probs[0], 'DR': probs[1], 'Glaucoma': probs[2]}

    def check_bias(self, race, disease):
        """Looks up the 'Villain' stats from your CSV"""
        row = self.audit_df[(self.audit_df['Race'] == race) & (self.audit_df['Disease'] == disease)]
        if row.empty: return 0.0, "UNKNOWN"
        
        fnr = float(row.iloc[0]['FNR'])
        risk = "CRITICAL" if fnr > 0.80 else "HIGH" if fnr > 0.50 else "LOW"
        return fnr, risk

    def diagnose(self, img_path, patient_meta):
        print(f"\n===================================================")
        print(f"PATIENT CASE: {patient_meta['ID']} | Race: {patient_meta['Race']} | Age: {patient_meta['Age']}")
        print(f"===================================================")
        
        # STEP 1: VISION
        print(f"1. [VISION TOOL] Scanning Retina...")
        raw_results = self.run_inference(img_path)
        
        # STEP 2: REASONING LOOP
        for disease, prob in raw_results.items():
            print(f"\n--- Analyzing for {disease} ---")
            print(f"   > Raw Model Confidence: {prob:.1%}")
            
            # STEP 3: EQUITY AUDIT
            fnr, risk_level = self.check_bias(patient_meta['Race'], disease)
            print(f"   > [EQUITY TOOL] Bias Audit: Risk is {risk_level} (Hist. FNR: {fnr:.2f})")
            
            # STEP 4: DECISION LOGIC (The "Agent")
            decision = "Negative (Healthy)"
            color = "\033[92m" # Green
            
            # ADAPTIVE THRESHOLDING
            threshold = 0.50
            if risk_level == "CRITICAL":
                threshold = 0.15 # Massive correction for AMD/Black
                print(f"   > [PLANNER] ALERT: Applying Adaptive Threshold (0.50 -> 0.15)")
            elif risk_level == "HIGH":
                threshold = 0.30
                print(f"   > [PLANNER] WARN: Lowering Threshold (0.50 -> 0.30)")
            
            if prob >= threshold:
                decision = "POSITIVE (REFER)"
                color = "\033[91m" # Red
                
            print(f"   > FINAL DIAGNOSIS: {color}{decision}\033[0m")
            
            if risk_level == "CRITICAL" and prob < 0.5 and prob >= 0.15:
                print(f"   *** AGENT INTERVENTION: Standard model missed this! Agent caught it. ***")

if __name__ == "__main__":
    agent = EquiAgent()
    
    # Run the "Villain" Scenario (Black Patient with AMD)
    # We extracted this image specifically because we know it's an AMD case
    agent.diagnose(DEMO_IMAGE, {"ID": "B-999", "Race": "Black", "Age": 72})