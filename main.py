import os
import argparse
import json
import sys

# Ensure python can find your folders
sys.path.append(os.getcwd())

from tools.vision import VisionTool
from tools.retrieval import PatientRetrieval

# --- CONFIGURATION ---
# Update these paths to match your cluster environment
MODEL_PATH = "checkpoints/best_fair_eye_model.pth" 
DATA_CSV_PATH = "data/dr_dataset_10k.csv"

def run_pipeline(image_path, patient_metadata):
    print(f"\n{'='*20} INITIALIZING PIPELINE {'='*20}")
    
    # 1. LOAD TOOLS
    # We initialize them here for the demo, but in production, 
    # you'd load them once outside the loop to save time.
    try:
        vision_tool = VisionTool(MODEL_PATH)
        retrieval_tool = PatientRetrieval(DATA_CSV_PATH)
    except Exception as e:
        print(f"[CRITICAL ERROR] Failed to load tools: {e}")
        return

    print(f"\n{'='*20} PROCESSING PATIENT {'='*20}")
    print(f"Patient Profile: {json.dumps(patient_metadata, indent=2)}")

    # 2. STEP A: VISION (The "Specialist")
    print(f"\n[Step A] Running Vision Tool on {os.path.basename(image_path)}...")
    vision_result = vision_tool.predict(image_path)
    
    print(f" >>> Vision Output: {json.dumps(vision_result, indent=2)}")

    # 3. STEP B: RETRIEVAL (The "Evidence")
    print(f"\n[Step B] Running Retrieval Tool...")
    evidence = retrieval_tool.retrieve_evidence(patient_metadata)
    
    print(f" >>> Evidence Output: {json.dumps(evidence, indent=2)}")

    # 4. STEP C: THE AGENT (Placeholder)
    print(f"\n[Step C] Agent Debate (COMING SOON)")
    print("In Phase 2, the LLM will compare Step A and Step B.")
    
    # Simple Heuristic Check for Dev
    if vision_result.get('probability', 0) < 0.1 and evidence.get('flag') == 'HIGH_RISK_GROUP':
        print("\n[PREVIEW] RESULT: This case would trigger an AGENT OVERRIDE.")
    else:
        print("\n[PREVIEW] RESULT: Consensus achieved (or insufficient evidence to override).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Equi-Agent Dev Pipeline")
    parser.add_argument("--image", type=str, required=True, help="Path to a test image")
    
    # We will simulate metadata via CLI or just hardcode a test case for now
    args = parser.parse_args()

    # MOCK PATIENT DATA (The "Hard" Case)
    # This represents a Black female patient where your model likely fails.
    test_patient = {
        "age": 68,
        "gender": "female",
        "race": "black"
    }

    if os.path.exists(args.image):
        run_pipeline(args.image, test_patient)
    else:
        print(f"Error: Image not found at {args.image}")