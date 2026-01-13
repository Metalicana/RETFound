import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix

# --- IMPORTS (Adjust if your file structure is different) ---
from tools.vision import VisionTool
from agents.ophthalmic_agent import OphthalmicAgent 
# We reuse the dataset class you wrote earlier
from train_vision import FairVisionNPZ 

# --- CONFIGURATION ---
DATA_ROOT = "/home/ab575577/projects_spring_2026/HarvardFairVision30K/FairVision"
TEST_LIMIT = 100  # Total patients to eval (Save money during dev)
OUTPUT_FILE = "results_fairness_audit.csv"

# --- HELPER: CONVERT AGENT OUTPUT TO BINARY ---
def parse_agent_decision(response_text):
    """
    Parses the LLM's natural language output into a binary label.
    Expected LLM output contains: "FINAL_DECISION: REFER" or "FINAL_DECISION: DISMISS"
    """
    text = response_text.lower()
    if "refer" in text or "urgent" in text or "high risk" in text:
        return 1
    elif "dismiss" in text or "observe" in text or "healthy" in text:
        return 0
    else:
        return 0 # Default to negative if unclear (conservative)

def run_evaluation():
    print(f"--- STARTING FAIRNESS EVALUATION (N={TEST_LIMIT}) ---")
    
    # 1. SETUP
    # Load the Raw Vision Tool (The Baseline)
    vision_baseline = VisionTool(model_path="checkpoints/best_fair_eye_model.pth")
    
    # Load the Agent (The Proposed Solution)
    # We tweak the agent to be more concise for eval
    agent = OphthalmicAgent()
    
    # Load Dataset (Test Split)
    # We assume 'amd' for this test, but you can loop through diseases
    test_ds = FairVisionNPZ(DATA_ROOT, split='Validation', transform=None) 
    
    # Stratified Sampling: We want equal mix of races for the audit
    # In a real run, use the full test set. For dev, we pick manually.
    indices = list(range(len(test_ds)))
    np.random.shuffle(indices)
    subset_indices = indices[:TEST_LIMIT]
    
    results = []

    print("Running Inference...")
    for i in tqdm(subset_indices):
        try:
            # A. GET DATA
            # We need raw access to the file path for the agent
            item = test_ds.files[i] 
            image_path = item['path']
            
            # Load metadata manually (since dataset __getitem__ returns tensors)
            # This is a hack to get the metadata text for the prompt
            data = np.load(image_path)
            
            # Extract Truth
            # Logic: If ANY disease label is 1, patient is Sick (1)
            # You might need to adjust this depending on which disease you are auditing
            is_sick = 0
            # Check AMD
            amd_cond = str(data['amd_condition']) if 'amd_condition' in data else ''
            if test_ds.amd_map.get(amd_cond, 0.) == 1.0: is_sick = 1
            
            # Check Race
            race_code = int(data['race']) if 'race' in data else 0
            race_map = {1: "White", 2: "Black", 3: "Asian", 4: "Hispanic"} # Verify this map with your README
            race_str = race_map.get(race_code, "Unknown")
            
            # Check Age
            age = float(data['age']) if 'age' in data else 65.0
            gender = "Female" if data.get('gender', 0) == 0 else "Male"

            # B. RUN BASELINE (Vision Only)
            # We use the raw probability from the tool
            vis_out = vision_baseline.predict(image_path)
            # Assume we are testing AMD for this example (index 0)
            # If testing DR, use index 1.
            vis_prob = vis_out["amd_prob"] 
            vis_pred = 1 if vis_prob > 0.5 else 0

            # C. RUN AGENT (The Intervention)
            # Construct the clinical note
            note = f"{age} year old {race_str} {gender}. Screening for AMD."
            
            # We inject a strict instruction for parsing
            prompt_suffix = "\nProvide your final answer as exactly 'FINAL_DECISION: REFER' or 'FINAL_DECISION: DISMISS'."
            
            # Call Agent (This costs money!)
            agent_response = agent.consult(note + prompt_suffix, image_path)
            agent_pred = parse_agent_decision(agent_response)

            # D. LOG RESULT
            results.append({
                "race": race_str,
                "ground_truth": is_sick,
                "baseline_prob": vis_prob,
                "baseline_pred": vis_pred,
                "agent_pred": agent_pred,
                "agent_reasoning": agent_response[:50] + "..." # Save snippet
            })

        except Exception as e:
            print(f"Skipped index {i}: {e}")

    # 3. CALCULATE METRICS
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved detailed results to {OUTPUT_FILE}")

    # Fairness Audit Table
    print("\n=== FAIRNESS AUDIT REPORT ===")
    for race in df['race'].unique():
        sub = df[df['race'] == race]
        if len(sub) == 0: continue
        
        # Baseline Metrics
        base_acc = accuracy_score(sub['ground_truth'], sub['baseline_pred'])
        # Agent Metrics
        agent_acc = accuracy_score(sub['ground_truth'], sub['agent_pred'])
        
        # False Negative Rate (The Killer Metric)
        # sick_patients = sub[sub['ground_truth'] == 1]
        # base_fnr = 1.0 - accuracy_score(sick_patients['ground_truth'], sick_patients['baseline_pred']) if len(sick_patients) > 0 else 0
        
        print(f"[{race.upper()}] (n={len(sub)})")
        print(f"  Baseline Accuracy: {base_acc:.2%}")
        print(f"  Agent Accuracy:    {agent_acc:.2%}  <-- DELTA: {agent_acc - base_acc:+.1%}")
        print("-" * 30)

if __name__ == "__main__":
    run_evaluation()