import os
import json
import pandas as pd
# Import the class you just shared
from agents.ophthalmic_agent import OphthalmicAgent 

# --- PATHS ---
EQUITY_STATS_PATH = "/home/ab575577/projects_spring_2026/OMEGA/RETFound/data/equity_stats.json"
HERO_JSON_PATH = "glaucoma_multimodal_failures/glaucoma_hero_metadata.json"
# Folder containing the original slo_fundus JPEGs
IMAGE_DIR = "/home/ab575577/projects_spring_2026/HarvardFairVision30K/FairVision/Glaucoma/Validation/"

def run_grant_evaluation():
    # 1. Load the knowledge bases
    if not os.path.exists(EQUITY_STATS_PATH):
        print(f"Error: {EQUITY_STATS_PATH} not found.")
        return

    with open(EQUITY_STATS_PATH, "r") as f:
        equity_stats = json.load(f)
    
    with open(HERO_JSON_PATH, "r") as f:
        hero_cases = json.load(f)

    # Initialize the Ophthalmic Agent
    agent = OphthalmicAgent()
    results = []

    print(f"🚀 Processing {len(hero_cases)} Failure Cases for Grant Evidence...")

    for case in hero_cases:
        # 2. Extract specific metadata for this patient
        patient_id = case['id']
        race = case['race']
        age = case['age']
        md_val = case['md']
        vision_prob = case['vision_prob']
        
        # 3. Pull demographic-specific clinical knowledge
        race_key = race.capitalize() # 'black' -> 'Black'
        risk_info = equity_stats['Glaucoma'].get(race_key, {})
        
        # 4. CONSTRUCT THE CLINICAL NOTE
        # This is the 'User Input' for the Agent
        clinical_note = (
            f"ID: {patient_id}. Patient is a {age}yo {race}. "
            f"Mean Deviation (MD) from visual field testing is {md_val}. "
            f"The primary screening model reported a probability of {vision_prob:.2%}."
        )
        
        # Inject the 'Hidden' Equity knowledge into the note
        if risk_info:
            clinical_note += (
                f"\n[DEMOGRAPHIC CONTEXT]: Risk Level {risk_info.get('risk_level')}. "
                f"{risk_info.get('clinical_note')} Action suggested: {risk_info.get('action')}."
            )

        # 5. Locate the Image
        # Based on your setup: slo_fundus_00001.jpg
        img_path = os.path.join(IMAGE_DIR, f"slo_fundus_{patient_id}.jpg")

        # 6. RUN AGENT CONSULTATION
        print(f"\n--- EVALUATING CASE {patient_id} ({race}) ---")
        print(f"Vision Expert Prob: {vision_prob:.2%}")
        
        try:
            agent_output = agent.consult(clinical_note, img_path)
            
            # Store comparative results
            results.append({
                "case_id": patient_id,
                "race": race,
                "vision_only_prob": vision_prob,
                "ground_truth": "Sick" if case['actual_label'] == 1 else "Healthy",
                "agent_reasoning_and_diagnosis": agent_output
            })
            
        except Exception as e:
            print(f"Failed to process {patient_id}: {e}")

    # 7. Export the "Grant Table"
    with open("grant_comparison_results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print("\n✅ Evaluation complete. Check 'grant_comparison_results.json' for the Agent's fixes.")

if __name__ == "__main__":
    run_grant_evaluation()