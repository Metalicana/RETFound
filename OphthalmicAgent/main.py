# OphthalmicAgent/main.py

from Orchestrator.state import AgentState
from BioProfilerAgent.bio_profiler import BioProfiler
from EquityAgent.equity_agent import EquityAgent
from VisionAgent.vision import VisionSpecialist
from VisionAgent.linear_probing_oct3 import FairVisionNPZ
from FunctionalInterpretationAgent.function_interpreter import FunctionalSpecialist
from GuidelinesAgent.guidelines_agent import GuidelinesAgent
from SafetyAgent.safety_agent import SafetyAgent
from Orchestrator.ophthalmic_agent import Orchestrator

from data.loader import GenericEyeLoader

import numpy as np
from PIL import Image
import pandas as pd
import json
import re

OUTPUT_CSV = "ophthalmic_performance_results_apr20.csv"

#Intiializing agents
profiler = BioProfiler()
equity_agent = EquityAgent()
vision_agent = VisionSpecialist("./weights/oct_model_best.pth", "./weights/slo_model_best.pth")
functional_agent = FunctionalSpecialist()
ophthalmic_agent = Orchestrator()
safety_agent = SafetyAgent()
guidelines_agent = GuidelinesAgent()

def run_diagnostic_pipeline(patient_data):
    
    #BIO PROFILER AGENT
    print("\n\n--- Sending Meta Data to Bio Profiler Agent ---")
    final_state["clinical_narrative"] = profiler.generate_narrative(final_state["metadata"])
    print("\n\nNarrative Ready: ")
    print(f"{final_state['clinical_narrative']}") 
    print("\n" + "-"*30)
    
    #VISION AGENT
    print("\n\n--- Sending Visual Data to Vision Specialist ---")
    final_state["vision_opinion"] = vision_agent.analyze(patient_data['oct_img'], patient_data['fundus_img'], final_state)
    print("\n\nProbabilities Ready: ")
    print("\n\nOCT DIAGNOSIS USING FINETUNED RETFOUND")
    oct_diagnosis = final_state['oct_diagnosis']
    stage = oct_diagnosis['AMD']['Predicted_Stage']
    if(stage == 3):
      amd_probability = oct_diagnosis['AMD']['Advanced'] * 100
    elif(stage == 2):
      amd_probability = oct_diagnosis['AMD']['Intermediate'] * 100
    elif(stage == 1):
      amd_probability = oct_diagnosis['AMD']['Early'] * 100
    else:
      amd_probability = 50
    
    retfound_scores = f"AMD Diagnosis: Stage {oct_diagnosis['AMD']['Predicted_Stage']}, Probability: {amd_probability}%\nDR Risk: {oct_diagnosis['DR']['Probability'] * 100:.2f}% \nGlaucoma Risk: {oct_diagnosis['Glaucoma']['Probability'] * 100:.2f}%"
    print(retfound_scores)
    
    print("\nSLO DIAGNOSIS USING FINETUNED MIRAGE")
    slo_diagnosis = final_state['slo_diagnosis']
    stage = slo_diagnosis['AMD']['Predicted_Stage']
    if(stage == 3):
      amd_probability = slo_diagnosis['AMD']['Advanced'] * 100
    elif(stage == 2):
      amd_probability = slo_diagnosis['AMD']['Intermediate'] * 100
    elif(stage == 1):
      amd_probability = slo_diagnosis['AMD']['Early'] * 100
    else:
      amd_probability = 50
    
    mirage_scores = f"AMD Diagnosis: Stage {slo_diagnosis['AMD']['Predicted_Stage']}, Probability: {amd_probability}% \nDR Risk: {slo_diagnosis['DR']['Probability'] * 100:.2f}% \nGlaucoma Risk: {slo_diagnosis['Glaucoma']['Probability'] * 100:.2f}%"
    print(mirage_scores)
    
    print("\n\nVision Specialist's Output Ready: ")
    print(f"{final_state['vision_opinion']['summary']}")
    
    print("\n" + "-"*30)
    
    #FUNCTIONAL INTERPRETATION AGENT
    print("\n\n--- Sending Visual Field Data to Functional Interpretation Agent ---")
    final_state["functional_opinion"] = functional_agent.analyze(final_state)
    print("\n\nFunctional Vision Interpreter's Output Ready: ")
    print(f"{final_state['functional_opinion']['summary']}")

    print("\n" + "-"*30)   
    
    #EQUITY AGENT
    print("\n\n--- Sending Narrative and Visual Findings to Equity Agent ---")
    equity_input = f"""
### PATIENT DATA 
Patient narrative: {final_state['clinical_narrative']}
    
### AI MODEL OUTPUTS
Model Probabilities - RETFOUND: 
{retfound_scores} 
    
Model Probabilities - MIRAGE: 
{mirage_scores}"""

    final_state["equity_opinion"] = equity_agent.analyze_patients(equity_input, output_format="text")
    print("\n\nEquity Agent's Output Ready: ")
    print(f"{final_state['equity_opinion']}") 
    print("\n" + "-"*30)
    
    #GUIDELINES AGENT
    print("\n\n--- Sending Query to Guidelines Agent ---")
    note = f"""
Patient Narrative: {final_state['clinical_narrative']} 
Functional Interpretation Agent output: {final_state['functional_opinion']['summary']}"""

    final_state['guidelines'] = guidelines_agent.consult_note(note, max_results=5, diagnosis_only=True,)
    print("\n\n Guidelines Agent's Output   Ready: ")
    print(f"{final_state['guidelines']}")
    
    print("\n" + "-"*30) 
    
    #ORCHESTRATOR
    print("\n\n--- Sending Case to Orchestrator ---")
    final_state["final_diagnosis"] = ophthalmic_agent.analyze(final_state)
    print("\n\n Ophthalmic Agent's Final Diagnosis Ready: ")
    print(f"{final_state['final_diagnosis']['decision']}")
    print("\n" + "-"*30)  
    
    #SAFETY AGENT
    print("\n\n--- Sending Case to Safety Agent ---")
    final_state["safety_output"] = safety_agent.run(final_state)
    print("\n\n Safety Agent's Output Ready: ")
    print(f"{final_state['safety_output']}")
    print("\n" + "-"*30)  
    
def parse_agent_labels(final_state):
    label_text = final_state['final_diagnosis']['labels']
    
    patterns = {
        "AMD": r"AMD_STAGE:\s*(\d)",
        "DR": r"DR_DETECTED:\s*(\d)",
        "GL": r"GLAUCOMA_DETECTED:\s*(\d)"
    }
    
    results = {}
    
    try:
        for key, pattern in patterns.items():
            match = re.search(pattern, label_text, re.IGNORECASE)
            if match:
                results[key] = int(match.group(1))
            else:
                print(f"Warning: Could not find {key} in label text!")
                results[key] = 0 # Default fallback
        return results
    except Exception as e:
        print(f"Parsing Error: {e}")
        return None
        
def initialize_state(patient_data):
    # Initialize the State
    state: AgentState = {
        "patient_id": patient_data['metadata']['filename'],
        "metadata": patient_data['metadata'],
        "fundus_img": np.array(patient_data['fundus_img']),
        "oct_img": np.array(patient_data['oct_img']),
        "clinical_narrative": "",
        "oct_diagnosis": None,
        "slo_diagnosis": None,
        "vision_opinion": {},
        "functional_opinion": {},
        "final_diagnosis": {},
        "fairness_flag": False,
        "safety_output": "",
        "guidelines": "",
        "equity_opinion:": ""
    }
    
    Image.fromarray(state['oct_img']).save("check_my_work.png")
    Image.fromarray(state['fundus_img']).save("check_my_work2.png")
    return state
    
if __name__ == "__main__":

    results = []
    
    BASE_PATH = "data/"
    diseases = ['Glaucoma', 'AMD', 'DR']
    
    for disease in diseases: 
      loader = GenericEyeLoader(BASE_PATH)  
      df = loader.get_metadata(disease)
      
      test_rows = df[df['use'] == 'test']
      
      if not test_rows.empty: 
          for i in range(1):     
          
            try: 
              patient_record = loader.load_patient(disease, test_rows.iloc[i])
        
              final_state = initialize_state(patient_record)
              run_diagnostic_pipeline(patient_record)
              
              pred_labels = parse_agent_labels(final_state)
  
              ground_truth = patient_record['stage']
              
              row = {
                    "Filename": patient_record['directory'],
                    "Task_Folder": disease,
                    "Ground_Truth": ground_truth,
                    "Pred_AMD": pred_labels["AMD"] if pred_labels else -1,
                    "Pred_DR": pred_labels["DR"] if pred_labels else -1,
                    "Pred_GL": pred_labels["GL"] if pred_labels else -1,
                    }
                    
              if "AMD" in disease:
                  # Grades the stage (0-3)
                  row["Is_Correct"] = (pred_labels["AMD"] == ground_truth)
              
              elif "DR" in disease:
                  # Grades binary detection (0 or 1)
                  row["Is_Correct"] = (pred_labels["DR"] == ground_truth)
              
              elif "Glaucoma" in disease:
                  # Grades binary detection (0 or 1)
                  row["Is_Correct"] = (pred_labels["GL"] == ground_truth)
              
              results.append(row)
              
              df = pd.DataFrame(results)
              df.to_csv(OUTPUT_CSV, index=False)
            
              print(f"\nDisease folder is {disease} and ground truth is {ground_truth}, example number is {i}")
              print("\nEND OF EXAMPLE")
              print("\n" + "-"*30)
              
            except Exception as e:
              # Catching the content filter specifically
              print(f"!!! Error at Index {i}. Skipping...")
              row = {
                    "Filename": patient_record['directory'],
                    "Task_Folder": disease,
                    "Ground_Truth": -1,
                    "Pred_AMD": -1,  
                    "Pred_DR": -1,    
                    "Pred_GL": -1,    
                    "Is_Correct": "N/A" # 
                  }
              results.append(row)
              
              df = pd.DataFrame(results)
              df.to_csv(OUTPUT_CSV, index=False)    
            
      else:
          print("No test data found to process.")