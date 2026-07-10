########################################################################################### AMD (to be) #################################################


## OphthalmicAgent/main.py
#
##change 5
#
#from Orchestrator.state import AgentState
#from BioProfilerAgent.bio_profiler import BioProfiler
##from EquityAgent.equity_agent import EquityAgent
#from EquityAgent.compute_demographic_reliability_score import main
#from VisionAgent.vision import VisionSpecialist
#
#from VisionAgent.vision_oct import VisionSpecialistOct
#from VisionAgent.vision_slo import VisionSpecialistSlo
#
#from VisionAgent.linear_probing_oct3 import FairVisionNPZ
#from FunctionalInterpretationAgent.function_interpreter import FunctionalSpecialist
#from GuidelinesAgent.guidelines_agent import GuidelinesAgent
#from SafetyAgent.safety_agent import SafetyAgent
#from Orchestrator.new import Orchestrator
#
#from data.loader import ExcelEyeLoader
#
#from types import SimpleNamespace
#from pathlib import Path
#
#import numpy as np
#from PIL import Image
#import pandas as pd
#import json
#import re
#
#OUTPUT_CSV = "ophthalmic_performance_results_jun29_dr.csv" #CHANGE
#
##Intiializing agents
#profiler = BioProfiler()
##equity_agent = EquityAgent()
##vision_agent = VisionSpecialist("./weights/oct_model_best_all_binary.pth", "./weights/slo_model_best_all_binary.pth")
#
#vision_agent_oct = VisionSpecialistOct("./weights/oct_model_8_slices_not_center.pth")
#vision_agent_slo = VisionSpecialistSlo("./weights/oct_model_best_all_binary.pth", "./weights/slo_model_best_all_binary.pth")
#
#functional_agent = FunctionalSpecialist()
#ophthalmic_agent = Orchestrator()
#safety_agent = SafetyAgent()
#guidelines_agent = GuidelinesAgent()
#
#def run_diagnostic_pipeline(patient_data):
#
#    final_state["retfound_scores"] = vision_agent_oct.analyze(patient_data['oct_img'], final_state)
##    probability = final_state['oct_diagnosis']['Glaucoma']['Prob_Pct']
##    probability = final_state['oct_diagnosis']['AMD']['Prob_Pct']
#    probability = final_state['oct_diagnosis']['DR']['Prob_Pct'] #CHANGE
#    
#    print("\n\nProbabilities Ready: ")
#    print("\nOCT DIAGNOSIS USING FINETUNED RETFOUND")
#    retfound_scores = final_state["retfound_scores"] 
#    print(retfound_scores)
#    
#    if probability >= 90:
##        final_state['final_diagnosis']['labels'] = "GLAUCOMA_DETECTED: 1"
##        final_state['final_diagnosis']['labels'] = "AMD_DETECTED: 1"
#        final_state['final_diagnosis']['labels'] = "DR_DETECTED: 1" #CHANGE
#    elif probability <= 10:
##        final_state['final_diagnosis']['labels'] = "GLAUCOMA_DETECTED: 0"
##        final_state['final_diagnosis']['labels'] = "AMD_DETECTED: 0"
#        final_state['final_diagnosis']['labels'] = "DR_DETECTED: 0" #CHANGE
#    else:
#    
#        print("STARTING AGENTIC PIPELINE")
#        #BIO PROFILER AGENT
#        print("\n\n--- Sending Meta Data to Bio Profiler Agent ---")
#        final_state["clinical_narrative"] = profiler.generate_narrative(final_state["metadata"])
#        print("\n\nNarrative Ready: ")
#        print(f"{final_state['clinical_narrative']}") 
#        print("\n" + "-"*30)
#        
#        #VISION AGENT
#        print("\n\n--- Sending Visual Data to Vision Specialists ---")
#
#        final_state["vision_opinion_slo"], v_cdr = vision_agent_slo.analyze(final_state['fundus_img'], final_state)   
#        print("\n\nVision Specialist SLO's Output Ready: ")
#        print(f"{final_state['vision_opinion_slo']}")
#        print("\n" + "-"*30)
#          
#        print("\n\n--- Calculating Model Trust Score ---")
#        if final_state["metadata"]['Age'] >= 60:
#          age_group = "older"
#        elif final_state["metadata"]['Age'] < 40:
#          age_group = "younger"
#        else:
#          age_group = "middle-aged"
#        
#        
#        args = SimpleNamespace(
#            subgroup_reliability_csv=Path("/lustre/fs1/home/yu395012/RETFound/OphthalmicAgent/_extras/CSVs/demographic_reliability_subgroup_model_scores.csv"),
#            task=final_state["metadata"]["Task_Folder"].lower(),
#            model="retfound_oct",
#            age_group=age_group,
#            race=final_state["metadata"]["Race"],
#            gender=final_state["metadata"]["Gender"],
#            score_column="final_R_bad",
#            json=False,
#            priors_json=None,
#            support_csv=None,
#            k=50.0,
#            fnr_weight=0.35,
#            fpr_weight=0.25,
#            ece_weight=0.15,
#            auroc_weight=0.15,
#            f1_weight=0.10,
#            raw_local_lambdas=False,
#        )
#        
#        trust_score = 1 - main(args)
#        
#        print(f"Trust Score is: {trust_score}")
#        
#        #ORCHESTRATOR
#        print("\n\n--- Sending Case to Orchestrator ---")
#        final_state["final_diagnosis"] = ophthalmic_agent.analyze(final_state, probability, v_cdr, trust_score)
#        print("\n\n Ophthalmic Agent's Final Diagnosis Ready: ")
#        print(f"{final_state['final_diagnosis']['decision']}")
#        print("\n" + "-"*30)  
#
##    #FUNCTIONAL INTERPRETATION AGENT
##    print("\n\n--- Sending Visual Field Data to Functional Interpretation Agent ---")
##    final_state["functional_opinion"] = functional_agent.analyze(final_state)
##    print("\n\nFunctional Vision Interpreter's Output Ready: ")
##    print(f"{final_state['functional_opinion']['summary']}")
##    print("\n" + "-"*30)   
#    
##    #EQUITY AGENT
##    print("\n\n--- Sending Narrative and Visual Findings to Equity Agent ---")
##    equity_input = f"""
##    ### PATIENT DATA 
##    Patient narrative: {final_state['clinical_narrative']}
##        
##    ### AI MODEL OUTPUTS
##    {retfound_scores} 
##         
##    {mirage_scores}"""
##
##    final_state["equity_opinion"] = equity_agent.analyze_patients(equity_input, output_format="text")
##    print("\n\nEquity Agent's Output Ready: ")
##    print(f"{final_state['equity_opinion']}") 
##    print("\n" + "-"*30)
#      
#
#        
##    #GUIDELINES AGENT
##    print("\n\n--- Sending Query to Guidelines Agent ---")
##    note = f"""
##    Patient Narrative: {final_state['clinical_narrative']} 
##    Functional Interpretation Agent output: {final_state['functional_opinion']['summary']}"""
##
##    final_state['guidelines'] = guidelines_agent.consult_note(note, max_results=5, diagnosis_only=True,)
##    print("\n\n Guidelines Agent's Output   Ready: ")
##    print(f"{final_state['guidelines']}")
##    
##    print("\n" + "-"*30) 
##    #SAFETY AGENT
##    print("\n\n--- Sending Case to Safety Agent ---")
##    final_state["safety_output"] = safety_agent.run(final_state)
##    print("\n\n Safety Agent's Output Ready: ")
##    print(f"{final_state['safety_output']}")
##    print("\n" + "-"*30)  
#    
#def parse_agent_labels(final_state):
#    label_text = final_state['final_diagnosis']['labels']
#    
#    print(f"Label Text {label_text}")
#    
#    patterns = {
#    "AMD": r"AMD_DETECTED:\s*(.+)",
#    "DR": r"DR_DETECTED:\s*(.+)",
#    "GL": r"GLAUCOMA_DETECTED:\s*(.+)"
#    }
#    
#    results = {}
#    
#    for key, pattern in patterns.items():
#        match = re.search(pattern, label_text, re.IGNORECASE)
#        if match:
#            value_match = re.search(r"-?\d+", match.group(1))
#            results[key] = int(value_match.group()) if value_match else -1
#        else:
#            results[key] = -1
#    
#    return results
#       
#def initialize_state(patient_data):
#    # Initialize the State
#    state: AgentState = {
#        "patient_id": patient_data['metadata']['filename'],
#        "metadata": patient_data['metadata'],
#        "fundus_img": np.array(patient_data['fundus_img']),
#        "oct_img": np.array(patient_data['oct_img']),
#        "clinical_narrative": "",
#        "oct_diagnosis": None,
#        "slo_diagnosis": None,
#        "vision_opinion": {},
#        
#        "vision_opinion_oct": {},
#        "vision_opinion_slo": {},
#        
#        "functional_opinion": {},
#        "final_diagnosis": {},
#        "fairness_flag": False,
#        "safety_output": "",
#        "guidelines": "",
#        "equity_opinion:": ""
#    }
#
#    return state
#    
#if __name__ == "__main__":
#    
#    EXCEL_PATH = "./data/fairvision_250each.csv"
#    
#    results = []
#    
#    # Initialize the updated Excel Loader
#    loader = ExcelEyeLoader(EXCEL_PATH)
#    
#    # Target conditions to run evaluation blocks on
##    diseases = ['Glaucoma', 'AMD', 'DR']
##    diseases = ['Glaucoma']
##    diseases = ['AMD']
#    diseases = ['DR'] #CHANGE
#    
#    for disease in diseases:
#        # Pull up to 250 records from the master Excel per disease category
#        test_rows = loader.get_records_by_disease(disease, limit=250)
#        
#        if not test_rows.empty:
#            # Iterate through every row explicitly returned by the Excel filter block
#            for index, current_row in test_rows.iterrows():
#                
#                # Pre-extract basic row demographics for error-handling resilience
#                # Safe checking for case-variations using dictionary transformations
#                row_dict = {k.lower(): v for k, v in current_row.to_dict().items()}
#                
#                age = row_dict.get('age', 'Unknown')
#                gender = row_dict.get('gender', 'Unknown')
#                race = row_dict.get('race', 'Unknown')
#                ethnicity = row_dict.get('ethnicity', 'Unknown')
#                
#                patient_record = None
#                
#                try:
#                    # Load structural files and map binary ground truth via row paths
#                    patient_record = loader.load_patient_from_excel_row(current_row)
#                    
#                    print(list(patient_record['metadata'].keys()))
#                    # Core pipeline logic execution
#                    final_state = initialize_state(patient_record)
#                    temp = patient_record['metadata']['filename']
#                        
##                    print(f"patient id: {temp}")
##                    if temp == "data/Glaucoma/Test/data_07010.npz": 
##                      Image.fromarray(np.array(patient_record['oct_img'])).convert('RGB').save("oct_slice.png")
##                      Image.fromarray(np.array(patient_record['fundus_img'])).convert('RGB').save("slo.png")
##                      
##                      continue
##      
#                    
#                    run_diagnostic_pipeline(patient_record)
#                    
#                    pred_labels = parse_agent_labels(final_state)
#                    ground_truth = patient_record['stage']
#                    
#                    row = {
#                        "Filename": patient_record['directory'],
#                        "Task_Folder": disease,
#                        "Age": age,
#                        "Gender": gender,
#                        "Race": race,
#                        "Ethnicity": ethnicity,
#                        "Ground_Truth": ground_truth,
#                        "Pred_AMD": pred_labels.get("AMD", -1) if pred_labels else -1,
#                        "Pred_DR": pred_labels.get("DR", -1) if pred_labels else -1,
#                        "Pred_GL": pred_labels.get("GL", -1) if pred_labels else -1,
#                    }
#                    
#                    # Select corresponding validation prediction
#                    if "AMD" in disease:
#                        pred = row["Pred_AMD"]
#                    elif "DR" in disease:
#                        pred = row["Pred_DR"]
#                    elif "Glaucoma" in disease:
#                        pred = row["Pred_GL"]
#                    else:
#                        pred = -1
#                    
#                    # Compute accuracy profiles
#                    if pred == -1 or ground_truth == -1:
#                        row["Is_Correct"] = -1
#                    else:
#                        row["Is_Correct"] = int(pred == ground_truth)
#                    
#                    results.append(row)
#                    
#                    # Inline checkpoint auto-saving
#                    out_df = pd.DataFrame(results)
#                    out_df.to_csv(OUTPUT_CSV, index=False)
#                    
#                    print(f"\nDisease: {disease} | Ground Truth: {ground_truth} | Row Index Checked: {index}")
#                    print("END OF EXAMPLE\n" + "-"*30)
#                      
#                except Exception as e:
#                      print(f"!!! Error processing row index {index} in {disease}. Skipping... Details: {e}")
#                      
#                      # Safely map fallback structural variables during exception failures
#                      fallback_filename = patient_record['directory'] if patient_record else row_dict.get('filepath', 'Error')
#                      fallback_gt = patient_record['stage'] if patient_record else row_dict.get('ground_truth', -1)
#                      
#                      row = {
#                          "Filename": fallback_filename,
#                          "Task_Folder": disease,
#                          "Age": age,
#                          "Gender": gender,
#                          "Race": race,
#                          "Ethnicity": ethnicity,
#                          "Ground_Truth": fallback_gt,
#                          "Pred_AMD": -1,  
#                          "Pred_DR": -1,    
#                          "Pred_GL": -1,    
#                          "Is_Correct": -1
#                      }
#                      results.append(row)
#                      
#                      out_df = pd.DataFrame(results)  
#                      out_df.to_csv(OUTPUT_CSV, index=False)    
#                    
#        else:
#            print(f"No custom Excel rows found to process for: {disease}")
##
##################################################################################################### DR #################################################
#
#
## OphthalmicAgent/main.py
#
##change 5
#
#from Orchestrator.state import AgentState
#from BioProfilerAgent.bio_profiler import BioProfiler
##from EquityAgent.equity_agent import EquityAgent
#from EquityAgent.compute_demographic_reliability_score import main
#from VisionAgent.vision import VisionSpecialist
#
#from VisionAgent.vision_oct import VisionSpecialistOct
#from VisionAgent.vision_slo import VisionSpecialistSlo
#
#from VisionAgent.linear_probing_oct3 import FairVisionNPZ
#from FunctionalInterpretationAgent.function_interpreter import FunctionalSpecialist
#from GuidelinesAgent.guidelines_agent import GuidelinesAgent
#from SafetyAgent.safety_agent import SafetyAgent
#from Orchestrator.new import Orchestrator
#
#from data.loader import ExcelEyeLoader
#
#from types import SimpleNamespace
#from pathlib import Path
#
#import numpy as np
#from PIL import Image
#import pandas as pd
#import json
#import re
#
#OUTPUT_CSV = "ophthalmic_performance_results_jul04_dr_1.csv" #CHANGE
#
##Intiializing agents
#profiler = BioProfiler()
##equity_agent = EquityAgent()
##vision_agent = VisionSpecialist("./weights/oct_model_best_all_binary.pth", "./weights/slo_model_best_all_binary.pth")
#
#vision_agent_oct = VisionSpecialistOct("./weights/oct_model_8_slices_not_center.pth")
#vision_agent_slo = VisionSpecialistSlo("./weights/oct_model_best_all_binary.pth", "./weights/slo_model_best_all_binary.pth")
#
#functional_agent = FunctionalSpecialist()
#ophthalmic_agent = Orchestrator()
#safety_agent = SafetyAgent()
#guidelines_agent = GuidelinesAgent()
#
#def run_diagnostic_pipeline(patient_data):
#
#    final_state["retfound_scores"], final_state["vision_opinion_oct"] = vision_agent_oct.analyze(patient_data['oct_img'], patient_data['middle_oct'], final_state)
##    probability = final_state['oct_diagnosis']['Glaucoma']['Prob_Pct']
##    probability = final_state['oct_diagnosis']['AMD']['Prob_Pct']
#    probability = final_state['oct_diagnosis']['DR']['Prob_Pct'] #CHANGE
#    
#    print("\n\nProbabilities Ready: ")
#    print("\nOCT DIAGNOSIS USING FINETUNED RETFOUND")
#    retfound_scores = final_state["retfound_scores"] 
#    print(retfound_scores)
#    
#    if probability >= 90:
##        final_state['final_diagnosis']['labels'] = "GLAUCOMA_DETECTED: 1"
##        final_state['final_diagnosis']['labels'] = "AMD_DETECTED: 1"
#        final_state['final_diagnosis']['labels'] = "DR_DETECTED: 1" #CHANGE
#    elif probability <= 10:
##        final_state['final_diagnosis']['labels'] = "GLAUCOMA_DETECTED: 0"
##        final_state['final_diagnosis']['labels'] = "AMD_DETECTED: 0"
#        final_state['final_diagnosis']['labels'] = "DR_DETECTED: 0" #CHANGE
#    else:
#    
#      print("STARTING AGENTIC PIPELINE")
#      #BIO PROFILER AGENT
#      print("\n\n--- Sending Meta Data to Bio Profiler Agent ---")
#      final_state["clinical_narrative"] = profiler.generate_narrative(final_state["metadata"])
#      print("\n\nNarrative Ready: ")
#      print(f"{final_state['clinical_narrative']}") 
#      print("\n" + "-"*30)
#      
#      #VISION AGENT
#      print("\n\n--- Sending Visual Data to Vision Specialists ---")
#  
#      final_state["vision_opinion_slo"], v_cdr = vision_agent_slo.analyze(final_state['fundus_img'], final_state)   
#      print("\n\nVision Specialist SLO's Output Ready: ")
#      print(f"{final_state['vision_opinion_slo']}")
#      print("\n" + "-"*30)
#       
#      
#      print("\n\nVision Specialist OCT's Output Ready: ")
#      print(f"{final_state['vision_opinion_oct']}")
#      print("\n" + "-"*30)
#        
#      print("\n\n--- Calculating Model Trust Score ---")
#      if final_state["metadata"]['Age'] >= 60:
#        age_group = "older"
#      elif final_state["metadata"]['Age'] < 40:
#        age_group = "younger"
#      else:
#        age_group = "middle-aged"
#      
#        
#      args = SimpleNamespace(
#          subgroup_reliability_csv=Path("/lustre/fs1/home/yu395012/RETFound/OphthalmicAgent/_extras/CSVs/demographic_reliability_subgroup_model_scores.csv"),
#          task=final_state["metadata"]["Task_Folder"].lower(),
#          model="retfound_oct",
#          age_group=age_group,
#          race=final_state["metadata"]["Race"],
#          gender=final_state["metadata"]["Gender"],
#          score_column="final_R_bad",
#          json=False,
#          priors_json=None,
#          support_csv=None,
#          k=50.0,
#          fnr_weight=0.35,
#          fpr_weight=0.25,
#          ece_weight=0.15,
#          auroc_weight=0.15,
#          f1_weight=0.10,
#          raw_local_lambdas=False,
#      )
#      
#      trust_score = 1 - main(args)
#      
#      print(f"Trust Score is: {trust_score}")
#      
#      #ORCHESTRATOR
#      print("\n\n--- Sending Case to Orchestrator ---")
#      final_state["final_diagnosis"] = ophthalmic_agent.analyze(final_state, probability, v_cdr, trust_score)
#      print("\n\n Ophthalmic Agent's Final Diagnosis Ready: ")
#      print(f"{final_state['final_diagnosis']['decision']}")
#      print("\n" + "-"*30)  
#
##    #FUNCTIONAL INTERPRETATION AGENT
##    print("\n\n--- Sending Visual Field Data to Functional Interpretation Agent ---")
##    final_state["functional_opinion"] = functional_agent.analyze(final_state)
##    print("\n\nFunctional Vision Interpreter's Output Ready: ")
##    print(f"{final_state['functional_opinion']['summary']}")
##    print("\n" + "-"*30)   
#    
##    #EQUITY AGENT
##    print("\n\n--- Sending Narrative and Visual Findings to Equity Agent ---")
##    equity_input = f"""
##    ### PATIENT DATA 
##    Patient narrative: {final_state['clinical_narrative']}
##        
##    ### AI MODEL OUTPUTS
##    {retfound_scores} 
##         
##    {mirage_scores}"""
##
##    final_state["equity_opinion"] = equity_agent.analyze_patients(equity_input, output_format="text")
##    print("\n\nEquity Agent's Output Ready: ")
##    print(f"{final_state['equity_opinion']}") 
##    print("\n" + "-"*30)
#      
#
#        
##    #GUIDELINES AGENT
##    print("\n\n--- Sending Query to Guidelines Agent ---")
##    note = f"""
##    Patient Narrative: {final_state['clinical_narrative']} 
##    Functional Interpretation Agent output: {final_state['functional_opinion']['summary']}"""
##
##    final_state['guidelines'] = guidelines_agent.consult_note(note, max_results=5, diagnosis_only=True,)
##    print("\n\n Guidelines Agent's Output   Ready: ")
##    print(f"{final_state['guidelines']}")
##    
##    print("\n" + "-"*30) 
##    #SAFETY AGENT
##    print("\n\n--- Sending Case to Safety Agent ---")
##    final_state["safety_output"] = safety_agent.run(final_state)
##    print("\n\n Safety Agent's Output Ready: ")
##    print(f"{final_state['safety_output']}")
##    print("\n" + "-"*30)  
#    
#def parse_agent_labels(final_state):
#    label_text = final_state['final_diagnosis']['labels']
#    
#    print(f"Label Text {label_text}")
#    
#    patterns = {
#    "AMD": r"AMD_DETECTED:\s*(.+)",
#    "DR": r"DR_DETECTED:\s*(.+)",
#    "GL": r"GLAUCOMA_DETECTED:\s*(.+)"
#    }
#    
#    results = {}
#    
#    for key, pattern in patterns.items():
#        match = re.search(pattern, label_text, re.IGNORECASE)
#        if match:
#            value_match = re.search(r"-?\d+", match.group(1))
#            results[key] = int(value_match.group()) if value_match else -1
#        else:
#            results[key] = -1
#    
#    return results
#       
#def initialize_state(patient_data):
#    # Initialize the State
#    state: AgentState = {
#        "patient_id": patient_data['metadata']['filename'],
#        "metadata": patient_data['metadata'],
#        "fundus_img": np.array(patient_data['fundus_img']),
#        "oct_img": np.array(patient_data['oct_img']),
#        "clinical_narrative": "",
#        "oct_diagnosis": None,
#        "slo_diagnosis": None,
#        "vision_opinion": {},
#        
#        "vision_opinion_oct": {},
#        "vision_opinion_slo": {},
#        
#        "functional_opinion": {},
#        "final_diagnosis": {},
#        "fairness_flag": False,
#        "safety_output": "",
#        "guidelines": "",
#        "equity_opinion:": ""
#    }
#
#    return state
#    
#if __name__ == "__main__":
#    
#    EXCEL_PATH = "./data/fairvision_250each.csv"
#    
#    results = []
#    
#    # Initialize the updated Excel Loader
#    loader = ExcelEyeLoader(EXCEL_PATH)
#    
#    # Target conditions to run evaluation blocks on
##    diseases = ['Glaucoma', 'AMD', 'DR']
##    diseases = ['Glaucoma']
##    diseases = ['AMD']
#    diseases = ['DR'] #CHANGE
#    
#    for disease in diseases:
#        # Pull up to 250 records from the master Excel per disease category
#        test_rows = loader.get_records_by_disease(disease, limit=250)
#        
#        if not test_rows.empty:
#            # Iterate through every row explicitly returned by the Excel filter block
#            for index, current_row in test_rows.iterrows():
#                
#                # Pre-extract basic row demographics for error-handling resilience
#                # Safe checking for case-variations using dictionary transformations
#                row_dict = {k.lower(): v for k, v in current_row.to_dict().items()}
#                
#                age = row_dict.get('age', 'Unknown')
#                gender = row_dict.get('gender', 'Unknown')
#                race = row_dict.get('race', 'Unknown')
#                ethnicity = row_dict.get('ethnicity', 'Unknown')
#                
#                patient_record = None
#                
#                try:
#                    # Load structural files and map binary ground truth via row paths
#                    patient_record = loader.load_patient_from_excel_row(current_row)
#                    
#                    print(list(patient_record['metadata'].keys()))
#                    # Core pipeline logic execution
#                    final_state = initialize_state(patient_record)
#                    temp = patient_record['metadata']['filename']
#                        
##                    print(f"patient id: {temp}")
##                    if temp == "data/Glaucoma/Test/data_07010.npz": 
##                      Image.fromarray(np.array(patient_record['oct_img'])).convert('RGB').save("oct_slice.png")
##                      Image.fromarray(np.array(patient_record['fundus_img'])).convert('RGB').save("slo.png")
##                      
##                      continue
##      
#                    
#                    run_diagnostic_pipeline(patient_record)
#                    
#                    pred_labels = parse_agent_labels(final_state)
#                    ground_truth = patient_record['stage']
#                    
#                    row = {
#                        "Filename": patient_record['directory'],
#                        "Task_Folder": disease,
#                        "Age": age,
#                        "Gender": gender,
#                        "Race": race,
#                        "Ethnicity": ethnicity,
#                        "Ground_Truth": ground_truth,
#                        "Pred_AMD": pred_labels.get("AMD", -1) if pred_labels else -1,
#                        "Pred_DR": pred_labels.get("DR", -1) if pred_labels else -1,
#                        "Pred_GL": pred_labels.get("GL", -1) if pred_labels else -1,
#                    }
#                    
#                    # Select corresponding validation prediction
#                    if "AMD" in disease:
#                        pred = row["Pred_AMD"]
#                    elif "DR" in disease:
#                        pred = row["Pred_DR"]
#                    elif "Glaucoma" in disease:
#                        pred = row["Pred_GL"]
#                    else:
#                        pred = -1
#                    
#                    # Compute accuracy profiles
#                    if pred == -1 or ground_truth == -1:
#                        row["Is_Correct"] = -1
#                    else:
#                        row["Is_Correct"] = int(pred == ground_truth)
#                    
#                    results.append(row)
#                    
#                    # Inline checkpoint auto-saving
#                    out_df = pd.DataFrame(results)
#                    out_df.to_csv(OUTPUT_CSV, index=False)
#                    
#                    print(f"\nDisease: {disease} | Ground Truth: {ground_truth} | Row Index Checked: {index}")
#                    print("END OF EXAMPLE\n" + "-"*30)
#                      
#                except Exception as e:
#                      print(f"!!! Error processing row index {index} in {disease}. Skipping... Details: {e}")
#                      
#                      # Safely map fallback structural variables during exception failures
#                      fallback_filename = patient_record['directory'] if patient_record else row_dict.get('filepath', 'Error')
#                      fallback_gt = patient_record['stage'] if patient_record else row_dict.get('ground_truth', -1)
#                      
#                      row = {
#                          "Filename": fallback_filename,
#                          "Task_Folder": disease,
#                          "Age": age,
#                          "Gender": gender,
#                          "Race": race,
#                          "Ethnicity": ethnicity,
#                          "Ground_Truth": fallback_gt,
#                          "Pred_AMD": -1,  
#                          "Pred_DR": -1,    
#                          "Pred_GL": -1,    
#                          "Is_Correct": -1
#                      }
#                      results.append(row)
#                      
#                      out_df = pd.DataFrame(results)  
#                      out_df.to_csv(OUTPUT_CSV, index=False)    
#                    
#        else:
#            print(f"No custom Excel rows found to process for: {disease}")
#
#

#################################################################################################### DR #################################################


# OphthalmicAgent/main.py

#change 5

from Orchestrator.state import AgentState
from BioProfilerAgent.bio_profiler import BioProfiler
#from EquityAgent.equity_agent import EquityAgent
from EquityAgent.compute_demographic_reliability_score import main
from VisionAgent.vision import VisionSpecialist

from VisionAgent.vision_oct import VisionSpecialistOct
from VisionAgent.vision_slo import VisionSpecialistSlo

from VisionAgent.linear_probing_oct3 import FairVisionNPZ
from FunctionalInterpretationAgent.function_interpreter import FunctionalSpecialist
from GuidelinesAgent.guidelines_agent import GuidelinesAgent
from SafetyAgent.safety_agent import SafetyAgent
from Orchestrator.new import Orchestrator

from data.loader import ExcelEyeLoader

from types import SimpleNamespace
from pathlib import Path

import numpy as np
from PIL import Image
import pandas as pd
import json
import re

OUTPUT_CSV = "ophthalmic_performance_results_jul04_glaucoma_4.csv" #CHANGE

#Intiializing agents
profiler = BioProfiler()
#equity_agent = EquityAgent()
#vision_agent = VisionSpecialist("./weights/oct_model_best_all_binary.pth", "./weights/slo_model_best_all_binary.pth")

vision_agent_oct = VisionSpecialistOct("./weights/oct_model_8_slices_not_center.pth")
vision_agent_slo = VisionSpecialistSlo("./weights/oct_model_best_all_binary.pth", "./weights/slo_model_best_all_binary.pth")

functional_agent = FunctionalSpecialist()
ophthalmic_agent = Orchestrator()
safety_agent = SafetyAgent()
guidelines_agent = GuidelinesAgent()

def run_diagnostic_pipeline(patient_data):

    final_state["retfound_scores"], final_state["vision_opinion_oct"] = vision_agent_oct.analyze(patient_data['oct_img'], patient_data['middle_oct'], final_state)
    probability = final_state['oct_diagnosis']['Glaucoma']['Prob_Pct']
#    probability = final_state['oct_diagnosis']['AMD']['Prob_Pct']
#    probability = final_state['oct_diagnosis']['DR']['Prob_Pct'] #CHANGE
    
    print("\n\nProbabilities Ready: ")
    print("\nOCT DIAGNOSIS USING FINETUNED RETFOUND")
    retfound_scores = final_state["retfound_scores"] 
    print(retfound_scores)
    
    if probability >= 90:
        final_state['final_diagnosis']['labels'] = "GLAUCOMA_DETECTED: 1"
#        final_state['final_diagnosis']['labels'] = "AMD_DETECTED: 1"
#        final_state['final_diagnosis']['labels'] = "DR_DETECTED: 1" #CHANGE
    elif probability <= 10:
        final_state['final_diagnosis']['labels'] = "GLAUCOMA_DETECTED: 0"
#        final_state['final_diagnosis']['labels'] = "AMD_DETECTED: 0"
#        final_state['final_diagnosis']['labels'] = "DR_DETECTED: 0" #CHANGE
    else:
    
      print("STARTING AGENTIC PIPELINE")
      #BIO PROFILER AGENT
      print("\n\n--- Sending Meta Data to Bio Profiler Agent ---")
      final_state["clinical_narrative"] = profiler.generate_narrative(final_state["metadata"])
      print("\n\nNarrative Ready: ")
      print(f"{final_state['clinical_narrative']}") 
      print("\n" + "-"*30)
      
      #VISION AGENT
      print("\n\n--- Sending Visual Data to Vision Specialists ---")
  
      final_state["vision_opinion_slo"], v_cdr = vision_agent_slo.analyze(final_state['fundus_img'], final_state)   
      print("\n\nVision Specialist SLO's Output Ready: ")
      print(f"{final_state['vision_opinion_slo']}")
      print("\n" + "-"*30)
      
      print("\n\nVision Specialist OCT's Output Ready: ")
      print(f"{final_state['vision_opinion_oct']}")
      print("\n" + "-"*30)
        
      print("\n\n--- Calculating Model Trust Score ---")
      if final_state["metadata"]['Age'] >= 60:
        age_group = "older"
      elif final_state["metadata"]['Age'] < 40:
        age_group = "younger"
      else:
        age_group = "middle-aged"
      
        
      args = SimpleNamespace(
          subgroup_reliability_csv=Path("/lustre/fs1/home/yu395012/RETFound/OphthalmicAgent/_extras/CSVs/demographic_reliability_subgroup_model_scores.csv"),
          task=final_state["metadata"]["Task_Folder"].lower(),
          model="retfound_oct",
          age_group=age_group,
          race=final_state["metadata"]["Race"],
          gender=final_state["metadata"]["Gender"],
          score_column="final_R_bad",
          json=False,
          priors_json=None,
          support_csv=None,
          k=50.0,
          fnr_weight=0.35,
          fpr_weight=0.25,
          ece_weight=0.15,
          auroc_weight=0.15,
          f1_weight=0.10,
          raw_local_lambdas=False,
      )
      
      trust_score = 1 - main(args)
      
      print(f"Trust Score is: {trust_score}")
      
      #ORCHESTRATOR
      print("\n\n--- Sending Case to Orchestrator ---")
      final_state["final_diagnosis"] = ophthalmic_agent.analyze(final_state, probability, v_cdr, trust_score)
      print("\n\n Ophthalmic Agent's Final Diagnosis Ready: ")
      print(f"{final_state['final_diagnosis']['decision']}")
      print("\n" + "-"*30)  

#    #FUNCTIONAL INTERPRETATION AGENT
#    print("\n\n--- Sending Visual Field Data to Functional Interpretation Agent ---")
#    final_state["functional_opinion"] = functional_agent.analyze(final_state)
#    print("\n\nFunctional Vision Interpreter's Output Ready: ")
#    print(f"{final_state['functional_opinion']['summary']}")
#    print("\n" + "-"*30)   
    
#    #EQUITY AGENT
#    print("\n\n--- Sending Narrative and Visual Findings to Equity Agent ---")
#    equity_input = f"""
#    ### PATIENT DATA 
#    Patient narrative: {final_state['clinical_narrative']}
#        
#    ### AI MODEL OUTPUTS
#    {retfound_scores} 
#         
#    {mirage_scores}"""
#
#    final_state["equity_opinion"] = equity_agent.analyze_patients(equity_input, output_format="text")
#    print("\n\nEquity Agent's Output Ready: ")
#    print(f"{final_state['equity_opinion']}") 
#    print("\n" + "-"*30)
      

        
#    #GUIDELINES AGENT
#    print("\n\n--- Sending Query to Guidelines Agent ---")
#    note = f"""
#    Patient Narrative: {final_state['clinical_narrative']} 
#    Functional Interpretation Agent output: {final_state['functional_opinion']['summary']}"""
#
#    final_state['guidelines'] = guidelines_agent.consult_note(note, max_results=5, diagnosis_only=True,)
#    print("\n\n Guidelines Agent's Output   Ready: ")
#    print(f"{final_state['guidelines']}")
#    
#    print("\n" + "-"*30) 
#    #SAFETY AGENT
#    print("\n\n--- Sending Case to Safety Agent ---")
#    final_state["safety_output"] = safety_agent.run(final_state)
#    print("\n\n Safety Agent's Output Ready: ")
#    print(f"{final_state['safety_output']}")
#    print("\n" + "-"*30)  
    
def parse_agent_labels(final_state):
    label_text = final_state['final_diagnosis']['labels']
    
    print(f"Label Text {label_text}")
    
    patterns = {
    "AMD": r"AMD_DETECTED:\s*(.+)",
    "DR": r"DR_DETECTED:\s*(.+)",
    "GL": r"GLAUCOMA_DETECTED:\s*(.+)"
    }
    
    results = {}
    
    for key, pattern in patterns.items():
        match = re.search(pattern, label_text, re.IGNORECASE)
        if match:
            value_match = re.search(r"-?\d+", match.group(1))
            results[key] = int(value_match.group()) if value_match else -1
        else:
            results[key] = -1
    
    return results
       
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
        
        "vision_opinion_oct": {},
        "vision_opinion_slo": {},
        
        "functional_opinion": {},
        "final_diagnosis": {},
        "fairness_flag": False,
        "safety_output": "",
        "guidelines": "",
        "equity_opinion:": ""
    }

    return state
    
if __name__ == "__main__":
    
    EXCEL_PATH = "./data/fairvision_250each.csv"
    
    results = []
    
    # Initialize the updated Excel Loader
    loader = ExcelEyeLoader(EXCEL_PATH)
    
    # Target conditions to run evaluation blocks on
#    diseases = ['Glaucoma', 'AMD', 'DR']
    diseases = ['Glaucoma']
#    diseases = ['AMD']
#    diseases = ['DR'] #CHANGE
    
    for disease in diseases:
        # Pull up to 250 records from the master Excel per disease category
        test_rows = loader.get_records_by_disease(disease, limit=250)
        
        if not test_rows.empty:
            # Iterate through every row explicitly returned by the Excel filter block
            for index, current_row in test_rows.iterrows():
                
                # Pre-extract basic row demographics for error-handling resilience
                # Safe checking for case-variations using dictionary transformations
                row_dict = {k.lower(): v for k, v in current_row.to_dict().items()}
                
                age = row_dict.get('age', 'Unknown')
                gender = row_dict.get('gender', 'Unknown')
                race = row_dict.get('race', 'Unknown')
                ethnicity = row_dict.get('ethnicity', 'Unknown')
                
                patient_record = None
                
                try:
                    # Load structural files and map binary ground truth via row paths
                    patient_record = loader.load_patient_from_excel_row(current_row)
                    
                    print(list(patient_record['metadata'].keys()))
                    # Core pipeline logic execution
                    final_state = initialize_state(patient_record)
                    temp = patient_record['metadata']['filename']
                        
#                    print(f"patient id: {temp}")
#                    if temp == "data/Glaucoma/Test/data_07010.npz": 
#                      Image.fromarray(np.array(patient_record['oct_img'])).convert('RGB').save("oct_slice.png")
#                      Image.fromarray(np.array(patient_record['fundus_img'])).convert('RGB').save("slo.png")
#                      
#                      continue
#      
                    
                    run_diagnostic_pipeline(patient_record)
                    
                    pred_labels = parse_agent_labels(final_state)
                    ground_truth = patient_record['stage']
                    
                    row = {
                        "Filename": patient_record['directory'],
                        "Task_Folder": disease,
                        "Age": age,
                        "Gender": gender,
                        "Race": race,
                        "Ethnicity": ethnicity,
                        "Ground_Truth": ground_truth,
                        "Pred_AMD": pred_labels.get("AMD", -1) if pred_labels else -1,
                        "Pred_DR": pred_labels.get("DR", -1) if pred_labels else -1,
                        "Pred_GL": pred_labels.get("GL", -1) if pred_labels else -1,
                    }
                    
                    # Select corresponding validation prediction
                    if "AMD" in disease:
                        pred = row["Pred_AMD"]
                    elif "DR" in disease:
                        pred = row["Pred_DR"]
                    elif "Glaucoma" in disease:
                        pred = row["Pred_GL"]
                    else:
                        pred = -1
                    
                    # Compute accuracy profiles
                    if pred == -1 or ground_truth == -1:
                        row["Is_Correct"] = -1
                    else:
                        row["Is_Correct"] = int(pred == ground_truth)
                    
                    results.append(row)
                    
                    # Inline checkpoint auto-saving
                    out_df = pd.DataFrame(results)
                    out_df.to_csv(OUTPUT_CSV, index=False)
                    
                    print(f"\nDisease: {disease} | Ground Truth: {ground_truth} | Row Index Checked: {index}")
                    print("END OF EXAMPLE\n" + "-"*30)
                      
                except Exception as e:
                      print(f"!!! Error processing row index {index} in {disease}. Skipping... Details: {e}")
                      
                      # Safely map fallback structural variables during exception failures
                      fallback_filename = patient_record['directory'] if patient_record else row_dict.get('filepath', 'Error')
                      fallback_gt = patient_record['stage'] if patient_record else row_dict.get('ground_truth', -1)
                      
                      row = {
                          "Filename": fallback_filename,
                          "Task_Folder": disease,
                          "Age": age,
                          "Gender": gender,
                          "Race": race,
                          "Ethnicity": ethnicity,
                          "Ground_Truth": fallback_gt,
                          "Pred_AMD": -1,  
                          "Pred_DR": -1,    
                          "Pred_GL": -1,    
                          "Is_Correct": -1
                      }
                      results.append(row)
                      
                      out_df = pd.DataFrame(results)  
                      out_df.to_csv(OUTPUT_CSV, index=False)    
                    
        else:
            print(f"No custom Excel rows found to process for: {disease}")