# OphthalmicAgent/main.py

from state import AgentState
from BioProfilerAgent.bio_profiler import BioProfiler
from VisionAgent.vision_specialist import VisionSpecialist
from data.loader import GenericEyeLoader
from drAgent.dr_agent import DRSpecialist
import numpy as np
from PIL import Image

#Intiializing agents
profiler = BioProfiler()
vision_agent = VisionSpecialist("VisionAgent/weights/RETFound_mae_natureOCT.pth")
dr_specialist = DRSpecialist()

def run_diagnostic_pipeline(patient_data):
    # Initialize the State
    state: AgentState = {
        "patient_id": patient_data['metadata']['filename'],
        "metadata": patient_data['metadata'],
        "fundus_img": np.array(patient_data['fundus_img']),
        "clinical_narrative": "",
        "vision_features": None,
        "specialist_opinions": [],
        "final_diagnosis": "",
        "fairness_flag": False
    }

    # Step 1: Bio-Profiler creates the story
    print("Running Bio-Profiler...")
    state["clinical_narrative"] = profiler.generate_narrative(state["metadata"])

    # Step 2: Vision Specialist extracts features
    print("Running Vision Specialist...")
    state["vision_features"] = vision_agent.get_features(patient_data['oct_tensors'])

    return state


# --- Integration Test Logic ---
if __name__ == "__main__":
    # 1. Setup paths
    BASE_PATH = "/lustre/fs1/home/yu395012/OphthalmicAgent/data"
    
    disease = 'DR'
    loader = GenericEyeLoader(BASE_PATH)
    df = loader.get_metadata(disease)
    
    # Grab the first test patient
    test_rows = df[df['use'] == 'test']
    if not test_rows.empty:
        patient_record = loader.load_patient(disease, test_rows.iloc[0])
#        metadata = patient_record['metadata']
#        oct_data = patient_record['oct_tensors']
    else:
        print("No test data found to process.")
        

final_state = run_diagnostic_pipeline(patient_record)
print(f"Narrative Ready: {final_state['clinical_narrative'][:50]}...")
print(f"Features Ready: {final_state['vision_features'].shape}")
print("Fundus image stored in state")

#final_state["fundus_img"].save("state_check_direct.jpg")
img = Image.fromarray(final_state["fundus_img"].astype('uint8'), 'RGB')
img.save("state_check.jpg")

# 3. TRIGGER THE AGENT
print("--- Sending Case to DR Specialist ---")
opinion = dr_specialist.analyze(final_state)

# 4. Store the outcome back in the state
final_state["specialist_opinions"].append(opinion)

# 5. Review the result
print(f"DR Specialist Decision: {opinion['decision']}")