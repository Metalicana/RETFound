import os
import sys
import torch
import json
import time
from PIL import Image
from torchvision import transforms
from openai import AzureOpenAI
from serpapi import GoogleSearch 
import models_vit 

# --- NEW: IMPORT YOUR CUSTOM TOOL ---
from agent_tools import AVAILABLE_TOOLS, TOOL_DEFINITIONS

# --- CONFIGURATION ---
MODEL_PATH = "best_fair_eye_model.pth"
# We expect the demo image to be passed in or set here
DEFAULT_DEMO_IMAGE = "demo_case.jpg" 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 3

# --- CREDENTIALS (FROM ENV) ---
AZURE_KEY = os.environ.get("AZURE_OPENAI_KEY")
SERPAPI_KEY = os.environ.get("SERPAPI_KEY")

# Hardcoded non-secrets
AZURE_ENDPOINT = "https://azure-openai-radi.cognitiveservices.azure.com/" # Update if different
AZURE_MODEL = "gpt-4" # Corrected from gpt-5.1
API_VERSION = "2024-02-15-preview"

# Safety Check
if not AZURE_KEY or not SERPAPI_KEY:
    print("\n[ERROR] Missing API Keys!")
    print("export AZURE_OPENAI_KEY='your-key'")
    print("export SERPAPI_KEY='your-key'")
    sys.exit(1)

try:
    client = AzureOpenAI(
        api_version=API_VERSION,
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_KEY,
    )
except Exception as e:
    print(f"[ERROR] Failed to initialize Azure Client: {e}")
    sys.exit(1)

# --- 1. THE VISION SPECIALIST (RETFound) ---
class VisionSpecialist:
    def __init__(self):
        print("[System] Initializing Vision Specialist (RETFound)...")
        self.device = DEVICE
        # Corrected Model Definition
        self.model = models_vit.vit_large_patch16(
            img_size=224, num_classes=NUM_CLASSES, drop_path_rate=0.2, global_pool=True
        ).to(self.device)
        
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=self.device)
            state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
            # Remove head if needed, but assuming fine-tuned weights match
            self.model.load_state_dict(state_dict, strict=False)
            self.model.eval()
        except FileNotFoundError:
            print(f"[ERROR] Model file '{MODEL_PATH}' not found. Using random weights for test.")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def analyze(self, img_path):
        try:
            img = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"[ERROR] Demo image '{img_path}' not found.")
            return 0.5 # Neutral fallback
            
        img_t = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(img_t)
            probs = torch.sigmoid(outputs).cpu().numpy()[0]
        # Returning Glaucoma probability (Index 2) or AMD (Index 0) depending on demo
        # For this script, let's assume index 2 is Glaucoma
        return float(probs[2]) 

# --- 2. THE EPIDEMIOLOGIST (Search) ---
class EpidemiologistAgent:
    def consult(self, race, age, disease):
        query = f"prevalence and misdiagnosis of {disease} in {race} patients over {age}"
        print(f"   > [Epidemiologist] Searching: '{query}'...")
        
        try:
            params = {"q": query, "api_key": SERPAPI_KEY, "num": 3}
            search = GoogleSearch(params)
            results = search.get_dict()
            evidence = ""
            if "organic_results" in results:
                for idx, result in enumerate(results["organic_results"][:2]):
                    evidence += f"- {result.get('title')}: {result.get('snippet')}\n"
            else:
                evidence = "No search results found."
        except Exception as e:
            evidence = f"Search failed: {str(e)}"
        return evidence

# --- 3. THE CLINICIAN (Static) ---
class ClinicalAgent:
    def consult(self, disease, risk_profile):
        print(f"   > [Clinician] Retrieving Guidelines for {disease}...")
        return "AAO Guidelines: For High Risk patients, lower the referral threshold. Verify structural damage if IOP is normal."

# --- 4. THE CHAIR (Active Agent) ---
class TheCouncil:
    def __init__(self):
        self.vision = VisionSpecialist()
        self.epi = EpidemiologistAgent()
        self.clinic = ClinicalAgent()
        
    def convene_council(self, img_path, patient):
        print(f"\n=== CONVENING DIAGNOSTIC COUNCIL ===")
        print(f"Patient: {patient['ID']} | {patient['Race']} | {patient['Age']} years old")
        
        # 1. Gather Initial Opinions
        glaucoma_score = self.vision.analyze(img_path)
        print(f"   > [Vision Specialist] Glaucoma Probability: {glaucoma_score:.1%}")
        
        epi_data = self.epi.consult(patient['Race'], patient['Age'], "Glaucoma")
        clinical_data = self.clinic.consult("Glaucoma", "High")
        
        # 2. Build the "Case File" for GPT
        system_prompt = """
        You are the Chair of a Medical Diagnostic Council.
        Your goal is to decide: REFER or DISMISS.
        
        You have access to a tool: 'tool_glaucoma_check'. 
        USE IT if the Vision Specialist's score is ambiguous (e.g., 30-70%) OR if the Epidemiologist suggests high risk of bias.
        Do NOT guess. If you need physical proof (CDR), call the tool.
        """
        
        user_prompt = f"""
        CASE DATA:
        - Image Path: {img_path}
        - Patient: {patient['Age']}y {patient['Race']}
        - Vision Model Score: {glaucoma_score:.1%}
        - Epidemiology: {epi_data}
        - Guidelines: {clinical_data}
        
        Decide. If uncertain, verify with the tool.
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # 3. The Active Loop (Thinking -> Tool Call -> Final Answer)
        print(f"\n   > [The Chair] Deliberating (and checking tools)...")
        
        response = client.chat.completions.create(
            model=AZURE_MODEL,
            messages=messages,
            tools=TOOL_DEFINITIONS, # Inject your tool here
            tool_choice="auto"
        )
        
        response_msg = response.choices[0].message
        messages.append(response_msg) # Add to history
        
        # 4. Handle Tool Call
        if response_msg.tool_calls:
            print(f"   [!] The Chair is pausing to use a tool...")
            
            for tool_call in response_msg.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
                
                if func_name in AVAILABLE_TOOLS:
                    print(f"   [>] Executing {func_name} on {func_args.get('image_path')}...")
                    
                    # Execute the Python Function
                    tool_func = AVAILABLE_TOOLS[func_name]
                    observation = tool_func(**func_args)
                    
                    print(f"   [<] Observation: {observation}")
                    
                    # Feed back to GPT
                    messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": func_name,
                        "content": observation
                    })
            
            # 5. Get Final Verdict after Tool
            print(f"   > [The Chair] Synthesizing final decision...")
            final_response = client.chat.completions.create(
                model=AZURE_MODEL,
                messages=messages
            )
            print("\n=== FINAL COUNCIL DECISION ===")
            print(final_response.choices[0].message.content)
            
        else:
            # If no tool was called
            print("\n=== FINAL COUNCIL DECISION ===")
            print(response_msg.content)

if __name__ == "__main__":
    # Ensure a demo image exists
    if not os.path.exists(DEFAULT_DEMO_IMAGE):
        # Fallback to the known Glaucoma path
        fallback = "/home/ab575577/projects_spring_2026/HarvardFairVision30K/FairVision/Glaucoma/Training/slo_fundus_00001.jpg"
        if os.path.exists(fallback):
            DEFAULT_DEMO_IMAGE = fallback
        else:
            print("Warning: No demo image found. Please set correct path.")

    council = TheCouncil()
    council.convene_council(DEFAULT_DEMO_IMAGE, {"ID": "B-999", "Race": "Black", "Age": 72})