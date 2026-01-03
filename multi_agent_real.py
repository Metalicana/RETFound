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

# --- IMPORT YOUR CUSTOM TOOL ---
# This imports the registry you built in agent_tools.py
from agent_tools import AVAILABLE_TOOLS, TOOL_DEFINITIONS

# --- CONFIGURATION ---
MODEL_PATH = "best_fair_eye_model.pth"
# Use a known Glaucoma image that might be tricky (dark/SLO)
DEFAULT_DEMO_IMAGE = "/home/ab575577/projects_spring_2026/HarvardFairVision30K/FairVision/Glaucoma/Training/slo_fundus_00001.jpg" 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 3

# --- CREDENTIALS ---
AZURE_KEY = os.environ.get("AZURE_OPENAI_KEY")
SERPAPI_KEY = os.environ.get("SERPAPI_KEY")
AZURE_ENDPOINT = "https://azure-openai-radi.cognitiveservices.azure.com/" 
AZURE_MODEL = "gpt-4" 
API_VERSION = "2024-02-15-preview"

if not AZURE_KEY or not SERPAPI_KEY:
    print("[ERROR] Export AZURE_OPENAI_KEY and SERPAPI_KEY first!")
    sys.exit(1)

client = AzureOpenAI(
    api_version=API_VERSION,
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_KEY,
)

# --- 1. VISION SPECIALIST (Glaucoma Focused) ---
class VisionSpecialist:
    def __init__(self):
        print("[System] Initializing Vision Specialist (RETFound)...")
        self.device = DEVICE
        self.model = models_vit.vit_large_patch16(
            img_size=224, num_classes=NUM_CLASSES, drop_path_rate=0.2, global_pool=True
        ).to(self.device)
        
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=self.device)
            state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
            self.model.load_state_dict(state_dict, strict=False)
            self.model.eval()
        except Exception:
            print(f"[WARNING] Model weights not found at {MODEL_PATH}. Using random init (for demo flow testing).")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def analyze(self, img_path):
        try:
            img = Image.open(img_path).convert('RGB')
        except:
            return 0.5 
            
        img_t = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(img_t)
            probs = torch.sigmoid(outputs).cpu().numpy()[0]
        
        # --- CRITICAL CHANGE: Return Glaucoma Probability (Index 2) ---
        return float(probs[2]) 

# --- 2. EPIDEMIOLOGIST ---
class EpidemiologistAgent:
    def consult(self, race, age):
        # Specific query for Glaucoma disparities
        query = f"misdiagnosis of Normal Tension Glaucoma in {race} patients"
        print(f"   > [Epidemiologist] Searching: '{query}'...")
        
        try:
            params = {"q": query, "api_key": SERPAPI_KEY, "num": 3}
            search = GoogleSearch(params)
            results = search.get_dict()
            evidence = ""
            if "organic_results" in results:
                for result in results["organic_results"][:2]:
                    evidence += f"- {result.get('title')}: {result.get('snippet')}\n"
            else:
                evidence = "No search results found."
        except Exception as e:
            evidence = "Search unavailable."
        return evidence

# --- 3. THE CLINICIAN ---
class ClinicalAgent:
    def consult(self):
        # Static knowledge injection
        return "Guideline: Glaucoma can exist with Normal Intraocular Pressure (IOP). Structural analysis (CDR) is required for confirmation in high-risk groups."

# --- 4. THE CHAIR (Active Agent) ---
class TheCouncil:
    def __init__(self):
        self.vision = VisionSpecialist()
        self.epi = EpidemiologistAgent()
        self.clinic = ClinicalAgent()
        
    def convene_council(self, img_path, patient):
        print(f"\n=== 🏛️ CONVENING DIAGNOSTIC COUNCIL ===")
        print(f"Patient: {patient['ID']} | {patient['Race']} | {patient['Age']} years old")
        
        # 1. Gather Opinions
        glaucoma_score = self.vision.analyze(img_path)
        print(f"   > [Vision Specialist] Glaucoma Confidence: {glaucoma_score:.1%}")
        
        epi_data = self.epi.consult(patient['Race'], patient['Age'])
        clinical_data = self.clinic.consult()
        
        # 2. System Prompt: INSTRUCTING THE AGENT WHEN TO USE THE TOOL
        system_prompt = """
        You are the Chair of a Medical Diagnostic Council.
        
        DECISION PROTOCOL:
        1. If Vision Confidence is > 90%, you may decide immediately.
        2. IF Vision Confidence is LOW/AMBIGUOUS (< 80%) OR the Epidemiology suggests bias/risk:
           - YOU MUST VERIFY with the tool 'tool_glaucoma_check'.
           - This tool measures the Cup-to-Disc Ratio (CDR).
           - High CDR (> 0.6) confirms Glaucoma even if the model is unsure.
        
        Goal: Correctly Diagnose REFER or DISMISS.
        """
        
        user_prompt = f"""
        CASE FILE:
        - Image: {img_path}
        - Patient: {patient['Age']}y {patient['Race']}
        - Vision Model Score: {glaucoma_score:.1%}
        - Epidemiology: {epi_data}
        - Guidelines: {clinical_data}
        
        The user suspects Normal Tension Glaucoma. Proceed.
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        print(f"\n   > [The Chair] Deliberating...")
        
        # 3. First LLM Call (Does it want to use the tool?)
        response = client.chat.completions.create(
            model=AZURE_MODEL,
            messages=messages,
            tools=TOOL_DEFINITIONS, #
            tool_choice="auto"
        )
        
        response_msg = response.choices[0].message
        messages.append(response_msg)
        
        # 4. Handle Tool Call
        if response_msg.tool_calls:
            print(f"   [!] INTERVENTION: The Chair is requesting a structural check...")
            
            for tool_call in response_msg.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
                
                if func_name in AVAILABLE_TOOLS:
                    print(f"   [>] Running {func_name} on image...")
                    
                    # Run the SegFormer Tool
                    tool_func = AVAILABLE_TOOLS[func_name]
                    observation = tool_func(**func_args)
                    
                    print(f"   [<] Tool Observation: {observation}")
                    
                    messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": func_name,
                        "content": observation
                    })
            
            # 5. Final Synthesis
            final_response = client.chat.completions.create(
                model=AZURE_MODEL,
                messages=messages
            )
            print("\n=== FINAL VERDICT ===")
            print(final_response.choices[0].message.content)
            
        else:
            print("\n=== FINAL VERDICT ===")
            print(response_msg.content)

if __name__ == "__main__":
    # Ensure this image exists!
    council = TheCouncil()
    # High Risk Demographic + Glaucoma Image
    council.convene_council(DEFAULT_DEMO_IMAGE, {"ID": "G-102", "Race": "Black", "Age": 68})