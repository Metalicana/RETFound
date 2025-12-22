import os
import json
import random
import re
from openai import AzureOpenAI
from serpapi import GoogleSearch # Official search library

# --- CONFIGURATION (UPDATE THESE) ---
# Azure Setup
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://azure-openai-xxxxxx.cognitiveservices.azure.com/")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "your-azure-key")
DEPLOYMENT_NAME = "gpt-4" 

# Search Setup (Get a free key from serpapi.com if you don't have one, or swap for your preferred API)
SERPAPI_KEY = os.getenv("SERPAPI_KEY", "your-serpapi-key")

# --- 1. THE TOOL BELT ---

class MockVisionExpert:
    """
    Simulates RETFound output. 
    In Phase 2, you will replace this class with 'RealVisionExpert' from real_tools.py.
    """
    def analyze(self, image_path):
        print(f"   [VisionTool] Scanning {image_path} with RETFound (Simulated)...")
        # Simulating a "Complex Glaucoma" case for testing
        return {
            "findings": {
                "AMD": 0.05,
                "Diabetic Retinopathy": 0.12,
                "Glaucoma": 0.78, # High probability
                "Cataract": 0.10
            },
            "features": "Enlarged vertical cup-to-disc ratio (0.7). Inferior rim thinning detected.",
            "model_version": "RETFound_FairVision_Finteune_v1"
        }

class LiteratureSearchTool:
    """
    Real-time access to Medical Guidelines (RAG/Search).
    Uses Google Search to find AAO/PubMed contexts.
    """
    def search(self, query):
        print(f"   [SearchTool] Searching medical literature for: '{query}'...")
        
        if not SERPAPI_KEY or "your-serpapi-key" in SERPAPI_KEY:
            # Fallback if no key provided
            return "Simulated Search Result: AAO Guidelines (2024) recommend treating IOP > 21mmHg if optic nerve damage is present. African American patients often have thinner corneas, affecting IOP readings."

        try:
            params = {
                "engine": "google",
                "q": f"ophthalmology guidelines {query}", # Bias search towards medical info
                "api_key": SERPAPI_KEY,
                "num": 3
            }
            search = GoogleSearch(params)
            results = search.get_dict()
            
            snippets = []
            if "organic_results" in results:
                for item in results["organic_results"]:
                    snippets.append(f"- {item.get('title')}: {item.get('snippet')}")
            
            return "\n".join(snippets)
        except Exception as e:
            return f"Search Error: {e}"

class EpidemiologyStats:
    """
    The 'Fairness' Knowledge Base. 
    Hardcoded facts from Nature Medicine / Lancet to ensure Equity.
    """
    def get_risk_context(self, disease, race):
        print(f"   [EpidemiologyTool] Checking risk profile for {disease} in {race} patients...")
        stats = {
            "Glaucoma": {
                "Black": "CRITICAL: 4-5x higher prevalence. Onset 10 years earlier. Blindness risk 6x higher than White patients.",
                "Hispanic": "High prevalence. Open-angle glaucoma is leading cause of blindness.",
                "Asian": "Higher risk of Normal Tension Glaucoma (normal IOP but nerve damage)."
            },
            "Diabetic Retinopathy": {
                "Black": "2x risk of incident DR compared to White patients.",
                "Hispanic": "Higher rates of vision-threatening proliferative DR."
            }
        }
        
        # Simple lookup
        race_key = "White" # Default
        if "Black" in race or "African" in race: race_key = "Black"
        elif "Hispanic" in race or "Latino" in race: race_key = "Hispanic"
        elif "Asian" in race: race_key = "Asian"

        if disease in stats and race_key in stats[disease]:
            return stats[disease][race_key]
        return "Standard risk profile."

# --- 2. THE AGENT ORCHESTRATOR ---

class SOTAMedAgent:
    def __init__(self):
        self.client = AzureOpenAI(
            api_key=AZURE_API_KEY,
            api_version="2024-02-15-preview", # Update to match your endpoint version
            azure_endpoint=AZURE_ENDPOINT
        )
        
        # Initialize Tools
        self.vision = MockVisionExpert()
        self.search = LiteratureSearchTool()
        self.stats = EpidemiologyStats()
        
        # Tool Definitions for GPT-4
        self.tools_schema = [
            {
                "type": "function",
                "function": {
                    "name": "analyze_image",
                    "description": "Use RETFound model to detect disease probabilities in fundus/OCT images.",
                    "parameters": {"type": "object", "properties": {"image_path": {"type": "string"}}}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "consult_guidelines",
                    "description": "Search medical literature (AAO, PubMed) for treatment guidelines or drug interactions.",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "check_equity_risk",
                    "description": "Retrieve specific racial/demographic risk factors for a suspected disease.",
                    "parameters": {"type": "object", "properties": {"disease": {"type": "string"}, "race": {"type": "string"}}}
                }
            }
        ]

    def run_diagnosis(self, patient_note, image_path):
        # 1. System Prompt (The "Nature Medicine" Logic)
        system_prompt = """
        You are 'FairView', an advanced autonomous Ophthalmology Agent.
        
        CORE PROTOCOL:
        1. **Gather Evidence:** Always analyze the image first.
        2. **Hypothesize:** Based on vision data + patient note, form a hypothesis.
        3. **Equity Check:** YOU MUST check if the patient's demographics affect the risk profile using 'check_equity_risk'.
        4. **Verify Guidelines:** If the case is serious or ambiguous, search for the latest AAO guidelines using 'consult_guidelines'.
        5. **Final Triage:** Output a diagnosis, severity, and urgency level.
        
        Tone: Professional, Clinical, Evidence-Based.
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Please diagnose this patient. Notes: {patient_note}. Image: {image_path}"}
        ]

        print("\n--- AGENT STARTING WORKFLOW ---")
        
        # Loop for Multi-Step Reasoning (ReAct)
        for i in range(6): # Max 6 steps
            response = self.client.chat.completions.create(
                model=DEPLOYMENT_NAME,
                messages=messages,
                tools=self.tools_schema,
                tool_choice="auto"
            )
            
            msg = response.choices[0].message
            messages.append(msg)
            
            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                    fname = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)
                    print(f" -> [DECISION] Agent calls tool: {fname} with {args}")
                    
                    # Execute Tool
                    result = ""
                    if fname == "analyze_image":
                        result = self.vision.analyze(args["image_path"])
                    elif fname == "consult_guidelines":
                        result = self.search.search(args["query"])
                    elif fname == "check_equity_risk":
                        result = self.stats.get_risk_context(args["disease"], args["race"])
                    
                    # Feed result back to Brain
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result)
                    })
            else:
                # No more tools -> Final Answer
                print("--- AGENT FINISHED ---")
                return msg.content
                
        return "Error: Max steps exceeded."

# --- RUN IT ---
if __name__ == "__main__":
    agent = SOTAMedAgent()
    
    # Test Case: The "Equity Trap"
    # A Black patient with borderline Glaucoma. A standard model might miss this.
    # The Agent should catch it because of the 'check_equity_risk' tool.
    patient_note = "45-year-old African American male. Family history of blindness. IOP 22mmHg."
    image_path = "patient_scan_001.jpg"
    
    diagnosis = agent.run_diagnosis(patient_note, image_path)
    print("\nFINAL REPORT:\n", diagnosis)