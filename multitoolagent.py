import os
import json
import re
from openai import AzureOpenAI

# --- CONFIG ---
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
DEPLOYMENT = "gpt-4"

# --- 1. THE TOOL BELT ---

class ToolRegistry:
    """The 'Bag of Tricks' the Agent can access."""
    
    def vision_expert(self, image_path):
        # Your RETFound logic goes here
        # Mocking for now:
        return {"finding": "Likely Glaucoma", "confidence": 0.88, "features": "Enlarged cup"}

    def morphology_measurer(self, image_path):
        # Hypothetical SAM-Med tool
        return {"cup_to_disc_ratio": 0.75, "rim_thinning": True}

    def guideline_retriever(self, query):
        # RAG Search mock
        if "Glaucoma" in query:
            return "AAO Guidelines: IOP > 21 mmHg with optic nerve damage requires treatment."
        return "No specific guidelines found."

    def patient_history_scanner(self, note_text):
        # Extract Intraocular Pressure (IOP)
        iop = re.search(r"IOP\s*[:=]\s*(\d+)", note_text)
        return {"IOP": iop.group(1) if iop else "Unknown"}

# --- 2. THE ORCHESTRATOR (The "Pro" Agent) ---

class SOTAMedAgent:
    def __init__(self):
        self.client = AzureOpenAI(
            api_key=API_KEY, 
            api_version="2024-02-15-preview", 
            azure_endpoint=AZURE_ENDPOINT
        )
        self.tools = ToolRegistry()
        
        # DEFINITION OF TOOLS FOR THE LLM
        self.tool_definitions = [
            {
                "type": "function",
                "function": {
                    "name": "vision_expert",
                    "description": "Analyze retinal image for disease probabilities.",
                    "parameters": {"type": "object", "properties": {"image_path": {"type": "string"}}}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "morphology_measurer",
                    "description": "Measure physical features like Cup-to-Disc ratio.",
                    "parameters": {"type": "object", "properties": {"image_path": {"type": "string"}}}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "guideline_retriever",
                    "description": "Search clinical guidelines for treatment standards.",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}
                }
            },
             {
                "type": "function",
                "function": {
                    "name": "patient_history_scanner",
                    "description": "Extract vitals like IOP from text notes.",
                    "parameters": {"type": "object", "properties": {"note_text": {"type": "string"}}}
                }
            }
        ]

    def solve_case(self, patient_note, image_path):
        # INITIAL MEMORY
        messages = [
            {"role": "system", "content": """
             You are an Advanced Ophthalmology Agent. 
             You have access to specialized tools. 
             DO NOT GUESS. Use tools to gather evidence before diagnosing.
             
             Standard Workflow:
             1. Scan patient history for vitals (IOP, Diabetes).
             2. Analyze the image with VisionExpert.
             3. If Vision is ambiguous, use MorphologyMeasurer to get exact numbers.
             4. Check Guidelines for the confirmed condition.
             5. Output Final Diagnosis.
             """},
            {"role": "user", "content": f"Diagnose this case. Note: {patient_note}. Image: {image_path}"}
        ]

        print(f"--- STARTING DIAGNOSIS ---")
        
        # MAX STEPS (To prevent infinite loops)
        for step in range(5):
            response = self.client.chat.completions.create(
                model=DEPLOYMENT,
                messages=messages,
                tools=self.tool_definitions,
                tool_choice="auto" 
            )
            
            msg = response.choices[0].message
            messages.append(msg) # Add agent's thought to memory

            # IF THE AGENT WANTS TO USE A TOOL
            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                    func_name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)
                    
                    print(f" -> Agent executing: {func_name} with {args}")
                    
                    # EXECUTE TOOL
                    if func_name == "vision_expert":
                        result = self.tools.vision_expert(args['image_path'])
                    elif func_name == "morphology_measurer":
                        result = self.tools.morphology_measurer(args['image_path'])
                    elif func_name == "guideline_retriever":
                        result = self.tools.guideline_retriever(args['query'])
                    elif func_name == "patient_history_scanner":
                        result = self.tools.patient_history_scanner(args['note_text'])
                    else:
                        result = "Error: Tool not found."

                    # FEEDBACK RESULT TO AGENT
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result)
                    })
            else:
                # NO TOOLS CALLED -> FINAL ANSWER
                print(f"--- DIAGNOSIS COMPLETE ---")
                return msg.content

        return "Error: Maximum steps reached."

# --- TEST DRIVE ---
if __name__ == "__main__":
    agent = SOTAMedAgent()
    
    # Complex Case: Confusing Glaucoma
    note = "55yo Male, complains of tunnel vision. IOP measured at 24mmHg."
    img = "test_eye.jpg"
    
    final_report = agent.solve_case(note, img)
    print("\nFINAL REPORT:\n", final_report)