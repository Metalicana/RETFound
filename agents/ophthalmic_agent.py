import os
import json
from openai import AzureOpenAI
from tools.registry import ToolRegistry

# --- CONFIGURATION ---
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_KEY = os.getenv("AZURE_OPENAI_API_KEY")
DEPLOYMENT = "gpt-5.1"

class OphthalmicAgent:
    def __init__(self):
        self.client = AzureOpenAI(
            api_key=AZURE_KEY,
            api_version="2024-12-01-preview",
            azure_endpoint=AZURE_ENDPOINT
        )
        self.registry = ToolRegistry()
        self.tools_schema = self.registry.get_schemas()

    def consult(self, patient_note, image_path):
        """
        The main loop. 
        Input: "Patient is 45yo Black male...", "/path/to/img.jpg"
        Output: Final Diagnosis String
        """
        
        # 1. The System Persona (The "Chief Resident")
        system_prompt = """
        You are Equi-Agent, an expert Ophthalmic Consultant.
        Your goal is to provide a diagnosis and management plan that is accurate AND fair.

        PROTOCOL:
        1. Always consult the 'Vision Expert' first to get the image analysis.
        2. Identify the patient's demographic profile from the notes.
        3. Consult the 'Equity Expert' to check if the Vision Expert might be biased or if specific risk factors exist.
        4. Synthesize: 
           - If Vision says 'Healthy' but Equity says 'High FNR Risk', trigger a referral.
           - If Vision says 'Sick' and Equity confirms, treat urgently.
        
        Output Style: Clinical, decisive, justifying overrides explicitly.
        """

        # Initialize Conversation
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Consult Request. Note: {patient_note}. Image Path: {image_path}"}
        ]

        print(f"\n--- CONSULTATION START: {patient_note[:30]}... ---")

        # 2. The Reasoning Loop (Max 5 turns to prevent infinite loops)
        for i in range(5):
            response = self.client.chat.completions.create(
                model=DEPLOYMENT,
                messages=messages,
                tools=self.tools_schema,
                tool_choice="auto" # Let GPT decide when to call tools
            )

            msg = response.choices[0].message
            messages.append(msg)

            # A. Did the Agent ask for a tool?
            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                    # Execute
                    result = self.registry.execute(tool_call)
                    
                    # Feed back the result
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result)
                    })
                    print(f" -> [Tool Result] {json.dumps(result)[:100]}...") # Log brief output
            
            # B. Did the Agent finish? (No tool calls)
            else:
                print("--- FINAL DIAGNOSIS GENERATED ---")
                return msg.content

        return "Error: Agent stuck in reasoning loop."

# --- DEV TEST ---
if __name__ == "__main__":
    # Simulate a run
    agent = OphthalmicAgent()
    
    # Test Case: The "Normal Tension Glaucoma" Trap
    note = "62 year old Asian female. IOP 18mmHg (Normal). Complains of peripheral vision loss."
    img = "/home/ab575577/projects_spring_2026/HarvardFairVision30K/FairVision/Glaucoma/Training/slo_fundus_00001.jpg"
    
    final_report = agent.consult(note, img)
    print("\n" + final_report)