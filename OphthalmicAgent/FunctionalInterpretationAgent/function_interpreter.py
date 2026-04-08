import os
from openai import AzureOpenAI
from dotenv import load_dotenv


load_dotenv() 
api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
DEPLOYMENT = "gpt-5.1"
        
class FunctionalSpecialist:
    def __init__(self, model_client=None):
        
        self.model_client = AzureOpenAI(
          azure_endpoint = endpoint, 
          api_key = api_key,
          api_version="2024-12-01-preview"
          )

    def analyze(self, state):
        
        metadata = state['metadata']
        
        md_value = metadata.get('md')
        md_score = f"{md_value} dB" if md_value is not None else "Not Available"
        
        narrative = state['clinical_narrative']
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a Specialist in Functional Vision Assessment. Your role is to interpret "
                    "perimetry data, specifically Mean Deviation (MD) scores, and correlate them "
                    "with patient history. You translate numerical deficits into clinical severity "
                    "(Early, Moderate, or Advanced loss).\n\n"
                    "CRITICAL: You must conclude your report with a section titled '[EXECUTIVE SUMMARY]' "
                    "containing 3-5 bullet points for the Lead Ophthalmic Surgeon."
                )
            },
            {
                "role": "user",
                "content": f"""
                ### INPUT DATA
                - **Mean Deviation (MD):** {md_score}
                - **Patient Narrative:** {narrative}

                ### TASK
                1. Interpret the MD score severity. (Normal: > -2dB, Early: -2 to -6dB, Moderate: -6 to -12dB, Advanced: < -12dB).
                2. Explain how this functional loss aligns with the symptoms described in the narrative.
                3. Provide a 'Functional Status Report'.
                4. Conclude with the [EXECUTIVE SUMMARY].
                """
            }
        ]

        response = self.model_client.chat.completions.create(
            model=DEPLOYMENT,
            messages=messages,
            temperature=0.3 # Lower temperature for consistent clinical interpretation
        )
        
        full_content = response.choices[0].message.content
        
        if "[EXECUTIVE SUMMARY]" in full_content:
            summary = full_content.split("[EXECUTIVE SUMMARY]")[-1].strip()
        else:
            summary = full_content # Fallback

        return {
            "agent": "Functional Specialist",
            "full_report": full_content,
            "summary": summary
        }
        
        return 