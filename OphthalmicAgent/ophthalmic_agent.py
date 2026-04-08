import os
from openai import AzureOpenAI
from dotenv import load_dotenv
import re


load_dotenv() 
api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
DEPLOYMENT = "gpt-5.1"
        
class Orchestrator:
    def __init__(self, model_client=None):
        
        self.model_client = AzureOpenAI(
          azure_endpoint = endpoint, 
          api_key = api_key,
          api_version="2024-12-01-preview"
          )

    def analyze(self, state):
        
        narrative = state['clinical_narrative']
        vision_summary = state['vision_opinion']['summary']
        functional_summary = state['functional_opinion']['summary']
        race = state['metadata']['race']
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a Lead Ophthalmic Surgeon. Your role is to provide a final Clinicopathologic Correlation "
                    "by synthesizing Specialist Summaries into a definitive diagnosis.\n\n"
                    
                    "DECISION HIERARCHY & TIE-BREAKING:\n"
                    "1. BASELINE: Start with the numerical probabilities (p-values) provided in the Vision Summary. "
                    "A p-value > 0.50 generally indicates a positive finding.\n"
                    
                    "2. OVERRIDE PROTOCOL FOR AMD: 
                    "a. If the Foundation Models (RETFound/MIRAGE) predict a lower "
                    "stage (e.g., Stage 0 or 1), but the Vision Specialist identifies clear visual evidence of a "
                    "higher stage (e.g., 'Small Drusen' for Stage 1 or 'Large Mounded Drusen/Pigment Clumping' for Stage 2), "
                    "you MUST override the numerical models and assign the higher stage.\n\n"
                    
                    "b. If both Foundation Models (RETFound and MIRAGE) "
                    "agree on an advanced stage (Stage 2 or 3), do NOT override them based on the Vision Specialist's "
                    "narrative alone. In cases of FM agreement on high stages, prioritize the FMs' numerical detection "
                    "unless the Vision Specialist reports a clear image artifact.\n\n"
                    
                    "c. TIE-BREAKING: If RETFound and MIRAGE disagree on the stage, use the Vision Specialist's "
                    "description of drusen size (Small vs. Mounded/Confluent) as the definitive tie-breaker."
                            
                    "3. PROTOCOL FOR DR: Trust RETFound (OCT) p-values for DR. If the Vision Specialist confirms "
                    "ANY intraretinal fluid or hemorrhages, classify DR_DETECTED as 1 to maintain high screening recall.\n"
                    
                    "4. PROTOCOL FOR GLAUCOMA: For Glaucoma, only confirm the diagnosis if RETFound p > 0.50 "
                    "IMPORTANT: Prioritize output of functional specialist for glaucoma"
                )
            },
            {
                "role": "user",
                "content": f"""
                ### SPECIALIST EVIDENCE LOG
                - **Clinical Narrative**: {narrative}
                - **Vision Specialist Verification**: {vision_summary}
                - **Functional Specialist Findings**: {functional_summary}
                
                ### DIAGNOSTIC TASK
                1. **Review Alignment**: Check the 'Alignment Status' in the Vision Summary (Agree/Conflict).
                2. **Execute Tie-Break**: Execute tie-break according to the decision rules described for each disease.
                3. **Final Impression**: Provide a one-sentence summary of the case.
        
                ### OUTPUT FORMAT
                [LABELS]
                AMD_STAGE: [0, 1, 2, or 3]
                DR_DETECTED: [0 or 1]
                GLAUCOMA_DETECTED: [0 or 1]
                [/LABELS]
        
                """
            }
        ]

        response = self.model_client.chat.completions.create(
            model=DEPLOYMENT,
            messages=messages,
            temperature=0.3
        )

        raw_response = response.choices[0].message.content
        
        query_match = re.search(r"\[QUERY\](.*?)\[/QUERY\]", raw_response, re.DOTALL)
        if query_match:
            pubmed_query = query_match.group(1).strip()
            final_decision = re.sub(r"\[QUERY\].*?\[/QUERY\]", "", raw_response, flags=re.DOTALL).strip()
        else:
            pubmed_query = "Ophthalmology treatment guidelines 2026"
            final_decision = raw_response
            
        label_match = re.search(r"\[LABELS\](.*?)\[/LABELS\]", raw_response, re.DOTALL)
        if label_match:
            output_labels = label_match.group(1).strip()
        else:
            output_labels = "AMD_STAGE: 0, DR_DETECTED: 0, GLAUCOMA_DETECTED: 0"
        
        return {
            "agent": "Ophthalmic_Master",
            "decision": final_decision,
            "labels": output_labels,
            "pubmed_query": pubmed_query
        }