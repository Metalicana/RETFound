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

    def analyze(self, state, retfound_scores, mirage_scores):
        
        narrative = state['clinical_narrative']
        vision_summary = state['vision_opinion']['summary']
        functional_summary = state['functional_opinion']['summary']
        equity_output = state['equity_opinion']
        guidelines = state['guidelines']
    
        messages = [
            {
                "role": "system",
                    "content": ( """ 
                    You are a Lead Ophthalmic Surgeon. You must synthesize specialist findings into a final diagnosis using the following strict hierarchy:
                    
                    1. **THE GLAUCOMA "FUNCTIONAL BRAKE"** (Mandatory)
                       - If the Functional Agent reports a Normal field (MD > -2.0 dB), you must output 0 for Glaucoma.
                    
                    2. **THE AMD "PATHOLOGY SIGNAL" RULE**
                       - Do not just look at the "Primary Prediction" stage. Analyze the "[!] TOTAL PATHOLOGY SIGNAL" (the sum of Stages 1, 2, and 3).
                       - If the Total Pathology Signal exceeds the RECOMMENDED_THRESHOLD (e.g., 35%) from the Equity Agent, you must consider the case                           POSITIVE for AMD.
                       - To determine the stage: Look at the distribution mass. If the mass is concentrated in Stage 2, and the Vision Specialist confirms                       "lumpy" or "textured" RPE, you must assign Stage 2.
                    
                    3. **THE AMD "CONSENSUS RULE"**
                       - To prevent "Stage 0" calls for subtle Stage 2 cases:
                       - If BOTH RetFound and MIRAGE show a Total Pathology Signal > 60%, do NOT output Stage 0, even if the Vision Specialist reports a                         clear image.
                       - In this case, defer to the specific stage (1, 2, or 3) that has the highest relative probability among the disease stages.
                    
                    4. **DYNAMIC THRESHOLDING & WEIGHTING**
                       - Apply the RECOMMENDED_THRESHOLD from the Equity Audit to all AMD and DR probabilities.
                       - Prioritize the PRIMARY_MODEL identified by the Equity Agent during conflicts.
                    
                    5. THE "PROXY" EVALUATION AND BENCHMARK LABELING:
                    - If the Vision Specialist says "Macula not in view," check the AI Models.       
                    - If RETFound (OCT) and MIRAGE (SLO) both agree on a stage (e.g., both say Stage 2), accept the AI's diagnosis. 
                    - For benchmark scoring, always provide forced diagnostic labels whenever any model evidence is available.
                    - Human escalation/safety concern belongs in FINAL_IMPRESSION, not as an abstention label.
                    - Only output -1 if the Vision Specialist says "Image Unreadable" AND both AI models have confidence < 40%. """

                )
            },
            {
                "role": "user",
                "content": f"""
                ### MULTI-AGENT CASE INPUTS
                - **Clinical Narrative**: {narrative}
                - **RetFound (OCT) Distribution**: {retfound_scores}
                - **MIRAGE (SLO) Distribution**: {mirage_scores}
                - **Vision Specialist Findings**: {vision_summary}
                - **Functional Specialist Findings**: {functional_summary}
                - **Equity Agent Audit**: {equity_output}
                - **Clinical Guidelines**: {guidelines}
                
                ### DIAGNOSTIC TASK:
                1. **Extract Constraints**: Identify the RECOMMENDED_THRESHOLD and PRIMARY_MODEL.
                2. **Glaucoma Check**: Apply the "Functional Brake" (MD > -2.0 dB check).
                3. **AMD Signal Analysis**: 
                   - Sum the disease probabilities (Stages 1-3). 
                   - Does this "Total Pathology Signal" exceed the threshold?
                   - If yes, cross-reference with the Vision Specialist's "Independent Stage" and morphological descriptions.
                4. **Final Staging**: Use the Consensus Rule to ensure intermediate cases aren't downgraded to Stage 0.
                
                ### OUTPUT FORMAT
                [LABELS]
                AMD_STAGE: [-1, 0, 1, 2, or 3]
                DR_DETECTED: [-1, 0 or 1]
                GLAUCOMA_DETECTED: [-1, 0 or 1]
                [/LABELS]
                
                FINAL_IMPRESSION: [One-sentence clinical summary of the final decision logic. If human review is recommended, state it here while preserving the forced labels above.]
        
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
