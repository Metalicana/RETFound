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
#        vision_summary = state['vision_opinion']['summary']
        
        vision_summary_oct = state['vision_opinion_oct']['summary']
        vision_summary_slo = state['vision_opinion_slo']['summary']
        
#        functional_summary = state['functional_opinion']['summary']
#        equity_output = state['equity_opinion']
#        guidelines = state['guidelines']
 
        ### FOR GLAUCOMA ONLY
        
#        messages = [
#        {
#            "role": "system",
#            "content": ( 
#                """You are a Lead Ophthalmic Surgeon specializing in Glaucomatous Optic Neuropathy. Your core mandate is to act as a **Precision Gatekeeper**, minimizing False Positives by resolving extreme conflicts between text descriptions and fine-tuned numerical AI models.
#    
#    ### CONFLICT RESOLUTION ARCHITECTURE (ANTI-FALSE POSITIVE FILTER)
#    
#    You must process the patient's case using the following mathematical and structural hierarchy:
#    
#    1. **THE STRUCTURAL DOUBLE-CONFIRMATION RULE (Baseline Threshold = 50%)**
#       - A raw probability >= 0.50 from an AI model is considered a 'positive' disease signal.
#       - **True Positive Criteria:** If BOTH RETFound (OCT) and MIRAGE (SLO) agree that the probability is >= 0.50, AND the Vision Specialist describes active structural damage (e.g., focal notching, wedge defects, or a severe vertical Cup-to-Disc Ratio >= 0.75), classify the case as POSITIVE (1).
#    
#    2. **THE "BENIGN VARIANT" BRAKE (Intercepting Vision Agent False Positives)**
#       - The Vision Agent is prone to over-interpreting normal physiological variants (like large benign macro-cups or naturally thin nerve fiber layers) as pathological.
#       - **THE OVERRIDE:** If EITHER of the following conditions is met, you must execute a strict manual override and classify the case as HEALTHY/NEGATIVE (0):
#         * Condition A: BOTH RETFound (OCT) and MIRAGE (SLO) models output a probability *below* 45%, even if the Vision Specialist text claims to see structural changes.
#         * Condition B: The Vision Specialist explicitly notes that the neuroretinal rim tissue 'is uniform, pink, or obeys the ISNT rule', or that the OCT profile 'preserves its standard double-hump peaks'. These are definitive markers of health that negate borderline AI scores.
#    
#    3. **MODALITY DISAGREEMENT TIE-BREAKING**
#       - If RETFound (OCT) and MIRAGE (SLO) disagree (one is high, one is low):
#         * Defer strictly to the **RETFound (OCT) probability** as your gold-standard structural anchor. 
#         * If the OCT probability is < 0.50, default to NEGATIVE (0) unless the Vision Specialist describes a *highly localized, active focal defect* (such as an active splinter hemorrhage or a distinct inferior rim notch).
#    
#    4. **FORCED BENCHMARK CONSTRAINTS**
#       - For algorithmic tracking, you must output a definitive binary choice: 0 (Low Risk/Healthy) or 1 (High Risk/Pathology). 
#       - If you override the Vision Specialist because the AI models are completely silent, document this reasoning clearly in your FINAL_IMPRESSION, but enforce the clean `0` label.
#       - Only output an unreadable label (-1) if the Vision Specialist explicitly states "Image Completely Unreadable" AND both deep learning models show complete confidence collapse (< 30%)."""
#            )
#        },
#        {
#            "role": "user",
#            "content": f"""
#            ### MULTI-AGENT CASE INPUTS
#            - **Clinical Narrative**: {narrative}
#            - **RetFound (OCT) Binary Scores**: {retfound_scores}
#            - **MIRAGE (SLO) Binary Scores**: {mirage_scores}
#            - **Vision Specialist Visual Audit**: {vision_summary}
#            
#            ### DIAGNOSTIC TASK:
#            1. **Check for "Benign Variant" Mismatches**: Evaluate if the Vision Specialist's positive notes are being contradicted by low probabilities (<45%) across both deep learning backbones. If yes, trigger the 'Benign Variant Brake' and classify as 0.
#            2. **Cross-Reference Healthy Biomarkers**: Look for keyword anchors in the Vision Summary like "pink rim", "obeys ISNT", or "double-hump profile" to dismiss false alarms.
#            3. **Resolve Modality Disagreements**: If the AI backbones conflict, leverage the 3D peripapillary OCT (RETFound) score as the primary diagnostic anchor over the 2D fundus projection.
#            
#            ### REQUIRED OUTPUT FORMAT
#            You must provide your final consensus exactly within the formatted tags below:
#            
#            [LABELS]
#            GLAUCOMA_DETECTED: [-1, 0 or 1]
#            [/LABELS]
#            
#            FINAL_IMPRESSION: [Provide a concise, one-sentence clinical summary of your final decision logic. Explicitly state if you triggered a manual override to correct a structural false alarm based on model-tissue alignment.]
#            """
#        }
#    ]
#        

#2. HIGH-CONFIDENCE AGREEEMENT: If the RETFound score is very high (>= 0.75) OR the Vision Specialists describe clear tissue damage (like a "notch" in the rim or a "flat/thin" nerve layer), classify as POSITIVE (1).
#3. HEALTHY SAFETY OVERRIDE: If the RETFound score is very low (< 0.20) OR if the Vision Specialist explicitly notes that the rim is "uniform/healthy" AND the OCT shows a "normal double-hump shape", classify as HEALTHY (0)."""    

        messages = [
            {
                "role": "system",
                "content": (
                    """You are a Lead Ophthalmic Surgeon. Your task is to weigh the numerical RETFound AI score against the SLO Specialist and OCT Specialist's text description to choose a final binary label: 0 (Healthy) or 1 (Glaucoma).

NOISE FALLBACK: If BOTH specialists state their images are too noisy, classify as -1."""
                )
            },
            {
                "role": "user",
                "content": f"""
            ### MULTI-AGENT CASE INPUTS
            - **RetFound (OCT) Score**: {retfound_scores}
            - **OCT Specialist Report**: {vision_summary_oct}
            - **SLO Specialist Report**: {vision_summary_slo}
            
            ### DIAGNOSTIC TASK:
            1. Check for text noise failures to determine if you need to execute the noise fallback policy.
            2. IMPORTANT: Output 1 (positive glaucoma) if RETFound scores are high (>80%) and IGNORE SLO/OCT findings.
            3. IMPORTANT: Output 0 (healthy) if RETFound scores are low (<21%) and IGNORE SLO/OCT findings.
            3. For all other cases, weigh the isolated structural findings against the RETFound scores to deliver a high-precision binary consensus.
            
            ### REQUIRED OUTPUT FORMAT
            You must provide your final consensus exactly within the formatted tags below:
            
            [LABELS]
            GLAUCOMA_DETECTED: [-1, 0 or 1]
            [/LABELS]
            
            FINAL_IMPRESSION: [Provide a concise explanation of why you chose this final label.]
            """
            }
        ]
        
        response = self.model_client.chat.completions.create(
            model=DEPLOYMENT,
            messages=messages,
            temperature=0.3
        )

        raw_response = response.choices[0].message.content
        
#        query_match = re.search(r"\[QUERY\](.*?)\[/QUERY\]", raw_response, re.DOTALL)
#        if query_match:
#            pubmed_query = query_match.group(1).strip()
#            final_decision = re.sub(r"\[QUERY\].*?\[/QUERY\]", "", raw_response, flags=re.DOTALL).strip()
#        else:
#            pubmed_query = "Ophthalmology treatment guidelines 2026"
#            final_decision = raw_response
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
#            "pubmed_query": pubmed_query
        }
