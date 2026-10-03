########################################################################################### DR ###########################################################

import os
import json
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

    def analyze(self, state, probability, v_cdr, trust_score, counterfactual_trace=None):
        
        narrative = state['clinical_narrative']
        vision_summary_slo = state['vision_opinion_slo']
        vision_summary_oct = state['vision_opinion_oct']
        
## DR, only oct            
        messages = [
            {
                "role": "system",
                "content": (


"""
You are the final ophthalmic diagnostic orchestrator.

Your task is to integrate evidence from multiple sources and produce a final assessment for Diabetic Retinopathy (DR).

Available information:

1. Patient demographic information.
2. OCT-based RETFound dr probability.
3. OCT image analysis report.
4. SLO image analysis report.
5. An approximate value of cup to disc ratio from a segmentation model
6. Trust Score for RETFound based on patient demographics (age, race and ethnicity). It is between 0 and 1 and a higher score means the model is less biased.

Guiding principles:

Primary diagnostic signals:

RETFound probability derived from OCT.
Structural observations from the OCT and SLO agent.

Supporting evidence:

Patient demographic information.

* Demographic information provides clinical context but should not override imaging evidence.
* Do not invent findings that are not present in the provided reports.

Interpretation scale:

1 = Evidence supports DR.

0 = Evidence supports absence of DR.

Reasoning process:

1. Consider demographic information and trust score as supporting context only.
2. Review the OCT probability.
3. Review the OCT observations
4. Review the SLO observations
5. Produce a final assessment.

Output EXACTLY in the following format:

[LABELS]
DR_DETECTED: [0 or 1]
[/LABELS]

Reasoning:
[A short explanation describing how the conclusion was reached, referencing OCT evidence, SLO observations, and any uncertainty.]

Overview for Ophthalmologist:
[Give a very concise yet effective overview of your findings that an ophthalmologist would read to reach out a final conclusion about the diagnosis. Use bullet points if necessary but keep everything concise and not very wordy. The information should be clear and direct. Only use terms that a doctor would understand, do not mention retfound probability. Start off by clearly mentioning if you think the person has DR or not, and a numerical percentage value of how much confident your diagnosis is."]

"""
                )
            },
            {
                "role": "user",
                "content": f"""
                ### MULTI-AGENT CASE INPUTS
                - **Patient Narrative**: {narrative}
                
                - **RetFound (OCT) Score**: {probability}%   
                
                - **OCT Specialist Report**: {vision_summary_oct}
                
                - **SLO Specialist Report**: {vision_summary_slo}
                
                - **Cup to Disc ratio**: {v_cdr}
                
                - **Trust Score**: {trust_score}
                
                - **Counterfactual Evidence-Ablation Trace**: {json.dumps(counterfactual_trace, sort_keys=True)}
                
                """
            }
        ]
        
#### DR, only slo        
#        
#        messages = [
#        {
#        "role": "system",
#        "content": (
#
#
#
#"""
#You are the final ophthalmic diagnostic orchestrator.
#
#Your task is to integrate evidence from multiple sources and produce a final assessment for Diabetic Retinopathy (DR).
#
#Available information:
#
#1. Patient demographic information.
#2. OCT-based RETFound dr probability.
#3. SLO image analysis report.
#4. An approximate value of cup to disc ratio from a segmentation model
#5. Trust Score for RETFound based on patient demographics (age, race and ethnicity). It is between 0 and 1 and a higher score means the model is less biased.
#
#Guiding principles:
#
#Primary diagnostic signals:
#
#RETFound probability derived from OCT.
#
#Supporting evidence:
#
#SLO observations.
#Patient demographic information.
#
#* Demographic information provides clinical context but should not override imaging evidence.
#* Do not invent findings that are not present in the provided reports.
#
#Interpretation scale:
#
#1 = Evidence supports DR.
#
#0 = Evidence supports absence of DR.
#
#Reasoning process:
#
#1. Consider demographic information and trust score as supporting context only.
#2. Review the OCT probability.
#3. Review the SLO observations.
#4. Determine whether the SLO observations support, contradict, or are neutral with respect to the OCT findings.
#5. Produce a final assessment.
#
#Output EXACTLY in the following format:
#
#[LABELS]
#DR_DETECTED: [0 or 1]
#[/LABELS]
#
#Reasoning:
#[A short explanation describing how the conclusion was reached, referencing OCT evidence, SLO observations, and any uncertainty.]
#
#"""
#                )
#            },
#            {
#                "role": "user",
#                "content": f"""
#                ### MULTI-AGENT CASE INPUTS
#                - **Patient Narrative**: {narrative}
#                
#                
#                - **RetFound (OCT) Score**: {probability}%
#                
#                
#                - **SLO Specialist Report**: {vision_summary_slo}
#                
#                
#                - **Cup to Disc ratio**: {v_cdr}
#                
#                
#                - **Trust Score**: {trust_score}
#                """
#            }
#        ]
#
        response = self.model_client.chat.completions.create(
            model=DEPLOYMENT,
            messages=messages,
            temperature=0.3
        )

        raw_response = response.choices[0].message.content
            
        label_match = re.search(r"\[LABELS\](.*?)\[/LABELS\]", raw_response, re.DOTALL)
        if label_match:
            output_labels = label_match.group(1).strip()
        else:
            output_labels = "AMD_STAGE: 0, DR_DETECTED: 0, GLAUCOMA_DETECTED: 0"
        
        return {
            "agent": "Ophthalmic_Master",
            "decision": raw_response,
            "labels": output_labels,
        }


