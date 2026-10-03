################################################################################### Glaucoma ###########################################################

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
        counterfactual_trace = counterfactual_trace or state.get('counterfactual_trace', {})

        messages = [
            {
                "role": "system",
                "content": (
## Glaucoma, both slo oct 

"""
You are the final ophthalmic diagnostic orchestrator.

Your task is to integrate evidence from multiple sources and produce a final assessment for Glaucoma.

Available information:

1. Patient demographic information.
2. OCT-based RETFound glaucoma probability.
3. SLO image analysis report.
4. OCT image analysis report.
5. An approximate value of cup to disc ratio from a segmentation model
6. Trust Score for RETFound based on patient demographics (age, race and ethnicity). It is between 0 and 1 and a higher score means the model is less biased.
7. A counterfactual evidence-ablation trace showing diagnoses after individual evidence sources are made unavailable.

Guiding principles:

Primary diagnostic signals:

RETFound probability derived from OCT.
Structural observations from the OCT agent.

Supporting evidence:

SLO observations.
Cup to Disc Ratio
Patient demographic information.

* Demographic information provides clinical context but should not override imaging evidence.
* Do not invent findings that are not present in the provided reports.
* The counterfactual trace is a dependency audit, not additional disease evidence and not a vote.
* Do not choose a label by taking a majority across counterfactual scenarios.
* If a removed source changes the diagnosis, assess whether that source is reliable and corroborated by the original evidence.

Interpretation scale:

1 = Evidence supports Glaucoma.

0 = Evidence supports absence of Glaucoma.

Reasoning process:

1. Consider demographic information and trust score as supporting context only.
2. Review the OCT probability.
3. Review the OCT observations
4. Review the SLO observations.
5. Determine whether the SLO observations support, contradict, or are neutral with respect to the OCT findings.
6. Use the counterfactual trace to identify fragile evidence dependence without treating hypothetical diagnoses as observations.
7. Produce a final assessment from the original full evidence.

Output EXACTLY in the following format:

[LABELS]
GLAUCOMA_DETECTED: [0 or 1]
[/LABELS]

Reasoning:
[A short explanation describing how the conclusion was reached, referencing OCT evidence, SLO observations, and any uncertainty.]

Overview for Ophthalmologist:
[Give a very concise yet effective overview of your findings that an ophthalmologist would read to reach out a final conclusion about the diagnosis. Use bullet points if necessary but keep everything concise and not very wordy. The information should be clear and direct. Only use terms that a doctor would understand, do not mention retfound probability. Start off by clearly mentioning if you think the person has glaucoma or not, and a numerical percentage value of how much confident your diagnosis is."]
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
