import json
import os
import re

from dotenv import load_dotenv
from openai import AzureOpenAI


load_dotenv()


class DrishtiOrchestrator:
    def __init__(self, model_client=None):
        self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.6-luna")
        self.model_client = model_client or AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        )

    def analyze(self, probability, cfp_report, cdr, counterfactual_trace=None):
        response = self.model_client.chat.completions.create(
            model=self.deployment,
#            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are the final glaucoma diagnostic orchestrator for a CFP-only pipeline. "
                        "Your task is to integrate evidence from multiple sources and produce a final assessment for Glaucoma."
                        "Available information:"
                        "1. CFP-based RETFound glaucoma probability"
                        "2. CFP image analysis report"
                        "3. An approximate value of cup to disc ratio from a segmentation model"
                        "4. A counterfactual evidence-ablation trace showing diagnoses after individual evidence sources are made unavailable."
                        
                        
                        """
                        * Do not invent findings that are not present in the provided reports.
                        * The counterfactual trace is a dependency audit, not additional disease evidence and not a vote.
                        * Do not choose a label by taking a majority across counterfactual scenarios.
                        * If a removed source changes the diagnosis, assess whether that source is reliable and corroborated by the original evidence.
                        * If cup to disc ratio is greater than 0.48, consider it suspicious and look for more signs for positive glaucoma"
                        """
                        
                        "Return exactly:\n"
                        "[LABELS]\nGLAUCOMA_DETECTED: [0 or 1]\n[/LABELS]\n\nReasoning:\n"
                        "[brief explanation that explicitly discusses CDR, corroborating optic-disc "
                        "features, image/segmentation credibility, and the RETFound-CFP score]"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"RETFound-CFP glaucoma probability: {probability}%\n\n"
                        f"CFP specialist report: {cfp_report}\n\n"
                        f"Calculated vertical cup-to-disc ratio: {cdr if cdr is not None else 'Not Available'}\n\n"
                        "Counterfactual trace: " + json.dumps(counterfactual_trace or {}, sort_keys=True)
                    ),
                },
            ],
        )
        raw = response.choices[0].message.content
        match = re.search(r"\[LABELS\](.*?)\[/LABELS\]", raw, re.DOTALL)
        labels = match.group(1).strip() if match else "GLAUCOMA_DETECTED: -1"
        return {"agent": "Drishti_CFP_Orchestrator", "decision": raw, "labels": labels}
