import json
import os
import re

from dotenv import load_dotenv
from openai import AzureOpenAI


load_dotenv()


class DrishtiOrchestrator:
    def __init__(self, model_client=None):
        self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.1")
        self.model_client = model_client or AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        )

    def analyze(self, probability, cfp_report, cdr, counterfactual_trace=None):
        response = self.model_client.chat.completions.create(
            model=self.deployment,
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are the final glaucoma diagnostic orchestrator for a CFP-only pipeline. "
                        "The fine-tuned RETFound-CFP head is known to be under-sensitive and may output "
                        "artificially low glaucoma probabilities for positive cases. Therefore, do not let "
                        "a low RETFound-CFP probability automatically override convincing optic-disc "
                        "evidence. Give the calculated vertical cup-to-disc ratio and corroborating CFP "
                        "optic-disc features substantial diagnostic weight. A vertical CDR above 0.7 is "
                        "strongly suspicious when the segmentation and disc image are credible; a CDR from "
                        "0.6 to 0.7 is concerning when accompanied by rim thinning, notching, vessel changes, "
                        "or disc hemorrhage. A smaller CDR is reassuring but does not alone exclude glaucoma. "
                        "CDR can be influenced by physiologic disc size and segmentation error, so check it "
                        "against image quality and the CFP specialist's independent observations. Treat a "
                        "missing CDR as unavailable evidence, not a normal result. Use RETFound-CFP as an "
                        "additional signal rather than an absolute gate. The counterfactual trace is a "
                        "dependency audit, not a vote. Do not invent OCT, SLO, demographic, clinical-history, "
                        "or fellow-eye evidence. Return exactly:\n"
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
