"""Final GPT-5.1 orchestrator for GDP OCT + RNFLT testing."""

import json
import os
import re

from dotenv import load_dotenv
from openai import AzureOpenAI


load_dotenv()


class GDPOrchestrator:
    def __init__(self, model_client=None):
        self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.1")
        self.client = model_client or AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        )

    def analyze(self, demographics, probability, oct_report, rnflt_report, rnflt_statistics):
        response = self.client.chat.completions.create(
            model=self.deployment,
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are the final glaucoma diagnostic orchestrator for GDP. Integrate the "
                        "RETFound-OCT glaucoma probability, independent OCT visual report, RNFLT thickness-map "
                        "report, RNFLT summary statistics, and basic demographics. Imaging is the diagnostic "
                        "evidence. Age, gender, and race are contextual metadata only and must not be treated "
                        "as anatomical proof of glaucoma. Require coherent structural evidence when overriding "
                        "a strong RETFound score, explicitly discuss agreement or discordance between OCT and "
                        "RNFLT, and treat poor or missing evidence as uncertainty rather than normality. Do not "
                        "invent SLO, CFP, CDR, intraocular pressure, symptoms, history, or visual-field findings. "
                        "Return exactly:\n[LABELS]\nGLAUCOMA_DETECTED: [0 or 1]\n[/LABELS]\n\n"
                        "Reasoning:\n[brief explanation referencing RETFound-OCT, OCT observations, RNFLT, "
                        "and uncertainty]"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Demographics: {json.dumps(demographics, sort_keys=True)}\n\n"
                        f"RETFound-OCT glaucoma probability: {probability}%\n\n"
                        f"OCT specialist report: {oct_report}\n\n"
                        f"RNFLT specialist report: {rnflt_report}\n\n"
                        f"RNFLT summary statistics: {json.dumps(rnflt_statistics, sort_keys=True)}"
                    ),
                },
            ],
        )
        raw = response.choices[0].message.content or ""
        match = re.search(r"\[LABELS\](.*?)\[/LABELS\]", raw, re.DOTALL)
        labels = match.group(1).strip() if match else "GLAUCOMA_DETECTED: -1"
        return {"decision": raw, "labels": labels}
