from __future__ import annotations

import json
import os
import logging
from typing import List, Dict, Any, Optional

try:
    from openai import AzureOpenAI
except Exception as e:
    raise ImportError("openai package is required. Install via `pip install openai`") from e

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class EquityAgent:
    """Agent that queries an Azure OpenAI LLM and applies equity-aware prompting.

    It creates a single prompt that contains patient data and explicit instructions
    about known demographic discrepancies (for example higher false negatives
    for glaucoma in Black patients using `retfound` base model) and asks the LLM
    to compensate accordingly in risk estimates and recommendations.
    """

    def __init__(self):
        self.deployment = "gpt-5.1"
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = "2024-12-01-preview"

        if not all([self.deployment, self.azure_endpoint, self.api_key]):
            raise ValueError(
                "Missing Azure OpenAI configuration. Set AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, and AZURE_OPENAI_DEPLOYMENT."
            )

        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.azure_endpoint,
        )

    def analyze_patients(self, patients: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Send patients to the LLM and return equity-aware recommendations.

        The assistant is asked to return strict JSON: an array of objects with keys:
        - `id` : original patient id (if any)
        - `diagnosis` : high-level likely findings
        - `risk_score` : 0..1 clinical risk probability (float)
        - `adjusted_for_equity` : boolean
        - `notes` : short text explaining adjustments

        """
        # Restore: full prompt and all patients, but keep system/user split and temp 0.0
        import re
        patient_blob = json.dumps(patients, ensure_ascii=False)
        """
        system_message = {
            "role": "system",
            "content": (
                "You are a clinical decision-support assistant focused on ophthalmology. "
                "Return only valid JSON as output."
            ),
        }
        """

        user_message = {
            "role": "user",
            "content": (
                "Here are patient records (JSON array). For each, provide the requested JSON output. "
                "Patients may have fields such as id, race, sex, age, location, and imaging_findings. "
                "Pay special attention to **ALL** demographics (race, ethnicity, age, sex, location) when assessing risk and recommendations, considering known disparities in ophthalmology.\n\n"
                f"PATIENTS_JSON:\n{patient_blob}\n\n"
                "Rules:\n"
                "1) Adjust risk estimates and recommendations based on demographic factors: e.g., increase sensitivity for glaucoma in Black, Hispanic, and Asian patients; consider higher AMD risk in White and older patients; account for socioeconomic factors via location.\n"
                "2) Where evidence from patient fields is insufficient, state uncertainty and recommend low-risk, high-value follow-up. Suggest surrogate ways to triage risk if possible.\n"
                "3) Output must be valid JSON only; do not include extra commentary."
                "4) Instead of just qualitative labels, provide a **numeric risk estimate** for their respective eye disease's progression in this patient. Give a **5-year risk probability** as a percentage. Include a **confidence interval** if appropriate (e.g., 25–35%). Explain briefly **why this numeric risk was chosen**, referencing patient age, race, optic nerve findings, and missing data."
                "5) Assess all relevant eye diseases for each patient, allowing for multiple coexisting conditions, and provide separate numeric 5-year risk estimates for each disease."
                "6) Where patient fields are missing, explicitly estimate or impute numeric risk using population averages or validated surrogate measures, and adjust confidence intervals accordingly."
                "Output can only be 2000 characters maximum"
            ),
        }
        messages=[user_message]

        try:
            resp = self.client.chat.completions.create(
                model=self.deployment,
                messages=messages,
                temperature=0.0,
                max_completion_tokens=800,
            )
        except Exception as exc:
            logger.exception("LLM call failed")
            raise

        assistant_text = resp.choices[0].message.content

        try:
            return json.loads(assistant_text)
        except Exception:
            m = re.search(r'(\[.*\])', assistant_text, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(1))
                except Exception:
                    pass
            return assistant_text
