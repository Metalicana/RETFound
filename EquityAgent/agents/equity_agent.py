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

    def __init__(self, calibration_db_path: Optional[str] = None):
        self.deployment = "gpt-5.1"
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = "2024-12-01-preview"

        # Optional small on-disk JSON database with historical model performance
        # and demographics. This is intended to capture limitations of the
        # base imaging model (e.g., `retfound` having higher false negatives
        # in Black patients) so that downstream LLM reasoning can calibrate
        # risk estimates.
        #
        # Expected JSON format (list of records), e.g.:
        # [
        #   {
        #     "id": "P001",
        #     "race": "Black",
        #     "sex": "Female",
        #     "age": 68,
        #     "location": "Urban",
        #     "imaging_findings": "optic_disc_cupping",
        #     "ai_risk_percentage": 8.0,           # prior model estimate
        #     "true_diagnosis": "glaucoma"        # ground truth outcome
        #   },
        #   ...
        # ]
        #
        # Path can be provided directly or via environment variable
        # EQUITY_AGENT_DB_PATH. The file itself is optional.
        self.calibration_db_path: Optional[str] = (
            calibration_db_path
            or os.getenv("EQUITY_AGENT_DB_PATH")
            or os.path.join(os.path.dirname(__file__), "dummy_db.json")
        )
        self.bias_database: Optional[List[Dict[str, Any]]] = None

        if not all([self.deployment, self.azure_endpoint, self.api_key]):
            raise ValueError(
                "Missing Azure OpenAI configuration. Set AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, and AZURE_OPENAI_DEPLOYMENT."
            )

        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.azure_endpoint,
        )

    def _load_calibration_db(self) -> Optional[List[Dict[str, Any]]]:
        """Load the optional JSON calibration database, if present.

        The database should be small (for example, a few dozen records) so it
        can be safely inlined into the LLM prompt when available. If the file
        is missing or cannot be parsed, the agent simply proceeds without it.
        """
        if not self.calibration_db_path:
            return None

        if not os.path.exists(self.calibration_db_path):
            logger.info("No calibration DB found at %s; continuing without it.", self.calibration_db_path)
            return None

        try:
            with open(self.calibration_db_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                logger.warning("Calibration DB at %s is not a list; ignoring.", self.calibration_db_path)
                return None
            logger.info("Loaded calibration DB with %d records from %s", len(data), self.calibration_db_path)
            return data
        except Exception:
            logger.exception("Failed to load calibration DB from %s", self.calibration_db_path)
            return None

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

        # Lazily load the optional bias / calibration database so the LLM can
        # explicitly reason about systematic errors of the base model
        # (e.g., `retfound` having higher false-negative rates in Black
        # patients) across demographics.
        if self.bias_database is None:
            self.bias_database = self._load_calibration_db()

        patient_blob = json.dumps(patients, ensure_ascii=False)
        calibration_blob = (
            json.dumps(self.bias_database, ensure_ascii=False)
            if self.bias_database is not None
            else None
        )
        """
        system_message = {
            "role": "system",
            "content": (
                "You are a clinical decision-support assistant focused on ophthalmology. "
                "Return only valid JSON as output."
            ),
        }
        """

        # Base description explicitly references the underlying imaging model and
        # its limitations. When available, a small calibration database of past
        # predictions vs. true outcomes is also provided for subgroup-aware
        # recalibration.
        base_content = (
            "You are an equity-aware ophthalmology decision-support assistant built on top of an imaging model called 'retfound'. "
            "Retfound is known to have higher false-negative rates for glaucoma and other optic nerve diseases in Black patients, and potential calibration issues across other demographic subgroups. "
            "You are given patient records and, when available, a small JSON calibration database of prior patients that includes demographics, AI-assigned risk percentages from retfound, and the true disease diagnoses. "
            "Use this database to understand and correct for systematic errors of the base model across race, ethnicity, age, sex, and location, with particular attention to historically under-served groups. "
            "Here are current patient records (JSON array). For each, provide the requested JSON output. "
            "Patients may have fields such as id, race, sex, age, location, and imaging_findings. "
            "Pay special attention to ALL demographics (race, ethnicity, age, sex, location) when assessing risk and recommendations, considering known disparities in ophthalmology.\n\n"
            f"PATIENTS_JSON:\n{patient_blob}\n\n"
        )

        if calibration_blob is not None:
            base_content += (
                "CALIBRATION_DB_JSON (historical retfound performance with true outcomes):\n"
                f"{calibration_blob}\n\n"
            )
        else:
            base_content += (
                "CALIBRATION_DB_JSON: null (no on-disk calibration database loaded). "
                "Still, explicitly reason about known limitations of retfound, especially higher false-negative risk in Black patients, and compensate accordingly in your risk estimates.\n\n"
            )

        base_content += (
            "Rules:\n"
            "1) Explicitly adjust risk estimates and recommendations based on demographic factors AND observed systematic errors of the base model in the calibration database: for example, increase sensitivity for glaucoma in Black, Hispanic, and Asian patients when prior data show under-detection; consider higher AMD risk in White and older patients; account for socioeconomic factors via location.\n"
            "2) Where evidence from patient fields or calibration data is insufficient, state uncertainty and recommend low-risk, high-value follow-up. Suggest surrogate ways to triage risk if possible.\n"
            "3) Output must be valid JSON only; do not include extra commentary."
            "4) Instead of just qualitative labels, provide a numeric risk estimate for each relevant eye disease's progression in this patient. Give a 5-year risk probability as a percentage. Include a confidence interval if appropriate (e.g., 25–35%). Explain briefly why this numeric risk was chosen, referencing patient age, race, optic nerve findings, base-model limitations, calibration data, and missing data."
            "5) Assess all relevant eye diseases for each patient, allowing for multiple coexisting conditions, and provide separate numeric 5-year risk estimates for each disease."
            "6) Where patient fields are missing, explicitly estimate or impute numeric risk using population averages, calibration database signals, or validated surrogate measures, and adjust confidence intervals accordingly."
            "7) Output can only be 1000 characters maximum."
        )

        user_message = {
            "role": "user",
            "content": base_content,
        }
        messages=[user_message]

        # Call Azure OpenAI Chat Completions using chat completion endpoint
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
