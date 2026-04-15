from __future__ import annotations

import json
import logging
import os
from typing import Any, Literal, Optional

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

    def __init__(self, deployment: Optional[str] = None):
        self.deployment = deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.1")
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("AZURE_OPENAI_API_BASE")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

        if not all([self.azure_endpoint, self.api_key]):
            raise ValueError(
                "Missing Azure OpenAI configuration. Set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT (or AZURE_OPENAI_API_BASE)."
            )

        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.azure_endpoint,
        )

        base_dir = os.path.dirname(os.path.dirname(__file__))
        self.calibration_summary = self._load_calibration_jsons(
            {
                "mirage": os.path.join(base_dir, "model_specific_data", "equity_mirage_calibration.json"),
                "retfound": os.path.join(base_dir, "model_specific_data", "equity_retfound_calibration.json"),
            }
        )

    def _load_calibration_jsons(self, json_paths: dict[str, str]) -> dict[str, Any]:
        summary: dict[str, Any] = {"models": {}}
        for model_name, path in json_paths.items():
            if not os.path.exists(path):
                logger.warning(f"Calibration JSON not found: {path}")
                continue
            with open(path, encoding="utf-8") as file_handle:
                summary["models"][model_name] = json.load(file_handle)

        logger.info(f"Loaded calibration JSON for {len(summary['models'])} models.")
        return summary

    def _format_patient_input(self, patients: Any) -> str:
        if isinstance(patients, str):
            return patients.strip()
        if isinstance(patients, list):
            sections = []
            for index, patient in enumerate(patients, start=1):
                if isinstance(patient, str):
                    body = patient.strip()
                else:
                    body = json.dumps(patient, ensure_ascii=False)
                sections.append(f"PATIENT_{index}:\n{body}")
            return "\n\n".join(sections)
        return str(patients)

    def analyze_patients(self, patients: Any, output_format: Literal["json", "text"] = "json") -> Any:
        """Send patient summaries to the LLM and return either JSON or concise text."""
        import re

        if output_format not in {"json", "text"}:
            raise ValueError("output_format must be 'json' or 'text'.")

        patient_blob = self._format_patient_input(patients)
        calibration_blob = (
            json.dumps(self.calibration_summary, ensure_ascii=False)
            if self.calibration_summary is not None
            else None
        )

        base_content = (
            "You are an equity-aware ophthalmology decision-support assistant that integrates outputs from two primary vision models: mirage and retfound. "
            "The patient input is a freeform paragraph per patient and may include demographics, imaging findings attributed to mirage, imaging findings attributed to retfound, and findings from other upstream models or clinical context. "
            "Extract the relevant patient facts from each paragraph, including demographics, symptoms, imaging findings, model-specific concerns, and any prior model outputs. "
            "You are also given precomputed JSON calibration data for mirage and retfound. Each model includes false_positive and false_negative rates by disease, race, gender, and age_group. "
            "Use the calibration summary to reason about systematic error by model, disease, race, gender, and age group. If the patient paragraph mentions location or access-to-care context, use that as additional clinical and equity context, but do not invent calibration data that is not present in the JSON summary. "
            "Here are the current patient summaries:\n\n"
            f"PATIENTS_INPUT:\n{patient_blob}\n\n"
        )

        if calibration_blob is not None:
            base_content += (
                "CALIBRATION_SUMMARY_JSON (loaded from precomputed mirage and retfound calibration JSON files):\n"
                f"{calibration_blob}\n\n"
            )
        else:
            base_content += (
                "CALIBRATION_SUMMARY_JSON: null (no calibration JSON data could be loaded). "
                "Still, explicitly reason about likely calibration limitations across mirage and retfound and compensate conservatively in your risk estimates.\n\n"
            )

        base_content += (
            "Rules:\n"
            "1) Explicitly compare mirage and retfound findings when both are present. Use the calibration summary to up-weight or down-weight each model's concern based on the disease and the patient's demographic profile.\n"
            "2) Use disease-specific reasoning. Do not treat mirage or retfound predictions for unrelated diseases as evidence for the current disease.\n"
            "3) Where calibration data shows higher false-negative risk for a subgroup, increase sensitivity and recommend appropriate follow-up rather than dismissing the finding.\n"
            "4) Where evidence from patient fields or calibration data is insufficient, state uncertainty and recommend low-risk, high-value follow-up. Suggest surrogate ways to triage risk if possible.\n"
            "5) Assess only the most relevant eye diseases for each patient and keep the full response under 1200 characters.\n"
        )

        if output_format == "json":
            base_content += (
                "6) Output must be valid JSON only and no prose outside the JSON. Return an array with one object per patient. Each object must contain patient_id and disease_summaries.\n"
                "7) disease_summaries must be an array of short objects with disease, rationale, confidence, and recommended_actions. Keep rationale and recommended_actions brief."
            )
        else:
            base_content += (
                "6) Output must be plain text only, not JSON or markdown.\n"
                "7) Write one short paragraph per patient. Start with the patient ID, then summarize the main likely disease concern, key evidence from mirage and retfound, the equity-aware calibration caveat if relevant, confidence, and the most important next step."
            )

        user_message = {
            "role": "user",
            "content": base_content,
        }
        messages = [user_message]

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
        if isinstance(assistant_text, list):
            assistant_text = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in assistant_text
            )

        assistant_text = assistant_text.strip()

        if output_format == "text":
            return assistant_text

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
