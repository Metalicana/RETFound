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
#logging.basicConfig(level=logging.INFO)

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
                "mirage": os.path.join(base_dir, "EquityAgent/JSONs", "equity_mirage_calibration.json"),
                "retfound": os.path.join(base_dir, "EquityAgent/JSONs", "equity_retfound_calibration.json"),
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

#        logger.info(f"Loaded calibration JSON for {len(summary['models'])} models.")
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

#        system_content = (
#            "You are a Clinical Equity Auditor. Your task is to translate decimal error rates (Calibration Data) "
#            "into percentage-based diagnostic thresholds for the Orchestrator.\n\n"
#            
#            "SCALE TRANSLATION PROTOCOL:\n"
#            "- Calibration Input: Decimals (e.g., 0.15 = 15% error rate).\n"
#            "- Orchestrator Output: Percentages (e.g., 35%, 50%, or 65% threshold).\n\n"
#            
#            "CORE MISSION: You prevent model-reliability failures. Use subgroup membership only to identify "
#            "which validation-derived error priors apply; never treat demographics as direct disease evidence. "
#            "Recommend sensitivity shifts when reliable priors show high false-negative risk, and precision shifts "
#            "or down-weighting when reliable priors show high false-positive risk with low false-negative risk."
#        )
#
#        # BASE CONTENT: Incorporating the new Distribution and Pathology Signal data
#        base_content = (
#            "### TASK OVERVIEW\n"
#            "Audit the patient's AI data against empirical calibration JSONs. You must determine if a "
#            "'Sensitivity Shift' is required to prevent missing early/intermediate pathology.\n\n"
#            
#            "### DATA INPUTS\n"
#            "1) PATIENT_CONTEXT (Full Distributions & [!] TOTAL PATHOLOGY SIGNAL):\n"
#            f"{patient_blob}\n\n"
#        )
#
#        if calibration_blob is not None:
#            base_content += (
#                "2) CALIBRATION_DATA (Rates in Decimals):\n"
#                f"{calibration_blob}\n\n"
#                "INSTRUCTION: If any reliable FN rate for the patient's subgroup is > 0.15, prioritize SENSITIVITY. "
#                "If FP is high and FN is low, prioritize PRECISION instead. If estimates are unstable, fall back to global reliability.\n\n"
#            )
#        else:
#            base_content += (
#                "2) CALIBRATION_DATA: NULL.\n"
#                "ACTION: Historical reliability for this demographic is unknown. Assume high FN risk by default.\n\n"
#            )
#
#        base_content += (
#            "### RULES FOR EQUITY AUDITING\n"
#            "1) **SENSITIVITY_SHIFT (Threshold = 35%)**: \n"
#            "   - Trigger this if FN rate > 0.15.\n"
#            "   - MANDATORY: If a 35% shift is active, and the '[!] TOTAL PATHOLOGY SIGNAL' (Sum of Stages 1-3) "
#            "is > 35%, command the Orchestrator to investigate Stage 1 or 2, even if Stage 0 is the single highest probability.\n\n"
#            
#            "2) **PRECISION_SHIFT (Threshold = 65%)**: \n"
#            "   - Trigger this if FP rate > 0.15 and FN rate is low (< 0.10).\n\n"
#            
#            "3) **PRIMARY_MODEL SELECTION**: \n"
#            "   - Compare RETFOUND and MIRAGE decimal error rates. Favor the model with the lowest cumulative error "
#            "for this specific patient's race/age/gender.\n\n"
#            
#            "### REQUIRED OUTPUT FORMAT\n"
#            "[BIAS_AUDIT_REPORT]\n"
#            "- RISK_TYPE: [FN Risk / FP Risk / Minimal Risk]\n"
#            "- RECOMMENDED_THRESHOLD: [35%, 50%, or 65%]\n"
#            "- PRIMARY_MODEL: [MIRAGE or RETFOUND]\n"
#            "- ORCHESTRATOR_ADVICE: [Justify the threshold shift by citing the specific FN/FP rates and the "
#            "Total Pathology Signal found in the distribution.]\n"
#            "[/BIAS_AUDIT_REPORT]"
#        )
  
        ## FOR GLAUCOMA ONLY
        system_content = (
            "You are a Clinical Equity Auditor specializing in Glaucoma Screening. Your task is to translate "
            "decimal error rates (Calibration Data) into percentage-based diagnostic decision thresholds for the Orchestrator.\n\n"
            
            "SCALE TRANSLATION PROTOCOL:\n"
            "- Calibration Input: Decimals (e.g., 0.15 = 15% subgroup error rate).\n"
            "- Orchestrator Output: Binary Decision Percentages (e.g., 35%, 50%, or 65% threshold updates).\n\n"
            
            "CORE MISSION: You prevent model-reliability failures across diverse demographic subgroups. Glaucoma cause "
            "irreversible vision loss, meaning False Negatives carry extreme clinical risk. Use subgroup membership solely "
            "to identify which validation-derived error priors apply; never treat race, age, or gender as direct disease evidence. "
            "Recommend lowering the threshold (Sensitivity Shift) when reliable priors show a historical risk of missing glaucoma cases "
            "in this subgroup."
        )

        # BASE CONTENT: Standardized for Binary Glaucoma Probability and Demographics
        base_content = (
            "### TASK OVERVIEW\n"
            "Audit the patient's AI output scores against empirical calibration JSON data. You must determine if a "
            "'Sensitivity Shift' is required to lower the threshold and force the Orchestrator to catch borderline or early glaucoma.\n\n"
            
            "### DATA INPUTS\n"
            "1) PATIENT_CONTEXT (Demographics, RETFound/MIRAGE Disease Probabilities, and History):\n"
            f"{patient_blob}\n\n"
        )

        if calibration_blob is not None:
            base_content += (
                "2) CALIBRATION_DATA (Subgroup Error Rates in Decimals):\n"
                f"{calibration_blob}\n\n"
                "INSTRUCTION: If any validated False Negative (FN) rate for the patient's demographic subgroup is > 0.15, "
                "prioritize SENSITIVITY. If the False Positive (FP) rate is high (>0.20) and the FN rate is exceptionally "
                "low (<0.08), shift toward PRECISION to avoid over-referral. Otherwise, fall back to global calibration profiles.\n\n"
            )
        else:
            base_content += (
                "2) CALIBRATION_DATA: NULL.\n"
                "ACTION: Historical classification reliability for this demographic subgroup is unmapped. In glaucoma screening, "
                "missing disease leads to irreversible loss. Assume high FN risk by default and aggressively force sensitivity.\n\n"
            )

        base_content += (
            "### RULES FOR EQUITY AUDITING (BINARY GLAUCOMA)\n"
            "1) **SENSITIVITY_SHIFT (Decision Threshold = 35%)**: \n"
            "   - Trigger this if the subgroup FN rate > 0.15 OR if calibration data is NULL.\n"
            "   - MANDATORY LOGIC: If a 35% sensitivity shift is activated, command the Orchestrator to flag this patient "
            "     as 'Glaucoma Positive' if EITHER model (RETFound or MIRAGE) outputs a raw probability greater than 35%.\n\n"
            
            "2) **PRECISION_SHIFT (Decision Threshold = 65%)**: \n"
            "   - Trigger this ONLY if the subgroup FP rate > 0.20 and the subgroup FN rate is extremely low (< 0.08).\n"
            "   - This prevents over-taxing clinics with false alarms in historically over-flagged populations.\n\n"
            
            "3) **DEFAULT EVALUATION (Decision Threshold = 50%)**: \n"
            "   - Standard operational baseline when subgroup error dynamics are balanced and within tolerance.\n\n"
            
            "4) **PRIMARY_MODEL SELECTION**: \n"
            "   - Compare RETFound (OCT Specialist) and MIRAGE (SLO Specialist) subgroup calibration matrices. "
            "     Favor the model modality that demonstrates the lowest historical False Negative rate for this specific patient's demographic profile.\n\n"
            
            "### REQUIRED OUTPUT FORMAT\n"
            "[BIAS_AUDIT_REPORT]\n"
            "- RISK_TYPE: [FN Risk / FP Risk / Balanced Risk]\n"
            "- RECOMMENDED_THRESHOLD: [35%, 50%, or 65%]\n"
            "- PRIMARY_MODEL: [MIRAGE or RETFOUND]\n"
            "- ORCHESTRATOR_ADVICE: [Justify the adjusted decision threshold by directly citing the specific subgroup FN/FP rates "
            "and comparing the raw clinical model probabilities passed in the patient data context.]\n"
            "[/BIAS_AUDIT_REPORT]"
        )
        
        
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": base_content}
        ]
        
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
