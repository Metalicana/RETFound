import os
import json
from typing import Any, Dict, Optional

from openai import AzureOpenAI


SAFETY_AGENT_SYSTEM_PROMPT = """
You are the final Safety & Uncertainty Agent in a multi‑step AI pipeline
for general eye disease assessment (e.g., glaucoma, diabetic retinopathy,
AMD, other optic neuropathies).

Your role is not to make a new diagnosis, but to:
- Audit the consistency and safety of all prior agents' outputs.
- Identify discordances, edge cases, missing information, or sources of
  uncertainty that could cause harm if ignored.
- Recommend conservative actions when risk of harm from a missed
  diagnosis or inappropriate reassurance is non‑trivial.

You will be given a single JSON object that aggregates all previous
agents' findings, scores, and recommendations.

GENERAL PRINCIPLES
- Err on the side of patient safety and conservative management.
- Pay special attention to:
  - Discordance between structural findings (e.g., OCT, fundus) and
    functional findings (e.g., visual field, acuity).
  - Large gaps between AI confidence scores and clinical risk factors.
  - Poor image quality, missing modalities, or obvious data issues.
  - High‑risk patient contexts (e.g., advanced age, high IOP, strong
    family history, monocular patients, rapidly changing symptoms).
- You may endorse the pipeline's final label, or recommend that it be
  overridden, upgraded in severity, or deferred to human review.

OUTPUT FORMAT
- Write a short, clinically‑oriented narrative in natural language,
  3–8 sentences, that explains your safety reasoning.
- Explicitly mention any important discordances you detect.
- Be very clear when you are recommending an override.
- End with a single line in the following format, in ALL CAPS:

  SAFETY_DECISION: <ACCEPT | OVERRIDE | ESCALATE_TO_HUMAN | INSUFFICIENT_DATA>

Where:
- ACCEPT: No substantial safety concern; current final label is
  acceptable given the evidence.
- OVERRIDE: You recommend changing the pipeline's final label in a more
  conservative direction (e.g., from "healthy" to "possible disease").
- ESCALATE_TO_HUMAN: You cannot safely decide; human review is needed
  due to serious uncertainty or complexity.
- INSUFFICIENT_DATA: Data quality/availability is too poor to support
  any confident downstream decision.

Follow these rules strictly.
"""

class SafetyAgent:
    def __init__(self) -> None:
        self.deployment = "gpt-5.1"
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = "2024-12-01-preview"

        if not all([self.deployment, self.endpoint, self.api_key]):
            raise ValueError(
                "Missing Azure OpenAI configuration. Set AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, and AZURE_OPENAI_DEPLOYMENT."
            )

        self.client = AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version=self.api_version,
        )

    def run(self,pipeline_state: Dict[str, Any],*,clinician_notes: Optional[str] = None,temperature: float = 0.1,max_tokens: int = 400,) -> str:
        """Run the safety agent on the full pipeline state.

        Parameters
        ----------
        pipeline_state:
            Structured JSON‑like dict containing all upstream agents'
            findings, scores, and recommendations.
        clinician_notes:
            Optional free‑text context (e.g., brief history or comments
            from a clinician) that should inform safety decisions.
        temperature:
            Sampling temperature for the model (keep low for stability).
        max_tokens:
            Maximum tokens in the safety agent's response.

        Returns
        -------
        str
            Safety narrative ending with a SAFETY_DECISION line.
        """

        user_instructions = {
            "task": "Final safety and uncertainty audit of the eye-disease pipeline output.",
            "instructions": """
You are given the aggregated JSON output of all previous AI agents in
the pipeline. Carefully read all sections (raw measurements, image
quality, risk scores, intermediate labels, and final recommendation).

Identify:
- Any discordance between structural and functional findings.
- Any mismatch between AI confidence and clinical risk context.
- Any reasons to distrust the data (artifacts, missing modalities,
  out-of-range values, etc.).

Then, produce a concise narrative safety assessment and end with one
SAFETY_DECISION line as specified in your system prompt.
""".strip(),
        }

        if clinician_notes:
            user_instructions["clinician_notes"] = clinician_notes

        messages = [
            {"role": "system", "content": SAFETY_AGENT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Safety audit instructions and metadata:\n" +
                    json.dumps(user_instructions, indent=2)
                ),
            },
            {
                "role": "user",
                "content": (
                    "Aggregated pipeline JSON from previous agents:\n" +
                    json.dumps(pipeline_state, indent=2)
                ),
            },
        ]

        response = self.client.chat.completions.create(
            model=self.deployment,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )

        content = response.choices[0].message.content or ""
        return content.strip()
